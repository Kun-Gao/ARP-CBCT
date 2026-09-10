from __future__ import annotations

import torch
from torch import nn

from ..geometry import ProjectionGeometry, ReconstructionGrid
from .decoder3d import Decoder3D
from .encoder2d import MultiScaleEncoder2D
from .geometry_conditioning import GeometryFiLM
from .physics_branch import PhysicsDecoderFusion
from .primitive_attention import GeometryAwareAttention
from .primitive_interaction import PrimitiveInteraction
from .primitive_proposal_v2 import StructuredRayPrimitiveProposalV2
from .primitive_query import MultiScalePrimitiveQuery
from .primitive_refinement_v2 import StructuredPrimitiveRefinementV2
from .primitive_splatting import PrimitiveSplatting


class ARPCBCTV2(nn.Module):
    """Structured hypotheses: Projection -> Ray -> Primitive -> Interaction -> Volume."""

    def __init__(self, cfg: dict):
        super().__init__()
        channels = cfg["encoder_channels"]
        dim = cfg["primitive_dim"]
        hypotheses = cfg.get("depth_hypotheses", [0.2, 0.5, 0.8])
        max_delta = cfg.get("max_delta_alpha", 0.1)
        confidence_min = cfg.get("confidence_min", 0.1)
        self.projection_input_scale = float(cfg.get("projection_input_scale", 32.0))
        self.encoder = MultiScaleEncoder2D(
            channels,
            cfg.get("encoder_blocks_per_stage", 2),
            cfg.get("encoder_view_chunk_size"),
        )
        self.film = GeometryFiLM(channels, cfg.get("geometry_embed_dim", 96))
        self.proposal = StructuredRayPrimitiveProposalV2(
            channels[1], dim, cfg["primitive_budget"], cfg["sigma_min_mm"], cfg["sigma_max_mm"], hypotheses,
            max_delta, tuple(cfg.get("detector_strata_hw", [8, 16])), cfg.get("max_rays_per_cell", 2),
            cfg.get("sigma_base_mm", 4.0), cfg.get("support_log_scale", 0.75), confidence_min,
            cfg.get("sigma_min_spacing_multiplier", 1.0), cfg.get("support_mode", "adaptive"),
            cfg.get("support_perpendicular_multiplier", 1.0), cfg.get("support_parallel_interval_multiplier", 0.25),
            cfg.get("density_scale_mm_inv", 1.0),
        )
        self.query = MultiScalePrimitiveQuery({f"s{i + 1}": c for i, c in enumerate(channels)}, cfg["query_scales"], dim)
        self.attention = GeometryAwareAttention(dim)
        self.interaction = PrimitiveInteraction(dim, cfg["interaction_k"], cfg["knn_method"], cfg["knn_chunk_size"])
        self.refine = StructuredPrimitiveRefinementV2(
            dim, max_delta, cfg["sigma_min_mm"], cfg["sigma_max_mm"], cfg.get("support_refine_log_scale", 0.25), confidence_min,
            cfg.get("support_mode", "adaptive"),
            cfg.get("support_residual_log_scale", 0.6931471805599453),
        )
        self.splat = PrimitiveSplatting(cfg["splat_chunk_size"], max_chunk_voxels=cfg.get("splat_max_chunk_voxels", 2_000_000))
        self.decoder = Decoder3D(
            dim,
            cfg.get("decoder_base_channels", 48),
            cfg.get("decoder_bottleneck_blocks", 6),
            cfg.get("decoder_gradient_checkpointing", True),
        )
        decoder_amp_dtype = cfg.get("decoder_amp_dtype", "bfloat16")
        if decoder_amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError(f"Unsupported decoder_amp_dtype: {decoder_amp_dtype}")
        self.decoder_amp_dtype = getattr(torch, decoder_amp_dtype)
        self.fusion = PhysicsDecoderFusion(
            cfg.get("output_softplus_beta", 10.0),
            cfg.get("output_activation", "softplus"),
            cfg.get("output_residual_scale", 1.0),
        )
        self.debug_assertions = bool(cfg.get("debug_assertions", True))

    def forward(self, projections: torch.Tensor, view_mask: torch.Tensor, geometry: ProjectionGeometry, grid: ReconstructionGrid) -> dict:
        if projections.ndim != 5 or projections.shape[:2] != view_mask.shape:
            raise ValueError("Expected projections [B,V,1,H,W] and view_mask [B,V]")
        geometry = geometry.to(projections.device, projections.dtype)
        features = self.film(self.encoder(projections / self.projection_input_scale), geometry)
        proposal_primitives = self.proposal(features["s2"], geometry, grid, view_mask)
        view_features, query_metadata = self.query(proposal_primitives, features, geometry, view_mask)
        fused, weights = self.attention(proposal_primitives, view_features, geometry, query_metadata["valid"])
        interacted = self.interaction(proposal_primitives.updated(feature=proposal_primitives.feature + fused))
        primitives = self.refine(interacted)
        latent, physics, coverage = self.splat(primitives, grid)
        # The high-resolution 3D decoder can exceed FP16's finite range after
        # otherwise valid optimizer updates. BF16 retains 16-bit activation
        # storage while providing FP32-like exponent range.
        with torch.cuda.amp.autocast(
            enabled=torch.is_autocast_enabled() and latent.is_cuda,
            dtype=self.decoder_amp_dtype,
        ):
            residual = self.decoder(latent)
        volume = self.fusion(physics, residual)

        ray_delta = primitives.position - primitives.ray_origin
        ray_deviation = torch.linalg.vector_norm(
            ray_delta - (ray_delta * primitives.ray_direction).sum(-1, keepdim=True) * primitives.ray_direction, dim=-1
        )
        box_min, box_max = grid.bounds_xyz(projections.device, projections.dtype)
        fov_violation_count = ((primitives.position < box_min - 2e-3) | (primitives.position > box_max + 2e-3)).any(-1).sum()
        source_mask = torch.arange(projections.shape[1], device=projections.device)[None, :, None] == primitives.source_view[:, None]
        source_attention = (weights * source_mask).sum(1)
        non_source_attention = (weights * ~source_mask).sum(1)
        if self.debug_assertions:
            if not torch.isfinite(residual).all():
                raise FloatingPointError(f"Non-finite CORE_V2 decoder residual ({residual.dtype})")
            if not torch.isfinite(volume).all() or not torch.isfinite(primitives.position).all():
                raise FloatingPointError("Non-finite CORE_V2 forward output")
            if fov_violation_count:
                raise AssertionError("Primitive outside common reconstruction FOV")
            if ray_deviation.max() > 2e-3:
                raise AssertionError(f"Primitive left its measurement ray: {ray_deviation.max().item():.6g} mm")
            if primitives.confidence.min() + 1e-6 < self.proposal.confidence_min:
                raise AssertionError("Primitive confidence crossed the configured V2 lower bound")
        return {
            "volume": volume,
            "physics_volume": physics,
            "decoder_residual": residual,
            "primitive_positions": primitives.position,
            "primitive_support": primitives.support,
            "primitive_support_basis": primitives.support_basis,
            "primitive_support_base": primitives.support_base,
            "primitive_support_scale": primitives.support_scale,
            "primitive_confidence": primitives.confidence,
            "coverage": coverage,
            "primitive_latent": latent,
            "primitives": primitives,
            "proposal_primitives": proposal_primitives,
            "post_interaction_primitives": interacted,
            "attention": weights,
            "source_attention_mean": source_attention.mean(),
            "non_source_attention_mean": non_source_attention.mean(),
            "sigma_min_mm": projections.new_tensor(self.proposal.sigma_min_mm),
            "sigma_max_mm": projections.new_tensor(self.proposal.sigma_max_mm),
            "query_metadata": query_metadata,
            "primitive_ray_deviation_max_mm": ray_deviation.max(),
            "primitive_fov_violation_count": fov_violation_count,
        }
