"""Complete ARP-CBCT architecture used in the reported experiments."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import get_default_config
from .geometry import ProjectionGeometry, ReconstructionGrid
from .models.decoder import BalancedHierarchicalDecoder3D
from .models.encoder import MultiScaleEncoder2D
from .models.fusion import PhysicsDecoderFusion
from .models.geometry_conditioning import GeometryFiLM
from .models.multiscale_splatting import MultiScaleSplatOutput, PrimitiveMultiScaleSplatting
from .models.primitive_attention import GeometryAwareAttention
from .models.primitive_interaction import PrimitiveInteraction
from .models.primitive_proposal import RayPrimitiveProposal
from .models.primitive_query import MultiScalePrimitiveQuery
from .models.primitive_refinement import PrimitiveRefinement


class ARPCBCT(nn.Module):
    """Adaptive Ray Primitive Representation for sparse-view CBCT.

    Projection tensors have shape ``[B, V, 1, H, W]``. Volume arrays use
    z-y-x indexing; physical vectors use world x-y-z coordinates in mm.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        cfg = get_default_config() if config is None else config
        channels = cfg["encoder_channels"]
        primitive_dim = int(cfg["primitive_dim"])
        hypotheses = cfg.get("depth_hypotheses", [0.2, 0.5, 0.8])
        max_delta = cfg.get("max_delta_alpha", 0.1)
        confidence_min = cfg.get("confidence_min", 0.1)

        self.projection_input_scale = float(cfg.get("projection_input_scale", 32.0))
        self.encoder = MultiScaleEncoder2D(
            channels, cfg.get("encoder_blocks_per_stage", 2), cfg.get("encoder_view_chunk_size")
        )
        self.film = GeometryFiLM(channels, cfg.get("geometry_embed_dim", 96))
        self.proposal = RayPrimitiveProposal(
            channels[1], primitive_dim, cfg["primitive_budget"], cfg["sigma_min_mm"],
            cfg["sigma_max_mm"], hypotheses, max_delta,
            tuple(cfg.get("detector_strata_hw", [8, 16])), cfg.get("max_rays_per_cell", 2),
            cfg.get("sigma_base_mm", 4.0), cfg.get("support_log_scale", 0.75),
            confidence_min, cfg.get("sigma_min_spacing_multiplier", 1.0),
            cfg.get("support_mode", "adaptive"), cfg.get("support_perpendicular_multiplier", 1.0),
            cfg.get("support_parallel_interval_multiplier", 0.25), cfg.get("density_scale_mm_inv", 1.0),
        )
        feature_channels = {f"s{i + 1}": c for i, c in enumerate(channels)}
        self.query = MultiScalePrimitiveQuery(feature_channels, cfg["query_scales"], primitive_dim)
        self.attention = GeometryAwareAttention(primitive_dim)
        self.interaction = PrimitiveInteraction(
            primitive_dim, cfg["interaction_k"], cfg["knn_method"], cfg["knn_chunk_size"]
        )
        self.refine = PrimitiveRefinement(
            primitive_dim, max_delta, cfg["sigma_min_mm"], cfg["sigma_max_mm"],
            cfg.get("support_refine_log_scale", 0.25), confidence_min,
            cfg.get("support_mode", "adaptive"), cfg.get("support_residual_log_scale", 0.6931471805599453),
        )

        coarse_channels = int(cfg.get("coarse_channels", primitive_dim))
        mid_channels = int(cfg.get("mid_channels", 64))
        self.splat = PrimitiveMultiScaleSplatting(
            primitive_dim=primitive_dim,
            coarse_channels=coarse_channels,
            mid_channels=mid_channels,
            coarse_shape_zyx=tuple(cfg.get("coarse_shape_zyx", (36, 64, 64))),
            mid_shape_zyx=tuple(cfg.get("mid_shape_zyx", (72, 128, 128))),
            chunk_size=int(cfg["splat_chunk_size"]),
            max_chunk_voxels=int(cfg.get("splat_max_chunk_voxels", 2_000_000)),
        )
        self.decoder = BalancedHierarchicalDecoder3D(
            coarse_channels=coarse_channels,
            mid_channels=mid_channels,
            high_channels=int(cfg.get("high_channels", 24)),
            coarse_blocks=int(cfg.get("coarse_blocks", 10)),
            mid_blocks=int(cfg.get("mid_blocks", 4)),
            high_blocks=int(cfg.get("high_blocks", 2)),
            gradient_checkpointing=bool(cfg.get("decoder_gradient_checkpointing", False)),
            max_fullres_channels=int(cfg.get("max_fullres_channels", 32)),
        )
        decoder_amp_dtype = cfg.get("decoder_amp_dtype", "bfloat16")
        if decoder_amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError(f"Unsupported decoder_amp_dtype: {decoder_amp_dtype}")
        self.decoder_amp_dtype = getattr(torch, decoder_amp_dtype)
        self.decoder_force_fp32 = bool(cfg.get("decoder_force_fp32", False))
        self.fusion = PhysicsDecoderFusion(
            cfg.get("output_softplus_beta", 10.0),
            cfg.get("output_activation", "softplus"),
            cfg.get("output_residual_scale", 1.0),
        )
        self.debug_assertions = bool(cfg.get("debug_assertions", True))

    @staticmethod
    def _physics_and_coverage(
        splat: MultiScaleSplatOutput,
        output_shape_zyx: tuple[int, int, int],
        epsilon: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coarse_density = F.interpolate(
            splat.coarse_density, size=splat.mid_density.shape[-3:], mode="trilinear", align_corners=False
        )
        coarse_weight = F.interpolate(
            splat.coarse_weight, size=splat.mid_weight.shape[-3:], mode="trilinear", align_corners=False
        )
        combined_weight = coarse_weight + splat.mid_weight
        combined_density = (
            coarse_density * coarse_weight + splat.mid_density * splat.mid_weight
        ) / (combined_weight + epsilon)
        physics = F.interpolate(combined_density, size=output_shape_zyx, mode="trilinear", align_corners=False)
        coverage = F.interpolate(combined_weight, size=output_shape_zyx, mode="trilinear", align_corners=False)
        return physics, coverage

    def forward(
        self,
        projections: torch.Tensor,
        view_mask: torch.Tensor,
        geometry: ProjectionGeometry,
        grid: ReconstructionGrid,
    ) -> dict[str, Any]:
        """Reconstruct a dense volume from sparse cone-beam projections."""
        if projections.ndim != 5 or projections.shape[:2] != view_mask.shape:
            raise ValueError("Expected projections [B,V,1,H,W] and view_mask [B,V]")

        geometry = geometry.to(projections.device, projections.dtype)
        features = self.film(
            self.encoder(projections / self.projection_input_scale), geometry, grid.center_world_xyz_mm
        )
        proposal_primitives = self.proposal(features["s2"], geometry, grid, view_mask)
        view_features, query_metadata = self.query(proposal_primitives, features, geometry, view_mask)
        fused, weights = self.attention(proposal_primitives, view_features, geometry, query_metadata["valid"])
        interacted = self.interaction(proposal_primitives.updated(feature=proposal_primitives.feature + fused))
        primitives = self.refine(interacted)

        splat = self.splat(primitives, grid)

        if self.decoder_force_fp32:
            fp32_splat = replace(
                splat,
                coarse_feature=splat.coarse_feature.float(), coarse_density=splat.coarse_density.float(),
                coarse_weight=splat.coarse_weight.float(), mid_feature=splat.mid_feature.float(),
                mid_density=splat.mid_density.float(), mid_weight=splat.mid_weight.float(),
            )
            with torch.cuda.amp.autocast(enabled=False):
                residual, decoder_shapes = self.decoder(fp32_splat, grid.shape_zyx)
                physics, coverage = self._physics_and_coverage(fp32_splat, grid.shape_zyx)
        else:
            with torch.cuda.amp.autocast(
                enabled=torch.is_autocast_enabled() and splat.coarse_feature.is_cuda,
                dtype=self.decoder_amp_dtype,
            ):
                residual, decoder_shapes = self.decoder(splat, grid.shape_zyx)
                physics, coverage = self._physics_and_coverage(splat, grid.shape_zyx)

        fusion_preactivation = self.fusion.preactivation(physics, residual)
        with torch.cuda.amp.autocast(enabled=False):
            volume = self.fusion.activate(fusion_preactivation)

        ray_delta = primitives.position - primitives.ray_origin
        ray_deviation = torch.linalg.vector_norm(
            ray_delta - (ray_delta * primitives.ray_direction).sum(-1, keepdim=True) * primitives.ray_direction,
            dim=-1,
        )
        box_min, box_max = grid.bounds_xyz(projections.device, projections.dtype)
        fov_violation_count = (
            ((primitives.position < box_min - 2e-3) | (primitives.position > box_max + 2e-3)).any(-1).sum()
        )
        source_mask = (
            torch.arange(projections.shape[1], device=projections.device)[None, :, None]
            == primitives.source_view[:, None]
        )
        source_attention = (weights * source_mask).sum(1)
        non_source_attention = (weights * ~source_mask).sum(1)

        if self.debug_assertions:
            if not torch.isfinite(residual).all() or not torch.isfinite(volume).all():
                raise FloatingPointError("Non-finite ARP-CBCT volume output")
            if fov_violation_count:
                raise AssertionError("Primitive outside common reconstruction FOV")
            if ray_deviation.max() > 2e-3:
                raise AssertionError(f"Primitive left its measurement ray: {ray_deviation.max().item():.6g} mm")
            if primitives.confidence.min() + 1e-6 < self.proposal.confidence_min:
                raise AssertionError("Primitive confidence crossed its configured lower bound")
            for shape in decoder_shapes.values():
                if shape[1] >= 64 and tuple(shape[-3:]) == tuple(grid.shape_zyx):
                    raise AssertionError(f"Forbidden full-resolution high-channel tensor: {shape}")

        output: dict[str, Any] = {
            "volume": volume,
            "physics_volume": physics,
            "decoder_residual": residual,
            "decoder_residual_physical": self.fusion.residual_scale * residual,
            "fusion_preactivation": fusion_preactivation,
            "primitive_positions": primitives.position,
            "primitive_support": primitives.support,
            "primitive_support_basis": primitives.support_basis,
            "primitive_support_base": primitives.support_base,
            "primitive_support_scale": primitives.support_scale,
            "primitive_confidence": primitives.confidence,
            "coverage": coverage,
            "coarse_coverage": splat.coarse_weight,
            "mid_coverage": splat.mid_weight,
            "primitive_latent": splat.mid_feature,
            "primitive_latent_coarse": splat.coarse_feature,
            "primitive_latent_mid": splat.mid_feature,
            "decoder_activation_shapes": decoder_shapes,
            "primitives": primitives,
            "proposal_primitives": proposal_primitives,
            "post_interaction_primitives": interacted,
            "interaction_knn_indices": self.interaction.last_indices,
            "attention": weights,
            "source_attention_mean": source_attention.mean(),
            "non_source_attention_mean": non_source_attention.mean(),
            "sigma_min_mm": projections.new_tensor(self.proposal.sigma_min_mm),
            "sigma_max_mm": projections.new_tensor(self.proposal.sigma_max_mm),
            "query_metadata": query_metadata,
            "primitive_ray_deviation_max_mm": ray_deviation.max(),
            "primitive_fov_violation_count": fov_violation_count,
        }
        output.update(self.decoder.last_balance_diagnostics)
        return output


def build_model(config: dict[str, Any] | None = None) -> ARPCBCT:
    """Build the complete model used for the reported experiments."""
    cfg = get_default_config() if config is None else config
    architecture = cfg.get("architecture", "arp_cbct")
    if architecture != "arp_cbct":
        raise ValueError(f"Unsupported architecture: {architecture!r}")
    return ARPCBCT(cfg)


ARP_CBCT = ARPCBCT
