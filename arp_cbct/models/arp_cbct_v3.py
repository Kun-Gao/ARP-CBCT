from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from ..geometry import ProjectionGeometry, ReconstructionGrid
from .arp_cbct_v2 import ARPCBCTV2
from .hierarchical_decoder3d import HierarchicalDecoder3D
from .multiscale_splatting import MultiScaleSplatOutput, PrimitiveMultiScaleSplatting


class ARPCBCTV3(ARPCBCTV2):
    """V3-A: unchanged V2 ray primitives with multi-scale volume recovery."""

    REPRESENTATION_MODES = {
        "NORMAL",
        "ZERO_PRIMITIVES",
        "SHUFFLED_PRIMITIVES",
        "ZERO_MID_SPLAT",
        "ZERO_COARSE_SPLAT",
        "ZERO_PRIMITIVE_FEATURE",
        "SHUFFLED_PRIMITIVE_FEATURE",
        "PHYSICS_ONLY",
        "DECODER_ONLY",
        "ZERO_COARSE",
        "ZERO_MID",
        "SHUFFLE_MID_SPLAT",
        "SHUFFLE_COARSE_SPLAT",
    }

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        primitive_dim = int(cfg["primitive_dim"])
        coarse_channels = int(cfg.get("v3_coarse_channels", primitive_dim))
        mid_channels = int(cfg.get("v3_mid_channels", 64))
        self.splat = PrimitiveMultiScaleSplatting(
            primitive_dim=primitive_dim,
            coarse_channels=coarse_channels,
            mid_channels=mid_channels,
            coarse_shape_zyx=tuple(cfg.get("v3_coarse_shape_zyx", (36, 64, 64))),
            mid_shape_zyx=tuple(cfg.get("v3_mid_shape_zyx", (72, 128, 128))),
            chunk_size=int(cfg["splat_chunk_size"]),
            max_chunk_voxels=int(cfg.get("splat_max_chunk_voxels", 2_000_000)),
        )
        self.decoder = HierarchicalDecoder3D(
            coarse_channels=coarse_channels,
            mid_channels=mid_channels,
            high_channels=int(cfg.get("v3_high_channels", 24)),
            coarse_blocks=int(cfg.get("v3_coarse_blocks", 10)),
            mid_blocks=int(cfg.get("v3_mid_blocks", 4)),
            high_blocks=int(cfg.get("v3_high_blocks", 2)),
            gradient_checkpointing=bool(cfg.get("decoder_gradient_checkpointing", False)),
            max_fullres_channels=int(cfg.get("v3_max_fullres_channels", 32)),
        )
        # The full-resolution 3D recovery path has a substantially larger
        # activation range than the 2D/primitive path.  Keeping only this
        # branch in FP32 avoids late-training FP16 overflows while preserving
        # AMP for the majority of the model.
        self.decoder_force_fp32 = bool(cfg.get("decoder_force_fp32", False))

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

    @staticmethod
    def _representation_primitives(primitives, mode: str):
        if mode == "NORMAL" or mode in {
            "ZERO_MID_SPLAT", "ZERO_COARSE_SPLAT", "ZERO_COARSE", "ZERO_MID", "PHYSICS_ONLY", "DECODER_ONLY",
            "SHUFFLE_MID_SPLAT", "SHUFFLE_COARSE_SPLAT"
        }:
            return primitives
        if mode == "ZERO_PRIMITIVE_FEATURE":
            return primitives.updated(feature=torch.zeros_like(primitives.feature))
        if mode == "SHUFFLED_PRIMITIVE_FEATURE":
            return primitives.updated(feature=torch.roll(primitives.feature, shifts=1, dims=1))
        if mode == "ZERO_PRIMITIVES":
            return primitives.updated(
                feature=torch.zeros_like(primitives.feature),
                density=torch.zeros_like(primitives.density),
                confidence=torch.zeros_like(primitives.confidence),
            )
        if mode == "SHUFFLED_PRIMITIVES":
            return primitives.updated(
                feature=torch.roll(primitives.feature, shifts=1, dims=1),
                density=torch.roll(primitives.density, shifts=1, dims=1),
                confidence=torch.roll(primitives.confidence, shifts=1, dims=1),
            )
        raise ValueError(f"Unknown V3 representation mode: {mode!r}")

    def forward(
        self,
        projections: torch.Tensor,
        view_mask: torch.Tensor,
        geometry: ProjectionGeometry,
        grid: ReconstructionGrid,
        representation_mode: str = "NORMAL",
    ) -> dict:
        mode = representation_mode.upper()
        if mode not in self.REPRESENTATION_MODES:
            raise ValueError(f"Unknown V3 representation mode: {representation_mode!r}")
        if projections.ndim != 5 or projections.shape[:2] != view_mask.shape:
            raise ValueError("Expected projections [B,V,1,H,W] and view_mask [B,V]")
        geometry = geometry.to(projections.device, projections.dtype)
        features = self.film(
            self.encoder(projections / self.projection_input_scale),
            geometry,
            grid.center_world_xyz_mm,
        )
        proposal_primitives = self.proposal(features["s2"], geometry, grid, view_mask)
        view_features, query_metadata = self.query(proposal_primitives, features, geometry, view_mask)
        fused, weights = self.attention(proposal_primitives, view_features, geometry, query_metadata["valid"])
        interacted = self.interaction(proposal_primitives.updated(feature=proposal_primitives.feature + fused))
        primitives = self.refine(interacted)

        splat = self.splat(self._representation_primitives(primitives, mode), grid)
        if mode in {"ZERO_COARSE_SPLAT", "ZERO_COARSE"}:
            splat = splat.zero_scale("coarse")
        elif mode in {"ZERO_MID_SPLAT", "ZERO_MID"}:
            splat = splat.zero_scale("mid")
        elif mode == "SHUFFLE_MID_SPLAT":
            splat = replace(
                splat,
                mid_feature=torch.roll(splat.mid_feature.flatten(2), shifts=7919, dims=2).reshape_as(splat.mid_feature),
                mid_density=torch.roll(splat.mid_density.flatten(2), shifts=7919, dims=2).reshape_as(splat.mid_density),
                mid_weight=torch.roll(splat.mid_weight.flatten(2), shifts=7919, dims=2).reshape_as(splat.mid_weight),
            )
        elif mode == "SHUFFLE_COARSE_SPLAT":
            splat = replace(
                splat,
                coarse_feature=torch.roll(splat.coarse_feature.flatten(2), shifts=1543, dims=2).reshape_as(splat.coarse_feature),
                coarse_density=torch.roll(splat.coarse_density.flatten(2), shifts=1543, dims=2).reshape_as(splat.coarse_density),
                coarse_weight=torch.roll(splat.coarse_weight.flatten(2), shifts=1543, dims=2).reshape_as(splat.coarse_weight),
            )

        if self.decoder_force_fp32:
            fp32_splat = replace(
                splat,
                coarse_feature=splat.coarse_feature.float(),
                coarse_density=splat.coarse_density.float(),
                coarse_weight=splat.coarse_weight.float(),
                mid_feature=splat.mid_feature.float(),
                mid_density=splat.mid_density.float(),
                mid_weight=splat.mid_weight.float(),
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
        fusion_physics = torch.zeros_like(physics) if mode == "DECODER_ONLY" else physics
        fusion_residual = torch.zeros_like(residual) if mode == "PHYSICS_ONLY" else residual
        fusion_preactivation = self.fusion.preactivation(fusion_physics, fusion_residual)
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
                raise FloatingPointError("Non-finite ARP-V3 volume recovery output")
            if fov_violation_count:
                raise AssertionError("Primitive outside common reconstruction FOV")
            if ray_deviation.max() > 2e-3:
                raise AssertionError(f"Primitive left its measurement ray: {ray_deviation.max().item():.6g} mm")
            if primitives.confidence.min() + 1e-6 < self.proposal.confidence_min:
                raise AssertionError("Primitive confidence crossed the configured V2 lower bound")
            for shape in decoder_shapes.values():
                if shape[1] >= 64 and tuple(shape[-3:]) == tuple(grid.shape_zyx):
                    raise AssertionError(f"Forbidden V3 full-resolution high-channel tensor: {shape}")

        return {
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
            "representation_mode": mode,
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
