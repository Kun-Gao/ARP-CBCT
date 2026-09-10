"""Configuration of the ARP-CBCT model used for the reported experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_MODEL_CONFIG: dict[str, Any] = {
    "architecture": "arp_cbct_v4_balanced",
    "encoder_channels": [64, 96, 128, 192],
    "encoder_blocks_per_stage": 2,
    "encoder_view_chunk_size": 16,
    "geometry_embed_dim": 96,
    "projection_input_scale": 32.0,
    "primitive_dim": 128,
    "primitive_budget": 8192,
    "depth_hypotheses": [0.2, 0.5, 0.8],
    "max_delta_alpha": 0.1,
    "detector_strata_hw": [8, 16],
    "max_rays_per_cell": 2,
    "query_scales": ["s2", "s3", "s4"],
    "support_mode": "geometry_grounded",
    "support_perpendicular_multiplier": 0.5,
    "support_parallel_interval_multiplier": 1.0 / 6.0,
    "support_residual_log_scale": 0.6931471805599453,
    "sigma_min_mm": 2.0,
    "sigma_max_mm": 24.0,
    "sigma_base_mm": 4.0,
    "sigma_min_spacing_multiplier": 1.0,
    "support_log_scale": 0.75,
    "support_refine_log_scale": 0.25,
    "confidence_min": 0.1,
    "density_scale_mm_inv": 0.022,
    "interaction_k": 16,
    "knn_method": "chunked_exact",
    "knn_chunk_size": 4096,
    "splat_chunk_size": 512,
    "splat_max_chunk_voxels": 16_000_000,
    "v3_coarse_shape_zyx": [36, 64, 64],
    "v3_mid_shape_zyx": [72, 128, 128],
    "v3_coarse_channels": 128,
    "v3_mid_channels": 64,
    "v3_high_channels": 24,
    "v3_coarse_blocks": 10,
    "v3_mid_blocks": 4,
    "v3_high_blocks": 2,
    "v3_max_fullres_channels": 32,
    "decoder_gradient_checkpointing": False,
    "decoder_amp_dtype": "float16",
    "decoder_force_fp32": True,
    "output_softplus_beta": 10.0,
    "output_activation": "relu",
    "output_residual_scale": 0.022,
    "debug_assertions": True,
}


def get_default_config() -> dict[str, Any]:
    """Return an independent copy of the formal ARP-CBCT model config."""

    return deepcopy(_MODEL_CONFIG)
