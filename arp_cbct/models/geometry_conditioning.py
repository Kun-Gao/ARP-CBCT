from __future__ import annotations

import torch
from torch import nn

from ..geometry import ProjectionGeometry


def geometry_vectors(
    geometry: ProjectionGeometry,
    volume_center_world_xyz_mm: tuple[float, float, float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Batch-safe per-frame scanner vectors in a volume-centered physical frame."""
    g = geometry.with_batch()
    dsd = torch.linalg.vector_norm(g.detector_center_world - g.source_position_world, dim=-1, keepdim=True).clamp_min(1.0)
    if volume_center_world_xyz_mm is None:
        center = torch.zeros(3, device=g.source_position_world.device, dtype=g.source_position_world.dtype)
    else:
        center = torch.as_tensor(
            volume_center_world_xyz_mm,
            device=g.source_position_world.device,
            dtype=g.source_position_world.dtype,
        )
    spacing = g.detector_spacing_uv_mm
    return torch.cat(
        (
            (g.source_position_world - center) / dsd,
            (g.detector_center_world - center) / dsd,
            g.detector_u_axis_world,
            g.detector_v_axis_world,
            torch.log1p(spacing),
            torch.log1p(dsd) / 10.0,
        ),
        -1,
    )


class GeometryFiLM(nn.Module):
    def __init__(self, channels: list[int], embed_dim: int = 96):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(15, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(embed_dim, 2 * c) for c in channels])

    def forward(
        self,
        features: dict[str, torch.Tensor],
        geometry: ProjectionGeometry,
        volume_center_world_xyz_mm: tuple[float, float, float] | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        embedding = self.embed(geometry_vectors(geometry, volume_center_world_xyz_mm))
        output = {}
        for i, (key, x) in enumerate(features.items()):
            gamma, beta = self.heads[i](embedding).chunk(2, -1)
            output[key] = (1 + gamma[..., None, None]) * x + beta[..., None, None]
        return output
