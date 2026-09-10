from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..geometry import ProjectionGeometry, project_world_to_detector
from .primitive_proposal import PrimitiveSet


class MultiScalePrimitiveQuery(nn.Module):
    """Reproject primitive centers and query every view without dense 3D lifting."""

    def __init__(self, channels: dict[str, int], scales: list[str], out_dim: int):
        super().__init__()
        self.scales = scales
        self.proj = nn.ModuleDict({scale: nn.Linear(channels[scale], out_dim) for scale in scales})

    def forward(self, primitives: PrimitiveSet, features: dict[str, torch.Tensor], geometry: ProjectionGeometry, view_mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
        row, col, depth = project_world_to_detector(primitives.position, geometry)
        hd, wd = geometry.detector_shape_hw
        valid = (depth > 0) & (row >= 0) & (row <= hd - 1) & (col >= 0) & (col <= wd - 1) & view_mask[:, :, None]
        per_scale = []
        for scale in self.scales:
            x = features[scale]
            b, v, c, h, w = x.shape
            gx = 2 * col / (wd - 1) - 1
            gy = 2 * row / (hd - 1) - 1
            sample_grid = torch.stack((gx, gy), -1).reshape(b * v, -1, 1, 2)
            sampled = F.grid_sample(x.flatten(0, 1), sample_grid, align_corners=True, padding_mode="zeros")
            sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, v, -1, c)
            per_scale.append(self.proj[scale](sampled))
        stacked = torch.stack(per_scale)
        fused = stacked.mean(0)
        with torch.no_grad():
            flattened = stacked.detach().flatten(1)
            scale_norm = flattened.square().mean(-1).sqrt().float()
            scale_variance = flattened.var(-1, unbiased=False).float()
            dot = flattened @ flattened.transpose(0, 1)
            length = flattened.square().sum(-1).sqrt().clamp_min(1e-12)
            scale_correlation = (dot / (length[:, None] * length[None])).float()
        return fused, {
            "valid": valid,
            "projected_rc": torch.stack((row, col), -1),
            "detector_depth": depth,
            "query_scale_names": tuple(self.scales),
            "query_scale_norm": scale_norm.detach(),
            "query_scale_variance": scale_variance.detach(),
            "query_scale_correlation": scale_correlation.detach(),
        }
