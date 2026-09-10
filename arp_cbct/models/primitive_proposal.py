from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn

from ..geometry import ProjectionGeometry, ReconstructionGrid, detector_pixels_to_rays, ray_aabb_intersection


@dataclass
class PrimitiveSet:
    position: torch.Tensor
    support: torch.Tensor
    feature: torch.Tensor
    density: torch.Tensor
    confidence: torch.Tensor
    source_view: torch.Tensor
    ray_origin: torch.Tensor
    ray_direction: torch.Tensor
    t: torch.Tensor
    t_near: torch.Tensor
    t_far: torch.Tensor
    detector_row: torch.Tensor
    detector_col: torch.Tensor
    ray_id: torch.Tensor | None = None
    hypothesis_index: torch.Tensor | None = None
    alpha_base: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    detector_cell: torch.Tensor | None = None
    support_basis: torch.Tensor | None = None
    support_base: torch.Tensor | None = None
    support_scale: torch.Tensor | None = None

    def updated(self, **kwargs) -> "PrimitiveSet":
        return replace(self, **kwargs)


class AdaptiveRayPrimitiveProposal(nn.Module):
    """Select valid detector rays and place one adaptive-depth primitive per ray."""

    def __init__(self, in_dim: int, primitive_dim: int, budget: int, sigma_min: float, sigma_max: float):
        super().__init__()
        self.budget = int(budget)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.score = nn.Conv2d(in_dim, 1, 1)
        self.params = nn.Conv2d(in_dim, 1 + 3 + 1 + primitive_dim + 1, 1)

    def forward(self, x: torch.Tensor, geometry: ProjectionGeometry, grid: ReconstructionGrid, view_mask: torch.Tensor) -> PrimitiveSet:
        b, v, _, h, w = x.shape
        if geometry.with_batch().source_position_world.shape[:2] != (b, v):
            raise ValueError("Feature and geometry batch/view dimensions differ")
        flat = x.flatten(0, 1)
        scores = self.score(flat).reshape(b, v, h, w).sigmoid()
        raw_map = self.params(flat).reshape(b, v, -1, h, w)
        hd, wd = geometry.detector_shape_hw
        row_1d = (torch.arange(h, device=x.device, dtype=x.dtype) + 0.5) * hd / h - 0.5
        col_1d = (torch.arange(w, device=x.device, dtype=x.dtype) + 0.5) * wd / w - 0.5
        rr, cc = torch.meshgrid(row_1d, col_1d, indexing="ij")
        all_rows = rr.flatten().repeat(v)
        all_cols = cc.flatten().repeat(v)
        all_views = torch.arange(v, device=x.device).repeat_interleave(h * w)
        box_min, box_max = grid.bounds_xyz(x.device, x.dtype)

        results = []
        for bi in range(b):
            g = geometry.select_batch(bi)
            origins, directions = detector_pixels_to_rays(all_rows, all_cols, all_views, g)
            near, far, valid = ray_aabb_intersection(origins, directions, box_min, box_max)
            valid = valid & view_mask[bi, all_views]
            available = int(valid.sum().item())
            if available == 0:
                raise RuntimeError("No detector rays intersect the reconstruction FOV")
            k = min(self.budget, available)
            masked_scores = scores[bi].flatten().masked_fill(~valid, -torch.inf)
            values, indices = masked_scores.topk(k)
            view = all_views[indices]
            feature_index = indices % (h * w)
            row_index = feature_index // w
            col_index = feature_index % w
            raw = raw_map[bi, view, :, row_index, col_index]
            alpha = raw[:, 0].sigmoid()
            t = near[indices] + alpha * (far[indices] - near[indices])
            position = origins[indices] + t[:, None] * directions[indices]
            support = self.sigma_min + (self.sigma_max - self.sigma_min) * raw[:, 1:4].sigmoid()
            density = torch.nn.functional.softplus(raw[:, 4:5])
            feature = raw[:, 5:-1]
            confidence = values[:, None] * raw[:, -1:].sigmoid()
            results.append((position, support, feature, density, confidence, view, origins[indices], directions[indices], t, near[indices], far[indices], all_rows[indices], all_cols[indices]))

        fields = list(zip(*results))
        primitive = PrimitiveSet(*(torch.stack(field) for field in fields))
        if not torch.isfinite(primitive.position).all() or not torch.isfinite(primitive.support).all():
            raise FloatingPointError("Primitive proposal generated non-finite values")
        return primitive
