from __future__ import annotations

import math

import torch
from torch import nn

from ..geometry import ProjectionGeometry, ReconstructionGrid, detector_pixels_to_rays, ray_aabb_intersection
from .primitive_proposal import PrimitiveSet
from .support_geometry import geometry_grounded_support, ray_aligned_basis


class StructuredRayPrimitiveProposalV2(nn.Module):
    """Per-view, detector-stratified ray proposal with bounded depth hypotheses."""

    def __init__(
        self,
        in_dim: int,
        primitive_dim: int,
        budget: int,
        sigma_min_mm: float,
        sigma_max_mm: float,
        hypotheses: list[float],
        max_delta_alpha: float = 0.1,
        strata_hw: tuple[int, int] = (8, 16),
        max_rays_per_cell: int = 2,
        sigma_base_mm: float = 4.0,
        support_log_scale: float = 0.75,
        confidence_min: float = 0.1,
        sigma_min_spacing_multiplier: float = 1.0,
        support_mode: str = "adaptive",
        support_perpendicular_multiplier: float = 1.0,
        support_parallel_interval_multiplier: float = 0.25,
        density_scale_mm_inv: float = 1.0,
    ):
        super().__init__()
        if len(hypotheses) < 2:
            raise ValueError("V2 requires multiple depth hypotheses per selected ray")
        if sorted(hypotheses) != list(hypotheses):
            raise ValueError("Depth hypotheses must be ordered")
        gaps = [right - left for left, right in zip(hypotheses[:-1], hypotheses[1:])]
        if 2 * max_delta_alpha >= min(gaps):
            raise ValueError("Depth residual intervals overlap; hypothesis ordering is not guaranteed")
        if not (0 < confidence_min < 1):
            raise ValueError("confidence_min must lie in (0, 1)")
        if not (sigma_min_mm <= sigma_base_mm <= sigma_max_mm):
            raise ValueError("sigma_base_mm must lie inside support bounds")
        self.budget = int(budget)
        self.hypotheses_per_ray = len(hypotheses)
        self.max_delta_alpha = float(max_delta_alpha)
        self.strata_hw = tuple(int(value) for value in strata_hw)
        self.max_rays_per_cell = int(max_rays_per_cell)
        self.sigma_min_mm = float(sigma_min_mm)
        self.sigma_max_mm = float(sigma_max_mm)
        self.sigma_base_mm = float(sigma_base_mm)
        self.support_log_scale = float(support_log_scale)
        self.confidence_min = float(confidence_min)
        self.sigma_min_spacing_multiplier = float(sigma_min_spacing_multiplier)
        if support_mode not in ("adaptive", "fixed_base", "geometry_grounded"):
            raise ValueError(f"Unknown support_mode: {support_mode!r}")
        self.support_mode = support_mode
        self.support_perpendicular_multiplier = float(support_perpendicular_multiplier)
        self.support_parallel_interval_multiplier = float(support_parallel_interval_multiplier)
        self.density_scale_mm_inv = float(density_scale_mm_inv)
        if self.density_scale_mm_inv <= 0:
            raise ValueError("density_scale_mm_inv must be positive")
        self.register_buffer("alpha_bases", torch.tensor(hypotheses, dtype=torch.float32))

        self.score = nn.Conv2d(in_dim, 1, 1)
        self.base_feature = nn.Conv2d(in_dim, primitive_dim, 1)
        self.depth_delta = nn.Conv2d(in_dim, self.hypotheses_per_ray, 1)
        self.support_delta = nn.Conv2d(in_dim, 3 * self.hypotheses_per_ray, 1)
        self.density = nn.Conv2d(in_dim, self.hypotheses_per_ray, 1)
        self.confidence = nn.Conv2d(in_dim, self.hypotheses_per_ray, 1)
        self.depth_embedding = nn.Sequential(nn.Linear(1, primitive_dim), nn.GELU(), nn.Linear(primitive_dim, primitive_dim))
        self.geometry_embedding = nn.Sequential(nn.Linear(7, primitive_dim), nn.GELU(), nn.Linear(primitive_dim, primitive_dim))

    @staticmethod
    def _quota(total: int, active_views: torch.Tensor) -> dict[int, int]:
        active = active_views.nonzero(as_tuple=False).flatten().tolist()
        if not active:
            return {}
        base, remainder = divmod(total, len(active))
        return {view: base + (rank < remainder) for rank, view in enumerate(active)}

    def _stratified_topk(self, score: torch.Tensor, valid: torch.Tensor, quota: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Select cell winners first, then bounded extra candidates from each cell."""
        h, w = score.shape
        gh, gw = self.strata_hw
        row_cell = torch.div(torch.arange(h, device=score.device) * gh, h, rounding_mode="floor").clamp_max(gh - 1)
        col_cell = torch.div(torch.arange(w, device=score.device) * gw, w, rounding_mode="floor").clamp_max(gw - 1)
        cells = (row_cell[:, None] * gw + col_cell[None]).flatten()
        flat_score, flat_valid = score.flatten(), valid.flatten()
        available = int(flat_valid.sum().item())
        quota = min(int(quota), available)
        if quota <= 0:
            return torch.empty(0, dtype=torch.long, device=score.device), torch.empty(0, dtype=torch.long, device=score.device)

        by_rank: list[torch.Tensor] = []
        for cell in range(gh * gw):
            indices = ((cells == cell) & flat_valid).nonzero(as_tuple=False).flatten()
            if indices.numel():
                count = min(self.max_rays_per_cell, indices.numel())
                by_rank.append(indices[flat_score[indices].topk(count).indices])
        if not by_rank:
            raise RuntimeError("No valid detector candidates in spatial strata")

        selected: list[torch.Tensor] = []
        for rank in range(self.max_rays_per_cell):
            candidates = [indices[rank] for indices in by_rank if indices.numel() > rank]
            if not candidates:
                break
            candidates_t = torch.stack(candidates)
            remaining = quota - sum(chunk.numel() for chunk in selected)
            if remaining <= 0:
                break
            if candidates_t.numel() > remaining:
                candidates_t = candidates_t[flat_score[candidates_t].topk(remaining).indices]
            selected.append(candidates_t)
        chosen = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long, device=score.device)
        if chosen.numel() < quota:
            # This only occurs when too few cells contain valid rays. Preserve the
            # per-cell cap whenever possible, then fill deterministically.
            unused = flat_valid.clone()
            unused[chosen] = False
            remaining_indices = unused.nonzero(as_tuple=False).flatten()
            fill = remaining_indices[flat_score[remaining_indices].topk(quota - chosen.numel()).indices]
            chosen = torch.cat((chosen, fill))
        return chosen, cells[chosen]

    def forward(self, x: torch.Tensor, geometry: ProjectionGeometry, grid: ReconstructionGrid, view_mask: torch.Tensor) -> PrimitiveSet:
        b, v, _, h, w = x.shape
        if geometry.with_batch().source_position_world.shape[:2] != (b, v):
            raise ValueError("Feature and geometry batch/view dimensions differ")
        required_sigma_min = self.sigma_min_spacing_multiplier * min(grid.spacing_zyx_mm)
        if self.sigma_min_mm + 1e-6 < required_sigma_min:
            raise ValueError(
                f"sigma_min_mm={self.sigma_min_mm} is below the configured grid-aware lower bound {required_sigma_min:.6g}"
            )

        flat = x.flatten(0, 1)
        scores = self.score(flat).reshape(b, v, h, w).sigmoid()
        base_features = self.base_feature(flat).reshape(b, v, -1, h, w)
        depth_raw = self.depth_delta(flat).reshape(b, v, self.hypotheses_per_ray, h, w)
        support_raw = self.support_delta(flat).reshape(b, v, self.hypotheses_per_ray, 3, h, w)
        density_raw = self.density(flat).reshape(b, v, self.hypotheses_per_ray, h, w)
        confidence_raw = self.confidence(flat).reshape(b, v, self.hypotheses_per_ray, h, w)

        hd, wd = geometry.detector_shape_hw
        row_1d = (torch.arange(h, device=x.device, dtype=x.dtype) + 0.5) * hd / h - 0.5
        col_1d = (torch.arange(w, device=x.device, dtype=x.dtype) + 0.5) * wd / w - 0.5
        rr, cc = torch.meshgrid(row_1d, col_1d, indexing="ij")
        box_min, box_max = grid.bounds_xyz(x.device, x.dtype)
        total_rays = self.budget // self.hypotheses_per_ray
        alpha_bases = self.alpha_bases.to(device=x.device, dtype=x.dtype)
        spatial_scale = max(grid.shape_zyx[i] * grid.spacing_zyx_mm[i] for i in range(3))

        results = []
        for bi in range(b):
            g = geometry.select_batch(bi)
            quotas = self._quota(total_rays, view_mask[bi])
            selected_views, selected_local, selected_cells = [], [], []
            selected_origins, selected_directions, selected_near, selected_far = [], [], [], []
            for view, quota in quotas.items():
                local = torch.arange(h * w, device=x.device)
                view_ids = torch.full((h * w,), view, device=x.device, dtype=torch.long)
                origins, directions = detector_pixels_to_rays(rr.flatten(), cc.flatten(), view_ids, g)
                near, far, valid = ray_aabb_intersection(origins, directions, box_min, box_max)
                indices, cells = self._stratified_topk(scores[bi, view], valid.reshape(h, w), quota)
                selected_views.append(torch.full_like(indices, view))
                selected_local.append(indices)
                selected_cells.append(cells)
                selected_origins.append(origins[indices])
                selected_directions.append(directions[indices])
                selected_near.append(near[indices])
                selected_far.append(far[indices])
            if not selected_local:
                raise RuntimeError("No active view produced valid rays")

            ray_view = torch.cat(selected_views)
            local_index = torch.cat(selected_local)
            ray_cell = torch.cat(selected_cells)
            ray_origin = torch.cat(selected_origins)
            ray_direction = torch.cat(selected_directions)
            ray_near = torch.cat(selected_near)
            ray_far = torch.cat(selected_far)
            ray_count = ray_view.numel()
            row_index, col_index = local_index // w, local_index % w

            base = base_features[bi, ray_view, :, row_index, col_index]
            raw_alpha = depth_raw[bi, ray_view, :, row_index, col_index]
            raw_support = support_raw[bi, ray_view, :, :, row_index, col_index]
            raw_density = density_raw[bi, ray_view, :, row_index, col_index]
            raw_confidence = confidence_raw[bi, ray_view, :, row_index, col_index]
            selected_score = scores[bi, ray_view, row_index, col_index]

            alpha_base = alpha_bases[None].expand(ray_count, -1)
            alpha = alpha_base + self.max_delta_alpha * torch.tanh(raw_alpha)
            near = ray_near[:, None].expand_as(alpha)
            far = ray_far[:, None].expand_as(alpha)
            t = near + alpha * (far - near)
            origin = ray_origin[:, None].expand(-1, self.hypotheses_per_ray, -1)
            direction = ray_direction[:, None].expand_as(origin)
            position = origin + t[..., None] * direction
            if self.support_mode == "fixed_base":
                support = torch.full_like(raw_support, self.sigma_base_mm).detach()
                support_basis = torch.eye(3, device=x.device, dtype=x.dtype)[None, None].expand(ray_count, self.hypotheses_per_ray, -1, -1)
            elif self.support_mode == "geometry_grounded":
                u_hint = g.detector_u_axis_world[ray_view]
                ray_basis = ray_aligned_basis(ray_direction, u_hint)
                support_basis = ray_basis[:, None].expand(-1, self.hypotheses_per_ray, -1, -1)
                detector_point = (
                    g.detector_center_world[ray_view]
                    + (cc.flatten()[local_index] - (wd - 1) / 2)[:, None] * g.detector_u_vector_world[ray_view]
                    + (rr.flatten()[local_index] - (hd - 1) / 2)[:, None] * g.detector_v_vector_world[ray_view]
                )
                dsd_ray = torch.linalg.vector_norm(detector_point - ray_origin, dim=-1)[:, None].expand_as(t)
                detector_stride_uv = g.detector_spacing_uv_mm[ray_view] * x.new_tensor((wd / w, hd / h))
                support = geometry_grounded_support(
                    t, near, far, dsd_ray, detector_stride_uv[:, None], alpha_bases,
                    self.support_perpendicular_multiplier, self.support_parallel_interval_multiplier,
                    max(self.sigma_min_mm, required_sigma_min), self.sigma_max_mm,
                )
            else:
                support = (self.sigma_base_mm * torch.exp(self.support_log_scale * torch.tanh(raw_support))).clamp(
                    self.sigma_min_mm, self.sigma_max_mm
                )
                support_basis = torch.eye(3, device=x.device, dtype=x.dtype)[None, None].expand(ray_count, self.hypotheses_per_ray, -1, -1)
            support_base = support.detach()
            support_scale = torch.ones_like(support)
            density = self.density_scale_mm_inv * torch.nn.functional.softplus(raw_density)[..., None]
            confidence_signal = 0.5 * raw_confidence.sigmoid() + 0.5 * selected_score[:, None]
            confidence = (self.confidence_min + (1 - self.confidence_min) * confidence_signal)[..., None]

            repeated_base = base[:, None].expand(-1, self.hypotheses_per_ray, -1)
            depth_embedding = self.depth_embedding(alpha_base[..., None])
            geometry_input = torch.cat(
                (
                    origin / spatial_scale,
                    direction,
                    alpha_base[..., None],
                ),
                dim=-1,
            )
            feature = repeated_base + depth_embedding + self.geometry_embedding(geometry_input)
            hypothesis = torch.arange(self.hypotheses_per_ray, device=x.device)[None].expand(ray_count, -1)
            ray_id = torch.arange(ray_count, device=x.device)[:, None].expand_as(hypothesis)

            def flatten_hypothesis(value: torch.Tensor) -> torch.Tensor:
                return value.reshape(-1, *value.shape[2:])

            results.append(
                (
                    flatten_hypothesis(position),
                    flatten_hypothesis(support),
                    flatten_hypothesis(feature),
                    flatten_hypothesis(density),
                    flatten_hypothesis(confidence),
                    ray_view[:, None].expand_as(hypothesis).reshape(-1),
                    flatten_hypothesis(origin),
                    flatten_hypothesis(direction),
                    t.reshape(-1),
                    near.reshape(-1),
                    far.reshape(-1),
                    rr.flatten()[local_index][:, None].expand_as(alpha).reshape(-1),
                    cc.flatten()[local_index][:, None].expand_as(alpha).reshape(-1),
                    ray_id.reshape(-1),
                    hypothesis.reshape(-1),
                    alpha_base.reshape(-1),
                    alpha.reshape(-1),
                    ray_cell[:, None].expand_as(hypothesis).reshape(-1),
                    support_basis.reshape(-1, 3, 3),
                    flatten_hypothesis(support_base),
                    flatten_hypothesis(support_scale),
                )
            )

        fields = list(zip(*results))
        primitive = PrimitiveSet(*(torch.stack(field) for field in fields))
        if not torch.isfinite(primitive.position).all() or not torch.isfinite(primitive.support).all():
            raise FloatingPointError("V2 primitive proposal generated non-finite values")
        return primitive
