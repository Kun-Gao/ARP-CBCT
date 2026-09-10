from __future__ import annotations

import torch
import torch.nn.functional as F


def ray_aligned_basis(ray_direction: torch.Tensor, transverse_hint: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Return row-wise [parallel, perpendicular-1, perpendicular-2] bases."""
    parallel = F.normalize(ray_direction, dim=-1)
    if transverse_hint is None:
        axes = torch.eye(3, device=parallel.device, dtype=parallel.dtype)
        choice = parallel.abs().argmin(-1)
        hint = axes[choice]
    else:
        hint = F.normalize(transverse_hint, dim=-1)
        degenerate = (hint * parallel).sum(-1).abs() > 1 - 1e-4
        axes = torch.eye(3, device=parallel.device, dtype=parallel.dtype)
        fallback = axes[parallel.abs().argmin(-1)]
        hint = torch.where(degenerate[..., None], fallback, hint)
    perpendicular_1 = hint - (hint * parallel).sum(-1, keepdim=True) * parallel
    perpendicular_1 = F.normalize(perpendicular_1, dim=-1, eps=eps)
    perpendicular_2 = F.normalize(torch.cross(parallel, perpendicular_1, dim=-1), dim=-1, eps=eps)
    return torch.stack((parallel, perpendicular_1, perpendicular_2), dim=-2)


def geometry_grounded_support(
    t: torch.Tensor,
    t_near: torch.Tensor,
    t_far: torch.Tensor,
    dsd_ray: torch.Tensor,
    detector_footprint_uv_mm: torch.Tensor,
    alpha_bases: torch.Tensor,
    perpendicular_multiplier: float,
    parallel_interval_multiplier: float,
    minimum_mm: float,
    maximum_mm: float,
) -> torch.Tensor:
    """Compute local [parallel, detector-u, transverse-v] Gaussian scales in mm."""
    depth_range = (t_far - t_near).clamp_min(1e-6)
    if alpha_bases.numel() < 2:
        raise ValueError("Multiple hypotheses are required for longitudinal support")
    left = torch.diff(alpha_bases, prepend=alpha_bases[:1] - (alpha_bases[1] - alpha_bases[0]))
    right = torch.diff(alpha_bases, append=alpha_bases[-1:] + (alpha_bases[-1] - alpha_bases[-2]))
    interval = torch.minimum(left, right).abs()
    parallel = depth_range * interval * float(parallel_interval_multiplier)
    magnification = t / dsd_ray.clamp_min(1e-6)
    perpendicular = detector_footprint_uv_mm * magnification[..., None] * float(perpendicular_multiplier)
    support = torch.cat((parallel[..., None], perpendicular), dim=-1)
    return support.clamp(float(minimum_mm), float(maximum_mm))
