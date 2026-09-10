from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ReconstructionGrid:
    """Common reconstruction grid; arrays are z-y-x and world vectors are x-y-z."""

    shape_zyx: tuple[int, int, int]
    spacing_zyx_mm: tuple[float, float, float]
    center_world_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ReconstructionGrid":
        center = cfg.get("center_world_xyz_mm", cfg.get("center_lps_mm", (0.0, 0.0, 0.0)))
        return cls(tuple(map(int, cfg["shape_zyx"])), tuple(map(float, cfg["spacing_zyx_mm"])), tuple(map(float, center)))

    def bounds_xyz(self, device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        """Return voxel-edge AABB, not just first/last voxel centers."""
        shape_xyz = torch.tensor(self.shape_zyx[::-1], device=device, dtype=dtype)
        spacing_xyz = torch.tensor(self.spacing_zyx_mm[::-1], device=device, dtype=dtype)
        center = torch.tensor(self.center_world_xyz_mm, device=device, dtype=dtype)
        half_extent = shape_xyz * spacing_xyz / 2
        return center - half_extent, center + half_extent

    def center_bounds_xyz(self, device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        """Return coordinates of the first and last voxel centers for grid_sample."""
        shape_xyz = torch.tensor(self.shape_zyx[::-1], device=device, dtype=dtype)
        spacing_xyz = torch.tensor(self.spacing_zyx_mm[::-1], device=device, dtype=dtype)
        center = torch.tensor(self.center_world_xyz_mm, device=device, dtype=dtype)
        half_extent = (shape_xyz - 1) * spacing_xyz / 2
        return center - half_extent, center + half_extent


@dataclass
class ProjectionGeometry:
    """Per-frame vector geometry in the validated common world coordinate frame.

    Detector u/v vectors point by one native detector pixel. Tensors are [B,V,3],
    or [V,3] before collation.
    """

    source_position_world: torch.Tensor
    detector_center_world: torch.Tensor
    detector_u_vector_world: torch.Tensor
    detector_v_vector_world: torch.Tensor
    detector_shape_hw: tuple[int, int]

    @classmethod
    def from_pose_vectors(cls, poses: torch.Tensor, detector_shape_hw: tuple[int, int]) -> "ProjectionGeometry":
        if poses.ndim != 2 or poses.shape[-1] != 12:
            raise ValueError(f"Expected pose vectors [V,12], got {tuple(poses.shape)}")
        if not torch.isfinite(poses).all():
            raise ValueError("Projection geometry contains NaN/Inf")
        return cls(poses[:, 0:3], poses[:, 3:6], poses[:, 6:9], poses[:, 9:12], detector_shape_hw)

    @property
    def batched(self) -> bool:
        return self.source_position_world.ndim == 3

    @property
    def detector_spacing_uv_mm(self) -> torch.Tensor:
        du = torch.linalg.vector_norm(self.detector_u_vector_world, dim=-1)
        dv = torch.linalg.vector_norm(self.detector_v_vector_world, dim=-1)
        return torch.stack((du, dv), -1)

    @property
    def detector_u_axis_world(self) -> torch.Tensor:
        return F.normalize(self.detector_u_vector_world, dim=-1)

    @property
    def detector_v_axis_world(self) -> torch.Tensor:
        return F.normalize(self.detector_v_vector_world, dim=-1)

    def to(self, device, dtype=None) -> "ProjectionGeometry":
        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return ProjectionGeometry(self.source_position_world.to(**kwargs), self.detector_center_world.to(**kwargs), self.detector_u_vector_world.to(**kwargs), self.detector_v_vector_world.to(**kwargs), self.detector_shape_hw)

    def with_batch(self) -> "ProjectionGeometry":
        if self.batched:
            return self
        return ProjectionGeometry(self.source_position_world.unsqueeze(0), self.detector_center_world.unsqueeze(0), self.detector_u_vector_world.unsqueeze(0), self.detector_v_vector_world.unsqueeze(0), self.detector_shape_hw)

    def select_batch(self, index: int) -> "ProjectionGeometry":
        if not self.batched:
            if index != 0:
                raise IndexError(index)
            return self
        return ProjectionGeometry(self.source_position_world[index], self.detector_center_world[index], self.detector_u_vector_world[index], self.detector_v_vector_world[index], self.detector_shape_hw)


def stack_projection_geometries(items: list[ProjectionGeometry]) -> ProjectionGeometry:
    if not items:
        raise ValueError("Cannot stack an empty geometry list")
    shape = items[0].detector_shape_hw
    views = items[0].source_position_world.shape[0]
    if any(x.detector_shape_hw != shape or x.source_position_world.shape[0] != views for x in items):
        raise ValueError("All batch items must have the same detector shape and view count")
    return ProjectionGeometry(torch.stack([x.source_position_world for x in items]), torch.stack([x.detector_center_world for x in items]), torch.stack([x.detector_u_vector_world for x in items]), torch.stack([x.detector_v_vector_world for x in items]), shape)


def ray_aabb_intersection(ray_origin: torch.Tensor, ray_direction: torch.Tensor, box_min: torch.Tensor, box_max: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-ray/AABB intersection using a parallel-safe slab formulation."""
    if ray_origin.shape != ray_direction.shape or ray_origin.shape[-1] != 3:
        raise ValueError("ray_origin and ray_direction must have identical [...,3] shape")
    box_min = torch.as_tensor(box_min, device=ray_origin.device, dtype=ray_origin.dtype)
    box_max = torch.as_tensor(box_max, device=ray_origin.device, dtype=ray_origin.dtype)
    parallel = ray_direction.abs() <= eps
    outside_parallel = parallel & ((ray_origin < box_min) | (ray_origin > box_max))
    safe_direction = torch.where(parallel, torch.ones_like(ray_direction), ray_direction)
    slab_a = (box_min - ray_origin) / safe_direction
    slab_b = (box_max - ray_origin) / safe_direction
    slab_near = torch.where(parallel, torch.full_like(slab_a, -torch.inf), torch.minimum(slab_a, slab_b))
    slab_far = torch.where(parallel, torch.full_like(slab_a, torch.inf), torch.maximum(slab_a, slab_b))
    t_near = slab_near.amax(-1).clamp_min(0)
    t_far = slab_far.amin(-1)
    valid = (~outside_parallel.any(-1)) & (t_far >= t_near) & (t_far > 0)
    return t_near, t_far, valid


def detector_pixels_to_rays(rows: torch.Tensor, cols: torch.Tensor, view_indices: torch.Tensor, geometry: ProjectionGeometry) -> tuple[torch.Tensor, torch.Tensor]:
    """Map native detector coordinates to normalized world-space rays."""
    if geometry.batched:
        raise ValueError("Select one batch item before mapping an irregular ray list")
    source = geometry.source_position_world[view_indices]
    center = geometry.detector_center_world[view_indices]
    u_vec = geometry.detector_u_vector_world[view_indices]
    v_vec = geometry.detector_v_vector_world[view_indices]
    hd, wd = geometry.detector_shape_hw
    points = center + (cols - (wd - 1) / 2).unsqueeze(-1) * u_vec + (rows - (hd - 1) / 2).unsqueeze(-1) * v_vec
    return source, F.normalize(points - source, dim=-1)


def project_world_to_detector(points: torch.Tensor, geometry: ProjectionGeometry, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project [B,N,3] points into each frame; return row/col/ray t [B,V,N]."""
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must be [B,N,3]")
    g = geometry.with_batch()
    if g.source_position_world.shape[0] != points.shape[0]:
        raise ValueError("Geometry batch does not match point batch")
    source, center = g.source_position_world, g.detector_center_world
    u_vec, v_vec = g.detector_u_vector_world, g.detector_v_vector_world
    normal = F.normalize(torch.cross(u_vec, v_vec, dim=-1), dim=-1)
    detector_forward = center - source
    normal = torch.where(((normal * detector_forward).sum(-1) < 0)[..., None], -normal, normal)
    ray = points[:, None] - source[:, :, None]
    numerator = (detector_forward * normal).sum(-1)[:, :, None]
    denominator = (ray * normal[:, :, None]).sum(-1)
    safe_denominator = torch.where(denominator.abs() < eps, torch.full_like(denominator, eps), denominator)
    ray_t = numerator / safe_denominator
    hit = source[:, :, None] + ray_t[..., None] * ray
    delta = hit - center[:, :, None]
    col_offset = (delta * u_vec[:, :, None]).sum(-1) / u_vec.square().sum(-1)[:, :, None]
    row_offset = (delta * v_vec[:, :, None]).sum(-1) / v_vec.square().sum(-1)[:, :, None]
    hd, wd = g.detector_shape_hw
    return row_offset + (hd - 1) / 2, col_offset + (wd - 1) / 2, ray_t
