from __future__ import annotations

import torch
from torch import nn

from ..geometry import ProjectionGeometry, ReconstructionGrid
from .primitive_types import PrimitiveSet


class GeometryAwareAttention(nn.Module):
    """Primitive-level cross-view fusion with explicit recorded-geometry bias."""

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.bias = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, 1))
        self.out = nn.Linear(dim, dim)

    def forward(self, p: PrimitiveSet, view_features: torch.Tensor, geometry: ProjectionGeometry, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, v, n, c = view_features.shape
        g = geometry.with_batch()
        query = self.q(p.feature)[:, None]
        key = self.k(view_features)
        value = self.v(view_features)
        source = g.source_position_world
        candidate_ray = torch.nn.functional.normalize(p.position[:, None] - source[:, :, None], dim=-1)
        source_index = p.source_view[:, None, :, None].expand(-1, 1, -1, 3)
        source_ray = candidate_ray.gather(1, source_index).expand(-1, v, -1, -1)
        ray_similarity = (candidate_ray * source_ray).sum(-1)
        center = source.mean(1, keepdim=True)
        source_radial = source - center
        source_azimuth = torch.atan2(source_radial[..., 1], source_radial[..., 0])
        source_angle = source_azimuth.gather(1, p.source_view)
        relative_angle = torch.remainder(source_azimuth[:, :, None] - source_angle[:, None] + torch.pi, 2 * torch.pi) - torch.pi
        dsd = torch.linalg.vector_norm(g.detector_center_world - source, dim=-1).median(1).values[:, None, None].clamp_min(1.0)
        source_distance = torch.linalg.vector_norm(p.position[:, None] - source[:, :, None], dim=-1) / dsd
        support = p.support.mean(-1)[:, None].expand(-1, v, -1) / g.detector_spacing_uv_mm.mean((1, 2))[:, None, None].clamp_min(1e-3)
        depth_fraction = ((p.t - p.t_near) / (p.t_far - p.t_near).clamp_min(1e-6))[:, None].expand(-1, v, -1)
        geom = torch.stack((relative_angle.sin(), relative_angle.cos(), ray_similarity, source_distance, support, depth_fraction), -1)
        logits = (query * key).sum(-1) / (c ** 0.5) + self.bias(geom).squeeze(-1)
        logits = logits.masked_fill(~valid, -torch.inf)
        no_valid = ~valid.any(1)
        logits = torch.where(no_valid[:, None], torch.zeros_like(logits), logits)
        weights = logits.softmax(1) * valid
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        fused = (weights[..., None] * value).sum(1)
        return self.out(fused), weights


class PrimitiveRefinement(nn.Module):
    """Refine depth only, preserving exact membership in the originating ray."""

    def __init__(self, dim: int, max_depth_delta_mm: float, sigma_min_mm: float, sigma_max_mm: float, max_support_delta_mm: float = 2.0):
        super().__init__()
        self.head = nn.Linear(dim, 1 + 3 + 1 + 1)
        self.max_depth_delta_mm = float(max_depth_delta_mm)
        self.max_support_delta_mm = float(max_support_delta_mm)
        self.sigma_min_mm = float(sigma_min_mm)
        self.sigma_max_mm = float(sigma_max_mm)

    def forward(self, p: PrimitiveSet, fused: torch.Tensor) -> PrimitiveSet:
        raw = self.head(fused)
        t = (p.t + self.max_depth_delta_mm * torch.tanh(raw[..., 0])).maximum(p.t_near).minimum(p.t_far)
        position = p.ray_origin + t[..., None] * p.ray_direction
        support = (p.support + self.max_support_delta_mm * torch.tanh(raw[..., 1:4])).clamp(self.sigma_min_mm, self.sigma_max_mm)
        density = (p.density * torch.exp(0.25 * torch.tanh(raw[..., 4:5]))).clamp_min(1e-8)
        confidence = (p.confidence + 0.25 * torch.tanh(raw[..., 5:6])).clamp(0, 1)
        return p.updated(position=position, t=t, support=support, feature=p.feature + fused, density=density, confidence=confidence)
