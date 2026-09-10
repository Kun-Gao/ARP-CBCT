from __future__ import annotations

import torch
from torch import nn

from .primitive_types import PrimitiveSet


class PrimitiveRefinement(nn.Module):
    """Interaction-conditioned refinement that cannot cross hypothesis intervals."""

    def __init__(
        self,
        dim: int,
        max_delta_alpha: float,
        sigma_min_mm: float,
        sigma_max_mm: float,
        support_log_scale: float = 0.25,
        confidence_min: float = 0.1,
        support_mode: str = "adaptive",
        support_residual_log_scale: float = 0.6931471805599453,
    ):
        super().__init__()
        self.head = nn.Linear(dim, 1 + 3 + 1 + 1 + dim)
        self.max_delta_alpha = float(max_delta_alpha)
        self.sigma_min_mm = float(sigma_min_mm)
        self.sigma_max_mm = float(sigma_max_mm)
        self.support_log_scale = float(support_log_scale)
        self.confidence_min = float(confidence_min)
        self.support_mode = support_mode
        self.support_residual_log_scale = float(support_residual_log_scale)
        if support_mode == "geometry_grounded":
            with torch.no_grad():
                self.head.weight[1:4].zero_()
                self.head.bias[1:4].zero_()

    def forward(self, p: PrimitiveSet) -> PrimitiveSet:
        if p.alpha is None or p.alpha_base is None:
            raise ValueError("Primitive refinement requires alpha and alpha_base")
        raw = self.head(p.feature)
        normalized = ((p.alpha - p.alpha_base) / self.max_delta_alpha).clamp(-0.999, 0.999)
        alpha = p.alpha_base + self.max_delta_alpha * torch.tanh(torch.atanh(normalized) + raw[..., 0])
        t = p.t_near + alpha * (p.t_far - p.t_near)
        position = p.ray_origin + t[..., None] * p.ray_direction
        if self.support_mode == "fixed_base":
            support = p.support.detach()
            support_scale = torch.ones_like(support)
        elif self.support_mode == "geometry_grounded":
            if p.support_base is None:
                raise ValueError("Geometry-grounded refinement requires support_base")
            support_scale = torch.exp(self.support_residual_log_scale * torch.tanh(raw[..., 1:4]))
            support = (p.support_base * support_scale).clamp(self.sigma_min_mm, self.sigma_max_mm)
        else:
            support = (p.support * torch.exp(self.support_log_scale * torch.tanh(raw[..., 1:4]))).clamp(
                self.sigma_min_mm, self.sigma_max_mm
            )
            support_scale = support / p.support.clamp_min(1e-6)
        density = (p.density * torch.exp(0.25 * torch.tanh(raw[..., 4:5]))).clamp_min(1e-8)
        confidence = (
            self.confidence_min
            + (1 - self.confidence_min)
            * torch.sigmoid(torch.logit(p.confidence.clamp(1e-5, 1 - 1e-5)) + raw[..., 5:6])
        )
        feature = p.feature + raw[..., 6:]
        return p.updated(position=position, t=t, alpha=alpha, support=support, support_scale=support_scale, density=density, confidence=confidence, feature=feature)
