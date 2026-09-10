from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PhysicsDecoderFusion(nn.Module):
    def __init__(
        self,
        softplus_beta: float = 10.0,
        output_activation: str = "softplus",
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.eta_logit = nn.Parameter(torch.tensor(0.0))
        self.softplus_beta = float(softplus_beta)
        self.output_activation = str(output_activation).lower()
        self.residual_scale = float(residual_scale)
        if self.output_activation not in {"softplus", "relu"}:
            raise ValueError(f"Unsupported output activation: {output_activation!r}")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")

    def preactivation(self, physics_density: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return physics_density + torch.sigmoid(self.eta_logit) * self.residual_scale * delta

    def activate(self, preactivation: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "relu":
            return F.relu(preactivation.float())
        return F.softplus(preactivation.float(), beta=self.softplus_beta)

    def forward(
        self,
        physics_density: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        preactivation = self.preactivation(physics_density, delta)
        with torch.cuda.amp.autocast(enabled=False):
            return self.activate(preactivation)
