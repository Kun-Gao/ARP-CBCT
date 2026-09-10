"""Structured data carried by adaptive ray primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


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
        """Return a copy with selected fields replaced."""
        return replace(self, **kwargs)
