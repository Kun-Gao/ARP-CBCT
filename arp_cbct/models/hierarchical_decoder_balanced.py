from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .hierarchical_decoder3d import HierarchicalDecoder3D, ResidualBlock3D, _groups
from .multiscale_splatting import MultiScaleSplatOutput


class BalancedHierarchicalDecoder3D(HierarchicalDecoder3D):
    """V3 decoder with independently normalized coarse and primitive mid paths."""

    def __init__(self, *args, mid_channels: int = 64, **kwargs):
        super().__init__(*args, mid_channels=mid_channels, **kwargs)
        del self.mid_fusion
        self.mid_coarse = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(mid_channels), mid_channels),
        )
        self.mid_primitive = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(mid_channels), mid_channels),
        )
        self.gamma_mid = nn.Parameter(torch.ones(mid_channels))
        self.last_balance_diagnostics: dict[str, torch.Tensor] = {}

    def forward(self, splat: MultiScaleSplatOutput, output_shape_zyx: tuple[int, int, int]):
        coarse = self._run(self.coarse_stem, splat.coarse_feature)
        coarse = self._run_blocks(self.coarse_blocks, coarse)
        mid_from_coarse = self._run(self.coarse_to_mid, coarse)
        mid_from_coarse = F.interpolate(mid_from_coarse, size=splat.mid_feature.shape[-3:], mode="trilinear", align_corners=False)
        coarse_contribution = self._run(self.mid_coarse, mid_from_coarse)
        primitive_contribution = self._run(self.mid_primitive, splat.mid_feature)
        scaled_mid = self.gamma_mid[None, :, None, None, None] * primitive_contribution
        mid = F.gelu(coarse_contribution + scaled_mid)
        mid = self._run_blocks(self.mid_blocks, mid)
        high = self._run(self.mid_to_high, mid)
        high = F.interpolate(high, size=output_shape_zyx, mode="trilinear", align_corners=False)
        if high.shape[1] > self.max_fullres_channels: raise AssertionError("Forbidden high-channel full-resolution tensor")
        high = self._run_blocks(self.high_blocks, high); residual = self.output(high)
        self.last_balance_diagnostics = {
            "gamma_mid_mean": self.gamma_mid.detach().mean(),
            "coarse_contribution_rms": coarse_contribution.detach().float().square().mean().sqrt(),
            "mid_contribution_rms": scaled_mid.detach().float().square().mean().sqrt(),
        }
        shapes = {"coarse": tuple(coarse.shape), "mid_from_coarse": tuple(mid_from_coarse.shape),
                  "mid_balanced": tuple(mid.shape), "high": tuple(high.shape), "output": tuple(residual.shape)}
        return residual, shapes
