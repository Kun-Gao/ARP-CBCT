from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from .multiscale_splatting import MultiScaleSplatOutput


def _groups(channels: int) -> int:
    return next(group for group in range(min(8, channels), 0, -1) if channels % group == 0)


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(value + self.net(value))


class HierarchicalDecoder3D(nn.Module):
    """Recover full resolution progressively while keeping full-res channels low."""

    def __init__(
        self,
        coarse_channels: int = 128,
        mid_channels: int = 64,
        high_channels: int = 24,
        coarse_blocks: int = 10,
        mid_blocks: int = 4,
        high_blocks: int = 2,
        gradient_checkpointing: bool = False,
        max_fullres_channels: int = 32,
    ):
        super().__init__()
        if high_channels > max_fullres_channels:
            raise ValueError(
                f"High-resolution channels ({high_channels}) exceed structural limit ({max_fullres_channels})"
            )
        self.high_channels = int(high_channels)
        self.max_fullres_channels = int(max_fullres_channels)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.coarse_stem = nn.Sequential(
            nn.Conv3d(coarse_channels, coarse_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(coarse_channels), coarse_channels),
            nn.GELU(),
        )
        self.coarse_blocks = nn.Sequential(*(ResidualBlock3D(coarse_channels) for _ in range(coarse_blocks)))
        self.coarse_to_mid = nn.Conv3d(coarse_channels, mid_channels, 3, padding=1, bias=False)
        self.mid_fusion = nn.Sequential(
            nn.Conv3d(2 * mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(mid_channels), mid_channels),
            nn.GELU(),
        )
        self.mid_blocks = nn.Sequential(*(ResidualBlock3D(mid_channels) for _ in range(mid_blocks)))
        # Channel reduction occurs at mid resolution, before any full-resolution tensor exists.
        self.mid_to_high = nn.Conv3d(mid_channels, high_channels, 3, padding=1, bias=False)
        self.high_blocks = nn.Sequential(*(ResidualBlock3D(high_channels) for _ in range(high_blocks)))
        self.output = nn.Conv3d(high_channels, 1, 1)

    def _run(self, module: nn.Module, value: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, value, use_reentrant=False)
        return module(value)

    def _run_blocks(self, blocks: nn.Sequential, value: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled() and len(blocks):
            return checkpoint_sequential(blocks, len(blocks), value, use_reentrant=False)
        return blocks(value)

    def forward(
        self,
        splat: MultiScaleSplatOutput,
        output_shape_zyx: tuple[int, int, int],
    ) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        coarse = self._run(self.coarse_stem, splat.coarse_feature)
        coarse = self._run_blocks(self.coarse_blocks, coarse)
        mid_from_coarse = self._run(self.coarse_to_mid, coarse)
        mid_from_coarse = F.interpolate(
            mid_from_coarse, size=splat.mid_feature.shape[-3:], mode="trilinear", align_corners=False
        )
        mid_concat = torch.cat((mid_from_coarse, splat.mid_feature), dim=1)
        mid = self._run(self.mid_fusion, mid_concat)
        mid = self._run_blocks(self.mid_blocks, mid)
        high = self._run(self.mid_to_high, mid)
        high = F.interpolate(high, size=output_shape_zyx, mode="trilinear", align_corners=False)
        if high.shape[1] > self.max_fullres_channels:
            raise AssertionError("Decoder created a forbidden high-channel full-resolution tensor")
        high = self._run_blocks(self.high_blocks, high)
        residual = self.output(high)
        shapes = {
            "coarse": tuple(coarse.shape),
            "mid_from_coarse": tuple(mid_from_coarse.shape),
            "mid_concat": tuple(mid_concat.shape),
            "mid": tuple(mid.shape),
            "high": tuple(high.shape),
            "output": tuple(residual.shape),
        }
        return residual, shapes
