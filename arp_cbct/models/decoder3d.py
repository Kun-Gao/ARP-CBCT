from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        self.net = nn.Sequential(nn.Conv3d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(groups, channels), nn.GELU(), nn.Conv3d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(groups, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class Decoder3D(nn.Module):
    """Moderate two-level ResUNet operating only on primitive-splatted features."""

    def __init__(
        self,
        in_dim: int,
        base_channels: int = 48,
        bottleneck_blocks: int = 6,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        b, m, high = base_channels, 2 * base_channels, 4 * base_channels
        self.stem = nn.Sequential(nn.Conv3d(in_dim, b, 3, padding=1, bias=False), nn.GroupNorm(min(8, b), b), nn.GELU(), ResidualBlock3D(b))
        self.down1 = nn.Sequential(nn.Conv3d(b, m, 3, stride=2, padding=1, bias=False), nn.GroupNorm(min(8, m), m), nn.GELU(), ResidualBlock3D(m))
        self.down2 = nn.Sequential(nn.Conv3d(m, high, 3, stride=2, padding=1, bias=False), nn.GroupNorm(min(8, high), high), nn.GELU())
        self.bottleneck = nn.Sequential(*(ResidualBlock3D(high) for _ in range(bottleneck_blocks)))
        self.up2 = nn.Conv3d(high, m, 3, padding=1, bias=False)
        self.fuse2 = nn.Sequential(nn.Conv3d(2 * m, m, 3, padding=1, bias=False), nn.GroupNorm(min(8, m), m), nn.GELU(), ResidualBlock3D(m))
        self.up1 = nn.Conv3d(m, b, 3, padding=1, bias=False)
        self.fuse1 = nn.Sequential(nn.Conv3d(2 * b, b, 3, padding=1, bias=False), nn.GroupNorm(min(8, b), b), nn.GELU(), ResidualBlock3D(b))
        self.output = nn.Conv3d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_checkpoint = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        run = (lambda module, value: checkpoint(module, value, use_reentrant=False)) if use_checkpoint else (lambda module, value: module(value))
        skip1 = run(self.stem, x)
        skip2 = run(self.down1, skip1)
        y = run(self.down2, skip2)
        y = checkpoint_sequential(self.bottleneck, len(self.bottleneck), y, use_reentrant=False) if use_checkpoint else self.bottleneck(y)
        y = self.up2(F.interpolate(y, size=skip2.shape[-3:], mode="trilinear", align_corners=False))
        y = run(self.fuse2, torch.cat((y, skip2), 1))
        y = self.up1(F.interpolate(y, size=skip1.shape[-3:], mode="trilinear", align_corners=False))
        return self.output(run(self.fuse1, torch.cat((y, skip1), 1)))
