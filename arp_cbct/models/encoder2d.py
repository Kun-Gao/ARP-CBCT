from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(groups, channels), nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(groups, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x + self.net(x))


class MultiScaleEncoder2D(nn.Module):
    """Capacity-aligned shared encoder; view identity remains an explicit axis."""

    def __init__(self, channels: list[int], blocks_per_stage: int = 2, view_chunk_size: int | None = None):
        super().__init__()
        self.view_chunk_size = None if view_chunk_size is None else int(view_chunk_size)
        if self.view_chunk_size is not None and self.view_chunk_size <= 0:
            raise ValueError("view_chunk_size must be positive or None")
        stages, input_channels = [], 1
        for channels_out in channels:
            groups = min(8, channels_out)
            layers = [nn.Conv2d(input_channels, channels_out, 3, stride=2, padding=1, bias=False), nn.GroupNorm(groups, channels_out), nn.GELU()]
            layers.extend(ResidualBlock(channels_out) for _ in range(blocks_per_stage))
            stages.append(nn.Sequential(*layers))
            input_channels = channels_out
        self.stages = nn.ModuleList(stages)

    def _forward_flat(self, y: torch.Tensor) -> dict[str, torch.Tensor]:
        output = {}
        for index, stage in enumerate(self.stages, 1):
            y = stage(y)
            output[f"s{index}"] = y
        return output

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, v, _, _, _ = x.shape
        chunk = self.view_chunk_size
        if chunk is None or v <= chunk:
            return {key: value.unflatten(0, (b, v)) for key, value in self._forward_flat(x.flatten(0, 1)).items()}
        per_scale: dict[str, list[torch.Tensor]] = {}
        for start in range(0, v, chunk):
            stop = min(start + chunk, v)
            encoded = self._forward_flat(x[:, start:stop].flatten(0, 1))
            for key, value in encoded.items():
                per_scale.setdefault(key, []).append(value.unflatten(0, (b, stop - start)))
        return {key: torch.cat(parts, dim=1) for key, parts in per_scale.items()}
