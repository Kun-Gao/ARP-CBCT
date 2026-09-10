from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..geometry import ReconstructionGrid
from .primitive_types import PrimitiveSet
from .primitive_splatting import PrimitiveSplatting


def resampled_grid(grid: ReconstructionGrid, shape_zyx: tuple[int, int, int]) -> ReconstructionGrid:
    """Return a centered grid with the same physical voxel-edge extent."""
    if len(shape_zyx) != 3 or any(int(value) <= 0 for value in shape_zyx):
        raise ValueError(f"Invalid latent grid shape: {shape_zyx}")
    spacing = tuple(
        float(old_spacing) * int(old_size) / int(new_size)
        for old_size, old_spacing, new_size in zip(grid.shape_zyx, grid.spacing_zyx_mm, shape_zyx)
    )
    return ReconstructionGrid(tuple(map(int, shape_zyx)), spacing, grid.center_world_xyz_mm)


@dataclass
class MultiScaleSplatOutput:
    coarse_feature: torch.Tensor
    coarse_density: torch.Tensor
    coarse_weight: torch.Tensor
    mid_feature: torch.Tensor
    mid_density: torch.Tensor
    mid_weight: torch.Tensor
    coarse_grid: ReconstructionGrid
    mid_grid: ReconstructionGrid

class PrimitiveMultiScaleSplatting(nn.Module):
    """Splat one continuous primitive set onto two physical grids.

    Position, support, density and confidence are shared. Learned linear feature
    projections only adapt channel width; they do not create scale-specific
    primitive geometry.
    """

    def __init__(
        self,
        primitive_dim: int,
        coarse_channels: int,
        mid_channels: int,
        coarse_shape_zyx: tuple[int, int, int],
        mid_shape_zyx: tuple[int, int, int],
        chunk_size: int = 256,
        max_chunk_voxels: int = 2_000_000,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.coarse_shape_zyx = tuple(map(int, coarse_shape_zyx))
        self.mid_shape_zyx = tuple(map(int, mid_shape_zyx))
        self.coarse_projection = (
            nn.Identity() if primitive_dim == coarse_channels else nn.Linear(primitive_dim, coarse_channels)
        )
        self.mid_projection = nn.Identity() if primitive_dim == mid_channels else nn.Linear(primitive_dim, mid_channels)
        self.splat = PrimitiveSplatting(
            chunk_size=chunk_size,
            epsilon=epsilon,
            max_chunk_voxels=max_chunk_voxels,
        )

    def forward(self, primitives: PrimitiveSet, target_grid: ReconstructionGrid) -> MultiScaleSplatOutput:
        coarse_grid = resampled_grid(target_grid, self.coarse_shape_zyx)
        mid_grid = resampled_grid(target_grid, self.mid_shape_zyx)
        coarse = self.splat(primitives.updated(feature=self.coarse_projection(primitives.feature)), coarse_grid)
        mid = self.splat(primitives.updated(feature=self.mid_projection(primitives.feature)), mid_grid)
        return MultiScaleSplatOutput(
            coarse_feature=coarse[0],
            coarse_density=coarse[1],
            coarse_weight=coarse[2],
            mid_feature=mid[0],
            mid_density=mid[1],
            mid_weight=mid[2],
            coarse_grid=coarse_grid,
            mid_grid=mid_grid,
        )
