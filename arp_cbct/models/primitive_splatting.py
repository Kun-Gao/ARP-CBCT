from __future__ import annotations

import torch
from torch import nn

from ..geometry import ReconstructionGrid
from .primitive_types import PrimitiveSet


def _chunk_contributions(position, support, confidence, center_lo, spacing_xyz, shape_zyx, start, requested_stop, max_chunk_voxels, support_basis=None):
    d, h, w = shape_zyx
    stop = requested_stop
    while True:
        sigma = support[start:stop].clamp_min(1e-4)
        basis = support_basis[start:stop] if support_basis is not None else torch.eye(3, device=position.device, dtype=position.dtype)[None].expand(stop - start, -1, -1)
        radius_mm = 3 * torch.sqrt((basis.square() * sigma[..., None].square()).sum(-2))
        radius_vox = torch.ceil(radius_mm / spacing_xyz).long()
        maximum = radius_vox.amax(0)
        count = int(((2 * maximum + 1).prod() * (stop - start)).item())
        if count <= max_chunk_voxels or stop - start == 1:
            break
        stop = start + max(1, (stop - start) // 2)
    ox = torch.arange(-maximum[0], maximum[0] + 1, device=position.device)
    oy = torch.arange(-maximum[1], maximum[1] + 1, device=position.device)
    oz = torch.arange(-maximum[2], maximum[2] + 1, device=position.device)
    zz, yy, xx = torch.meshgrid(oz, oy, ox, indexing="ij")
    offsets = torch.stack((xx, yy, zz), -1).reshape(-1, 3)
    centers = torch.round((position[start:stop] - center_lo) / spacing_xyz).long()
    indices = centers[:, None] + offsets[None]
    valid = (offsets[None].abs() <= radius_vox[:, None]).all(-1)
    valid &= ((indices >= 0) & (indices < torch.tensor((w, h, d), device=position.device))).all(-1)
    primitive_index, offset_index = valid.nonzero(as_tuple=True)
    selected_index = indices[primitive_index, offset_index]
    world = center_lo + selected_index.to(position.dtype) * spacing_xyz
    delta = world - position[start + primitive_index]
    selected_sigma = sigma[primitive_index]
    selected_basis = basis[primitive_index]
    local_delta = torch.einsum("nij,nj->ni", selected_basis, delta)
    gaussian_base = torch.exp(-0.5 * ((local_delta / selected_sigma) ** 2).sum(-1))
    gaussian = gaussian_base * confidence[start + primitive_index, 0]
    linear = selected_index[:, 2] * h * w + selected_index[:, 1] * w + selected_index[:, 0]
    return stop, primitive_index, linear, delta, local_delta, selected_sigma, selected_basis, gaussian_base, gaussian


class _MemoryBoundSplat(torch.autograd.Function):
    """Analytic backward avoids retaining every primitive-voxel contribution."""

    @staticmethod
    def forward(ctx, position, support, support_basis, feature, density, confidence, shape_zyx, spacing_zyx, center_xyz, chunk_size, max_chunk_voxels, channel_chunk, epsilon):
        b, n, c = feature.shape
        d, h, w = shape_zyx
        center = torch.tensor(center_xyz, device=position.device, dtype=position.dtype)
        spacing_xyz = torch.tensor(spacing_zyx[::-1], device=position.device, dtype=position.dtype)
        center_lo = center - (torch.tensor((w, h, d), device=position.device, dtype=position.dtype) - 1) * spacing_xyz / 2
        latent_num = torch.zeros((b, c, d * h * w), device=position.device, dtype=position.dtype)
        density_num = torch.zeros((b, 1, d * h * w), device=position.device, dtype=position.dtype)
        weight = torch.zeros((b, 1, d * h * w), device=position.device, dtype=position.dtype)
        for bi in range(b):
            start = 0
            while start < n:
                stop, primitive_index, linear, _, _, _, _, _, gaussian = _chunk_contributions(position[bi], support[bi], confidence[bi], center_lo, spacing_xyz, shape_zyx, start, min(start + chunk_size, n), max_chunk_voxels, support_basis[bi])
                global_index = start + primitive_index
                weight[bi, 0].scatter_add_(0, linear, gaussian)
                density_num[bi, 0].scatter_add_(0, linear, gaussian * density[bi, global_index, 0])
                for cs in range(0, c, channel_chunk):
                    ce = min(cs + channel_chunk, c)
                    latent_num[bi, cs:ce].scatter_add_(1, linear[None].expand(ce - cs, -1), (feature[bi, global_index, cs:ce] * gaussian[:, None]).T)
                start = stop
        denominator = weight + epsilon
        latent = latent_num / denominator
        attenuation = density_num / denominator
        ctx.save_for_backward(position, support, support_basis, feature, density, confidence, latent, attenuation, weight)
        ctx.args = (shape_zyx, spacing_zyx, center_xyz, chunk_size, max_chunk_voxels, epsilon)
        return latent.view(b, c, d, h, w), attenuation.view(b, 1, d, h, w), weight.view(b, 1, d, h, w)

    @staticmethod
    def backward(ctx, grad_latent, grad_attenuation, grad_weight):
        position, support, support_basis, feature, density, confidence, latent, attenuation, weight = ctx.saved_tensors
        shape_zyx, spacing_zyx, center_xyz, chunk_size, max_chunk_voxels, epsilon = ctx.args
        b, n, c = feature.shape
        d, h, w = shape_zyx
        center = torch.tensor(center_xyz, device=position.device, dtype=position.dtype)
        spacing_xyz = torch.tensor(spacing_zyx[::-1], device=position.device, dtype=position.dtype)
        center_lo = center - (torch.tensor((w, h, d), device=position.device, dtype=position.dtype) - 1) * spacing_xyz / 2
        grad_latent = torch.zeros_like(latent) if grad_latent is None else grad_latent.reshape_as(latent)
        grad_attenuation = torch.zeros_like(attenuation) if grad_attenuation is None else grad_attenuation.reshape_as(attenuation)
        grad_weight = torch.zeros_like(weight) if grad_weight is None else grad_weight.reshape_as(weight)
        denominator = weight + epsilon
        grad_weight_effective = grad_weight - (grad_latent * latent).sum(1, keepdim=True) / denominator - grad_attenuation * attenuation / denominator
        grad_position = torch.zeros_like(position)
        grad_support = torch.zeros_like(support)
        grad_feature = torch.zeros_like(feature)
        grad_density = torch.zeros_like(density)
        grad_confidence = torch.zeros_like(confidence)
        for bi in range(b):
            start = 0
            while start < n:
                stop, primitive_index, linear, delta, local_delta, sigma, basis, gaussian_base, gaussian = _chunk_contributions(position[bi], support[bi], confidence[bi], center_lo, spacing_xyz, shape_zyx, start, min(start + chunk_size, n), max_chunk_voxels, support_basis[bi])
                global_index = start + primitive_index
                grad_numerator_feature = (grad_latent[bi, :, linear] / denominator[bi, :, linear]).T
                grad_numerator_density = (grad_attenuation[bi, 0, linear] / denominator[bi, 0, linear])
                grad_g = (grad_numerator_feature * feature[bi, global_index]).sum(-1)
                grad_g += grad_numerator_density * density[bi, global_index, 0] + grad_weight_effective[bi, 0, linear]
                grad_feature[bi].index_add_(0, global_index, (gaussian[:, None] * grad_numerator_feature).to(grad_feature.dtype))
                grad_density[bi, :, 0].index_add_(0, global_index, (gaussian * grad_numerator_density).to(grad_density.dtype))
                grad_confidence[bi, :, 0].index_add_(0, global_index, (grad_g * gaussian_base).to(grad_confidence.dtype))
                scaled = grad_g * gaussian
                position_direction = torch.einsum("nij,ni->nj", basis, local_delta / sigma.square())
                grad_position[bi].index_add_(0, global_index, (scaled[:, None] * position_direction).to(grad_position.dtype))
                grad_support[bi].index_add_(0, global_index, (scaled[:, None] * local_delta.square() / sigma.pow(3)).to(grad_support.dtype))
                start = stop
        return grad_position, grad_support, None, grad_feature, grad_density, grad_confidence, None, None, None, None, None, None, None


class PrimitiveSplatting(nn.Module):
    """Chunked local 3-sigma splat with stable normalized attenuation."""

    def __init__(self, chunk_size: int = 256, epsilon: float = 1e-6, max_chunk_voxels: int = 2_000_000, channel_chunk: int = 16):
        super().__init__()
        self.chunk_size, self.epsilon = int(chunk_size), float(epsilon)
        self.max_chunk_voxels, self.channel_chunk = int(max_chunk_voxels), int(channel_chunk)

    def forward(self, p: PrimitiveSet, grid: ReconstructionGrid) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        basis = p.support_basis
        if basis is None:
            basis = torch.eye(3, device=p.position.device, dtype=p.position.dtype)[None, None].expand(*p.position.shape[:2], -1, -1)
        return _MemoryBoundSplat.apply(p.position, p.support, basis, p.feature, p.density, p.confidence, grid.shape_zyx, grid.spacing_zyx_mm, grid.center_world_xyz_mm, self.chunk_size, self.max_chunk_voxels, self.channel_chunk, self.epsilon)


def slow_reference_splat(p: PrimitiveSet, grid: ReconstructionGrid, epsilon: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, n, c = p.feature.shape
    d, h, w = grid.shape_zyx
    lo, _ = grid.center_bounds_xyz(p.position.device, p.position.dtype)
    spacing = torch.tensor(grid.spacing_zyx_mm[::-1], device=p.position.device, dtype=p.position.dtype)
    latent = torch.zeros((b, c, d, h, w), device=p.position.device, dtype=p.position.dtype)
    density_num = torch.zeros((b, 1, d, h, w), device=p.position.device, dtype=p.position.dtype)
    weight = torch.zeros((b, 1, d, h, w), device=p.position.device, dtype=p.position.dtype)
    for bi in range(b):
        for pi in range(n):
            sigma = p.support[bi, pi]
            basis = p.support_basis[bi, pi] if p.support_basis is not None else torch.eye(3, device=p.position.device, dtype=p.position.dtype)
            radius = torch.ceil(3 * torch.sqrt((basis.square() * sigma[:, None].square()).sum(0)) / spacing).long()
            center = torch.round((p.position[bi, pi] - lo) / spacing).long()
            x = torch.arange(max(0, int(center[0] - radius[0])), min(w, int(center[0] + radius[0] + 1)), device=p.position.device)
            y = torch.arange(max(0, int(center[1] - radius[1])), min(h, int(center[1] + radius[1] + 1)), device=p.position.device)
            z = torch.arange(max(0, int(center[2] - radius[2])), min(d, int(center[2] + radius[2] + 1)), device=p.position.device)
            zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
            xyz = torch.stack((xx, yy, zz), -1)
            world = lo + xyz * spacing
            local = torch.einsum("ij,...j->...i", basis, world - p.position[bi, pi])
            gaussian = torch.exp(-0.5 * ((local / sigma) ** 2).sum(-1)) * p.confidence[bi, pi, 0]
            weight[bi, 0, zz, yy, xx] += gaussian
            density_num[bi, 0, zz, yy, xx] += gaussian * p.density[bi, pi, 0]
            latent[bi, :, zz, yy, xx] += p.feature[bi, pi, :, None, None, None] * gaussian
    return latent / (weight + epsilon), density_num / (weight + epsilon), weight
