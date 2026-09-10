from __future__ import annotations

import torch
from torch import nn

from .primitive_proposal import PrimitiveSet


def knn_primitives(positions: torch.Tensor, k: int, method: str = "chunked_exact", chunk_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact KNN without ever materializing the complete N x N matrix."""
    if method != "chunked_exact":
        raise ValueError(f"CORE_V1 supports knn_method=chunked_exact, got {method!r}")
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must be [B,N,3]")
    b, n, _ = positions.shape
    if n < 2:
        raise ValueError("KNN interaction needs at least two primitives")
    k = min(int(k), n - 1)
    all_indices, all_distances = [], []
    for bi in range(b):
        batch_indices, batch_distances = [], []
        reference = positions[bi]
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            distance = torch.cdist(reference[start:stop], reference)
            local = torch.arange(stop - start, device=positions.device)
            distance[local, torch.arange(start, stop, device=positions.device)] = torch.inf
            values, indices = distance.topk(k, largest=False, sorted=True)
            batch_indices.append(indices)
            batch_distances.append(values)
        all_indices.append(torch.cat(batch_indices))
        all_distances.append(torch.cat(batch_distances))
    return torch.stack(all_indices), torch.stack(all_distances)


class PrimitiveInteraction(nn.Module):
    """Chunked exact local message passing over sparse primitives."""

    def __init__(self, dim: int, k: int = 8, method: str = "chunked_exact", chunk_size: int = 512):
        super().__init__()
        self.k, self.method, self.chunk_size = int(k), method, int(chunk_size)
        self.edge = nn.Sequential(nn.Linear(2 * dim + 4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.update = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, p: PrimitiveSet) -> PrimitiveSet:
        indices, _ = knn_primitives(p.position, self.k, self.method, self.chunk_size)
        self.last_indices = indices.detach()
        b, n, _ = p.position.shape
        outputs = []
        for bi in range(b):
            chunks = []
            for start in range(0, n, self.chunk_size):
                stop = min(start + self.chunk_size, n)
                idx = indices[bi, start:stop]
                neighbor = p.feature[bi][idx]
                relative = p.position[bi][idx] - p.position[bi, start:stop, None]
                distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
                center = p.feature[bi, start:stop, None].expand_as(neighbor)
                message = self.edge(torch.cat((center, neighbor, relative, distance), -1)).mean(1)
                chunks.append(p.feature[bi, start:stop] + self.update(torch.cat((p.feature[bi, start:stop], message), -1)))
            outputs.append(torch.cat(chunks))
        return p.updated(feature=torch.stack(outputs))
