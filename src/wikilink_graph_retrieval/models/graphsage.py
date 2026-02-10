from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GraphSageConfig:
    num_nodes: int
    dim: int
    layers: int
    neighbors: int  # 0 => all neighbors (deterministic)


class GraphSage(nn.Module):
    """
    GraphSAGE-style neighbor-mean aggregator with learnable node embeddings.

    This implementation is designed to be simple and deterministic for small graphs
    (neighbors=0 => full mean over adjacency list).
    """

    def __init__(self, cfg: GraphSageConfig, adjacency: Sequence[torch.Tensor]):
        super().__init__()
        self.cfg = cfg
        self.adjacency = adjacency

        self.emb = nn.Embedding(cfg.num_nodes, cfg.dim)
        self.self_linears = nn.ModuleList([nn.Linear(cfg.dim, cfg.dim, bias=False) for _ in range(cfg.layers)])
        self.neigh_linears = nn.ModuleList([nn.Linear(cfg.dim, cfg.dim, bias=False) for _ in range(cfg.layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(cfg.dim) for _ in range(cfg.layers)])

    def _sample_neighbors(
        self, node_ids: torch.Tensor, num_neighbors: int, generator: torch.Generator | None
    ) -> torch.Tensor:
        # Returns a padded [B, num_neighbors] tensor of neighbor ids (may include duplicates).
        out = torch.empty((node_ids.shape[0], num_neighbors), dtype=torch.long, device=node_ids.device)
        for i, nid in enumerate(node_ids.tolist()):
            neigh = self.adjacency[nid]
            if neigh.numel() == 0:
                out[i].fill_(nid)  # self-loop fallback
                continue
            if neigh.numel() >= num_neighbors:
                idx = torch.randint(0, neigh.numel(), (num_neighbors,), generator=generator, device=node_ids.device)
                out[i] = neigh.to(node_ids.device)[idx]
            else:
                # Repeat neighbors to fill.
                reps = (num_neighbors + neigh.numel() - 1) // neigh.numel()
                tiled = neigh.repeat(reps)[:num_neighbors]
                out[i] = tiled.to(node_ids.device)
        return out

    def embed_nodes(self, node_ids: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        h = self.emb(node_ids)
        for layer in range(self.cfg.layers):
            if self.cfg.neighbors == 0:
                neigh_ids = []
                for nid in node_ids.tolist():
                    neigh_ids.append(self.adjacency[nid])
                neigh_mean = []
                for i, ids in enumerate(neigh_ids):
                    if ids.numel() == 0:
                        neigh_mean.append(h[i])
                    else:
                        neigh_mean.append(self.emb(ids.to(node_ids.device)).mean(dim=0))
                neigh_mean = torch.stack(neigh_mean, dim=0)
            else:
                sampled = self._sample_neighbors(node_ids, self.cfg.neighbors, generator=generator)
                neigh_mean = self.emb(sampled).mean(dim=1)

            h = self.self_linears[layer](h) + self.neigh_linears[layer](neigh_mean)
            h = self.ln[layer](h)
            h = F.relu(h)
        return h


def link_prediction_loss(z_u: torch.Tensor, z_v: torch.Tensor, z_neg: torch.Tensor) -> torch.Tensor:
    """
    z_u: [B, D], z_v: [B, D] positive neighbors, z_neg: [B, Nneg, D]
    """
    pos = torch.sum(z_u * z_v, dim=-1)  # [B]
    neg = torch.einsum("bd,bnd->bn", z_u, z_neg)  # [B, Nneg]
    # Maximize log sigmoid(pos) + sum log sigmoid(-neg)
    loss = -torch.mean(F.logsigmoid(pos) + torch.sum(F.logsigmoid(-neg), dim=1))
    return loss

