from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FusionConfig:
    method: str  # "none" | "gated_residual" | "concat_mlp"
    embed_dim: int
    graph_dim: int


class GraphFusion(nn.Module):
    """
    Query-independent fusion of doc text embedding with a doc node embedding.

    This keeps doc embeddings indexable (precomputable).
    """

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.method not in ("none", "gated_residual", "concat_mlp"):
            raise ValueError(f"unknown fusion method: {cfg.method}")

        if cfg.method == "none":
            self.graph_proj = None
            self.gate = None
            self.mlp = None
        elif cfg.method == "gated_residual":
            self.graph_proj = nn.Linear(cfg.graph_dim, cfg.embed_dim, bias=False)
            self.gate = nn.Sequential(
                nn.Linear(cfg.embed_dim + cfg.embed_dim, cfg.embed_dim),
                nn.SiLU(),
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
                nn.Sigmoid(),
            )
            self.mlp = None
        else:
            self.graph_proj = nn.Linear(cfg.graph_dim, cfg.embed_dim, bias=False)
            self.gate = None
            self.mlp = nn.Sequential(
                nn.Linear(cfg.embed_dim + cfg.embed_dim, cfg.embed_dim),
                nn.SiLU(),
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
            )

    def forward(self, doc_text: torch.Tensor, doc_graph: torch.Tensor | None) -> torch.Tensor:
        if self.cfg.method == "none" or doc_graph is None:
            return doc_text

        g = self.graph_proj(doc_graph)
        if self.cfg.method == "gated_residual":
            gate = self.gate(torch.cat([doc_text, g], dim=-1))
            return doc_text + gate * g
        return self.mlp(torch.cat([doc_text, g], dim=-1))

