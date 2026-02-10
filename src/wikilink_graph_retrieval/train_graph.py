from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from wikilink_graph_retrieval.models.graphsage import GraphSage, GraphSageConfig, link_prediction_loss
from wikilink_graph_retrieval.utils import ensure_dir, save_json, set_reproducibility


class EdgeDataset(Dataset):
    def __init__(self, edges: np.ndarray):
        self.edges = edges.astype(np.int64)

    def __len__(self) -> int:
        return int(self.edges.shape[0])

    def __getitem__(self, idx: int) -> tuple[int, int]:
        u, v = self.edges[idx]
        return int(u), int(v)


def build_adjacency(num_nodes: int, edges: np.ndarray) -> list[torch.Tensor]:
    neigh = [[] for _ in range(num_nodes)]
    for u, v in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            neigh[int(u)].append(int(v))
    out = []
    for u in range(num_nodes):
        out.append(torch.tensor(neigh[u], dtype=torch.long))
    return out


def train_graphsage(
    *,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    out_dir: Path,
    seed: int,
    deterministic: bool,
    dim: int,
    layers: int,
    neighbors: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    neg_samples: int,
    device: torch.device,
) -> Path:
    ensure_dir(out_dir)
    set_reproducibility(seed, deterministic)

    num_nodes = int(nodes["doc_id"].max()) + 1
    edge_arr = edges[["src", "dst"]].to_numpy(dtype=np.int64)
    adjacency = build_adjacency(num_nodes, edge_arr)

    cfg = GraphSageConfig(num_nodes=num_nodes, dim=dim, layers=layers, neighbors=neighbors)
    model = GraphSage(cfg, adjacency).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = EdgeDataset(edge_arr)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    gen = torch.Generator(device=device.type)
    gen.manual_seed(seed)

    step = 0
    for epoch in range(epochs):
        model.train()
        for u, v in dl:
            u = u.to(device)
            v = v.to(device)
            # Negatives: random nodes.
            neg = torch.randint(0, num_nodes, (u.shape[0], neg_samples), generator=gen, device=device)

            z_u = model.embed_nodes(u, generator=gen)
            z_v = model.embed_nodes(v, generator=gen)
            z_neg = model.embed_nodes(neg.reshape(-1), generator=gen).reshape(u.shape[0], neg_samples, -1)

            loss = link_prediction_loss(z_u, z_v, z_neg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

    # Export deterministic node embeddings for all nodes.
    model.eval()
    with torch.no_grad():
        all_ids = torch.arange(num_nodes, device=device)
        emb = model.embed_nodes(all_ids, generator=gen).detach().cpu()

    emb_path = out_dir / "node_emb.pt"
    torch.save(emb, emb_path)
    save_json(
        out_dir / "meta.json",
        {
            "kind": "graphsage",
            "num_nodes": num_nodes,
            "dim": dim,
            "layers": layers,
            "neighbors": neighbors,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "neg_samples": neg_samples,
            "seed": seed,
            "deterministic": deterministic,
        },
    )
    return emb_path
