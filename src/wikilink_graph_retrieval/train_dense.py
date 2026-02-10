from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from wikilink_graph_retrieval.models.dense_retriever import DenseRetriever, DenseRetrieverConfig, inbatch_contrastive_loss
from wikilink_graph_retrieval.models.fusion import FusionConfig
from wikilink_graph_retrieval.models.text_encoder import TextEncoderConfig
from wikilink_graph_retrieval.tokenizer import HashTokenizer
from wikilink_graph_retrieval.utils import ensure_dir, resolve_device, save_json, set_reproducibility


class PairDataset(Dataset):
    def __init__(self, links: pd.DataFrame, doc_text_by_id: dict[int, str]):
        self.q = links["query_text"].astype(str).tolist()
        self.doc_id = links["doc_id"].astype(np.int64).tolist()
        self.doc_text_by_id = doc_text_by_id

    def __len__(self) -> int:
        return len(self.q)

    def __getitem__(self, idx: int) -> tuple[str, int, str]:
        q = self.q[idx]
        did = int(self.doc_id[idx])
        dtext = self.doc_text_by_id[did]
        return q, did, dtext


def _collate(tokenizer: HashTokenizer, batch: list[tuple[str, int, str]]) -> dict[str, Any]:
    q_texts = [b[0] for b in batch]
    doc_ids = np.array([b[1] for b in batch], dtype=np.int64)
    d_texts = [b[2] for b in batch]
    q = tokenizer.batch_encode(q_texts)
    d = tokenizer.batch_encode(d_texts)
    return {
        "q_input_ids": torch.from_numpy(q.input_ids),
        "q_attention_mask": torch.from_numpy(q.attention_mask.astype(np.int64)),
        "d_input_ids": torch.from_numpy(d.input_ids),
        "d_attention_mask": torch.from_numpy(d.attention_mask.astype(np.int64)),
        "doc_ids": torch.from_numpy(doc_ids),
    }


def _save_checkpoint(out_dir: Path, model: DenseRetriever, extra: dict[str, Any]) -> None:
    ensure_dir(out_dir)
    torch.save(model.state_dict(), out_dir / "model.pt")
    save_json(out_dir / "meta.json", extra)


def train_dense_retriever(
    *,
    name: str,
    corpus: pd.DataFrame,
    train_links: pd.DataFrame,
    out_dir: Path,
    seed: int,
    deterministic: bool,
    device: str,
    vocab_size: int,
    max_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    dropout: float,
    embed_dim: int,
    l2_normalize: bool,
    use_graph: bool,
    fusion_method: str,
    graph_emb_path: Path | None,
    init_from: Path | None,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    grad_clip_norm: float,
    log_every_steps: int,
) -> Path:
    ensure_dir(out_dir)
    set_reproducibility(seed, deterministic)
    tdev = resolve_device(device)

    doc_text_by_id = dict(zip(corpus["doc_id"].astype(int).tolist(), corpus["text"].astype(str).tolist()))

    tokenizer = HashTokenizer(vocab_size=vocab_size, max_len=max_len)
    cfg = DenseRetrieverConfig(
        text=TextEncoderConfig(
            vocab_size=vocab_size,
            max_len=max_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            embed_dim=embed_dim,
        ),
        l2_normalize=l2_normalize,
        fusion=FusionConfig(method=fusion_method, embed_dim=embed_dim, graph_dim=embed_dim),
    )
    model = DenseRetriever(cfg).to(tdev)

    if init_from is not None:
        sd = torch.load(init_from / "model.pt", map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # We expect fusion params to be missing when init'ing from text-only.
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading init checkpoint: {unexpected[:5]}")

    graph_emb = None
    if use_graph:
        if graph_emb_path is None:
            raise ValueError("use_graph=True requires graph_emb_path")
        graph_emb = torch.load(graph_emb_path, map_location="cpu")
        if graph_emb.ndim != 2:
            raise ValueError("graph_emb must be [num_nodes, dim]")
        if graph_emb.shape[1] != embed_dim:
            # We keep graph and text dims equal in this baseline for simplicity.
            raise ValueError(f"graph_emb dim {graph_emb.shape[1]} != embed_dim {embed_dim}")
        graph_emb = graph_emb.to(tdev)

    ds = PairDataset(train_links, doc_text_by_id)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=lambda b: _collate(tokenizer, b),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    step = 0
    for epoch in range(epochs):
        model.train()
        for batch in dl:
            q_ids = batch["q_input_ids"].to(tdev)
            q_mask = batch["q_attention_mask"].to(tdev)
            d_ids = batch["d_input_ids"].to(tdev)
            d_mask = batch["d_attention_mask"].to(tdev)
            doc_ids = batch["doc_ids"].to(tdev)

            q = model.encode_queries(q_ids, q_mask)
            if graph_emb is None:
                d = model.encode_docs(d_ids, d_mask, None)
            else:
                d_graph = graph_emb[doc_ids]
                d = model.encode_docs(d_ids, d_mask, d_graph)

            sim = model.similarity(q, d)
            loss = inbatch_contrastive_loss(sim, temperature=temperature)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            opt.step()

            step += 1
            if log_every_steps > 0 and (step % log_every_steps == 0):
                # Keep logging minimal; pipeline collects final artifacts in eval step.
                print(f"[{name}] epoch={epoch+1}/{epochs} step={step} loss={loss.item():.4f}")

    ckpt_dir = out_dir
    _save_checkpoint(
        ckpt_dir,
        model,
        {
            "kind": "dense_retriever",
            "name": name,
            "use_graph": bool(use_graph),
            "fusion": fusion_method,
            "seed": seed,
            "deterministic": deterministic,
            "device": str(tdev),
            "model_cfg": asdict(cfg),
            "train_cfg": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "temperature": temperature,
                "grad_clip_norm": grad_clip_norm,
            },
        },
    )
    return ckpt_dir
