"""Generate a toy synthetic dataset for smoke tests."""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from wikilink_graph_retrieval.data.io import write_parquet

_TOPICS = [
    "machine learning", "deep learning", "neural network", "natural language processing",
    "computer vision", "reinforcement learning", "graph neural network", "transformer",
    "attention mechanism", "convolutional network", "recurrent network", "generative model",
    "classification", "regression", "clustering", "dimensionality reduction",
    "optimization", "gradient descent", "backpropagation", "regularization",
    "data augmentation", "transfer learning", "pre-training", "fine-tuning",
    "tokenization", "embedding", "encoder", "decoder", "autoencoder", "variational",
]


def _generate_text(rng: random.Random, context_window_tokens: int) -> str:
    """Generate a synthetic text of roughly context_window_tokens tokens."""
    words = []
    for _ in range(context_window_tokens):
        topic = rng.choice(_TOPICS)
        words.append(topic.split()[rng.randrange(len(topic.split()))])
    return " ".join(words)


def build_toy_dataset(
    out_dir: str | Path,
    *,
    seed: int,
    num_docs: int,
    num_edges: int,
    train_links: int,
    val_links: int,
    test_links: int,
    context_window_tokens: int,
) -> None:
    """Build a synthetic toy retrieval dataset and write parquet files."""
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate target pages (corpus)
    docs = []
    for i in range(num_docs):
        docs.append({
            "doc_id": i,
            "title": f"Page_{i}",
            "text": _generate_text(rng, context_window_tokens),
        })
    corpus_df = pd.DataFrame(docs)
    write_parquet(corpus_df, out_dir / "target_pages.parquet")

    # Generate page link edges (graph structure)
    edge_set: set[Tuple[int, int]] = set()
    while len(edge_set) < num_edges:
        src = rng.randrange(num_docs)
        tgt = rng.randrange(num_docs)
        if src != tgt:
            edge_set.add((src, tgt))
    edges = [{"src": s, "dst": t} for s, t in edge_set]
    edges_df = pd.DataFrame(edges)
    write_parquet(edges_df, out_dir / "pagelinks_edges.parquet")

    # Generate train/val/test link retrieval pairs
    # Each link: (source_doc_id, context_text, target_doc_id)
    def _make_links(n: int) -> List[dict]:
        links = []
        for _ in range(n):
            src = rng.randrange(num_docs)
            # Pick a target that has an edge from src (if any), otherwise random
            src_targets = [t for s, t in edge_set if s == src]
            if src_targets:
                tgt = rng.choice(src_targets)
            else:
                tgt = rng.randrange(num_docs)
                if tgt == src:
                    tgt = (tgt + 1) % num_docs
            context = _generate_text(rng, context_window_tokens)
            links.append({
                "query_text": context,
                "doc_id": tgt,
                "source_doc_id": src,
            })
        return links

    train_df = pd.DataFrame(_make_links(train_links))
    val_df = pd.DataFrame(_make_links(val_links))
    test_df = pd.DataFrame(_make_links(test_links))

    write_parquet(train_df, out_dir / "train_links.parquet")
    write_parquet(val_df, out_dir / "val_links.parquet")
    write_parquet(test_df, out_dir / "test_links.parquet")

    print(f"Toy dataset: {num_docs} docs, {len(edge_set)} edges, "
          f"{train_links}/{val_links}/{test_links} train/val/test links -> {out_dir}")
