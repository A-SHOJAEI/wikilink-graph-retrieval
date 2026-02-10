from __future__ import annotations

from pathlib import Path

import pandas as pd

from wikilink_graph_retrieval.data.io import read_parquet
from wikilink_graph_retrieval.pipeline._common import load_project_config, parse_args
from wikilink_graph_retrieval.sparse_tfidf import build_tfidf_index
from wikilink_graph_retrieval.train_dense import train_dense_retriever
from wikilink_graph_retrieval.train_graph import train_graphsage
from wikilink_graph_retrieval.utils import resolve_device, save_json


def main() -> None:
    args = parse_args("Train baseline and dense retrievers.")
    cfg = load_project_config(args.config)

    corpus = read_parquet(cfg.paths.processed_dir / "target_pages.parquet")
    train_links = read_parquet(cfg.paths.processed_dir / "train_links.parquet")
    edges = read_parquet(cfg.paths.processed_dir / "pagelinks_edges.parquet")

    # 1) Baseline: sparse TF-IDF index build
    build_tfidf_index(corpus, index_dir=cfg.paths.index_dir / "tfidf", max_features=cfg.sparse.max_features)

    # 2) Graph embeddings (only if any experiment uses graph)
    need_graph = any((e.kind == "dense" and bool(e.use_graph)) for e in cfg.experiments)
    graph_emb_path = None
    if need_graph:
        if cfg.graph.dim != cfg.dense.embed_dim:
            raise ValueError(
                f"configs require graph.dim == dense.embed_dim for this baseline "
                f"(got {cfg.graph.dim} vs {cfg.dense.embed_dim})."
            )
        graph_dir = cfg.paths.runs_dir / "graph"
        graph_emb_path = train_graphsage(
            nodes=corpus[["doc_id"]],
            edges=edges,
            out_dir=graph_dir,
            seed=cfg.seed,
            deterministic=cfg.deterministic,
            dim=cfg.graph.dim,
            layers=cfg.graph.layers,
            neighbors=cfg.graph.neighbors,
            epochs=cfg.graph.epochs,
            batch_size=cfg.graph.batch_size,
            lr=cfg.graph.lr,
            weight_decay=cfg.graph.weight_decay,
            neg_samples=cfg.graph.neg_samples,
            device=resolve_device(cfg.device),
        )

    # 3) Dense models
    text_only_dir = None
    for e in cfg.experiments:
        if e.kind != "dense":
            continue
        out_dir = cfg.paths.runs_dir / e.name
        use_graph = bool(e.use_graph)
        fusion = e.fusion or ("gated_residual" if use_graph else "none")

        init_from = None
        if use_graph:
            # Initialize from the text-only ablation if present.
            if text_only_dir is not None:
                init_from = text_only_dir
        ckpt = train_dense_retriever(
            name=e.name,
            corpus=corpus,
            train_links=train_links,
            out_dir=out_dir,
            seed=cfg.seed,
            deterministic=cfg.deterministic,
            device=cfg.device,
            vocab_size=cfg.dense.vocab_size,
            max_len=cfg.dense.max_len,
            d_model=cfg.dense.d_model,
            n_layers=cfg.dense.n_layers,
            n_heads=cfg.dense.n_heads,
            dropout=cfg.dense.dropout,
            embed_dim=cfg.dense.embed_dim,
            l2_normalize=cfg.dense.l2_normalize,
            use_graph=use_graph,
            fusion_method=fusion,
            graph_emb_path=graph_emb_path,
            init_from=init_from,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            temperature=cfg.train.temperature,
            grad_clip_norm=cfg.train.grad_clip_norm,
            log_every_steps=cfg.train.log_every_steps,
        )
        if not use_graph:
            text_only_dir = ckpt

    save_json(cfg.paths.runs_dir / "last_train_config.json", {"config": str(args.config)})


if __name__ == "__main__":
    main()
