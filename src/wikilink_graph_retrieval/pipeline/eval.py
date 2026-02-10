from __future__ import annotations

from pathlib import Path

from wikilink_graph_retrieval.data.io import read_parquet
from wikilink_graph_retrieval.eval_dense import eval_dense_retriever
from wikilink_graph_retrieval.pipeline._common import load_project_config, parse_args
from wikilink_graph_retrieval.sparse_tfidf import eval_tfidf
from wikilink_graph_retrieval.utils import now_iso, save_json


def main() -> None:
    args = parse_args("Evaluate retrievers and write machine-readable results.")
    cfg = load_project_config(args.config)

    corpus = read_parquet(cfg.paths.processed_dir / "target_pages.parquet")
    test_links = read_parquet(cfg.paths.processed_dir / "test_links.parquet")

    graph_emb_path = cfg.paths.runs_dir / "graph" / "node_emb.pt"

    runs = []
    for e in cfg.experiments:
        if e.kind == "sparse_tfidf":
            out = eval_tfidf(index_dir=cfg.paths.index_dir / "tfidf", queries=test_links, k=100)
            runs.append({"name": e.name, "kind": e.kind, **out})
        elif e.kind == "dense":
            ckpt_dir = cfg.paths.runs_dir / e.name
            use_graph = bool(e.use_graph)
            fusion = e.fusion or ("gated_residual" if use_graph else "none")
            out = eval_dense_retriever(
                name=e.name,
                ckpt_dir=ckpt_dir,
                corpus=corpus,
                queries=test_links,
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
                graph_emb_path=graph_emb_path if use_graph else None,
                k=100,
            )
            runs.append({"name": e.name, "kind": e.kind, "use_graph": use_graph, "fusion": fusion, **out})
        else:
            raise ValueError(f"Unknown experiment kind: {e.kind}")

    results = {
        "generated_at": now_iso(),
        "config_path": str(args.config),
        "data_dir": str(cfg.paths.processed_dir),
        "runs": runs,
    }

    save_json(cfg.paths.artifacts_dir / "results.json", results)


if __name__ == "__main__":
    main()

