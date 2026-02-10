from __future__ import annotations

from pathlib import Path

from wikilink_graph_retrieval.pipeline._common import load_project_config, parse_args
from wikilink_graph_retrieval.utils import load_json, atomic_write_text


def _fmt(x: float | int | None) -> str:
    if x is None:
        return "-"
    if isinstance(x, int):
        return str(x)
    return f"{x:.4f}"


def main() -> None:
    args = parse_args("Generate a human-readable report from artifacts/results.json.")
    cfg = load_project_config(args.config)

    res_path = cfg.paths.artifacts_dir / "results.json"
    res = load_json(res_path)

    rows = []
    for r in res["runs"]:
        m = r.get("metrics", {})
        t = r.get("throughput", {})
        rows.append(
            {
                "name": r["name"],
                "kind": r["kind"],
                "recall@1": m.get("recall@1"),
                "recall@10": m.get("recall@10"),
                "recall@100": m.get("recall@100"),
                "mrr@100": m.get("mrr@100"),
                "ndcg@100": m.get("ndcg@100"),
                "ece_top1": m.get("ece_top1"),
                "docs/sec": t.get("docs_per_sec"),
                "queries/sec": t.get("queries_per_sec"),
            }
        )

    header = [
        "name",
        "kind",
        "recall@1",
        "recall@10",
        "recall@100",
        "mrr@100",
        "ndcg@100",
        "ece_top1",
        "docs/sec",
        "queries/sec",
    ]
    md = []
    md.append(f"# Wikilink Graph Retrieval Report\n")
    md.append(f"- Config: `{res.get('config_path')}`\n")
    md.append(f"- Data: `{res.get('data_dir')}`\n")
    md.append(f"- Generated: `{res.get('generated_at')}`\n")
    md.append("\n")

    md.append("## Results\n\n")
    md.append("| " + " | ".join(header) + " |\n")
    md.append("| " + " | ".join(["---"] * len(header)) + " |\n")
    for row in rows:
        md.append(
            "| "
            + " | ".join(
                [
                    str(row[h]) if h in ("name", "kind") else _fmt(row[h])
                    for h in header
                ]
            )
            + " |\n"
        )

    md.append("\n")
    md.append("## Notes\n\n")
    md.append("- `sparse_tfidf` is the lexical baseline described in the project plan.\n")
    md.append("- `dense_text_only` is the required ablation: graph embeddings + fusion disabled.\n")
    md.append("- `dense_text_graph` is the main model: text bi-encoder with graph-augmented doc representations (gated residual fusion).\n")
    md.append("- ECE is computed from an approximate softmax over the retrieved top-k candidate set.\n")
    md.append("\n")

    out_path = cfg.paths.artifacts_dir / "report.md"
    atomic_write_text(out_path, "".join(md))


if __name__ == "__main__":
    main()

