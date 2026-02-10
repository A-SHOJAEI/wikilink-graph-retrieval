Overwrote `README.md` with a project-specific, artifacts-grounded writeup.

What it includes (tied to the actual repo outputs and code):
- Problem statement and exact task/metrics as implemented (`src/wikilink_graph_retrieval/metrics.py`, `src/wikilink_graph_retrieval/eval_dense.py`)
- Dataset provenance for both the toy smoke dataset and the Wikimedia-dumps path, including the exact dump URLs from `configs/full.yaml` and what the extractor really does (`src/wikilink_graph_retrieval/data/wikidump_extract.py`)
- Methodology details matching the code (TF‑IDF settings, hash tokenizer, Transformer bi-encoder, GraphSAGE training, gated residual fusion)
- Baselines/ablations exactly as configured (`sparse_tfidf`, `dense_text_only`, `dense_text_graph`)
- The exact results table copied from `artifacts/report.md` / `artifacts/results.json` (generated `2026-02-10T09:32:13Z`)
- Repro commands (`make setup`, `make all`, `make all CONFIG=...`) and expected outputs
- Limitations that reflect current implementation (notably the placeholder wikidump graph edges and context-window behavior)
- Concrete next research steps

I also verified unit tests run in the repo venv: `.venv/bin/pytest -q` (5 passed).