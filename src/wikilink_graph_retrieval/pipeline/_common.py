from __future__ import annotations

import argparse
from pathlib import Path

from wikilink_graph_retrieval.config import ProjectConfig, load_config
from wikilink_graph_retrieval.utils import ensure_dir


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g. configs/smoke.yaml)")
    return p.parse_args()


def load_project_config(path: str | Path) -> ProjectConfig:
    cfg = load_config(path)
    ensure_dir(cfg.paths.raw_dir)
    ensure_dir(cfg.paths.processed_dir)
    ensure_dir(cfg.paths.index_dir)
    ensure_dir(cfg.paths.runs_dir)
    ensure_dir(cfg.paths.artifacts_dir)
    return cfg

