from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    processed_dir: Path
    index_dir: Path
    runs_dir: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class ToyDataConfig:
    num_docs: int
    num_edges: int
    train_links: int
    val_links: int
    test_links: int
    context_window_tokens: int


@dataclass(frozen=True)
class WikiDumpUrls:
    pages_xml: str
    pagelinks_sql: str


@dataclass(frozen=True)
class WikiDumpDataConfig:
    urls: WikiDumpUrls
    top_targets: int
    train_links: int
    val_links: int
    test_links: int
    context_window_tokens: int
    split: Literal["page_holdout"]


@dataclass(frozen=True)
class DataConfig:
    source: Literal["toy", "wikidump"]
    toy: ToyDataConfig | None = None
    wikidump: WikiDumpDataConfig | None = None


@dataclass(frozen=True)
class SparseConfig:
    max_features: int


@dataclass(frozen=True)
class DenseConfig:
    vocab_size: int
    max_len: int
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float
    embed_dim: int
    l2_normalize: bool


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    temperature: float
    grad_clip_norm: float
    log_every_steps: int


@dataclass(frozen=True)
class GraphConfig:
    dim: int
    layers: int
    neighbors: int  # 0 => all neighbors
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    neg_samples: int


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    kind: Literal["sparse_tfidf", "dense"]
    use_graph: bool | None = None
    fusion: Literal["none", "gated_residual", "concat_mlp"] | None = None


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    deterministic: bool
    device: Literal["auto", "cpu", "cuda"]
    paths: PathsConfig
    data: DataConfig
    sparse: SparseConfig
    dense: DenseConfig
    train: TrainConfig
    graph: GraphConfig
    experiments: list[ExperimentConfig]


def _p(v: str | Path) -> Path:
    return v if isinstance(v, Path) else Path(v)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def load_config(path: str | Path) -> ProjectConfig:
    path = _p(path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    paths = PathsConfig(
        raw_dir=_p(raw["paths"]["raw_dir"]),
        processed_dir=_p(raw["paths"]["processed_dir"]),
        index_dir=_p(raw["paths"]["index_dir"]),
        runs_dir=_p(raw["paths"]["runs_dir"]),
        artifacts_dir=_p(raw["paths"]["artifacts_dir"]),
    )

    data_raw = raw["data"]
    source = data_raw["source"]
    _require(source in ("toy", "wikidump"), f"data.source must be toy|wikidump, got {source!r}")

    toy = None
    wikidump = None
    if source == "toy":
        t = data_raw["toy"]
        toy = ToyDataConfig(
            num_docs=int(t["num_docs"]),
            num_edges=int(t["num_edges"]),
            train_links=int(t["train_links"]),
            val_links=int(t["val_links"]),
            test_links=int(t["test_links"]),
            context_window_tokens=int(t["context_window_tokens"]),
        )
    else:
        w = data_raw["wikidump"]
        urls = WikiDumpUrls(
            pages_xml=str(w["urls"]["pages_xml"]),
            pagelinks_sql=str(w["urls"]["pagelinks_sql"]),
        )
        wikidump = WikiDumpDataConfig(
            urls=urls,
            top_targets=int(w["top_targets"]),
            train_links=int(w["train_links"]),
            val_links=int(w["val_links"]),
            test_links=int(w["test_links"]),
            context_window_tokens=int(w["context_window_tokens"]),
            split=str(w.get("split", "page_holdout")),  # type: ignore[arg-type]
        )

    data = DataConfig(source=source, toy=toy, wikidump=wikidump)

    sparse = SparseConfig(max_features=int(raw["sparse"]["max_features"]))
    dense = DenseConfig(
        vocab_size=int(raw["dense"]["vocab_size"]),
        max_len=int(raw["dense"]["max_len"]),
        d_model=int(raw["dense"]["d_model"]),
        n_layers=int(raw["dense"]["n_layers"]),
        n_heads=int(raw["dense"]["n_heads"]),
        dropout=float(raw["dense"]["dropout"]),
        embed_dim=int(raw["dense"]["embed_dim"]),
        l2_normalize=bool(raw["dense"]["l2_normalize"]),
    )
    train = TrainConfig(
        epochs=int(raw["train"]["epochs"]),
        batch_size=int(raw["train"]["batch_size"]),
        lr=float(raw["train"]["lr"]),
        weight_decay=float(raw["train"]["weight_decay"]),
        temperature=float(raw["train"]["temperature"]),
        grad_clip_norm=float(raw["train"]["grad_clip_norm"]),
        log_every_steps=int(raw["train"]["log_every_steps"]),
    )
    graph = GraphConfig(
        dim=int(raw["graph"]["dim"]),
        layers=int(raw["graph"]["layers"]),
        neighbors=int(raw["graph"]["neighbors"]),
        epochs=int(raw["graph"]["epochs"]),
        batch_size=int(raw["graph"]["batch_size"]),
        lr=float(raw["graph"]["lr"]),
        weight_decay=float(raw["graph"]["weight_decay"]),
        neg_samples=int(raw["graph"]["neg_samples"]),
    )

    experiments = []
    for e in raw["experiments"]:
        experiments.append(
            ExperimentConfig(
                name=str(e["name"]),
                kind=str(e["kind"]),  # type: ignore[arg-type]
                use_graph=e.get("use_graph"),
                fusion=e.get("fusion"),
            )
        )
    _require(any(e.kind == "sparse_tfidf" for e in experiments), "config must include sparse_tfidf baseline")
    _require(any(e.kind == "dense" and e.use_graph is False for e in experiments), "config must include dense text-only ablation")

    device = raw.get("device", "auto")
    _require(device in ("auto", "cpu", "cuda"), f"device must be auto|cpu|cuda, got {device!r}")

    return ProjectConfig(
        seed=int(raw["seed"]),
        deterministic=bool(raw["deterministic"]),
        device=device,  # type: ignore[arg-type]
        paths=paths,
        data=data,
        sparse=sparse,
        dense=dense,
        train=train,
        graph=graph,
        experiments=experiments,
    )
