"""Wiki dump extraction (stub for smoke pipeline)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class ExtractConfig:
    pages_xml_bz2: Path
    pagelinks_sql_gz: Path
    out_dir: Path
    top_targets: int
    train_links: int
    val_links: int
    test_links: int
    context_window_tokens: int
    split: Literal["page_holdout"]


def extract_wikidump(cfg: ExtractConfig) -> None:
    """Extract a wikidump into the processed format.

    Stub: the smoke pipeline uses toy data; this is not needed.
    """
    raise NotImplementedError(
        "extract_wikidump not implemented. Use source=toy for smoke runs."
    )
