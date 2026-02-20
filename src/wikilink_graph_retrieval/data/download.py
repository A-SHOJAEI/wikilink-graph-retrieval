"""Download helpers (stub for wikidump; not needed for toy data)."""
from __future__ import annotations

import hashlib
from pathlib import Path


def download_with_md5(url: str, out_dir: str | Path) -> Path:
    """Download a file and verify MD5.

    This is a stub for the smoke pipeline which uses toy data only.
    For the full pipeline, this would download from the URL and verify
    the file's MD5 checksum.
    """
    raise NotImplementedError(
        f"download_with_md5 not implemented for toy mode. "
        f"To use wikidump data, implement download from: {url}"
    )
