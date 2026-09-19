"""Shared RNA import paths and streaming input hashing."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mg_paths import ROOT, EXTERNAL, OUTPUT

__all__ = ["ROOT", "EXTERNAL", "OUTPUT", "sha256_file"]


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
