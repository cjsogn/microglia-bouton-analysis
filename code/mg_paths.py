"""Portable input and output locations for the coauthor code release.

MG_PACKAGE_ROOT relocates code, MG_DATA_DIR relocates the deposited input bundle,
MG_EXTERNAL_DATA locates public RNA inputs, and MG_OUTPUT_DIR selects regenerated
outputs. Inputs are never overwritten. See README.md for the bundle layout.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = (
    Path(os.environ.get("MG_PACKAGE_ROOT") or Path(__file__).resolve().parents[1])
    .expanduser()
    .resolve()
)
BUNDLE = Path(os.environ.get("MG_BUNDLE_DIR") or ROOT.parent / "03_data").expanduser().resolve()
_default_data = BUNDLE / "coauthor_inputs" if BUNDLE.is_dir() else ROOT / "data"
DATA = Path(os.environ.get("MG_DATA_DIR") or _default_data).expanduser().resolve()
DERIVED = DATA / "derived"
PIPELINE = DATA / "pipeline"
MORPHOLOGY = DATA / "morphology"
SOURCE_DATA = DATA / "source_data"
_default_external = BUNDLE / "external" if BUNDLE.is_dir() else DATA / "external"
EXTERNAL = Path(os.environ.get("MG_EXTERNAL_DATA") or _default_external).expanduser().resolve()
OUTPUT = Path(os.environ.get("MG_OUTPUT_DIR") or ROOT / "outputs").expanduser().resolve()

# A misplaced environment variable must not turn a rerun into an input overwrite.
for protected in (DATA, EXTERNAL, BUNDLE):
    if OUTPUT == protected or protected in OUTPUT.parents:
        raise ValueError(f"MG_OUTPUT_DIR must be outside the input tree: {protected}")


def output_dir(*parts: str) -> Path:
    path = OUTPUT.joinpath(*parts).resolve()
    if path != OUTPUT and OUTPUT not in path.parents:
        raise ValueError("Output components must stay within MG_OUTPUT_DIR")
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_writable_output(path: Path) -> Path:
    """Reject output paths inside deposited data, including explicit CLI paths."""
    path = path.expanduser().resolve()
    for protected in (DATA, EXTERNAL, BUNDLE):
        if path == protected or protected in path.parents:
            raise ValueError(f"Output must be outside the input tree: {protected}")
    return path


def derived_input(name: str) -> Path:
    """Use an explicitly regenerated predecessor when present, else deposited data."""
    regenerated = OUTPUT / name
    return regenerated if regenerated.is_file() else DERIVED / name


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required input is missing: {path}. See README.md and set MG_DATA_DIR."
        )
    return path


if __name__ == "__main__":
    for name in ("ROOT", "DATA", "DERIVED", "PIPELINE", "MORPHOLOGY", "EXTERNAL", "OUTPUT"):
        print(f"{name:12s} {globals()[name]}")
