#!/usr/bin/env python3
"""Run the complete raw-count transcriptomic rebuild in checkpointed stages."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parent))  # so `common` resolves from any cwd

import argparse
import subprocess
import sys
from pathlib import Path

from common import ROOT, OUTPUT


def run(command):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run([str(x) for x in command], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "prepare", "de", "synthesis"], default="all")
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    # Stage 1 writes under MG_OUTPUT_DIR, so stages 2 and 3 must look there too. Reading the
    # checkout instead left stage 2 unable to find stage 1's output.

    derived = OUTPUT / "transcriptomics"
    de_dir = derived / "limma_voom"
    sensitivity_manifest = derived / "gse263094_cell_definition_sensitivity_manifest.tsv"
    sensitivity_de_dir = derived / "limma_voom_sensitivity"
    if args.stage in {"all", "prepare"}:
        run([sys.executable, scripts / "01_prepare_counts.py"])
    if args.stage in {"all", "de"}:
        if not sensitivity_manifest.is_file():
            raise FileNotFoundError(
                f"Required cell-definition sensitivity manifest: {sensitivity_manifest}"
            )
        de_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                "Rscript",
                scripts / "02_limma_voom.R",
                derived / "contrast_manifest.tsv",
                de_dir,
            ]
        )
        if sensitivity_manifest.exists():
            sensitivity_de_dir.mkdir(parents=True, exist_ok=True)
            run(
                [
                    "Rscript",
                    scripts / "02_limma_voom.R",
                    sensitivity_manifest,
                    sensitivity_de_dir,
                ]
            )
    if args.stage in {"all", "synthesis"}:
        run([sys.executable, scripts / "03_panel_synthesis.py"])


if __name__ == "__main__":
    main()
