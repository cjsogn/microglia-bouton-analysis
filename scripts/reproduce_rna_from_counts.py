"""Refit all deposited RNA contrasts without reclustering single-cell inputs.

Copies the verified count bundle into the output tree, rewrites only path fields,
and runs the retained limma-voom and panel-synthesis scripts. The source bundle
must contain the GSE244652 original-count primary release and its import marker.
"""

from __future__ import annotations
import os
import json

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "14"
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from mg_paths import ROOT, DERIVED, OUTPUT
import pandas as pd


def main():
    source = DERIVED / "transcriptomics"
    dest = OUTPUT / "transcriptomics"
    if dest.exists():
        raise FileExistsError(f"Use a fresh MG_OUTPUT_DIR to avoid mixing RNA runs: {dest}")
    marker = json.loads((source / "import_release.json").read_text())
    if (
        marker.get("primary_countsFromAbundance") != "no"
        or marker.get("fresh_imports_reconciled") != 4
    ):
        raise ValueError("The input bundle is not the verified MARS-seq primary release")
    shutil.copytree(source, dest)
    manifests = [
        ("contrast_manifest.tsv", "contrasts"),
        ("gse263094_cell_definition_sensitivity_manifest.tsv", "sensitivity_counts"),
    ]
    for name, folder in manifests:
        manifest = pd.read_csv(dest / name, sep="\t")
        for index, rec in manifest.iterrows():
            for col, filename in [
                ("counts_path", "counts.tsv.gz"),
                ("samples_path", "samples.tsv"),
            ]:
                path = dest / folder / rec.contrast_id / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
                manifest.loc[index, col] = str(path)
        manifest.to_csv(dest / name, sep="\t", index=False)
    scripts = ROOT / "code/transcriptomics"
    for stage in ("de", "synthesis"):
        subprocess.run(
            [sys.executable, str(scripts / "run_rebuild.py"), "--stage", stage], check=True
        )


if __name__ == "__main__":
    main()
