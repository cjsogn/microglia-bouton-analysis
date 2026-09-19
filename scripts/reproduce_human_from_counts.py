"""Refit the two separate human PD cohorts from deposited donor count matrices.

This runs every retained model and donor/annotation sensitivity in the original
R fitting scripts. It does not recluster nuclei, pool cohorts or change mouse FDR.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[name] = "14"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from mg_paths import BUNDLE, ROOT, OUTPUT, require_writable_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snpc-input", type=Path, default=BUNDLE / "corrected/human_pd")
    parser.add_argument(
        "--putamen-input", type=Path, default=BUNDLE / "xu_putamen/results/independent_annotation"
    )
    parser.add_argument("--out", type=Path, default=OUTPUT / "human_pd")
    args = parser.parse_args()
    out = require_writable_output(args.out)
    if out.exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    for folder, selections in (
        (args.snpc_input, ("broad", "final")),
        (args.putamen_input, ("primary", "harmony", "concordant")),
    ):
        for selection in selections:
            for filename in ("counts.tsv.gz", "samples.tsv"):
                path = folder / selection / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
    out.mkdir(parents=True)
    jobs = [
        (ROOT / "code/human_pd/02_human_limma.R", args.snpc_input, out / "SNpc"),
        (ROOT / "code/xu_putamen/code/05_donor_pseudobulk.R", args.putamen_input, out / "putamen"),
    ]
    for script, source, destination in jobs:
        subprocess.run(
            ["Rscript", str(script), str(source.resolve()), str(destination)], check=True
        )


if __name__ == "__main__":
    main()
