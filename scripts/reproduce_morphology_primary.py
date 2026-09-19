"""Refit only the retained primary branch-count contrasts from eligible-cell barcodes.

This short entry point calls the same functions as morph2_persistent_branches.py.
The full script also runs regional, density-adjusted and threshold-sweep analyses.
The primary threshold is recomputed inside every animal-label allocation.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/analysis")]
from mg_paths import output_dir
from morph2_persistent_branches import (
    CONTROL_PERCENTILE,
    load,
    per_animal,
    exact_contrast,
    exact_contrast_full_procedure,
)


def main():
    barcodes = load()
    threshold = float(
        np.percentile(barcodes.loc[barcodes.cond.eq("Control"), "persistence"], CONTROL_PERCENTILE)
    )
    cells, animals = per_animal(barcodes, threshold)
    rows = []
    for condition in ("LPS_72h", "LPS_60d"):
        standardized, conditional_p, conditional_floor = exact_contrast(
            animals.hit.to_numpy(), animals.cond.to_numpy(), condition
        )
        difference, p_value, floor = exact_contrast_full_procedure(barcodes, condition)
        rows.append(
            dict(
                contrast=f"{condition} vs Control",
                delta_sd=standardized,
                raw_diff_branches_per_cell=difference,
                exact_p=p_value,
                floor=floor,
                exact_p_conditional_on_observed_threshold=conditional_p,
                floor_conditional=conditional_floor,
            )
        )
    output = output_dir("morphology", "region_analysis")
    result = pd.DataFrame(rows)
    result.to_csv(output / "persistent_branches_treatment.tsv", sep="\t", index=False)
    animals.to_csv(output / "persistent_branches_animal_values.tsv", sep="\t", index=False)
    cells.to_csv(output / "persistent_branches_cell_values.tsv", sep="\t", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
