"""Assemble complete calibration runs without silently skipping a scheme.

This is a table-assembly step, not a simulation. The simulation cache must
contain all four named schemes and the explicitly selected pairwise scheme.
See docs/CALIBRATION.md for historical seeds and realized draw counts.
"""

from pathlib import Path
import os
import sys

ROOT = Path(os.environ.get("MG_PACKAGE_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "code"))
from mg_paths import output_dir, require_file
import pandas as pd
from statsmodels.stats.multitest import multipletests

SCHEMES = {
    "homog": "homogeneous_control",
    "matched": "matched_to_observed",
    "observed": "observed_between_animal",
    "sampling": "torus_sampling_only",
}
KEYS = ["endpoint", "test"]
COLS = [
    "endpoint",
    "test",
    "n_sim",
    "typeI_0p05",
    "typeI_0p01",
    "typeI_0p001",
    "nominal_p",
    "calibrated_p",
]


def checked_table(path, columns, keys, n):
    table = pd.read_csv(require_file(path), sep="\t")
    if not set(columns).issubset(table.columns):
        raise ValueError(f"Incomplete schema: {path}")
    if len(table) != n or table.duplicated(keys).any():
        raise ValueError(f"Incomplete or duplicated calibration rows: {path}")
    if table[columns].isna().any().any():
        raise ValueError(f"Missing calibration values: {path}")
    return table


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-mode", required=True, choices=tuple(SCHEMES))
    args = parser.parse_args()
    out = output_dir()
    cache = out / "_cache"
    frames = []
    for mode, label in SCHEMES.items():
        table = checked_table(cache / f"typeI_{mode}.tsv", COLS, KEYS, 8)[COLS].copy()
        table.insert(0, "calibration", label)
        table["inflation_at_0p05"] = table.typeI_0p05 / 0.05
        table["inflation_at_0p01"] = table.typeI_0p01 / 0.01
        frames.append(table)
    assembled = pd.concat(frames, ignore_index=True).sort_values(
        ["endpoint", "test", "calibration"]
    )
    pair_cols = [
        "endpoint",
        "scale",
        "comparison",
        "n_sim",
        "typeI_0p05",
        "typeI_0p01",
        "nominal_p",
        "calibrated_p",
    ]
    pairs = checked_table(
        cache / f"typeI_pairs_{args.pairwise_mode}.tsv",
        pair_cols,
        ["endpoint", "scale", "comparison"],
        20,
    ).copy()
    for _, group in pairs.groupby(["endpoint", "scale"]):
        if len(group) != 5:
            raise ValueError("Each pairwise multiplicity family must contain five comparisons")
        for source, target in [("nominal_p", "q_nominal"), ("calibrated_p", "q_calibrated")]:
            pairs.loc[group.index, target] = multipletests(group[source], method="fdr_bh")[1]
    variance = pd.read_csv(require_file(cache / "marker_null_variance.tsv"), sep="\t")
    # Write only after every required input has passed validation.
    assembled.to_csv(out / "baseline_ordering_typeI_calibration.tsv", sep="\t", index=False)
    pairs.sort_values(["endpoint", "scale", "comparison"]).to_csv(
        out / "baseline_ordering_pairwise_calibrated.tsv", sep="\t", index=False
    )
    variance.to_csv(out / "baseline_marker_null_variance.tsv", sep="\t", index=False)
    (out / "calibration_assembly_provenance.txt").write_text(
        f"Four schemes assembled. Pairwise source mode: {args.pairwise_mode}. "
        "BH families: five comparators within endpoint and scale. No simulations run here.\n"
    )


if __name__ == "__main__":
    main()
