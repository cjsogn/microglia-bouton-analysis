"""Compare rerun estimates with deposited results, including whole-transcriptome fits.

This checks identity of result tables, not whether the scientific model is valid.
Rows are matched by explicit keys. Missing rows, changed missingness and numerical
differences beyond the declared floating-point tolerance fail the check.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from mg_paths import BUNDLE, DERIVED, require_writable_output


def compare(source, output, keys):
    left = pd.read_csv(source, sep="\t")
    right = pd.read_csv(output, sep="\t")
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError(f"Duplicate comparison keys: {source.name}")
    joined = left.merge(
        right,
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_source", "_rerun"),
        indicator=True,
    )
    if not joined._merge.eq("both").all():
        raise AssertionError(f"Changed row set: {source.name}")
    columns = [c for c in left.select_dtypes(include="number") if c not in keys]
    missing_columns = set(columns) - set(right.columns)
    if missing_columns:
        raise AssertionError(f"Missing numeric columns in rerun: {sorted(missing_columns)}")
    if not columns:
        raise AssertionError(f"No numeric columns to compare: {source.name}")
    differences = {}
    for column in columns:
        x = joined[column + "_source"].to_numpy(dtype=float)
        y = joined[column + "_rerun"].to_numpy(dtype=float)
        np.testing.assert_allclose(
            x, y, rtol=1e-8, atol=1e-10, equal_nan=True, err_msg=f"{source.name}: {column}"
        )
        finite = np.isfinite(x) & np.isfinite(y)
        differences[column] = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else None
    return {
        "table": source.name,
        "rows": len(joined),
        "numeric_columns": len(columns),
        "maximum_absolute_differences": differences,
        "status": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imaging", type=Path)
    parser.add_argument("--rna", type=Path)
    parser.add_argument("--human", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not any((args.imaging, args.rna, args.human)):
        parser.error("Supply at least one result directory")
    jobs = []
    if args.imaging:
        for name, keys in [
            ("unbiased_baseline.tsv", ["dataset", "endpoint", "marker"]),
            ("unbiased_per_animal.tsv", ["dataset", "endpoint", "marker", "animal", "condition"]),
            (
                "unbiased_treatment_effects.tsv",
                ["dataset", "endpoint", "marker", "reference", "treatment"],
            ),
        ]:
            jobs.append((DERIVED / name, args.imaging / name, keys))
    if args.rna:
        for folder in ("limma_voom", "limma_voom_sensitivity"):
            sources = sorted((DERIVED / "transcriptomics" / folder).glob("*_limma_voom.tsv.gz"))
            if not sources:
                raise FileNotFoundError(f"No deposited RNA results in {folder}")
            for source in sources:
                jobs.append((source, args.rna / folder / source.name, ["contrast_id", "gene_id"]))
    if args.human:
        for source, output in [
            (BUNDLE / "corrected/human_pd/limma_voom", args.human / "SNpc"),
            (BUNDLE / "xu_putamen/results/differential_expression", args.human / "putamen"),
        ]:
            jobs.append(
                (
                    source / "panel_all_analyses.tsv",
                    output / "panel_all_analyses.tsv",
                    ["analysis", "gene"],
                )
            )
            tables = sorted(source.glob("*/all_genes.tsv.gz"))
            if not tables:
                raise FileNotFoundError(f"No deposited whole-gene results in {source}")
            for table in tables:
                jobs.append((table, output / table.parent.name / table.name, ["gene"]))
    checks = [compare(*job) for job in jobs]
    destination = require_writable_output(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"checks": checks, "rtol": 1e-8, "atol": 1e-10}, indent=2) + "\n"
    )
    print(f"PASS: {len(checks)} tables matched their deposited numerical results.")


if __name__ == "__main__":
    main()
