"""Descriptive identity-marker restrictions, not cell reannotation or DE tests."""

from pathlib import Path
import argparse
import json
import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mg_paths import BUNDLE, OUTPUT, require_writable_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BUNDLE / "corrected/moon_timecourse")
    parser.add_argument("--out", type=Path, default=OUTPUT / "moon_selection_checks")
    args = parser.parse_args()
    DATA = args.data_dir.resolve()
    OUT = require_writable_output(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(DATA / "selected_microglia_raw.h5ad")

    def positive(g):
        return np.asarray((a[:, g].X > 0).toarray()).ravel()

    markers = {g: positive(g) for g in ["P2ry12", "Tmem119", "S100a8", "S100a9", "Ccr2", "Ly6c2"]}
    double_s100 = markers["S100a8"] & markers["S100a9"]
    double_monocyte = markers["Ccr2"] & markers["Ly6c2"]
    masks = {
        "original_selection": np.ones(a.n_obs, dtype=bool),
        "exclude_dual_myeloid_markers": ~(double_s100 | double_monocyte),
        "dual_resident_markers": markers["P2ry12"] & markers["Tmem119"],
    }
    full = pd.read_csv(DATA / "expression_prevalence.tsv", sep="\t")
    genes = full.loc[full.feature_present, "gene"].drop_duplicates().tolist()
    normalized = a.copy()
    sc.pp.normalize_total(normalized, target_sum=1e4)
    sc.pp.log1p(normalized)
    normalized.X = normalized.X.astype(np.float64)
    results, counts = [], []
    for condition in ["PBS", "LPS_3d", "LPS_30d"]:
        c = a.obs.condition.eq(condition).to_numpy()
        counts.append(
            dict(
                condition=condition,
                original_cells=int(c.sum()),
                dual_s100=int((c & double_s100).sum()),
                dual_ccr2_ly6c2=int((c & double_monocyte).sum()),
                dual_p2ry12_tmem119=int((c & masks["dual_resident_markers"]).sum()),
            )
        )
        for label, mask in masks.items():
            ix = c & mask
            n = int(ix.sum())
            if n == 0:
                raise ValueError(f"No cells for {condition} / {label}")
            x = a[ix, genes].X
            expr = normalized[ix, genes].X
            pos = np.asarray((x > 0).sum(axis=0)).ravel()
            means = np.asarray(expr.mean(axis=0)).ravel()
            for g, k, m in zip(genes, pos, means):
                results.append(
                    dict(
                        gene=g,
                        condition=condition,
                        selection=label,
                        selected_microglia=n,
                        positive_microglia=int(k),
                        frac=float(k / n),
                        mean_log1p_10k=float(m),
                    )
                )
    r = pd.DataFrame(results)
    r.to_csv(OUT / "selection_sensitivity.tsv", sep="\t", index=False)
    pd.DataFrame(counts).to_csv(OUT / "identity_marker_coexpression.tsv", sep="\t", index=False)
    d = r[r.selection.eq("original_selection")].merge(
        full, on=["gene", "condition"], suffixes=("_check", "_source"), validate="one_to_one"
    )
    for col in ["selected_microglia", "positive_microglia", "frac", "mean_log1p_10k"]:
        assert np.allclose(d[col + "_check"], d[col + "_source"], rtol=2e-6, atol=2e-6)
    (OUT / "selection_diagnostic_notes.json").write_text(
        json.dumps(
            dict(
                rationale="Check dependence on overlapping myeloid markers and microglial marker detection without replacing the original cell selection.",
                primary_unchanged=True,
                excluded_from_primary=0,
                caveat="Restrictions can remove genuine inflammatory microglia and are not validated cell-type classifiers. Detection also depends on sequencing depth. No condition-level P values are calculated.",
                rules={
                    "exclude_dual_myeloid_markers": "Exclude cells with both S100a8 and S100a9 detected, or both Ccr2 and Ly6c2 detected.",
                    "dual_resident_markers": "Require both P2ry12 and Tmem119 detected.",
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print(pd.DataFrame(counts).to_string(index=False))


if __name__ == "__main__":
    main()
