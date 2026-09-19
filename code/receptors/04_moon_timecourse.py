"""Descriptive PBS/3-day/30-day microglial expression from GSE307796.

Retain and reproduce the published-package cell selection. Recover expression
from the full shared raw feature universe, without condition-wise gene filtering.
No biological replication is inferred from cells and no DE P values are produced.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

for variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"
os.environ["NUMBA_NUM_THREADS"] = "14"
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

CODE = Path(__file__).resolve().parents[2]
PACKAGE = CODE.parent
import sys

sys.path.insert(0, str(CODE / "code"))
from mg_paths import BUNDLE, OUTPUT, require_writable_output

HERE = Path(__file__).resolve().parent
CONDITIONS = {
    "PBS": "GSM9231553_PBS_filtered_feature_bc_matrix.h5",
    "LPS_3d": "GSM9231549_L3_filtered_feature_bc_matrix.h5",
    "LPS_30d": "GSM9231551_L30_filtered_feature_bc_matrix.h5",
}


def constants():
    """Load the unchanged selection markers and receptor catalogue."""
    definitions = json.loads((HERE / "selection_and_receptor_catalogue.json").read_text())
    return (
        definitions["selection_markers"],
        definitions["neurotransmitter_families"],
        definitions["broad_catalogue"],
    )


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUTPUT / "moon_timecourse")
    parser.add_argument(
        "--reference-table",
        type=Path,
        default=BUNDLE / "coauthor_inputs/external/receptor_prevalence_with_counts.csv",
        help="Existing receptor table used to verify the original cell selection.",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=BUNDLE
        / "original_analysis_inputs/pipeline/data/derived/transcriptomics/prespecified_gene_panel.tsv",
        help="Retained curated gene inventory. The historical filename does not imply preregistration.",
    )
    args = parser.parse_args()
    out = require_writable_output(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sc.settings.n_jobs = 14
    sc.settings.verbosity = 0
    definitions, nt, broad = constants()
    provenance = []

    def load(item):
        condition, name = item
        path = args.raw_dir / name
        a = sc.read_10x_h5(path)
        a.var_names_make_unique()
        a.obs_names = [f"{condition}::{b}" for b in a.obs_names]
        a.obs["condition"] = condition
        a.var["mt"] = a.var_names.str.startswith("mt-")
        sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
        before = a.n_obs
        a = a[(a.obs.pct_counts_mt < 15) & (a.obs.n_genes_by_counts > 200)].copy()
        return a, dict(
            condition=condition,
            raw_file=name,
            sha256=digest(path),
            raw_cells=before,
            qc_cells=a.n_obs,
            features=a.n_vars,
        )

    # Numba's installed workqueue cannot be entered from concurrent Python
    # threads. Process libraries sequentially and retain its 14-core numerical
    # execution inside each library, without changing any QC calculation.
    loaded = [load(item) for item in CONDITIONS.items()]
    raw = [x[0] for x in loaded]
    provenance = [x[1] for x in loaded]
    assert all(a.var_names.equals(raw[0].var_names) for a in raw), (
        "Feature identities must be reconciled explicitly."
    )
    filtered = []
    gene_retention = []
    for a in raw:
        count = np.asarray((a.X > 0).sum(axis=0)).ravel()
        gene_retention.append(pd.Series(count, index=a.var_names, name=a.obs.condition.iloc[0]))
        b = a[:, count >= 5].copy()
        filtered.append(b)
    legacy = sc.concat(filtered, join="inner", merge="same")
    sc.pp.normalize_total(legacy, target_sum=1e4)
    sc.pp.log1p(legacy)
    for name, score in [("MICROGLIA_MARKERS", "mg_score"), ("NON_MG_MARKERS", "nonmg_score")]:
        genes = [g for g in definitions[name] if g in legacy.var_names]
        sc.tl.score_genes(legacy, gene_list=genes, score_name=score, random_state=0)
    legacy.obs["mg_minus_nonmg"] = legacy.obs.mg_score - legacy.obs.nonmg_score
    selected = (legacy.obs.mg_score > 0.3) & (legacy.obs.mg_minus_nonmg > 0.2)
    legacy.obs["selected_microglia"] = selected
    expected = pd.read_csv(args.reference_table)
    checks = []
    for c in CONDITIONS:
        idx = selected & legacy.obs.condition.eq(c)
        n = int(idx.sum())
        assert n == int(expected.loc[expected.condition.eq(c), "selected_microglia"].iloc[0]), (
            c,
            n,
        )
        for r in expected[expected.condition.eq(c)].itertuples(index=False):
            values = legacy[idx, r.gene].X
            mean = float(values.mean())
            positive = int((values > 0).sum())
            assert positive == r.positive_microglia, (c, r.gene, positive)
            assert np.isclose(mean, r.mean, atol=2e-6, rtol=2e-6), (c, r.gene, mean, r.mean)
        checks.append(
            dict(condition=c, selected_microglia=n, legacy_prevalence_and_mean_reproduced=True)
        )
    print("Original cell selection and receptor tables reproduced.", flush=True)
    retention = pd.concat(gene_retention, axis=1)
    retention["retained_legacy_intersection"] = retention.index.isin(legacy.var_names)
    retention.to_csv(out / "gene_filter_audit.tsv", sep="\t", index_label="gene")
    selected_ids = legacy.obs.index[selected]
    raw_all = ad.concat(raw, join="inner", merge="same")
    raw_all = raw_all[selected_ids].copy()
    assert raw_all.obs_names.equals(selected_ids)
    cell_qc = legacy.obs.copy()
    cell_qc.to_csv(out / "cell_selection.tsv.gz", sep="\t", index_label="cell_id")
    normalized = raw_all.copy()
    sc.pp.normalize_total(normalized, target_sum=1e4)
    sc.pp.log1p(normalized)
    normalized.X = normalized.X.astype(np.float64)
    panel = pd.read_csv(args.panel, sep="\t")
    gene_column = next(c for c in ("gene", "symbol", "gene_symbol") if c in panel)
    panel_genes = panel[gene_column].astype(str).tolist()
    diagnostic = [
        "P2ry12",
        "Tmem119",
        "Hexb",
        "Csf1r",
        "Sall1",
        "Ccr2",
        "Ly6c2",
        "S100a8",
        "S100a9",
        "Mrc1",
        "Cd3e",
        "Mbp",
        "Snap25",
    ]
    genes = sorted(set(nt) | set(broad) | set(panel_genes) | set(diagnostic))
    rows = []
    for c in CONDITIONS:
        ix = raw_all.obs.condition.eq(c).to_numpy()
        n = int(ix.sum())
        for gene in genes:
            present = gene in raw_all.var_names
            values = raw_all[ix, gene].X if present else None
            positive = int((values > 0).sum()) if present else None
            mean = float(normalized[ix, gene].X.mean()) if present else None
            rows.append(
                dict(
                    gene=gene,
                    condition=c,
                    selected_microglia=n,
                    positive_microglia=positive,
                    frac=positive / n if present else None,
                    mean_log1p_10k=mean,
                    feature_present=present,
                    retained_legacy_intersection=gene in legacy.var_names,
                    neurotransmitter_family=nt.get(gene, ""),
                    broad_catalogue=gene in broad,
                    curated_transcript_panel=gene in panel_genes,
                    identity_diagnostic=gene in diagnostic,
                )
            )
    result = pd.DataFrame(rows)
    result.to_csv(out / "expression_prevalence.tsv", sep="\t", index=False)
    result[result.identity_diagnostic].to_csv(
        out / "cell_identity_diagnostics.tsv", sep="\t", index=False
    )
    # Portable selected-cell raw matrix supports independent regeneration of all summaries.
    raw_all.write_h5ad(out / "selected_microglia_raw.h5ad", compression="gzip")
    for source in provenance:
        source.update(next(c for c in checks if c["condition"] == source["condition"]))
    audit = dict(
        libraries=provenance,
        original_shared_gene_count=legacy.n_vars,
        full_shared_feature_count=raw_all.n_vars,
        n_summary_genes=len(genes),
        biological_libraries_per_condition=1,
        tests="None. Descriptive summaries only.",
        selection="Original shared-gene normalization and score thresholds retained exactly.",
        expression="Raw shared feature universe, normalized per cell to 10000 counts and log1p transformed. Means include zeros.",
        filtering="No condition-specific gene filter on expression outputs. Missing features remain missing, not biological zeros.",
        scanpy=sc.__version__,
        anndata=ad.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
        seed=0,
        workers=14,
        raw_reads_parallel=1,
        threading_note="Libraries processed sequentially to avoid concurrent Numba workqueue entry. Numerical worker limit remains 14.",
    )
    (out / "validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
