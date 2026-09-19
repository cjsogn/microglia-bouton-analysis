"""Independent reconciliation of annotation, pseudobulk and model exports."""

import json
import numpy as np
import pandas as pd
import scanpy as sc

from xu_paths import WORK as ROOT

A = ROOT / "results/independent_annotation"
D = ROOT / "results/differential_expression"
a = sc.read_h5ad(A / "immune_annotated.h5ad")
assert all(np.isfinite(x).all() for x in a.obsm.values())
assert np.isfinite(a.obs.doublet_score).all()
checks = []
for selection in ["primary", "harmony", "concordant"]:
    counts = pd.read_csv(A / selection / "counts.tsv.gz", sep="\t", index_col=0)
    samples = pd.read_csv(A / selection / "samples.tsv", sep="\t").set_index("donor_id")
    for donor in samples.index:
        ix = a.obs["microglia_" + selection] & (a.obs.donor == donor)
        expected = np.asarray(a.layers["counts"][ix.values].sum(axis=0, dtype=np.float64)).ravel()
        assert np.array_equal(expected, counts.loc[a.var_names, donor].values)
        assert int(ix.sum()) == samples.loc[donor, "n_cells"]
    checks.append(
        dict(check="pseudobulk_raw_count_reconciliation", selection=selection, status="PASS")
    )
panel = pd.read_csv(D / "panel_all_analyses.tsv", sep="\t")
for name in panel.analysis.unique():
    tab = pd.read_csv(D / name / "all_genes.tsv.gz", sep="\t")
    ps = tab["P.Value"].values
    order = np.argsort(ps)
    adjusted = np.minimum.accumulate((ps[order] * len(ps) / np.arange(1, len(ps) + 1))[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1)
    assert np.allclose(tab["adj.P.Val"].values[order], adjusted, rtol=1e-12, atol=1e-12)
    p = panel[(panel.analysis == name) & panel.tested].set_index("gene")
    assert np.allclose(p.logFC, tab.set_index("gene").loc[p.index, "logFC"])
    assert np.all((p["CI.L"] <= p.logFC) & (p.logFC <= p["CI.R"]))
    design = pd.read_csv(D / name / "design.tsv", sep="\t").iloc[:, 1:].values
    assert np.linalg.matrix_rank(design) == design.shape[1]
    expr = pd.read_csv(D / name / "normalized_logcpm.tsv.gz", sep="\t", index_col=0)
    assert np.isfinite(expr.values).all()
    checks.append(dict(check="BH_CI_rank_export_reconciliation", analysis=name, status="PASS"))
# Export a complete barcode-level decision table, including excluded nuclei.
q = pd.concat(
    [pd.read_csv(A / f"qc_{d}.csv") for d in sorted(a.obs.donor.unique())], ignore_index=True
)
q["annotation"] = "excluded_numerical_QC"
q.loc[q.pass_numeric_qc & q.predicted_doublet, "annotation"] = "predicted_doublet"
q.loc[q.retained, "annotation"] = "nonimmune_cluster"
lookup = a.obs.copy()
lookup["annotation"] = "microglia_primary"
lookup.loc[lookup.cluster.astype(str) == "0", "annotation"] = "lymphocyte"
lookup.loc[lookup.cluster.astype(str) == "1", "annotation"] = "PVM_BAM_candidate"
q = q.set_index("cell_id")
q.loc[lookup.index, "annotation"] = lookup.annotation
q["microglia_concordant"] = False
q.loc[lookup.index, "microglia_concordant"] = lookup.microglia_concordant
q.to_csv(A / "all_barcode_annotations.csv.gz")
q.groupby(["donor", "annotation"]).size().unstack(fill_value=0).to_csv(A / "annotation_funnel.csv")
checks.append(dict(check="finite_embeddings_scores_expression", status="PASS"))
(ROOT / "results/verification.json").write_text(json.dumps(checks, indent=2) + "\n")
print("All reconciliation checks passed")
