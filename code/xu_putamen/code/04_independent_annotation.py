"""Independent Xu annotation, not reproduction of the authors' cell labels.

Stages: prepare -> inspect all-cell markers -> immune -> inspect immune markers
-> finalize. Cluster selections are explicit command-line inputs after inspection.
PD effects are not computed by this script. Raw RNA counts remain unchanged.
"""

import os

for key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
]:
    os.environ[key] = "14"
import argparse
import gc
import json
import importlib.metadata
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import io
from threadpoolctl import threadpool_limits

from xu_paths import WORK as ROOT

OUT = ROOT / "results/independent_annotation"
OUT.mkdir(exist_ok=True)
SEED = 20260910
# These fixed exclusions belong to the cell-annotation procedure. Reporting
# additional genes must not change the selection of nuclei used for inference.
ANNOTATION_EXCLUSIONS = [
    "C3",
    "C3AR1",
    "C1QA",
    "TNF",
    "IL1B",
    "P2RY12",
    "TMEM119",
    "CX3CR1",
    "TREM2",
    "AXL",
    "MERTK",
]
MARKERS = {
    "Myeloid": ["CSF1R", "PTPRC", "TYROBP", "C1QB", "C1QC", "AIF1", "HEXB"],
    "Microglia": ["GPR34", "P2RY13", "SALL1", "SLC2A5", "FCRLS", "OLFML3"],
    "PVM": ["MRC1", "LYVE1", "CD163", "F13A1", "PF4", "MS4A7"],
    "T_NK": ["CD3D", "CD3E", "TRAC", "CD247", "NKG7"],
    "Astrocyte": ["AQP4", "SLC1A2", "SLC1A3", "ALDH1L1", "GFAP"],
    "Oligodendrocyte": ["MBP", "PLP1", "MOG", "MOBP", "MAG"],
    "OPC": ["PDGFRA", "VCAN", "PCDH15", "MEGF11"],
    "Neuron": ["SNAP25", "SYT1", "RBFOX3", "GAD1", "SLC17A7"],
    "Vascular": ["CLDN5", "FLT1", "VWF", "RGS5", "PDGFRB", "COL1A1"],
}
sc.settings.n_jobs = 14
sc.settings.verbosity = 2


def write_annotation_evidence(a, key, label):
    genes = [g for gs in MARKERS.values() for g in gs if g in a.var_names]
    rows = []
    for cluster in a.obs[key].cat.categories:
        ix = (a.obs[key] == cluster).values
        x = a[ix, genes].X
        for gene, mean, fraction in zip(
            genes, np.asarray(x.mean(0)).ravel(), np.asarray((x > 0).mean(0)).ravel()
        ):
            rows.append(
                dict(
                    cluster=cluster,
                    n_cells=int(ix.sum()),
                    gene=gene,
                    mean_log1p=mean,
                    fraction_detected=fraction,
                )
            )
    pd.DataFrame(rows).to_csv(OUT / f"{label}_marker_expression.csv", index=False)
    pd.crosstab(a.obs[key], a.obs.donor).to_csv(OUT / f"{label}_cluster_donors.csv")
    sc.tl.rank_genes_groups(
        a,
        groupby=key,
        method="t-test",
        n_genes=40,
        use_raw=False,
        mask_var=~a.var_names.isin(ANNOTATION_EXCLUSIONS),
    )
    sc.get.rank_genes_groups_df(a, group=None).to_csv(OUT / f"{label}_top_markers.csv", index=False)
    a.obs.to_csv(OUT / f"{label}_cells.csv.gz")


def embedding(a, label, resolution):
    sc.pp.highly_variable_genes(a, n_top_genes=3000, flavor="seurat", batch_key="donor")
    excluded = a.var_names.isin(
        ANNOTATION_EXCLUSIONS + ["XIST", "RPS4Y1", "DDX3Y", "UTY", "KDM5D"]
    ) | a.var_names.str.startswith(("MT-", "RPL", "RPS"))
    a.var.loc[excluded, "highly_variable"] = False
    b = a[:, a.var.highly_variable].copy()
    sc.pp.scale(b, max_value=10)
    sc.tl.pca(b, n_comps=40, svd_solver="arpack", random_state=SEED)
    assert np.isfinite(b.obsm["X_pca"]).all()
    a.obsm["X_pca"] = b.obsm["X_pca"]
    a.uns["pca_variance_ratio"] = b.uns["pca"]["variance_ratio"]
    del b
    gc.collect()
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=30, random_state=SEED)
    sc.tl.leiden(
        a,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=SEED,
        key_added="cluster",
    )
    sc.tl.umap(a, random_state=SEED)
    a.obs["UMAP1"] = a.obsm["X_umap"][:, 0]
    a.obs["UMAP2"] = a.obsm["X_umap"][:, 1]
    write_annotation_evidence(a, "cluster", label)


def prepare():
    parts = []
    qcs = []
    paths = sorted((ROOT / "raw/counts").glob("*_matrix.mtx.gz"))
    for path in paths:
        prefix = path.name.removesuffix("_matrix.mtx.gz")
        donor = prefix.split("_")[1]
        if donor.startswith("AD"):
            continue
        cache = OUT / f"qc_{donor}.h5ad"
        if cache.exists():
            a = sc.read_h5ad(cache)
            parts.append(a)
            qcs.append(pd.read_csv(OUT / f"qc_{donor}.csv"))
            continue
        print("Reading", donor, flush=True)
        features = pd.read_csv(path.with_name(prefix + "_features.tsv.gz"), sep="\t", header=None)
        barcodes = pd.read_csv(path.with_name(prefix + "_barcodes.tsv.gz"), sep="\t", header=None)[
            0
        ]
        x = io.mmread(path).T.tocsr().astype(np.float32)
        a = ad.AnnData(
            x,
            obs=pd.DataFrame(
                {"barcode": barcodes.values, "donor": donor},
                index=donor + "_" + barcodes.astype(str),
            ),
            var=pd.DataFrame(
                {"gene_id": features[0].values, "symbol": features[1].values},
                index=features[1].astype(str),
            ),
        )
        a.obs.index.name = "cell_id"
        a.var.index.name = "gene"
        a.var_names_make_unique(join="__duplicate")
        a.var["mt"] = a.var.symbol.str.startswith("MT-").values
        sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, inplace=True)
        keep = (
            (a.obs.n_genes_by_counts >= 500)
            & (a.obs.n_genes_by_counts <= 9000)
            & (a.obs.total_counts >= 500)
            & (a.obs.total_counts <= 70000)
            & (a.obs.pct_counts_mt <= 10)
        )
        qc = a.obs.copy()
        qc["pass_numeric_qc"] = keep
        qc["doublet_score"] = np.nan
        qc["predicted_doublet"] = False
        a = a[keep].copy()
        # 0.05 is Scrublet's algorithmic prior, not an observed doublet rate.
        sc.pp.scrublet(
            a,
            expected_doublet_rate=0.05,
            random_state=SEED,
            use_approx_neighbors=False,
            n_prin_comps=30,
            verbose=True,
        )
        if a.obs.predicted_doublet.isna().any() or not np.isfinite(a.obs.doublet_score).all():
            raise RuntimeError("Scrublet classification failed or produced nonfinite scores")
        qc.loc[a.obs_names, "doublet_score"] = a.obs.doublet_score
        qc.loc[a.obs_names, "predicted_doublet"] = a.obs.predicted_doublet
        qc["retained"] = qc.pass_numeric_qc & ~qc.predicted_doublet
        qc.to_csv(OUT / f"qc_{donor}.csv", index_label="cell_id")
        (OUT / f"scrublet_{donor}.json").write_text(
            json.dumps(
                {
                    k: float(a.uns["scrublet"][k])
                    for k in ["threshold", "expected_doublet_rate"]
                    if k in a.uns["scrublet"]
                },
                indent=2,
            )
        )
        a = a[~a.obs.predicted_doublet].copy()
        a.write_h5ad(cache, compression="gzip")
        parts.append(a)
        qcs.append(qc.reset_index())
    allcells = ad.concat(parts, join="inner", merge="same")
    del parts
    gc.collect()
    allcells.layers["counts"] = allcells.X.copy()
    sc.pp.filter_genes(allcells, min_cells=5)
    sc.pp.normalize_total(allcells, target_sum=10000)
    sc.pp.log1p(allcells)
    embedding(allcells, "allcells", 0.5)
    allcells.write_h5ad(OUT / "allcells.h5ad", compression="gzip")
    pd.concat(qcs, ignore_index=True).to_csv(OUT / "qc_all_cells.csv.gz", index=False)
    versions = {
        p: importlib.metadata.version(p)
        for p in [
            "scanpy",
            "anndata",
            "numpy",
            "scipy",
            "pandas",
            "igraph",
            "leidenalg",
            "harmonypy",
        ]
    }
    (OUT / "python_versions.json").write_text(json.dumps(versions, indent=2))


def immune(clusters):
    a = sc.read_h5ad(OUT / "allcells.h5ad")
    a = a[a.obs.cluster.isin(clusters)].copy()
    a.obs["parent_cluster"] = a.obs.cluster.astype(str)
    embedding(a, "immune", 0.5)
    # Additional resolution tests annotation boundaries, not disease significance.
    sc.tl.leiden(
        a,
        resolution=1.0,
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=SEED,
        key_added="cluster_fine",
    )
    write_annotation_evidence(a, "cluster_fine", "immune_fine")
    # Donor-corrected geometry is used only as an annotation sensitivity check.
    # Installed harmonypy 0.2.0 returns cells x PCs, while Scanpy 1.11's wrapper assumes
    # PCs x cells. Run the same algorithm directly and validate orientation.
    import harmonypy
    import torch

    torch.set_num_threads(14)
    ho = harmonypy.run_harmony(
        a.obsm["X_pca"], a.obs, ["donor"], random_state=SEED, max_iter_harmony=30
    )
    corrected = np.asarray(ho.Z_corr)
    if corrected.shape == a.obsm["X_pca"].T.shape:
        corrected = corrected.T
    assert corrected.shape == a.obsm["X_pca"].shape and np.isfinite(corrected).all()
    a.obsm["X_pca_harmony"] = corrected
    sc.pp.neighbors(
        a, use_rep="X_pca_harmony", n_neighbors=15, key_added="harmony", random_state=SEED
    )
    sc.tl.leiden(
        a,
        neighbors_key="harmony",
        resolution=0.5,
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=SEED,
        key_added="cluster_harmony",
    )
    write_annotation_evidence(a, "cluster_harmony", "immune_harmony")
    a.write_h5ad(OUT / "immune.h5ad", compression="gzip")


def finalize(clusters, harmony_clusters):
    a = sc.read_h5ad(OUT / "immune.h5ad")
    a.obs["microglia_primary"] = a.obs.cluster.isin(clusters)
    a.obs["microglia_harmony"] = a.obs.cluster_harmony.isin(harmony_clusters)
    a.obs["microglia_concordant"] = a.obs.microglia_primary & a.obs.microglia_harmony
    a.obs.to_csv(OUT / "final_immune_annotations.csv.gz", index_label="cell_id")
    for selection in ["primary", "harmony", "concordant"]:
        selected = a.obs["microglia_" + selection]
        folder = OUT / selection
        folder.mkdir(exist_ok=True)
        donors = sorted(a.obs.donor.unique())
        mat = np.column_stack(
            [
                np.asarray(
                    a.layers["counts"][selected.values & (a.obs.donor == d).values].sum(0)
                ).ravel()
                for d in donors
            ]
        ).astype(np.int64)
        counts = pd.DataFrame(mat, index=a.var_names, columns=donors)
        counts.index.name = "gene"
        counts.to_csv(folder / "counts.tsv.gz", sep="\t")
        meta = (
            pd.read_csv(ROOT / "results/published_donor_metadata.csv")
            .set_index("geo_sample_label")
            .loc[donors]
            .copy()
        )
        meta["n_cells"] = [int((selected & (a.obs.donor == d)).sum()) for d in donors]
        meta.index.name = "donor_id"
        meta.to_csv(folder / "samples.tsv", sep="\t")
    a.write_h5ad(OUT / "immune_annotated.h5ad", compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "immune", "finalize"])
    parser.add_argument("--clusters", nargs="+", default=[])
    parser.add_argument("--harmony-clusters", nargs="+", default=[])
    args = parser.parse_args()
    with threadpool_limits(limits=14):
        if args.stage == "prepare":
            prepare()
        elif args.stage == "immune":
            immune(args.clusters)
        else:
            finalize(args.clusters, args.harmony_clusters)
