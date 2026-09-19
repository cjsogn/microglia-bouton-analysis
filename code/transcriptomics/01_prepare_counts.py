#!/usr/bin/env python3
"""Prepare biological-sample count matrices for the molecular rebuild.

Inputs are local raw files.  Single-cell data are summed by independent library
after cell-type assignment; bulk/FACS/RiboTag data retain one column per
experimental sample.  No cell-level treatment test is performed here.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parent))  # so `common` resolves from any cwd

import json
import re
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from common import EXTERNAL, OUTPUT, sha256_file


# Raw count inputs are supplied separately from code in the full data bundle.
# MG_EXTERNAL_DATA can override its external/ folder. Each loader names the
# accession-specific subdirectory and verifies source files before aggregation.

DATA_ROOT = EXTERNAL
DERIVED = OUTPUT / "transcriptomics"
COUNTS_DIR = DERIVED / "contrasts"

PANEL = {
    "C1qa": ("Complement", "up"),
    "C1qb": ("Complement", "up"),
    "C1qc": ("Complement", "up"),
    "C3": ("Complement", "up"),
    "C3ar1": ("Complement_receptor", "up"),
    "Itgam": ("Complement_receptor", "up"),
    "Itgb2": ("Complement_receptor", "up"),
    "Trem2": ("Phagocytosis", "up"),
    "Tyrobp": ("Phagocytosis", "up"),
    "Mertk": ("TAM", "up"),
    "Axl": ("TAM", "up"),
    "Cd68": ("Phagolysosome", "up"),
    "Lgals3": ("Phagocytosis", "up"),
    "Apoe": ("Reactive_state", "up"),
    "Cst7": ("Reactive_state", "up"),
    "Spp1": ("Reactive_state", "up"),
    "Clec7a": ("Reactive_state", "up"),
    "P2ry12": ("Homeostatic", "down"),
    "Tmem119": ("Homeostatic", "down"),
    "Cx3cr1": ("Homeostatic", "down"),
    "Sall1": ("Homeostatic", "down"),
    "Adrb2": ("Adrenergic_restraint", "down"),
    "Il1b": ("Inflammatory", "up"),
    "Tnf": ("Inflammatory", "up"),
    "Ccl2": ("Chemotactic", "up"),
}


def record_source(path: Path, study: str, role: str, provenance: list[dict]):
    provenance.append(
        {
            "study": study,
            "role": role,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    )


def validated_harmony_coordinates(values, n_obs, n_components):
    """Normalize harmonypy pre/post-0.2 orientation and reject bad output."""
    coordinates = np.asarray(values)
    if coordinates.shape == (n_components, n_obs):
        coordinates = coordinates.T
    if coordinates.shape != (n_obs, n_components) or not np.isfinite(coordinates).all():
        raise RuntimeError(
            f"invalid Harmony coordinates: {coordinates.shape}, "
            f"finite={np.isfinite(coordinates).all()}"
        )
    return coordinates


def write_contrast(
    contrast_id: str,
    study: str,
    injury: str,
    region: str,
    regimen: str,
    technology: str,
    counts: pd.DataFrame,
    samples: pd.DataFrame,
    gene_id_type: str,
    manifest: list[dict],
    biological_unit: str = "one animal/sample per count-matrix column",
    output_base: Path | None = None,
    analysis_role: str = "primary",
):
    samples = samples.copy()
    samples["sample_id"] = samples.sample_id.astype(str)
    counts = counts.copy()
    counts.columns = counts.columns.astype(str)
    if samples.sample_id.duplicated().any() or counts.columns.duplicated().any():
        raise ValueError(f"duplicate sample IDs in {contrast_id}")
    missing = sorted(set(samples.sample_id) - set(counts.columns))
    if missing:
        raise ValueError(f"count matrix lacks samples in {contrast_id}: {missing}")
    if not counts.columns.equals(pd.Index(samples.sample_id)):
        counts = counts.loc[:, samples.sample_id]
    if counts.index.duplicated().any():
        counts = counts.groupby(level=0).sum()
    counts = counts.fillna(0)
    values = counts.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"nonfinite or negative counts in {contrast_id}")
    if set(samples.group) != {"Control", "Treatment"}:
        raise ValueError(f"{contrast_id} must have Control and Treatment")
    out = (output_base or COUNTS_DIR) / contrast_id
    out.mkdir(parents=True, exist_ok=True)
    table = counts.copy()
    table.insert(0, "gene_id", table.index)
    table.to_csv(out / "counts.tsv.gz", sep="\t", index=False, compression="gzip")
    samples.to_csv(out / "samples.tsv", sep="\t", index=False)
    manifest.append(
        {
            "contrast_id": contrast_id,
            "study": study,
            "injury": injury,
            "region": region,
            "regimen_or_time": regimen,
            "technology": technology,
            "biological_unit": biological_unit,
            "analysis_role": analysis_role,
            "gene_id_type": gene_id_type,
            "n_control": int(samples.group.eq("Control").sum()),
            "n_treatment": int(samples.group.eq("Treatment").sum()),
            "n_genes_input": int(len(counts)),
            "counts_path": str(out / "counts.tsv.gz"),
            "samples_path": str(out / "samples.tsv"),
        }
    )


def prepare_gse75246(manifest, provenance):
    study = "GSE75246"
    folder = DATA_ROOT / "GSE75246_Bhatt2018_LPS_sorted_cells"
    sample_meta = {
        "GSM1947162": (1, "microglia", "Treatment"),
        "GSM1947164": (2, "neuron", "Treatment"),
        "GSM1947165": (3, "microglia", "Control"),
        "GSM1947166": (4, "microglia", "Control"),
        "GSM1947167": (5, "astrocyte", "Control"),
        "GSM1947168": (6, "astrocyte", "Treatment"),
        "GSM1947169": (7, "microglia", "Treatment"),
        "GSM1947170": (8, "neuron", "Control"),
        "GSM1947171": (9, "neuron", "Control"),
        "GSM1947172": (10, "microglia", "Control"),
        "GSM1947173": (11, "astrocyte", "Control"),
        "GSM1947174": (12, "astrocyte", "Control"),
        "GSM1947175": (13, "neuron", "Control"),
        "GSM1947176": (14, "microglia", "Control"),
        "GSM1947177": (15, "astrocyte", "Control"),
        "GSM1947178": (16, "astrocyte", "Treatment"),
        "GSM1947179": (17, "microglia", "Treatment"),
        "GSM1947180": (18, "neuron", "Treatment"),
        "GSM1947181": (19, "microglia", "Treatment"),
        "GSM1947182": (20, "astrocyte", "Treatment"),
        "GSM1947183": (21, "neuron", "Control"),
        "GSM1947184": (22, "neuron", "Treatment"),
        "GSM1947185": (23, "microglia", "Control"),
        "GSM1947186": (24, "microglia", "Treatment"),
        "GSM1947187": (25, "neuron", "Treatment"),
        "GSM1947188": (26, "neuron", "Treatment"),
        "GSM1947189": (27, "neuron", "Control"),
        "GSM1947190": (28, "astrocyte", "Treatment"),
    }
    series = {}
    sample_rows = []
    for gsm, (number, cell_type, group) in sample_meta.items():
        if cell_type != "microglia":
            continue
        path = folder / f"{gsm}_sample{number}.txt.gz"
        record_source(path, study, "raw per-sample counts", provenance)
        data = pd.read_csv(path, sep="\t", skiprows=4, dtype={"ID_REF": str})
        series[gsm] = data.set_index("ID_REF")["count"].astype(float)
        sample_rows.append({"sample_id": gsm, "group": group, "cell_type": cell_type})
    counts = pd.DataFrame(series).fillna(0)
    samples = pd.DataFrame(sample_rows).sort_values(["group", "sample_id"])
    write_contrast(
        "gse75246_microglia_lps",
        study,
        "LPS",
        "hemicortex",
        "24h",
        "bulk RNA-seq, FACS-sorted CD11b+",
        counts,
        samples,
        "NCBI_Entrez_GeneID",
        manifest,
    )


def read_ribotag(path: Path):
    data = pd.read_csv(path, sep="\t", compression="gzip")
    if "gene_name" not in data:
        raise ValueError(f"gene_name missing in {path}")
    value_cols = [c for c in data if c not in {"gene_id", "gene_name"}]
    if len(value_cols) != 1:
        raise ValueError(f"unexpected count columns in {path}: {value_cols}")
    return data.groupby("gene_name")[value_cols[0]].sum().astype(float)


def prepare_gse213100(manifest, provenance):
    study = "GSE213100"
    folder = DATA_ROOT / "GSE213100_MPTP_celltype"
    groups = {
        "SN": {
            "Control": [
                "GSM6571656_MGSNCOUNTS1",
                "GSM6571657_MGSNCOUNTS2",
                "GSM6571658_MGSNCOUNTS3",
            ],
            "Treatment": [
                "GSM6571659_MGSNCOUNTM1",
                "GSM6571660_MGSNCOUNTM2",
                "GSM6571661_MGSNCOUNTM3",
            ],
        },
        "Striatum": {
            "Control": [
                "GSM6571662_MGSTRCOUNTS1",
                "GSM6571663_MGSTRCOUNTS2",
                "GSM6571664_MGSTRCOUNTS3",
                "GSM6571665_MGSTRCOUNTS4",
            ],
            "Treatment": [
                "GSM6571666_MGSTRCOUNTM1",
                "GSM6571667_MGSTRCOUNTM2",
                "GSM6571668_MGSTRCOUNTM3",
            ],
        },
    }
    for region, design in groups.items():
        series, rows = {}, []
        for group, ids in design.items():
            for sample_id in ids:
                path = folder / f"{sample_id}.txt.gz"
                record_source(path, study, f"raw microglia RiboTag counts ({region})", provenance)
                series[sample_id] = read_ribotag(path)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "group": group,
                        "cell_type": "microglia",
                        "region": region,
                    }
                )
        counts = pd.DataFrame(series).fillna(0)
        samples = pd.DataFrame(rows).sort_values(["group", "sample_id"])
        write_contrast(
            f"gse213100_microglia_{region.lower()}",
            study,
            "MPTP",
            region,
            "subchronic MPTP",
            "RiboTag RNA-seq",
            counts,
            samples,
            "mouse_gene_symbol",
            manifest,
        )


def load_tx2gene(path: Path):
    table = pd.read_csv(path, sep="\t")
    if table.shape[1] < 2:
        raise ValueError(path)
    mapping = dict(zip(table.iloc[:, 0].astype(str), table.iloc[:, 1].astype(str)))
    mapping_base = {k.rsplit(".", 1)[0]: v for k, v in mapping.items()}
    return mapping, mapping_base


def load_kallisto_unadjusted_diagnostic(path: Path, mapping, mapping_base):
    """Historical import for regression checks only, never revised inference."""
    with h5py.File(path, "r") as handle:
        ids = handle["aux/ids"][:]
        values = handle["est_counts"][:]
    genes = {}
    for raw_id, value in zip(ids, values):
        tx = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
        gene = mapping.get(tx) or mapping_base.get(tx.rsplit(".", 1)[0])
        if gene and gene != "nan":
            genes[gene] = genes.get(gene, 0.0) + float(value)
    return pd.Series(genes, dtype=float)


def prepare_gse244652(manifest, provenance):
    study = "GSE244652"
    folder = DATA_ROOT / "GSE244652_MPTP_sorted"
    tx_path = folder / "tx2gene_ensembl.tsv"
    record_source(tx_path, study, "transcript-to-gene map", provenance)
    mapping, mapping_base = load_tx2gene(tx_path)
    groups = {}
    for path in sorted(folder.glob("*.h5")):
        name = path.stem
        parts = name.split("_")
        if len(parts) < 4:
            continue
        code, cell_type = parts[1], parts[2]
        if cell_type != "CD11b":
            continue
        if code.startswith("Ctrl") or code.startswith("MPTP"):
            group = "Control" if code.startswith("Ctrl") else "Treatment"
            region_token = parts[3]
            region = (
                "Striatum"
                if region_token == "str"
                else "Midbrain"
                if region_token == "MdB"
                else region_token
            )
            regimen = "subacute"
        elif re.fullmatch(r"[CM]\d+[SM]", code):
            group = "Control" if code.startswith("C") else "Treatment"
            # In the encoded chronic-series filenames, the terminal letter is
            # anatomical: S=striatum, M=midbrain.  Historical code treated it
            # as regimen, which incorrectly pooled regions.
            region = "Striatum" if code.endswith("S") else "Midbrain"
            regimen = "chronic_probenecid"
        else:
            continue
        contrast = f"gse244652_microglia_{region.lower()}_{regimen}"
        groups.setdefault(contrast, {"region": region, "regimen": regimen, "samples": []})
        groups[contrast]["samples"].append((path, group))
    for contrast, info in sorted(groups.items()):
        rows = []
        for path, group in info["samples"]:
            record_source(
                path,
                study,
                f"Kallisto counts and effective lengths ({info['region']}, {info['regimen']})",
                provenance,
            )
            rows.append(
                {
                    "sample_id": path.name,
                    "group": group,
                    "cell_type": "CD11b+",
                    "region": info["region"],
                    "regimen": info["regimen"],
                    "input_path": str(path.resolve()),
                }
            )
        samples = pd.DataFrame(rows).sort_values(["group", "sample_id"])
        if samples.groupby("group").size().min() < 3:
            raise RuntimeError(
                f"unexpected GSE244652 design in {contrast}: {samples.groupby('group').size().to_dict()}"
            )
        out = COUNTS_DIR / contrast
        out.mkdir(parents=True, exist_ok=True)
        samples.to_csv(out / "samples.tsv", sep="\t", index=False)
        subprocess.run(
            [
                "Rscript",
                str(Path(__file__).with_name("00_import_kallisto.R")),
                str(out / "samples.tsv"),
                str(tx_path),
                str(out),
            ],
            check=True,
        )
        counts = pd.read_csv(out / "counts.tsv.gz", sep="\t", index_col="gene_id")
        write_contrast(
            contrast,
            study,
            "MPTP",
            info["region"],
            info["regimen"],
            "bulk MARS-seq, CD11b-selected myeloid-enriched; tximport original gene counts",
            counts,
            samples,
            "mouse_gene_symbol",
            manifest,
        )


def prepare_gse263094(manifest, provenance, seed):
    # Imports are delayed because scanpy is the heaviest dependency.
    tf_mock = types.ModuleType("tensorflow")
    tf_mock.__version__ = "0.0.0"
    sys.modules.setdefault("tensorflow", tf_mock)
    sys.modules.setdefault("tensorflow.python", types.ModuleType("tensorflow.python"))
    import scanpy as sc
    import harmonypy

    study = "GSE263094"
    folder = DATA_ROOT / "GSE263094"
    sample_conditions = {
        "GSM8185728_1": "PBS",
        "GSM8185733_6": "PBS",
        "GSM8185729_2": "LPSx1",
        "GSM8185734_7": "LPSx1",
        "GSM8185730_3": "LPSx2",
        "GSM8185735_8": "LPSx2",
    }
    adatas = []
    for sample_id, condition in sample_conditions.items():
        path = folder / f"{sample_id}-filtered_feature_bc_matrix.h5"
        record_source(path, study, "raw 10x count matrix", provenance)
        adata = sc.read_10x_h5(path)
        adata.var_names_make_unique()
        adata.obs_names = [f"{sample_id}_{x}" for x in adata.obs_names]
        adata.obs["sample_id"] = sample_id
        adata.obs["condition"] = condition
        adatas.append(adata)
    adata = sc.concat(adatas, join="outer")
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    n_raw = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, min_counts=500)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[(adata.obs.pct_counts_mt < 20) & (adata.obs.n_genes_by_counts < 8500)].copy()
    n_after_basic_qc = adata.n_obs
    # Match the source paper's doublet-removal step, applying Scrublet within
    # each independent library rather than across experimental conditions.
    sc.pp.scrublet(
        adata,
        batch_key="sample_id",
        random_state=seed,
        verbose=False,
    )
    predicted_doublet = adata.obs["predicted_doublet"].fillna(False).to_numpy(bool)
    n_predicted_doublets = int(predicted_doublet.sum())
    adata = adata[~predicted_doublet].copy()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=2000,
        batch_key="sample_id",
        flavor="seurat_v3",
        layer="counts",
    )
    sc.tl.pca(
        adata,
        n_comps=40,
        mask_var="highly_variable",
        svd_solver="arpack",
        random_state=seed,
    )
    pca_coordinates = np.asarray(adata.obsm["X_pca"])
    if pca_coordinates.shape != (adata.n_obs, 40) or not np.isfinite(pca_coordinates).all():
        raise RuntimeError(
            f"invalid PCA coordinates: {pca_coordinates.shape}, "
            f"finite={np.isfinite(pca_coordinates).all()}"
        )
    # harmonypy 0.2 returns cells x PCs, while Scanpy's compatibility wrapper
    # still transposes the pre-0.2 PCs x cells layout.  Call harmonypy directly
    # and accept either documented orientation only after an explicit check.
    harmony_out = harmonypy.run_harmony(
        pca_coordinates.astype(np.float64, copy=False),
        adata.obs,
        "sample_id",
        random_state=seed,
        device="cpu",
    )
    harmony_coordinates = validated_harmony_coordinates(
        harmony_out.Z_corr,
        adata.n_obs,
        40,
    )
    adata.obsm["X_pca_harmony"] = harmony_coordinates.astype(
        np.float32,
        copy=False,
    )
    sc.pp.neighbors(
        adata,
        n_pcs=40,
        use_rep="X_pca_harmony",
        random_state=seed,
    )
    # The source publication reported 17 broad whole-brain clusters.  A lower
    # resolution is used here for broad lineage assignment; microglial state
    # discovery is not the inferential purpose of this rebuild.
    sc.tl.leiden(adata, resolution=0.6, random_state=seed)
    adata.obs["leiden"] = adata.obs.leiden.astype(str)

    marker_genes = {
        "Microglia": ["Cx3cr1", "P2ry12", "Tmem119", "Hexb", "Fcrls", "Sall1", "Tgfbr1", "Olfml3"],
        "Monocyte_macrophage": ["Ccr2", "Ly6c2", "Lyz2", "Cd14", "Fcgr3", "Ms4a7", "Mrc1"],
        "Neutrophil": ["S100a8", "S100a9", "Lcn2", "Retnlg", "Ngp", "Camp", "Ly6g"],
        "Excitatory_neuron": ["Slc17a7", "Camk2a", "Neurod6", "Satb2"],
        "Inhibitory_neuron": ["Gad1", "Gad2", "Slc32a1"],
        "Astrocyte": ["Gfap", "Aqp4", "Aldh1l1", "Slc1a3"],
        "Oligodendrocyte": ["Mbp", "Plp1", "Mog"],
        "OPC": ["Pdgfra", "Cspg4"],
        "Endothelial": ["Pecam1", "Cldn5", "Flt1"],
        "Pericyte": ["Pdgfrb", "Rgs5"],
        "T_cell": ["Cd3d", "Cd3e", "Trbc1", "Trbc2"],
        "B_cell": ["Cd79a", "Ms4a1", "Cd74", "Cd37"],
        "Ependymal": ["Foxj1", "Pifo", "Tmem212", "Dynlrb2"],
    }
    clusters = sorted(adata.obs.leiden.unique(), key=lambda x: int(x))
    present_markers = sorted(
        {gene for genes in marker_genes.values() for gene in genes if gene in adata.var_names}
    )
    cluster_mean = pd.DataFrame(index=clusters, columns=present_markers, dtype=float)
    for cluster in clusters:
        mask = adata.obs.leiden.eq(cluster).to_numpy()
        cluster_mean.loc[cluster] = np.asarray(adata[mask, present_markers].X.mean(axis=0)).ravel()
    # Standardize each marker across clusters before averaging a lineage
    # signature.  This prevents signatures with different gene counts or
    # highly expressed genes from winning purely because of scale.
    gene_sd = cluster_mean.std(axis=0, ddof=0).replace(0, np.nan)
    marker_z = (
        cluster_mean.subtract(cluster_mean.mean(axis=0), axis=1)
        .divide(
            gene_sd,
            axis=1,
        )
        .fillna(0)
    )
    signature_scores = pd.DataFrame(index=clusters, dtype=float)
    raw_signature_means = pd.DataFrame(index=clusters, dtype=float)
    for cell_type, genes in marker_genes.items():
        genes = [gene for gene in genes if gene in present_markers]
        signature_scores[cell_type] = marker_z[genes].mean(axis=1)
        raw_signature_means[cell_type] = cluster_mean[genes].mean(axis=1)

    assignments = []
    cluster_types = {}
    for cluster in clusters:
        ordered = signature_scores.loc[cluster].sort_values(ascending=False)
        best, runner_up = ordered.index[:2]
        cluster_types[cluster] = best
        myeloid_margin = float(
            signature_scores.loc[cluster, "Microglia"]
            - signature_scores.loc[cluster, ["Monocyte_macrophage", "Neutrophil"]].max()
        )
        assignments.append(
            {
                "cluster": cluster,
                "assigned_cell_type": best,
                "runner_up_cell_type": runner_up,
                "signature_z_score": float(ordered.iloc[0]),
                "runner_up_z_score": float(ordered.iloc[1]),
                "assignment_margin": float(ordered.iloc[0] - ordered.iloc[1]),
                "microglia_vs_infiltrating_myeloid_margin": myeloid_margin,
                "n_cells": int(adata.obs.leiden.eq(cluster).sum()),
            }
        )
    assignment_df = pd.DataFrame(assignments)
    cluster_scores = []
    for cluster in clusters:
        for cell_type in marker_genes:
            cluster_scores.append(
                {
                    "cluster": cluster,
                    "cell_type": cell_type,
                    "signature_z_score": float(signature_scores.loc[cluster, cell_type]),
                    "mean_marker_score": float(raw_signature_means.loc[cluster, cell_type]),
                    "assigned": cluster_types[cluster] == cell_type,
                }
            )
    adata.obs["cell_type"] = adata.obs.leiden.map(cluster_types)
    mg = adata.obs.cell_type.eq("Microglia").to_numpy()
    strict_clusters = set(
        assignment_df.loc[
            assignment_df.assigned_cell_type.eq("Microglia")
            & assignment_df.microglia_vs_infiltrating_myeloid_margin.ge(0.25),
            "cluster",
        ].astype(str)
    )
    mg_strict = adata.obs.leiden.astype(str).isin(strict_clusters).to_numpy()
    counts_layer = adata.layers["counts"]
    pseudobulk = {}
    pseudobulk_strict = {}
    qc_rows = []
    for sample_id, condition in sample_conditions.items():
        sample_mask = adata.obs.sample_id.eq(sample_id).to_numpy()
        keep = sample_mask & mg
        keep_strict = sample_mask & mg_strict
        vector = counts_layer[keep].sum(axis=0)
        vector_strict = counts_layer[keep_strict].sum(axis=0)
        pseudobulk[sample_id] = np.asarray(vector).ravel()
        pseudobulk_strict[sample_id] = np.asarray(vector_strict).ravel()
        qc_rows.append(
            {
                "sample_id": sample_id,
                "condition": condition,
                "n_cells_after_qc": int(sample_mask.sum()),
                "n_microglia": int(keep.sum()),
                "n_microglia_resident_lineage_sensitivity": int(keep_strict.sum()),
                "library_size_microglia": float(np.asarray(vector).sum()),
                "library_size_microglia_resident_lineage_sensitivity": float(
                    np.asarray(vector_strict).sum()
                ),
            }
        )
    qc = pd.DataFrame(qc_rows)
    if qc.n_microglia.min() < 20 or qc.n_microglia_resident_lineage_sensitivity.min() < 20:
        raise RuntimeError(f"GSE263094 has <20 microglia in a sample: {qc.to_dict('records')}")
    counts = pd.DataFrame(pseudobulk, index=adata.var_names)
    counts_strict = pd.DataFrame(pseudobulk_strict, index=adata.var_names)
    qc.to_csv(DERIVED / "gse263094_pseudobulk_qc.tsv", sep="\t", index=False)
    pd.DataFrame(cluster_scores).to_csv(
        DERIVED / "gse263094_cluster_marker_scores.tsv", sep="\t", index=False
    )
    assignment_df.to_csv(DERIVED / "gse263094_cluster_assignments.tsv", sep="\t", index=False)
    composition = (
        adata.obs.groupby(["leiden", "cell_type", "sample_id", "condition"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    composition.to_csv(DERIVED / "gse263094_cluster_composition.tsv", sep="\t", index=False)
    (DERIVED / "gse263094_processing_summary.json").write_text(
        json.dumps(
            {
                "cells_raw": int(n_raw),
                "cells_after_basic_qc_before_scrublet": int(n_after_basic_qc),
                "predicted_doublets_removed": n_predicted_doublets,
                "cells_after_qc": int(adata.n_obs),
                "microglia": int(mg.sum()),
                "microglia_resident_lineage_sensitivity": int(mg_strict.sum()),
                "n_clusters": int(len(clusters)),
                "random_seed": int(seed),
                "pca_solver": "arpack with explicit finite/shape validation",
                "harmony_implementation": (
                    f"harmonypy {harmonypy.__version__} direct CPU call with "
                    "orientation, finite-value, and shape validation"
                ),
                "assignment": "Harmony-corrected unsupervised Leiden clusters assigned by standardized canonical lineage signatures including infiltrating myeloid competitors; no cell-level treatment test",
                "resident_lineage_sensitivity": "assigned Microglia and microglia-minus-max(monocyte/macrophage, neutrophil) signature margin >=0.25; sensitivity only because activation can reduce resident-lineage markers",
                "biological_unit": "one independently prepared pooled 10x library; three mice per library (two male, one female)",
            },
            indent=2,
        )
        + "\n"
    )
    sensitivity_manifest = []
    for treatment in ["LPSx1", "LPSx2"]:
        chosen = [s for s, c in sample_conditions.items() if c in {"PBS", treatment}]
        samples = pd.DataFrame(
            {
                "sample_id": chosen,
                "group": [
                    "Control" if sample_conditions[s] == "PBS" else "Treatment" for s in chosen
                ],
                "cell_type": "microglia",
                "condition_detail": [sample_conditions[s] for s in chosen],
            }
        ).sort_values(["group", "sample_id"])
        write_contrast(
            f"gse263094_microglia_{treatment.lower()}",
            study,
            "LPS",
            "whole brain",
            treatment,
            "single-cell RNA-seq raw-count pseudobulk",
            counts[chosen],
            samples,
            "mouse_gene_symbol",
            manifest,
            biological_unit=(
                "one independently prepared pooled 10x library per column; "
                "three mice/library (two male, one female)"
            ),
        )
        write_contrast(
            f"gse263094_microglia_{treatment.lower()}_resident_lineage_sensitivity",
            study,
            "LPS",
            "whole brain",
            treatment,
            "single-cell RNA-seq raw-count pseudobulk; resident-lineage sensitivity",
            counts_strict[chosen],
            samples,
            "mouse_gene_symbol",
            sensitivity_manifest,
            biological_unit=(
                "one independently prepared pooled 10x library per column; "
                "three mice/library (two male, one female)"
            ),
            output_base=DERIVED / "sensitivity_counts",
            analysis_role="cell_definition_sensitivity_not_independent",
        )
    pd.DataFrame(sensitivity_manifest).to_csv(
        DERIVED / "gse263094_cell_definition_sensitivity_manifest.tsv",
        sep="\t",
        index=False,
    )


def write_unreplicated_source_manifest():
    rows = [
        {
            "study": "GSE307796 / Moon 2025",
            "injury": "LPS",
            "status": "descriptive_only",
            "reason": "5-7 mice pooled into one 10x library per condition; no condition-level library replication",
            "allowed_use": "cell-state and expression-pattern description; no treatment DE p value",
        },
    ]
    DERIVED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        DERIVED / "transcriptomic_unreplicated_sources.tsv", sep="\t", index=False
    )


def build_panel_entrez_mapping():
    """Resolve only the prespecified symbols needed to label GSE75246.

    The complete count matrix remains keyed by the deposited Entrez IDs.  The
    small cached mapping makes later reruns network-independent and records the
    resolver and timestamp explicitly.
    """
    out = DERIVED / "panel_symbol_to_entrez.tsv"
    if out.exists():
        cached = pd.read_csv(out, sep="\t", dtype=str)
        if set(PANEL).issubset(set(cached.gene)):
            return
    import mygene

    client = mygene.MyGeneInfo()
    hits = client.querymany(
        list(PANEL),
        scopes="symbol",
        fields="symbol,entrezgene",
        species="mouse",
        as_dataframe=False,
        verbose=False,
    )
    rows = []
    for hit in hits:
        query = hit.get("query")
        entrez = hit.get("entrezgene")
        if query in PANEL and entrez is not None and not hit.get("notfound", False):
            rows.append(
                {
                    "gene": query,
                    "entrezgene": str(entrez),
                    "returned_symbol": hit.get("symbol"),
                    "resolver": "MyGene.info mouse symbol query",
                    "queried_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
    mapping = pd.DataFrame(rows).drop_duplicates("gene")
    missing = sorted(set(PANEL) - set(mapping.gene))
    if missing:
        raise RuntimeError(f"Could not resolve panel symbols to Entrez IDs: {missing}")
    mapping.sort_values("gene").to_csv(out, sep="\t", index=False)


# The one configuration value this stage needs. It was previously read from the imaging
# pipeline's configuration file, which had to be carried along for a single integer and named
# directories on the author's machine. It is fixed, so it is stated here.
RANDOM_SEED = 20260713


def main():
    DERIVED.mkdir(parents=True, exist_ok=True)
    COUNTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest, provenance = [], []
    panel = pd.DataFrame(
        [
            {"gene": gene, "mechanism": mechanism, "expected_direction": direction}
            for gene, (mechanism, direction) in PANEL.items()
        ]
    )
    panel.to_csv(DERIVED / "prespecified_gene_panel.tsv", sep="\t", index=False)
    build_panel_entrez_mapping()
    prepare_gse75246(manifest, provenance)
    prepare_gse213100(manifest, provenance)
    prepare_gse244652(manifest, provenance)
    prepare_gse263094(manifest, provenance, RANDOM_SEED)
    contrast_manifest = pd.DataFrame(manifest).sort_values(["injury", "study", "contrast_id"])
    contrast_manifest.to_csv(DERIVED / "contrast_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(provenance).drop_duplicates("path").sort_values(["study", "path"]).to_csv(
        DERIVED / "raw_input_provenance.tsv",
        sep="\t",
        index=False,
    )
    write_unreplicated_source_manifest()
    print(
        contrast_manifest[
            [
                "contrast_id",
                "study",
                "injury",
                "region",
                "regimen_or_time",
                "n_control",
                "n_treatment",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
