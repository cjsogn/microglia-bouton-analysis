#!/usr/bin/env python3
"""Dependency-aware descriptive synthesis of valid sample-level contrasts."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parent))  # so `common` resolves from any cwd

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import OUTPUT


# Read the count preparation and model outputs from the active run directory.
DERIVED = OUTPUT / "transcriptomics"
# Panel summaries are written alongside the other analysis outputs.
RESULTS = OUTPUT
RESULTS.mkdir(parents=True, exist_ok=True)
DE_DIR = DERIVED / "limma_voom"


def load_results():
    manifest = pd.read_csv(DERIVED / "contrast_manifest.tsv", sep="\t")
    panel = pd.read_csv(DERIVED / "prespecified_gene_panel.tsv", sep="\t")
    entrez = pd.read_csv(DERIVED / "panel_symbol_to_entrez.tsv", sep="\t", dtype=str)
    reverse_entrez = dict(zip(entrez.entrezgene.astype(str), entrez.gene))
    all_tables = []
    for rec in manifest.itertuples(index=False):
        path = DE_DIR / f"{rec.contrast_id}_limma_voom.tsv.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        table = pd.read_csv(path, sep="\t", dtype={"gene_id": str})
        if rec.gene_id_type == "NCBI_Entrez_GeneID":
            table["gene"] = table.gene_id.map(reverse_entrez)
        else:
            table["gene"] = table.gene_id
        all_tables.append(table)
    full = pd.concat(all_tables, ignore_index=True)
    full.to_parquet(DERIVED / "all_limma_voom_results.parquet", index=False, compression="zstd")

    rows = []
    panel_index = panel.set_index("gene")
    for rec in manifest.itertuples(index=False):
        sub = full[full.contrast_id.eq(rec.contrast_id)]
        by_gene = sub.dropna(subset=["gene"]).drop_duplicates("gene").set_index("gene")
        for gene, definition in panel_index.iterrows():
            if gene in by_gene.index:
                row = by_gene.loc[gene]
                rows.append(
                    {
                        "contrast_id": rec.contrast_id,
                        "study": rec.study,
                        "injury": rec.injury,
                        "region": rec.region,
                        "regimen_or_time": rec.regimen_or_time,
                        "n_control": rec.n_control,
                        "n_treatment": rec.n_treatment,
                        "gene": gene,
                        "mechanism": definition.mechanism,
                        "expected_direction": definition.expected_direction,
                        "log2fc": row.logFC,
                        "se_log2fc": row.se_log2fc,
                        "ci95_low": row.ci95_low,
                        "ci95_high": row.ci95_high,
                        "t": row.t,
                        "p_value": row["P.Value"],
                        "fdr_all_tested_genes": row["adj.P.Val"],
                        "ave_expression": row.AveExpr,
                        "tested": True,
                    }
                )
            else:
                rows.append(
                    {
                        "contrast_id": rec.contrast_id,
                        "study": rec.study,
                        "injury": rec.injury,
                        "region": rec.region,
                        "regimen_or_time": rec.regimen_or_time,
                        "n_control": rec.n_control,
                        "n_treatment": rec.n_treatment,
                        "gene": gene,
                        "mechanism": definition.mechanism,
                        "expected_direction": definition.expected_direction,
                        "tested": False,
                    }
                )
    panel_de = pd.DataFrame(rows)
    panel_de["aligned_log2fc"] = np.where(
        panel_de.expected_direction.eq("down"),
        -panel_de.log2fc,
        panel_de.log2fc,
    )
    panel_de["direction_matches_panel"] = panel_de.aligned_log2fc > 0
    panel_de.to_csv(
        RESULTS / "transcriptomic_biological_sample_panel_de.tsv", sep="\t", index=False
    )
    return manifest, panel, panel_de


def gse263094_cell_definition_sensitivity(panel):
    """Audit primary and resident-lineage-restricted pseudobulks.

    These are two definitions applied to the same pooled libraries, never two
    independent contrasts. Cell-set equality is checked explicitly so an
    unchanged selection is not overstated as a substantive sensitivity test.
    """
    manifest_path = DERIVED / "gse263094_cell_definition_sensitivity_manifest.tsv"
    if not manifest_path.exists():
        return None
    manifest = pd.read_csv(manifest_path, sep="\t")
    rows = []
    all_gene_pairs = []
    for rec in manifest.itertuples(index=False):
        primary_id = rec.contrast_id.replace(
            "_resident_lineage_sensitivity",
            "",
        )
        primary = pd.read_csv(
            DE_DIR / f"{primary_id}_limma_voom.tsv.gz",
            sep="\t",
        ).set_index("gene_id")
        restricted = pd.read_csv(
            DERIVED / "limma_voom_sensitivity" / f"{rec.contrast_id}_limma_voom.tsv.gz",
            sep="\t",
        ).set_index("gene_id")
        shared = primary.index.intersection(restricted.index)
        paired = pd.DataFrame(
            {
                "primary_log2fc": primary.loc[shared, "logFC"],
                "resident_lineage_sensitivity_log2fc": restricted.loc[shared, "logFC"],
            }
        ).dropna()
        all_gene_pairs.append(paired.assign(contrast_id=primary_id))
        for gene in panel.gene:
            if gene not in primary.index or gene not in restricted.index:
                continue
            rows.append(
                {
                    "contrast_id": primary_id,
                    "sensitivity_contrast_id": rec.contrast_id,
                    "gene": gene,
                    "primary_log2fc": float(primary.loc[gene, "logFC"]),
                    "resident_lineage_sensitivity_log2fc": float(restricted.loc[gene, "logFC"]),
                    "absolute_log2fc_difference": float(
                        abs(primary.loc[gene, "logFC"] - restricted.loc[gene, "logFC"])
                    ),
                    "same_direction": bool(
                        np.sign(primary.loc[gene, "logFC"])
                        == np.sign(restricted.loc[gene, "logFC"])
                    ),
                    "primary_fdr_all_genes": float(primary.loc[gene, "adj.P.Val"]),
                    "sensitivity_fdr_all_genes": float(restricted.loc[gene, "adj.P.Val"]),
                    "interpretation": (
                        "same pooled libraries; stricter rule retained the same "
                        "cells; classification-margin audit only"
                    ),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(
        RESULTS / "transcriptomic_gse263094_cell_definition_sensitivity.tsv",
        sep="\t",
        index=False,
    )
    gene_pair = pd.concat(all_gene_pairs, ignore_index=True)
    cell_qc = pd.read_csv(
        DERIVED / "gse263094_pseudobulk_qc.tsv",
        sep="\t",
    )
    required = {
        "n_microglia",
        "n_microglia_resident_lineage_sensitivity",
        "library_size_microglia",
        "library_size_microglia_resident_lineage_sensitivity",
    }
    if not required.issubset(cell_qc.columns):
        raise RuntimeError("GSE263094 cell-definition QC columns are incomplete")
    same_cell_counts = bool(
        cell_qc.n_microglia.eq(cell_qc.n_microglia_resident_lineage_sensitivity).all()
    )
    same_library_sizes = bool(
        np.allclose(
            cell_qc.library_size_microglia,
            cell_qc.library_size_microglia_resident_lineage_sensitivity,
            rtol=0,
            atol=0,
        )
    )
    summary = {
        "panel_rows_compared": int(len(out)),
        "panel_fraction_same_direction": float(out.same_direction.mean()),
        "panel_median_absolute_log2fc_difference": float(out.absolute_log2fc_difference.median()),
        "all_tested_gene_spearman_rho": float(
            spearmanr(
                gene_pair.primary_log2fc,
                gene_pair.resident_lineage_sensitivity_log2fc,
            ).statistic
        ),
        "primary_cells": int(cell_qc.n_microglia.sum()),
        "resident_lineage_sensitivity_cells": int(
            cell_qc.n_microglia_resident_lineage_sensitivity.sum()
        ),
        "same_cell_count_in_every_library": same_cell_counts,
        "same_pseudobulk_library_size_in_every_library": same_library_sizes,
        "distinct_cell_selection_perturbation": not (same_cell_counts and same_library_sizes),
        "independent_replication": False,
        "selection_caveat": (
            "the stricter resident-lineage margin retained exactly the same "
            "cells and counts in this dataset; this is a classification-margin "
            "audit, not a distinct sensitivity sample or superior ground truth"
        ),
    }
    (RESULTS / "transcriptomic_gse263094_cell_definition_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main():
    manifest, panel, panel_de = load_results()
    cell_definition_sensitivity = gse263094_cell_definition_sensitivity(panel)
    summary = {
        "independent_studies": int(manifest.study.nunique()),
        "contrasts": int(len(manifest)),
        "lps_studies": int(manifest.loc[manifest.injury.eq("LPS"), "study"].nunique()),
        "mptp_studies": int(manifest.loc[manifest.injury.eq("MPTP"), "study"].nunique()),
        "panel_genes": int(len(panel)),
        "inferential_meta_analysis_performed": False,
        "reason": "two independent studies per injury; dependent within-study contrasts",
        "gse263094_cell_definition_sensitivity": cell_definition_sensitivity,
    }
    (RESULTS / "transcriptomic_rebuild_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
