"""Extract a fixed curated panel from completed whole-transcriptome model fits.

This is result extraction, not a new differential-expression analysis. Estimates,
confidence intervals, P values and BH adjustments are copied from complete fits.
Missing or expression-filtered genes remain explicit untested rows.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from mg_paths import BUNDLE, DERIVED, require_writable_output


def read_table(path):
    return pd.read_csv(path, sep="\t", float_precision="round_trip")


def boolean_column(values):
    converted = values.astype(str).str.lower().map({"true": True, "false": False})
    if converted.isna().any():
        raise ValueError("Expected explicit TRUE/FALSE values")
    return converted.astype(bool)


def human_panels(folder, genes):
    """Return every requested gene for every fit, with its filtering status."""
    qc = read_table(folder / "analysis_qc.tsv")
    if qc.analysis.duplicated().any() or len(set(genes)) != len(genes):
        raise ValueError("Duplicate analysis names or panel genes")
    panels = []
    for rec in qc.itertuples(index=False):
        fit_dir = folder / rec.analysis
        fit = read_table(fit_dir / "all_genes.tsv.gz")
        filtering = read_table(fit_dir / "filtering.tsv")
        if fit.gene.duplicated().any() or filtering.gene.duplicated().any():
            raise ValueError(f"Duplicate gene identifiers in {fit_dir}")
        tested = set(filtering.loc[boolean_column(filtering.tested), "gene"])
        if set(fit.gene) != tested or len(tested) != rec.n_tested_genes:
            raise ValueError(f"Filtering and fitted gene sets disagree in {fit_dir}")
        panel = pd.DataFrame({"gene": genes}).merge(
            fit, on="gene", how="left", validate="one_to_one"
        )
        panel["analysis"] = rec.analysis
        panel["tested"] = panel.gene.isin(tested)
        panel["measured"] = panel.gene.isin(filtering.gene)
        for name in ["n_control", "n_pd", "n_tested_genes"]:
            panel[name] = getattr(rec, name)
        if not panel["P.Value"].notna().equals(panel.tested):
            raise ValueError(f"Missing P value for a tested gene in {fit_dir}")
        panels.append(panel)
    result = pd.concat(panels, ignore_index=True)
    counts = result.loc[result.tested & result["adj.P.Val"].lt(0.05)].groupby("analysis").size()
    qc["n_panel_fdr"] = qc.analysis.map(counts).fillna(0).astype(int)
    return result, qc


def mouse_panel(path, definition):
    table = read_table(path)
    if table.duplicated(["contrast_id", "gene"]).any():
        raise ValueError("Duplicate mouse contrast/gene rows")
    for name, block in table.groupby("contrast_id"):
        if set(block.gene) != set(definition.gene):
            raise ValueError(f"Incomplete curated panel in {name}")
    table = table.drop(
        columns=["expected_direction", "aligned_log2fc", "direction_matches_panel"], errors="ignore"
    )
    table["mechanism"] = table.gene.map(definition.set_index("gene").mechanism)
    table["tested"] = boolean_column(table.tested)
    if not table.log2fc.notna().equals(table.tested):
        raise ValueError("Mouse filtering status and estimates disagree")
    return table


def mptp_sensitivity(folder, genes):
    manifest = read_table(folder / "sensitivity_manifest.tsv").fillna({"omitted_sample": ""})
    rows = []
    for rec in manifest.itertuples(index=False):
        fit = read_table(
            folder / "fits" / rec.contrast_id / (rec.contrast_id + "_limma_voom.tsv.gz")
        )
        block = pd.DataFrame({"gene_id": genes}).merge(
            fit, on="gene_id", how="left", validate="one_to_one"
        )
        block["tested"] = block["P.Value"].notna()
        block["contrast_id"] = rec.contrast_id
        for name in ["original_contrast", "import_mode", "omitted_sample"]:
            block[name] = getattr(rec, name)
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mouse", type=Path, default=DERIVED / "transcriptomic_biological_sample_panel_de.tsv"
    )
    parser.add_argument("--snpc", type=Path, default=BUNDLE / "corrected/human_pd/limma_voom")
    parser.add_argument(
        "--putamen", type=Path, default=BUNDLE / "xu_putamen/results/differential_expression"
    )
    parser.add_argument("--mptp-sensitivity", type=Path, default=BUNDLE / "corrected/mptp_audit")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = require_writable_output(args.out)
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    definition = read_table(ROOT / "config/transcript_panel.tsv")
    mouse = mouse_panel(args.mouse, definition)
    human = {
        name: human_panels(folder, definition.gene.str.upper().tolist())
        for name, folder in [("SNpc", args.snpc), ("putamen", args.putamen)]
    }
    sensitivity = mptp_sensitivity(args.mptp_sensitivity, definition.gene.tolist())
    sources = [args.mouse, ROOT / "config/transcript_panel.tsv"]
    for folder in [args.snpc, args.putamen]:
        sources.extend(
            [
                folder / "analysis_qc.tsv",
                *sorted(folder.glob("*/all_genes.tsv.gz")),
                *sorted(folder.glob("*/filtering.tsv")),
            ]
        )
    sources.extend(
        [
            args.mptp_sensitivity / "sensitivity_manifest.tsv",
            *sorted((args.mptp_sensitivity / "fits").glob("*/*_limma_voom.tsv.gz")),
        ]
    )
    output.mkdir(parents=True)
    mouse.to_csv(output / "transcriptomic_biological_sample_panel_de.tsv", sep="\t", index=False)
    definition.to_csv(output / "prespecified_gene_panel.tsv", sep="\t", index=False)
    sensitivity.to_csv(output / "MPTP_panel_sensitivity.tsv", sep="\t", index=False)
    for name, (panel, qc) in human.items():
        (output / name).mkdir()
        panel.to_csv(output / name / "panel_all_analyses.tsv", sep="\t", index=False)
        qc.to_csv(output / name / "analysis_qc.tsv", sep="\t", index=False)
    record = {
        "refitted": False,
        "fdr_recalculated": False,
        "panel_genes": len(definition),
        "comparison_genes": int(definition.comparison_order.notna().sum()),
        "sources": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sources
        ],
    }
    (output / "extraction_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "sources"}, indent=2))


if __name__ == "__main__":
    main()
