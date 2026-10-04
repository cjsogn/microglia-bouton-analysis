"""Panel extraction preserves fitted inference and explicit missingness."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "panel_export", ROOT / "scripts/export_transcript_panels.py"
)
EXPORT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EXPORT)


def test_panel_definition_is_neutral_and_has_adrb2():
    panel = EXPORT.read_table(ROOT / "config/transcript_panel.tsv")
    assert len(panel) == 25 and panel.gene.is_unique
    assert panel.columns.tolist() == ["gene", "mechanism", "comparison_order"]
    comparison = panel.dropna(subset=["comparison_order"]).sort_values("comparison_order")
    assert len(comparison) == 12
    assert comparison.comparison_order.tolist() == list(range(1, 13))
    assert "Adrb2" in set(comparison.gene)


def test_human_extraction_copies_inference_and_marks_untested(tmp_path):
    folder = tmp_path / "fits"
    (folder / "primary").mkdir(parents=True)
    # Synthetic values test copying only. They are not scientific results.
    fit = pd.DataFrame(
        {
            "gene": ["ADRB2", "OTHER"],
            "logFC": [-0.4, 0.2],
            "P.Value": [0.031, 0.82],
            "adj.P.Val": [0.14, 0.91],
        }
    )
    fit.to_csv(folder / "primary/all_genes.tsv.gz", sep="\t", index=False)
    pd.DataFrame({"gene": ["ADRB2", "OTHER", "FILTERED"], "tested": [True, True, False]}).to_csv(
        folder / "primary/filtering.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        [{"analysis": "primary", "n_control": 4, "n_pd": 4, "n_tested_genes": 2, "n_panel_fdr": 0}]
    ).to_csv(folder / "analysis_qc.tsv", sep="\t", index=False)
    panel, _ = EXPORT.human_panels(folder, ["ADRB2", "FILTERED", "ABSENT"])
    assert panel.gene.tolist() == ["ADRB2", "FILTERED", "ABSENT"]
    assert panel.tested.tolist() == [True, False, False]
    assert panel.measured.tolist() == [True, True, False]
    for name in ["logFC", "P.Value", "adj.P.Val"]:
        assert panel.loc[0, name] == fit.loc[0, name]
        assert panel.loc[1:, name].isna().all()
    with pytest.raises(ValueError, match="Duplicate"):
        EXPORT.human_panels(folder, ["ADRB2", "ADRB2"])


def test_invalid_booleans_are_not_coerced_to_observed():
    with pytest.raises(ValueError, match="TRUE/FALSE"):
        EXPORT.boolean_column(pd.Series([True, None]))
