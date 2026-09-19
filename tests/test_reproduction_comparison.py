"""Synthetic fixtures for the deposited-table comparison contract."""

from pathlib import Path
import importlib.util

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "release_comparison", ROOT / "scripts/validate_reproduction.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_table(path, values):
    pd.DataFrame(values).to_csv(path, sep="\t", index=False)
    return path


def test_comparison_matches_keys_not_row_order(tmp_path):
    source = write_table(tmp_path / "source.tsv", {"key": ["a", "b"], "value": [1.0, 2.0]})
    rerun = write_table(tmp_path / "rerun.tsv", {"key": ["b", "a"], "value": [2.0, 1.0]})
    assert MODULE.compare(source, rerun, ["key"])["status"] == "PASS"


def test_comparison_rejects_missing_numeric_column(tmp_path):
    source = write_table(tmp_path / "source.tsv", {"key": ["a"], "value": [1.0]})
    rerun = write_table(tmp_path / "rerun.tsv", {"key": ["a"]})
    with pytest.raises(AssertionError, match="Missing numeric columns"):
        MODULE.compare(source, rerun, ["key"])


def test_comparison_rejects_changed_missingness(tmp_path):
    source = write_table(tmp_path / "source.tsv", {"key": ["a", "b"], "value": [1.0, 2.0]})
    rerun = write_table(tmp_path / "rerun.tsv", {"key": ["a", "b"], "value": [1.0, float("nan")]})
    with pytest.raises(AssertionError):
        MODULE.compare(source, rerun, ["key"])


def test_comparison_rejects_missing_rows(tmp_path):
    source = write_table(tmp_path / "source.tsv", {"key": ["a", "b"], "value": [1.0, 2.0]})
    rerun = write_table(tmp_path / "rerun.tsv", {"key": ["a"], "value": [1.0]})
    with pytest.raises(AssertionError, match="Changed row set"):
        MODULE.compare(source, rerun, ["key"])


def test_comparison_rejects_duplicate_keys(tmp_path):
    source = write_table(tmp_path / "source.tsv", {"key": ["a"], "value": [1.0]})
    rerun = write_table(tmp_path / "rerun.tsv", {"key": ["a", "a"], "value": [1.0, 1.0]})
    with pytest.raises(ValueError, match="Duplicate comparison keys"):
        MODULE.compare(source, rerun, ["key"])
