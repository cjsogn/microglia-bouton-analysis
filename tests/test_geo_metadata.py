"""Primary-record parsing does not conflate sample records or characteristics."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("geo_audit", ROOT / "scripts/verify_geo_metadata.py")
AUDIT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AUDIT)


def test_soft_sample_boundaries(tmp_path):
    path = tmp_path / "records.soft"
    path.write_text(
        "^SERIES = GSE_fixture\n^SAMPLE = GSM_fixture_a\n"
        "!Sample_title = donor_a\n!Sample_characteristics_ch1 = treatment: PBS\n"
        "^SAMPLE = GSM_fixture_b\n!Sample_title = donor_b\n"
        "!Sample_characteristics_ch1 = treatment: LPS\n"
        "^PLATFORM = GPL_fixture\n!Platform_title = array\n"
    )
    samples = AUDIT.read_soft(path)
    assert list(samples) == ["GSM_fixture_a", "GSM_fixture_b"]
    assert AUDIT.characteristics(samples["GSM_fixture_a"])["treatment"] == "PBS"
    assert AUDIT.characteristics(samples["GSM_fixture_b"])["treatment"] == "LPS"
    assert "!Platform_title" not in samples["GSM_fixture_b"]


def test_duplicate_characteristics_fail():
    with pytest.raises(ValueError, match="Repeated characteristic"):
        AUDIT.characteristics({"!Sample_characteristics_ch1": ["sex: F", "Sex: M"]})


def test_empty_record_fails(tmp_path):
    path = tmp_path / "empty.soft"
    path.write_text("^SERIES = GSE_fixture\n")
    with pytest.raises(ValueError, match="No sample metadata"):
        AUDIT.read_soft(path)
