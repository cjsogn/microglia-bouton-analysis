"""Software contract checks. Small arrays below are synthetic test fixtures."""

from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/analysis")]


def test_data_and_output_separation(tmp_path):
    env = os.environ.copy()
    env.update(
        MG_DATA_DIR=str(tmp_path / "inputs"), MG_OUTPUT_DIR=str(tmp_path / "inputs/replaced")
    )
    process = subprocess.run(
        [sys.executable, str(ROOT / "code/mg_paths.py")], env=env, capture_output=True, text=True
    )
    assert process.returncode != 0
    assert "outside the input tree" in process.stderr


def test_paths_work_from_other_directory(tmp_path):
    env = os.environ.copy()
    env.update(
        MG_PACKAGE_ROOT=str(ROOT),
        MG_DATA_DIR=str(tmp_path / "inputs"),
        MG_OUTPUT_DIR=str(tmp_path / "outputs"),
    )
    process = subprocess.run(
        [sys.executable, str(ROOT / "code/mg_paths.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert str(tmp_path / "inputs") in process.stdout
    assert str(tmp_path / "outputs") in process.stdout


def test_missing_imaging_file_fails(tmp_path, monkeypatch):
    import estimate_enrichment as model

    monkeypatch.setattr(model, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    pd.DataFrame(
        [
            dict(
                dataset="LPS",
                specification="uniform_standard",
                marker="TH",
                file_id="synthetic_missing",
                animal_id="test",
                condition="Control",
                region="test",
            )
        ]
    ).to_csv(tmp_path / "results/uniform_stack_summary.tsv", sep="\t", index=False)
    with pytest.raises(FileNotFoundError, match="Required input"):
        model.load("LPS", "TH", "complete")


def test_zero_branches_above_threshold_are_retained():
    import morph2_persistent_branches as model

    bars = pd.DataFrame(
        [
            dict(
                fov="fixture",
                label=1,
                animal="fixture",
                cond="Control",
                region="fixture",
                persistence=p,
            )
            for p in [1.0, 2.0]
        ]
    )
    cell, animal = model.per_animal(bars, 3.0)
    assert len(cell) == len(animal) == 1
    assert cell.hit.iloc[0] == animal.hit.iloc[0] == 0


def test_missing_eligible_cell_is_not_zero(tmp_path, monkeypatch):
    import morph2_persistent_branches as model

    monkeypatch.setattr(model, "MORPH", tmp_path)
    monkeypatch.setattr(model, "OUTPUT", tmp_path / "regenerated")
    (tmp_path / "output").mkdir()
    pd.DataFrame(
        [dict(fov="fixture", cell_id=1, animal="fixture", group="kiv", region="fixture")]
    ).to_csv(tmp_path / "output/features_3d_full_morphomics_v6_final.csv", index=False)
    pd.DataFrame(columns=["fov", "label", "persistence"]).to_csv(
        tmp_path / "output/topology_barcodes_v6.csv", index=False
    )
    with pytest.raises(ValueError, match="missing barcode"):
        model.load()


def test_harmony_orientation_is_validated():
    path = ROOT / "code/transcriptomics/01_prepare_counts.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("rna_prepare_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = module.validated_harmony_coordinates
    values = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(function(values, 4, 3), values.T)
    with pytest.raises(RuntimeError, match="invalid Harmony"):
        function(np.zeros((2, 2)), 4, 3)


def test_calibration_assembly_rejects_incomplete_table(tmp_path):
    import assemble_ordering_calibration as model

    path = tmp_path / "incomplete.tsv"
    pd.DataFrame([dict(endpoint="synthetic", test="synthetic", value=0)]).to_csv(
        path, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="Incomplete"):
        model.checked_table(path, model.COLS, model.KEYS, 8)


def test_calibration_assembly_requires_explicit_mode(tmp_path):
    env = os.environ.copy()
    env.update(MG_DATA_DIR=str(tmp_path / "inputs"), MG_OUTPUT_DIR=str(tmp_path / "outputs"))
    result = subprocess.run(
        [sys.executable, str(ROOT / "code/analysis/assemble_ordering_calibration.py")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--pairwise-mode" in result.stderr
    assert not (tmp_path / "outputs/baseline_ordering_typeI_calibration.tsv").exists()


def test_confidence_grid_does_not_report_truncated_or_disconnected_interval():
    from confidence_grid import checked_log_bounds

    grid = np.arange(-2, 2.1, 1)
    np.testing.assert_allclose(checked_log_bounds(grid, [-1, 0, 1]), np.exp([-1, 1]))
    for accepted, message in [
        ([-2, -1, 0], "boundary"),
        ([-1, 1], "Disconnected"),
        ([], "no grid"),
    ]:
        with pytest.raises(ValueError, match=message):
            checked_log_bounds(grid, accepted)


def test_precision_intervals_reject_invalid_levels():
    from precision_intervals import exact_interval

    for level in (0, 1):
        with pytest.raises(ValueError, match="Confidence level"):
            exact_interval([], [], "Control", "Treatment", level)
