"""Cell-mask processing checks using synthetic volumes and input manifests."""

from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from morphology_processing.preprocess import generate_theoretical_psf, preprocess_stack
from morphology_processing.segment import segment_stack, threshold_microglia_3d
from morphology_processing.split_cells import find_dt_seeds, split_cell, split_labels

spec = importlib.util.spec_from_file_location(
    "mask_runner_test", ROOT / "code/morphology_processing/run_masks.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_psf_is_positive_normalized_and_symmetric():
    psf = generate_theoretical_psf()
    assert psf.shape == (15, 31, 31)
    assert np.all(psf > 0)
    np.testing.assert_allclose(psf.sum(), 1.0, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(psf, psf[::-1, ::-1, ::-1])


def test_constant_volume_stays_empty():
    stack = np.ones((5, 12, 13), dtype=np.float32)
    assert not preprocess_stack(stack, generate_theoretical_psf()).any()
    assert not threshold_microglia_3d(stack).any()
    labels = segment_stack(stack)
    assert labels.shape == stack.shape
    assert labels.dtype == np.int32
    assert not labels.any()


def test_empty_mask_has_no_seeds_or_splits():
    labels = np.zeros((5, 12, 13), dtype=np.int32)
    assert find_dt_seeds(labels > 0) == []
    assert split_cell(labels > 0, []) is None
    result, split_count, new_count = split_labels(labels)
    np.testing.assert_array_equal(result, labels)
    assert split_count == new_count == 0


def test_splitting_preserves_foreground_and_assigns_distinct_labels():
    z, y, x = np.mgrid[:28, :110, :200]
    first = ((z - 14) * 0.5203) ** 2 + ((y - 55) * 0.1516) ** 2 + ((x - 60) * 0.1516) ** 2 < 4.8**2
    second = ((z - 14) * 0.5203) ** 2 + ((y - 55) * 0.1516) ** 2 + (
        (x - 135) * 0.1516
    ) ** 2 < 4.8**2
    labels = (first | second).astype(np.int32) * 7
    labels[14, 54:57, 88:108] = 7
    result, split_count, new_count = split_labels(labels)
    np.testing.assert_array_equal(result > 0, labels > 0)
    assert split_count == 1
    assert new_count == 2
    assert set(np.unique(result)) == {0, 8, 9}
    assert set(np.unique(labels)) == {0, 7}


def test_manifest_resolves_relative_inputs_and_rejects_duplicates(tmp_path):
    source = tmp_path / "source.tif"
    source.touch()
    manifest = tmp_path / "inputs.tsv"
    manifest.write_text("field_id\tpath\nfixture\tsource.tif\n")
    assert runner.read_manifest(manifest) == [dict(field_id="fixture", path=str(source))]
    manifest.write_text("field_id\tpath\nfixture\tsource.tif\nfixture2\tsource.tif\n")
    with pytest.raises(ValueError, match="Duplicate"):
        runner.read_manifest(manifest)


def test_manifest_rejects_output_path_components(tmp_path):
    manifest = tmp_path / "inputs.tsv"
    manifest.write_text("field_id\tpath\n../outside\tsource.tif\n")
    with pytest.raises(ValueError, match="Invalid field"):
        runner.read_manifest(manifest)


def test_manifest_requires_both_columns(tmp_path):
    manifest = tmp_path / "inputs.tsv"
    manifest.write_text("field_id\nfixture\n")
    with pytest.raises(ValueError, match="field_id and path"):
        runner.read_manifest(manifest)


def test_split_command_records_outputs_and_refuses_overwrites(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    labels = np.zeros((3, 6, 8), dtype=np.int32)
    labels[1, 2:4, 3:5] = 5
    for field in ("first", "second"):
        tifffile.imwrite(
            inputs / f"{field}.tif", labels, photometric="minisblack", metadata={"axes": "ZYX"}
        )
    manifest = inputs / "fields.tsv"
    manifest.write_text("field_id\tpath\nfirst\tfirst.tif\nsecond\tsecond.tif\n")
    output = tmp_path / "outputs"
    env = os.environ.copy()
    env.update(MG_DATA_DIR=str(inputs), MG_OUTPUT_DIR=str(tmp_path / "analysis_outputs"))
    command = [
        sys.executable,
        str(ROOT / "code/morphology_processing/run_masks.py"),
        "--manifest",
        str(manifest),
        "--out",
        str(output),
        "--stage",
        "split",
        "--workers",
        "2",
        "--max-active-stacks",
        "2",
    ]
    subprocess.run(command, env=env, cwd=tmp_path, check=True, capture_output=True, text=True)
    provenance = json.loads((output / "processing_provenance.json").read_text())
    assert provenance["stage"] == "split"
    assert provenance["concurrent_stacks"] == 2
    assert len(provenance["inputs_and_outputs"]) == 2
    for record in provenance["inputs_and_outputs"]:
        field = record["field_id"]
        result = output / "final_masks" / f"masks3d_{field}.tif"
        np.testing.assert_array_equal(tifffile.imread(result), labels)
        assert record["input_sha256"] == runner.digest(inputs / f"{field}.tif")
        assert record["final_mask_sha256"] == runner.digest(result)
    repeated = subprocess.run(command, env=env, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "FileExistsError" in repeated.stderr
