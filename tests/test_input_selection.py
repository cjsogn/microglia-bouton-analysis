"""Input staging and topology selection checks using synthetic file fixtures."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/analysis")]
spec = importlib.util.spec_from_file_location(
    "rna_runner", ROOT / "scripts/reproduce_rna_from_counts.py"
)
RNA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RNA)


@pytest.fixture
def rna_source(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    for name in RNA.REQUIRED_METADATA:
        (source / name).write_text("synthetic fixture\n")
    (source / "import_release.json").write_text(
        json.dumps({"primary_countsFromAbundance": "no", "fresh_imports_reconciled": 4})
    )
    for name, folder in RNA.MANIFESTS:
        cid = folder + "_fixture"
        target = source / folder / cid
        target.mkdir(parents=True)
        for filename in ("counts.tsv.gz", "samples.tsv"):
            (target / filename).write_bytes(b"synthetic input bytes\n")
        pd.DataFrame(
            [
                {
                    "contrast_id": cid,
                    "counts_path": "not-used/counts.tsv.gz",
                    "samples_path": "not-used/samples.tsv",
                    "analysis_role": "fixture",
                }
            ]
        ).to_csv(source / name, sep="\t", index=False)
    return source


def test_rna_staging_excludes_all_previous_results(rna_source, tmp_path):
    (rna_source / "limma_voom").mkdir()
    (rna_source / "limma_voom/old_result.tsv").write_text("must not be copied")
    (rna_source / "legacy_import_limma_voom").mkdir()
    (rna_source / "legacy_import_limma_voom/old_result.tsv").write_text("must not be copied")
    (rna_source / "all_limma_voom_results.parquet").write_text("must not be copied")
    before = {
        p.relative_to(rna_source): p.read_bytes() for p in rna_source.rglob("*") if p.is_file()
    }
    dest = RNA.prepare_inputs(rna_source, tmp_path / "run")
    expected = set(RNA.REQUIRED_METADATA) | {"staged_inputs.json"}
    for name, folder in RNA.MANIFESTS:
        expected.add(name)
        manifest = pd.read_csv(dest / name, sep="\t")
        assert manifest.analysis_role.tolist() == ["fixture"]
        for column, filename in (("counts_path", "counts.tsv.gz"), ("samples_path", "samples.tsv")):
            relative = Path(folder) / (folder + "_fixture") / filename
            expected.add(relative.as_posix())
            assert manifest[column].iloc[0] == str(dest / relative)
            assert (dest / relative).read_bytes() == before[relative]
    assert {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()} == expected
    assert before == {
        p.relative_to(rna_source): p.read_bytes() for p in rna_source.rglob("*") if p.is_file()
    }
    record = json.loads((dest / "staged_inputs.json").read_text())
    assert record["results_copied"] is False
    for item in record["files"]:
        assert item["sha256"] == hashlib.sha256((dest / item["file"]).read_bytes()).hexdigest()


def test_rna_staging_rejects_existing_destination(rna_source, tmp_path):
    dest = tmp_path / "run"
    dest.mkdir()
    with pytest.raises(FileExistsError):
        RNA.prepare_inputs(rna_source, dest)


def test_rna_staging_rejects_destination_inside_inputs(rna_source):
    with pytest.raises(ValueError, match="outside"):
        RNA.prepare_inputs(rna_source, rna_source / "run")


def test_rna_staging_checks_missing_inputs_before_creating_output(rna_source, tmp_path):
    (rna_source / "panel_symbol_to_entrez.tsv").unlink()
    with pytest.raises(FileNotFoundError):
        RNA.prepare_inputs(rna_source, tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("identifier", ["../escape", "/absolute", ".", "..", "folder\\escape"])
def test_rna_staging_rejects_unsafe_contrast_identifiers(rna_source, tmp_path, identifier):
    path = rna_source / "contrast_manifest.tsv"
    data = pd.read_csv(path, sep="\t")
    data.loc[0, "contrast_id"] = identifier
    data.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="Invalid contrast"):
        RNA.prepare_inputs(rna_source, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_rna_staging_rejects_duplicate_contrasts(rna_source, tmp_path):
    path = rna_source / "contrast_manifest.tsv"
    data = pd.read_csv(path, sep="\t")
    pd.concat([data, data]).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="duplicate RNA"):
        RNA.prepare_inputs(rna_source, tmp_path / "run")


def test_rna_staging_requires_original_count_import(rna_source, tmp_path):
    path = rna_source / "import_release.json"
    path.write_text(
        json.dumps(
            {"primary_countsFromAbundance": "lengthScaledTPM", "fresh_imports_reconciled": 4}
        )
    )
    with pytest.raises(ValueError, match="MARS-seq"):
        RNA.prepare_inputs(rna_source, tmp_path / "run")


@pytest.fixture
def topology_inputs(tmp_path, monkeypatch):
    import morph3_secondary_features as morphology

    monkeypatch.setattr(morphology, "MORPHOLOGY", tmp_path / "inputs")
    monkeypatch.setattr(morphology, "OUTPUT", tmp_path / "run")
    deposited = tmp_path / "inputs/output/topology_features_v6.csv"
    regenerated = tmp_path / "run/morphology/topology_features_v6.csv"
    for path in (deposited, regenerated):
        path.parent.mkdir(parents=True)
    deposited.write_text("deposited fixture\n")
    return morphology, deposited, regenerated


def test_topology_auto_selects_deposited_when_no_rerun_exists(topology_inputs):
    model, deposited, _ = topology_inputs
    assert model.select_topology_input() == (deposited, "deposited")


def test_topology_auto_prefers_regenerated_measurements(topology_inputs):
    model, _, regenerated = topology_inputs
    regenerated.write_text("regenerated fixture\n")
    assert model.select_topology_input() == (regenerated, "regenerated")


def test_topology_deposited_choice_ignores_regenerated_measurements(topology_inputs):
    model, deposited, regenerated = topology_inputs
    regenerated.write_text("regenerated fixture\n")
    assert model.select_topology_input("deposited") == (deposited, "deposited")


def test_topology_explicit_regenerated_choice_never_falls_back(topology_inputs):
    model, _, _ = topology_inputs
    with pytest.raises(FileNotFoundError):
        model.select_topology_input("regenerated")


def test_topology_explicit_path_never_falls_back(topology_inputs, tmp_path):
    model, deposited, _ = topology_inputs
    assert model.select_topology_input(explicit=deposited) == (deposited, "explicit")
    with pytest.raises(FileNotFoundError):
        model.select_topology_input(explicit=tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="either"):
        model.select_topology_input("regenerated", deposited)


def test_topology_provenance_records_selected_bytes(topology_inputs):
    model, deposited, _ = topology_inputs
    assert model.input_record(deposited) == {
        "path": str(deposited.resolve()),
        "sha256": hashlib.sha256(deposited.read_bytes()).hexdigest(),
    }
