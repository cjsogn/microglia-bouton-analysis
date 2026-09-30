"""Refit all deposited RNA contrasts without reclustering single-cell inputs.

Stages only the count matrices, sample metadata and required analysis inputs,
then runs limma-voom and panel synthesis. Existing results are never copied into
the new run. The input bundle must identify the original-count MARS-seq import.
"""

from __future__ import annotations
import os
import json
import hashlib

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "14"
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from mg_paths import ROOT, DERIVED, OUTPUT, require_file, require_writable_output
import pandas as pd

MANIFESTS = (
    ("contrast_manifest.tsv", "contrasts"),
    ("gse263094_cell_definition_sensitivity_manifest.tsv", "sensitivity_counts"),
)
REQUIRED_METADATA = (
    "import_release.json",
    "prespecified_gene_panel.tsv",
    "panel_symbol_to_entrez.tsv",
    "gse263094_pseudobulk_qc.tsv",
)


def prepare_inputs(source: Path, dest: Path) -> Path:
    """Validate and copy the explicit input set into a fresh run directory."""
    source = source.expanduser().resolve()
    dest = require_writable_output(dest)
    if dest == source or source in dest.parents:
        raise ValueError("RNA output must be outside the input directory")
    if dest.exists():
        raise FileExistsError(f"Use a fresh MG_OUTPUT_DIR to avoid mixing RNA runs: {dest}")
    marker = json.loads(require_file(source / "import_release.json").read_text())
    if (
        marker.get("primary_countsFromAbundance") != "no"
        or marker.get("fresh_imports_reconciled") != 4
    ):
        raise ValueError("The input bundle is not the verified MARS-seq primary release")
    copies = {Path(name) for name in REQUIRED_METADATA}
    manifests = {}
    for name, folder in MANIFESTS:
        manifest = pd.read_csv(require_file(source / name), sep="\t")
        if not {"contrast_id", "counts_path", "samples_path"}.issubset(manifest.columns):
            raise ValueError(f"Incomplete RNA manifest: {name}")
        if (
            manifest.empty
            or manifest.contrast_id.isna().any()
            or manifest.contrast_id.duplicated().any()
        ):
            raise ValueError(f"Empty or duplicate RNA contrasts: {name}")
        for index, rec in manifest.iterrows():
            if (
                not isinstance(rec.contrast_id, str)
                or rec.contrast_id in {"", ".", ".."}
                or Path(rec.contrast_id).name != rec.contrast_id
                or "\\" in rec.contrast_id
            ):
                raise ValueError(f"Invalid contrast identifier: {rec.contrast_id}")
            for col, filename in [
                ("counts_path", "counts.tsv.gz"),
                ("samples_path", "samples.tsv"),
            ]:
                relative = Path(folder) / rec.contrast_id / filename
                copies.add(relative)
                manifest.loc[index, col] = str(dest / relative)
        manifests[name] = manifest
    for relative in copies:
        require_file(source / relative)

    # No destination is created until every required input has been found.
    dest.mkdir(parents=True)
    provenance = []
    for relative in sorted(copies):
        original, target = source / relative, dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        with target.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        provenance.append({"file": relative.as_posix(), "sha256": digest})
    for name, manifest in manifests.items():
        manifest.to_csv(dest / name, sep="\t", index=False)
    (dest / "staged_inputs.json").write_text(
        json.dumps({"source": str(source), "files": provenance, "results_copied": False}, indent=2)
        + "\n"
    )
    return dest


def main():
    prepare_inputs(DERIVED / "transcriptomics", OUTPUT / "transcriptomics")
    scripts = ROOT / "code/transcriptomics"
    for stage in ("de", "synthesis"):
        subprocess.run(
            [sys.executable, str(scripts / "run_rebuild.py"), "--stage", stage], check=True
        )


if __name__ == "__main__":
    main()
