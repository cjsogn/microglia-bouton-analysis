"""Generate cell-morphology masks from explicitly selected image inputs.

Use a TSV or CSV manifest with field_id and path columns. Preprocess/all take
Iba1 acquisition files, segment takes restored volumes, and split takes initial
integer labels. Outputs always go to a new directory. This command does not
replace the retained eligible-cell measurements used by downstream statistics.
"""

from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import re
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mg_paths import require_writable_output
from morphology_processing import parameters
from morphology_processing.preprocess import generate_theoretical_psf, load_lsm, preprocess_stack
from morphology_processing.segment import segment_stack
from morphology_processing.split_cells import split_labels


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_manifest(path):
    """Validate unique identifiers and resolve every input before creating outputs."""
    path = path.expanduser().resolve()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ",")
        if not {"field_id", "path"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires field_id and path columns")
        rows = list(reader)
    if not rows:
        raise ValueError("The input manifest is empty")
    fields, inputs, records = set(), set(), []
    for row in rows:
        field = row["field_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", field):
            raise ValueError(f"Invalid field identifier: {field!r}")
        source = Path(row["path"]).expanduser()
        source = (source if source.is_absolute() else path.parent / source).resolve()
        if field in fields or source in inputs:
            raise ValueError(f"Duplicate field or input path: {field}")
        if not source.is_file():
            raise FileNotFoundError(source)
        fields.add(field)
        inputs.add(source)
        records.append(dict(field_id=field, path=str(source)))
    return records


def save_volume(path, values):
    tifffile.imwrite(
        path, values, compression="zlib", photometric="minisblack", metadata={"axes": "ZYX"}
    )


def process_one(job):
    row, stage, output = job
    output = Path(output)
    field = row["field_id"]
    source = Path(row["path"])
    record = dict(field_id=field, input_path=str(source), input_sha256=digest(source))
    if stage in {"preprocess", "all"}:
        values = preprocess_stack(load_lsm(source), generate_theoretical_psf()).astype(np.float32)
        path = output / "deconvolved_stacks" / f"deconv_{field}.tif"
        save_volume(path, values)
        record["deconvolved_sha256"] = digest(path)
    else:
        values = tifffile.imread(source)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(f"Invalid three-dimensional input: {field}")
    if stage in {"segment", "all"}:
        values = segment_stack(values)
        path = output / "initial_masks" / f"masks3d_{field}.tif"
        save_volume(path, values)
        record["initial_mask_sha256"] = digest(path)
    if stage in {"split", "all"}:
        if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
            raise ValueError(f"Splitting requires nonnegative integer cell labels: {field}")
        values, count, components = split_labels(values)
        path = output / "final_masks" / f"masks3d_{field}.tif"
        save_volume(path, values)
        record.update(
            final_mask_sha256=digest(path),
            split_input_labels=count,
            split_output_labels=components,
        )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stage", choices=("preprocess", "segment", "split", "all"), default="all")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("MG_WORKERS", "14")))
    parser.add_argument(
        "--max-active-stacks",
        type=int,
        default=1,
        help="Memory bound on concurrent volumes. Increase only after measuring a representative run.",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.max_active_stacks < 1:
        parser.error("Worker and active-stack limits must be positive")
    rows = read_manifest(args.manifest)
    out = require_writable_output(args.out)
    for row in rows:
        source = Path(row["path"])
        if out == source.parent or source.parent in out.parents:
            raise ValueError("Output must be outside the source image directory")
    out.mkdir(parents=True, exist_ok=False)
    stages = {
        "preprocess": ["deconvolved_stacks"],
        "segment": ["initial_masks"],
        "split": ["final_masks"],
        "all": ["deconvolved_stacks", "initial_masks", "final_masks"],
    }
    for directory in stages[args.stage]:
        (out / directory).mkdir()
    active = min(args.workers, args.max_active_stacks, len(rows))
    print(f"Processing {len(rows)} fields with {active} concurrent stacks", flush=True)
    jobs = [(row, args.stage, str(out)) for row in rows]
    with ProcessPoolExecutor(max_workers=active) as pool:
        records = []
        for record in pool.map(process_one, jobs, chunksize=1):
            records.append(record)
            print(record["field_id"], flush=True)
    settings = {name: value for name, value in vars(parameters).items() if name.isupper()}
    provenance = dict(
        stage=args.stage,
        manifest_sha256=digest(args.manifest),
        parameters=settings,
        concurrent_stacks=active,
        packages={name: version(name) for name in ("numpy", "scipy", "scikit-image", "tifffile")},
        source_sha256={
            path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        inputs_and_outputs=records,
    )
    (out / "processing_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
