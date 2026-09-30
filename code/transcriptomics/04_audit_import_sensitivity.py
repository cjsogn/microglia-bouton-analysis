"""Read original Kallisto metadata and prepare targeted MPTP refits."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shlex
import subprocess
import os
import argparse
import sys

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from mg_paths import require_writable_output

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--counts-dir", type=Path, required=True)
p.add_argument("--raw-dir", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
DATA = args.counts_dir
OUT = require_writable_output(args.out)
if OUT.exists():
    raise FileExistsError(f"Use a fresh output directory: {OUT}")
OUT.mkdir(parents=True)
manifest = pd.read_csv(DATA / "contrast_manifest.tsv", sep="\t")
jobs = []
audits = []
for _, row in manifest[manifest.study.eq("GSE244652")].iterrows():
    folder = DATA / "contrasts" / row.contrast_id
    samples = pd.read_csv(folder / "samples.tsv", sep="\t")
    for _, s in samples.iterrows():
        path = args.raw_dir / Path(s.input_path).name
        with h5py.File(path, "r") as f:
            call = f["aux/call"][0].decode()
            n = int(f["aux/num_processed"][0])
            c = np.asarray(f["est_counts"])
            audits.append(
                dict(
                    contrast_id=row.contrast_id,
                    sample_id=s.sample_id,
                    group=s.group,
                    command=call,
                    single_end="--single" in shlex.split(call),
                    kallisto_version=f["aux/kallisto_version"][0].decode(),
                    bootstrap_replicates=int(f["aux/num_bootstrap"][0]),
                    reads_processed=n,
                    estimated_pseudoaligned_fragments=float(c.sum()),
                    pseudoalignment_fraction=float(c.sum() / n),
                    detected_transcripts=int((c > 0).sum()),
                )
            )
    # Official tximport no-length counts already exist from the joint import.
    for mode, filename in [
        ("no_length", "counts.tsv.gz"),
        ("length_scaled", "length_scaled_counts_sensitivity.tsv.gz"),
    ]:
        counts = pd.read_csv(folder / filename, sep="\t")
        for omitted in [None] + samples.sample_id.tolist():
            suffix = "full" if omitted is None else "omit_" + omitted.split("_")[0]
            cid = row.contrast_id + "__" + mode + "__" + suffix
            target = OUT / "contrasts" / cid
            target.mkdir(parents=True, exist_ok=True)
            ss = samples if omitted is None else samples[samples.sample_id.ne(omitted)]
            ss.to_csv(target / "samples.tsv", sep="\t", index=False)
            counts[["gene_id"] + ss.sample_id.tolist()].to_csv(
                target / "counts.tsv.gz", sep="\t", index=False
            )
            d = row.to_dict()
            d.update(
                contrast_id=cid,
                original_contrast=row.contrast_id,
                import_mode=mode,
                omitted_sample=omitted or "",
                n_control=int(ss.group.eq("Control").sum()),
                n_treatment=int(ss.group.eq("Treatment").sum()),
                counts_path=str(target / "counts.tsv.gz"),
                samples_path=str(target / "samples.tsv"),
            )
            pd.DataFrame([d]).to_csv(target / "manifest.tsv", sep="\t", index=False)
            jobs.append(d)
pd.DataFrame(audits).to_csv(OUT / "original_kallisto_qc.tsv", sep="\t", index=False)
pd.DataFrame(jobs).to_csv(OUT / "sensitivity_manifest.tsv", sep="\t", index=False)
script = ROOT / "code/transcriptomics/02_limma_voom.R"


def fit(job):
    folder = OUT / "fits" / job["contrast_id"]
    folder.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    with (folder / "execution.log").open("w") as log:
        subprocess.run(
            [
                "Rscript",
                str(script),
                str(OUT / "contrasts" / job["contrast_id"] / "manifest.tsv"),
                str(folder),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env=env,
        )
    return job["contrast_id"]


with ThreadPoolExecutor(max_workers=14) as pool:
    for cid in pool.map(fit, jobs):
        print("Fit", cid, flush=True)
tables = []
genes = [
    "C3",
    "C3ar1",
    "C1qa",
    "Tnf",
    "Il1b",
    "P2ry12",
    "Tmem119",
    "Cx3cr1",
    "Trem2",
    "Axl",
    "Mertk",
]
for job in jobs:
    path = OUT / "fits" / job["contrast_id"] / (job["contrast_id"] + "_limma_voom.tsv.gz")
    d = pd.read_csv(path, sep="\t")
    d = d[d.gene_id.isin(genes)].copy()
    d["original_contrast"] = job["original_contrast"]
    d["import_mode"] = job["import_mode"]
    d["omitted_sample"] = job["omitted_sample"]
    tables.append(d)
pd.concat(tables).to_csv(OUT / "panel_sensitivity.tsv", sep="\t", index=False)
print("All import and leave-one-sample-out fits complete", flush=True)
