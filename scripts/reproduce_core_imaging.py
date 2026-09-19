"""Reproduce primary imaging estimates with independent jobs across 14 processes.

Uses the unchanged estimator, calibration seed, exact allocation test and
confidence-set grid in estimate_enrichment.py. Parallelism changes scheduling,
not random streams, endpoints or multiplicity families.
"""

from __future__ import annotations
import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "code"),
    str(Path(__file__).resolve().parents[1] / "code/analysis"),
]
import numpy as np
import pandas as pd
import estimate_enrichment as e


def one_job(job):
    dataset, endpoint, marker = job
    data = e.load(dataset, marker, endpoint)
    if data.empty:
        raise ValueError(f"No input stacks for {job}")
    reference = "Control" if dataset == "LPS" else "Sham"
    cal, cal_se = e.calibrate(data)
    animals = e.per_animal(data)
    if not np.isfinite(animals.ratio).all() or (animals.ratio <= 0).any():
        raise ValueError(f"Nonpositive or nonfinite animal ratio for {job}")
    animals["x"] = np.log(animals.ratio)
    baseline = animals[animals.cond == reference]
    row = dict(
        dataset=dataset,
        endpoint=endpoint,
        marker=marker,
        n_animals=len(baseline),
        enrichment=e.group_summary(baseline.ratio),
        null_expectation=cal,
        null_se=cal_se,
        geometric_mean=float(np.exp(baseline.x.mean())),
        pooled=float(data[data.cond == reference].o.sum() / data[data.cond == reference].e.sum()),
    )
    effects = []
    for ref, trt in e.CONTRASTS[dataset]:
        if not {ref, trt}.issubset(set(animals.cond)):
            raise ValueError(f"Missing expected group: {job}, {ref}, {trt}")
        ratio, p, low, high = e.exact_test(animals.x.values, animals.cond.values, ref, trt)
        effects.append(
            dict(
                dataset=dataset,
                endpoint=endpoint,
                marker=marker,
                reference=ref,
                treatment=trt,
                n_ref=int((animals.cond == ref).sum()),
                n_trt=int((animals.cond == trt).sum()),
                ratio=ratio,
                ci_low=low,
                ci_high=high,
                exact_p=p,
            )
        )
    animals = animals.rename(columns={"cond": "condition", "ratio": "enrichment"}).drop(columns="x")
    animals["dataset"], animals["endpoint"], animals["marker"] = job
    return row, effects, animals


def main():
    jobs = [
        (dataset, endpoint, marker)
        for dataset in ("LPS", "MPTP")
        for endpoint in ("complete", "proximity_contact")
        for marker in (e.MARKERS if dataset == "LPS" else ["TH", "vglut1"])
    ]
    baseline, effects, animals = [], [], []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("MG_WORKERS", "14"))) as pool:
        for job, (row, contrasts, pa) in zip(jobs, pool.map(one_job, jobs)):
            baseline.append(row)
            effects.extend(contrasts)
            animals.append(pa)
            print("Reproduced:", job, flush=True)
    effects = pd.DataFrame(effects)
    for _, idx in effects.groupby(["dataset", "endpoint"]).groups.items():
        effects.loc[idx, "bh_q"] = e.bh(effects.loc[idx, "exact_p"].values)
    pd.DataFrame(baseline).to_csv(e.OUT / "unbiased_baseline.tsv", sep="\t", index=False)
    effects.to_csv(e.OUT / "unbiased_treatment_effects.tsv", sep="\t", index=False)
    pd.concat(animals, ignore_index=True).to_csv(
        e.OUT / "unbiased_per_animal.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    main()
