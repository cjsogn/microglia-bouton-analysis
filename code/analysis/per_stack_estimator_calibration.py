# -*- coding: utf-8 -*-
"""What each estimator returns when the true enrichment is exactly 1.

The evaluated estimator is the natural per-stack statistic, the mean over stacks of
log((observed + 0.5) / (null mean + 0.5)), runs low for sparse markers, and quotes a calibrated
value for TH. This script computes it from the study's own torus arrays and writes it to a results
file, so the justification for pooling counts before forming ratios traces to data.

Each stack's observed count is replaced by a draw from that stack's own 1,000 torus shifts, which
makes the true enrichment 1 by construction. The per-stack statistic is then formed by averaging the per-stack log ratio within region, then over regions within
animal, then exponentiating the mean over animals.

The estimator the paper actually uses is calibrated on the same draws, so the two are compared on
identical data rather than on separate simulations: counts are pooled within animal and region, the
region ratios are averaged with equal weight, and the animal means are averaged. Both estimators are
calibrated on all three endpoints, contact, any overlap and complete inclusion.
"""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _P

PKG = _P(
    _os.environ.get("MG_PACKAGE_ROOT")  # override to relocate
    or next(
        p
        for p in _P(__file__).resolve().parents  # else the dir holding
        if (p / "code").is_dir()
    )
)  # code/, at any depth
for _d in (_P(__file__).resolve().parent, PKG / "code", PKG / "code/analysis"):
    _sys.path.insert(0, str(_d))  # siblings, and across analysis subdirectories

from mg_paths import (
    PIPELINE,
    output_dir,
    require_file,
)

import json

import numpy as np
import pandas as pd

ROOT = PIPELINE
OUT = output_dir()
KEY = {"complete": "n_complete", "proximity_contact": "n_proximity_contact", "any_overlap": "n_any"}
MARKERS = ["TH", "ChaT", "vglut1", "vglut2", "vgat", "gad65"]
NREP = 3000
SEED = 20260806


def load(marker, endpoint):
    S = ROOT / "data/derived/lps/uniform_standard/torus_null/summaries"
    iterations = ROOT / "data/derived/lps/uniform_standard/torus_null/iterations"
    st = pd.read_csv(ROOT / "results/uniform_stack_summary.tsv", sep="\t")
    st = st[
        (st.dataset == "LPS")
        & (st.specification == "uniform_standard")
        & (st.condition == "Control")
        & (st.marker.astype(str).str.lower() == marker.lower())
    ]
    rows = []
    for _, r in st.iterrows():
        fs, fi = S / f"{r.file_id}.json", iterations / f"{r.file_id}.npz"
        require_file(fs)
        require_file(fi)
        ep = json.loads(fs.read_text())[endpoint]
        rows.append(
            dict(
                animal=r.animal_id,
                region=r.region,
                e=float(ep["null_mean"]),
                null=np.load(fi)[KEY[endpoint]].astype(float),
            )
        )
    return pd.DataFrame(rows)


def per_stack_statistic(d, obs):
    """Per-stack log ratio, averaged within region, then over regions, then over animals."""
    t = pd.DataFrame(
        {
            "animal": d.animal.values,
            "region": d.region.values,
            "lr": np.log((obs + 0.5) / (d.e.values + 0.5)),
        }
    )
    per_region = t.groupby(["animal", "region"]).lr.mean().reset_index()
    per_animal = per_region.groupby("animal").lr.mean()
    return float(np.exp(per_animal.mean()))


def pooled_statistic(d, obs):
    """The reported estimator: pool counts within animal and region, then average the ratios."""
    t = pd.DataFrame(
        {"animal": d.animal.values, "region": d.region.values, "o": obs, "e": d.e.values}
    )
    g = t.groupby(["animal", "region"], as_index=False)[["o", "e"]].sum()
    g = g[g.e > 0]
    g["ratio"] = g.o / g.e
    return float(g.groupby("animal").ratio.mean().mean())


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for endpoint in ("complete", "proximity_contact", "any_overlap"):
        for m in MARKERS:
            d = load(m, endpoint)
            if d.empty:
                continue
            vals, pooled = [], []
            for _ in range(NREP):
                obs = np.array([nl[rng.integers(0, len(nl))] for nl in d.null.values], float)
                vals.append(per_stack_statistic(d, obs))
                pooled.append(pooled_statistic(d, obs))
            vals, pooled = np.asarray(vals), np.asarray(pooled)
            rows.append(
                dict(
                    endpoint=endpoint,
                    marker=m,
                    n_stacks=len(d),
                    n_rep=NREP,
                    per_stack_null_expectation=float(vals.mean()),
                    mc_se=float(vals.std(ddof=1) / np.sqrt(NREP)),
                    pooled_null_expectation=float(pooled.mean()),
                    pooled_mc_se=float(pooled.std(ddof=1) / np.sqrt(NREP)),
                    median_events_per_stack=float(np.median([nl.mean() for nl in d.null.values])),
                )
            )
            print(
                f"{endpoint:18s} {m:7s} per-stack {vals.mean():.4f} "
                f"(SE {vals.std(ddof=1) / np.sqrt(NREP):.4f})   pooled {pooled.mean():.4f} "
                f"(SE {pooled.std(ddof=1) / np.sqrt(NREP):.4f})   truth 1",
                flush=True,
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_stack_estimator_calibration.tsv", sep="\t", index=False)
    print(f"\nwritten {OUT / 'per_stack_estimator_calibration.tsv'}")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
