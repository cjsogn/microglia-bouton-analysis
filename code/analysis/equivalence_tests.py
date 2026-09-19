#!/usr/bin/env python3
"""Descriptive 90% exact-test confidence sets for the retained imaging contrasts.

The legacy filename is kept for compatibility. No equivalence margin, TOST test,
or claim of absence is generated. The estimator and confidence-set grid are
unchanged from the archived analysis. See docs/ANALYSIS_SCOPE.md.
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

import itertools
import json

import numpy as np
import pandas as pd

ROOT = PIPELINE
OUT = output_dir()


def per_animal(dataset, marker, endpoint):
    st = pd.read_csv(ROOT / "results/uniform_stack_summary.tsv", sep="\t")
    st = st[
        (st.dataset == dataset)
        & (st.specification == "uniform_standard")
        & (st.marker.astype(str).str.lower() == marker.lower())
    ]
    S = ROOT / f"data/derived/{dataset.lower()}/uniform_standard/torus_null/summaries"
    rows = []
    for _, r in st.iterrows():
        f = S / f"{r.file_id}.json"
        require_file(f)
        ep = json.loads(f.read_text())[endpoint]
        rows.append(
            dict(
                animal=r.animal_id,
                cond=r.condition,
                region=r.region,
                o=float(ep["observed"]),
                e=float(ep["null_mean"]),
            )
        )
    d = pd.DataFrame(rows)
    d = d.groupby(["animal", "cond", "region"], as_index=False)[["o", "e"]].sum()
    d = d[d.e > 0]
    d["ratio"] = d.o / d.e
    v = d.groupby(["animal", "cond"], as_index=False).ratio.mean()
    v["x"] = np.log(v.ratio)
    return v


def allocation_stats(x, g, ref, trt):
    x = np.asarray(x, float)
    g = np.asarray(g)
    k = np.isin(g, [ref, trt]) & np.isfinite(x)
    x, g = x[k], g[k]
    nt = int((g == trt).sum())
    n = len(x)
    combos = list(itertools.combinations(range(n), nt))
    return x, g, nt, n, combos


def exact_interval(x, g, ref, trt, level):
    """Confidence set by inverting the two-sided exact test at the requested level."""
    x, g, nt, n, combos = allocation_stats(x, g, ref, trt)
    obs = x[g == trt].mean() - x[g == ref].mean()
    keep = []
    for d in np.linspace(obs - 2.0, obs + 2.0, 1601):
        v = x.copy()
        v[g == trt] -= d
        o2 = v[g == trt].mean() - v[g == ref].mean()
        s = np.empty(len(combos))
        for i, idx in enumerate(combos):
            m = np.zeros(n, bool)
            m[list(idx)] = True
            s[i] = v[m].mean() - v[~m].mean()
        if np.mean(np.abs(s) >= abs(o2) - 1e-12) > (1 - level):
            keep.append(d)
    return (min(keep), max(keep)) if keep else (np.nan, np.nan)


def precision_summary(v, ref, trt):
    if not {ref, trt}.issubset(set(v["cond"])):
        raise ValueError("Required comparison group is absent")
    if not np.isfinite(v.x).all():
        raise ValueError("Nonfinite animal log enrichment")
    lo, hi = exact_interval(v.x.values, v["cond"].values, ref, trt, 0.90)
    estimate = v.loc[v["cond"] == trt, "x"].mean() - v.loc[v["cond"] == ref, "x"].mean()
    if not np.isfinite([estimate, lo, hi]).all():
        raise ValueError("Confidence-set inversion did not return finite limits")
    return dict(
        ratio=float(np.exp(estimate)), ci90_low=float(np.exp(lo)), ci90_high=float(np.exp(hi))
    )


def main():
    rows = []

    print("Descriptive precision intervals")
    cases = [
        ("LPS 72 h, TH contact", "LPS", "TH", "proximity_contact", "Control", "LPS_72h"),
        ("LPS 72 h, TH complete inclusion", "LPS", "TH", "complete", "Control", "LPS_72h"),
        ("MPTP, TH contact", "MPTP", "TH", "proximity_contact", "Sham", "MPTP"),
        ("MPTP, TH complete inclusion", "MPTP", "TH", "complete", "Sham", "MPTP"),
        ("MPTP, VGLUT1 contact", "MPTP", "vglut1", "proximity_contact", "Sham", "MPTP"),
    ]
    for label, ds, mk, ep, ref, trt in cases:
        v = per_animal(ds, mk, ep)
        r = precision_summary(v, ref, trt)
        rows.append(dict(claim=label, **r))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "precision_intervals.tsv", sep="\t", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
