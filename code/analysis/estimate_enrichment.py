# -*- coding: utf-8 -*-
"""Primary pooled-count enrichment and animal-level comparisons.

Observed counts and mean shifted counts are summed within animal, region and
marker before forming regional enrichment ratios. Regional ratios receive equal
weight within animal. Exact animal-label comparisons and inverted confidence
sets use these retained estimates.

The accompanying calibration compares this estimator with the per-stack
log-ratio alternative under the implemented torus reference. It does not
establish freedom from every spatial-reference or detection assumption."""

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
OUT.mkdir(parents=True, exist_ok=True)

KEY = {"complete": "n_complete", "proximity_contact": "n_proximity_contact", "any_overlap": "n_any"}
MARKERS = ["TH", "ChaT", "gad65", "vgat", "vglut1", "vglut2"]
CONTRASTS = {"LPS": [("Control", "LPS_72h"), ("Control", "LPS_60d")], "MPTP": [("Sham", "MPTP")]}
NREP = 800
SEED = 20260729


def load(dataset, marker, endpoint):
    S = ROOT / "data/derived" / dataset.lower() / "uniform_standard/torus_null/summaries"
    iterations = ROOT / "data/derived" / dataset.lower() / "uniform_standard/torus_null/iterations"
    stack = pd.read_csv(ROOT / "results/uniform_stack_summary.tsv", sep="\t")
    stack = stack[
        (stack.dataset == dataset)
        & (stack.specification == "uniform_standard")
        & (stack.marker.astype(str).str.lower() == marker.lower())
    ]
    rows = []
    for _, r in stack.iterrows():
        fs, fi = S / f"{r.file_id}.json", iterations / f"{r.file_id}.npz"
        require_file(fs)
        require_file(fi)
        ep = json.loads(fs.read_text())[endpoint]
        rows.append(
            dict(
                animal=r.animal_id,
                cond=r.condition,
                region=r.region,
                o=float(ep["observed"]),
                e=float(ep["null_mean"]),
                null=np.load(fi)[KEY[endpoint]].astype(float),
            )
        )
    return pd.DataFrame(rows)


def estimate(g, o):
    """Region-pooled counts, then the mean of region-level ratios (equal region weight)."""
    t = pd.DataFrame({"r": g.region.values, "o": o, "e": g.e.values}).groupby("r").sum()
    return float((t.o / t.e).mean())


def per_animal(d, use_null_draw=None, rng=None):
    rows = []
    for (a, c), g in d.groupby(["animal", "cond"]):
        if use_null_draw:
            o = np.array([nl[rng.integers(0, len(nl))] for nl in g.null.values], float)
        else:
            o = g.o.values
        rows.append(dict(animal=a, cond=c, ratio=estimate(g, o)))
    return pd.DataFrame(rows)


def group_summary(ratios):
    """Arithmetic mean of the per-animal ratios.

    The per-animal statistic is a mean of region-level ratios with fixed denominators, so it
    is linear in the observed counts and therefore exactly unbiased for the equal-region-weight
    enrichment at any event rate. Averaging those ratios arithmetically preserves that; taking
    a geometric mean across animals would reintroduce a Jensen bias of a few per cent in the
    sparsest marker and would then need a calibration constant to remove it.
    """
    return float(np.mean(np.asarray(ratios, dtype=float)))


def calibrate(d, nrep=NREP, summary=group_summary):
    """What the estimator returns when the truth is exactly 1.

    Each stack's observed count is replaced by a draw from that stack's own 1,000 torus shifts,
    which makes the true enrichment 1 by construction. Reported as a diagnostic: it is what
    shows the per-stack log-ratio statistic to be biased and this one not to be.
    """
    rng = np.random.default_rng(SEED)
    vals = [summary(per_animal(d, True, rng).ratio) for _ in range(nrep)]
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(nrep))


def exact_test(v, g, ref, trt):
    from confidence_grid import checked_log_bounds

    v = np.asarray(v, float)
    g = np.asarray(g)
    k = np.isin(g, [ref, trt]) & np.isfinite(v)
    v, g = v[k], g[k]
    nt = int((g == trt).sum())
    n = len(v)
    obs = v[g == trt].mean() - v[g == ref].mean()
    stats = []
    for idx in itertools.combinations(range(n), nt):
        m = np.zeros(n, bool)
        m[list(idx)] = True
        stats.append(v[m].mean() - v[~m].mean())
    stats = np.asarray(stats)
    p = float(np.mean(np.abs(stats) >= abs(obs) - 1e-12))
    # invert the same test for a confidence set on the additive log effect
    lo, hi = None, None
    grid = np.linspace(obs - 3.0, obs + 3.0, 1201)
    keep = []
    for delta in grid:
        vv = v.copy()
        vv[g == trt] -= delta
        o2 = vv[g == trt].mean() - vv[g == ref].mean()
        ss = []
        for idx in itertools.combinations(range(n), nt):
            m = np.zeros(n, bool)
            m[list(idx)] = True
            ss.append(vv[m].mean() - vv[~m].mean())
        ss = np.asarray(ss)
        if np.mean(np.abs(ss) >= abs(o2) - 1e-12) > 0.05:
            keep.append(delta)
    lo, hi = checked_log_bounds(grid, keep)
    return float(np.exp(obs)), p, lo, hi


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    r = p[o] * m / np.arange(1, m + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    q = np.empty(m)
    q[o] = np.clip(r, 0, 1)
    return q


def main():
    baseline, effects = [], []
    for dataset in ("LPS", "MPTP"):
        markers = MARKERS if dataset == "LPS" else ["TH", "vglut1"]
        ref = "Control" if dataset == "LPS" else "Sham"
        for endpoint in ("complete", "proximity_contact"):
            for m in markers:
                d = load(dataset, m, endpoint)
                if d.empty:
                    continue
                cal, cal_se = calibrate(d)
                pa = per_animal(d)
                pa["x"] = np.log(pa.ratio)
                b = pa[pa["cond"] == ref]
                baseline.append(
                    dict(
                        dataset=dataset,
                        endpoint=endpoint,
                        marker=m,
                        n_animals=len(b),
                        enrichment=group_summary(b.ratio),
                        null_expectation=cal,
                        null_se=cal_se,
                        geometric_mean=float(np.exp(b["x"].mean())),
                        pooled=float(d[d.cond == ref].o.sum() / d[d.cond == ref].e.sum()),
                    )
                )
                for r0, t0 in CONTRASTS[dataset]:
                    if t0 not in set(pa["cond"]):
                        continue
                    ratio, p, lo, hi = exact_test(pa["x"].values, pa["cond"].values, r0, t0)
                    effects.append(
                        dict(
                            dataset=dataset,
                            endpoint=endpoint,
                            marker=m,
                            reference=r0,
                            treatment=t0,
                            n_ref=int((pa["cond"] == r0).sum()),
                            n_trt=int((pa["cond"] == t0).sum()),
                            ratio=ratio,
                            ci_low=lo,
                            ci_high=hi,
                            exact_p=p,
                        )
                    )
                print(
                    f"{dataset:5s} {endpoint:18s} {m:7s} null={cal:.4f} "
                    f"base={group_summary(b.ratio):.3f}",
                    flush=True,
                )

    B = pd.DataFrame(baseline)
    E = pd.DataFrame(effects)
    for (ds, ep), idx in E.groupby(["dataset", "endpoint"]).groups.items():
        E.loc[idx, "bh_q"] = bh(E.loc[idx, "exact_p"].values)
    B.to_csv(OUT / "unbiased_baseline.tsv", sep="\t", index=False)
    E.to_csv(OUT / "unbiased_treatment_effects.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print("\n=== BASELINE (untreated / sham) ===")
    print(B.to_string(index=False))
    print("\n=== TREATMENT EFFECTS ===")
    print(E.to_string(index=False))


if __name__ == "__main__":
    main()
