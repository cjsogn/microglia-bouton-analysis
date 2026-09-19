# -*- coding: utf-8 -*-
"""Direct LPS timepoint comparisons using the primary animal-level estimator.

The same exact animal-label test and confidence-set inversion compare the
two independent time groups. This directly tests a temporal contrast rather
than comparing their separate treatment-versus-control significance labels.
The module also records the allocation-grid resolution for its fixed statistic."""

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
MARKERS = ["TH", "ChaT", "vglut1", "vglut2", "vgat", "gad65"]


def per_animal(marker, endpoint):
    stack = pd.read_csv(ROOT / "results/uniform_stack_summary.tsv", sep="\t")
    stack = stack[
        (stack.dataset == "LPS")
        & (stack.specification == "uniform_standard")
        & (stack.marker.astype(str).str.lower() == marker.lower())
    ]
    S = ROOT / "data/derived/lps/uniform_standard/torus_null/summaries"
    rows = []
    for _, r in stack.iterrows():
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


def exact(x, g, ref, trt):
    """Exact allocation test, with the confidence set obtained by inverting the same test."""
    from confidence_grid import checked_log_bounds

    x = np.asarray(x, float)
    g = np.asarray(g)
    k = np.isin(g, [ref, trt]) & np.isfinite(x)
    x, g = x[k], g[k]
    nt = int((g == trt).sum())
    n = len(x)
    obs = x[g == trt].mean() - x[g == ref].mean()
    combos = list(itertools.combinations(range(n), nt))

    def stats(v):
        out = np.empty(len(combos))
        for i, idx in enumerate(combos):
            m = np.zeros(n, bool)
            m[list(idx)] = True
            out[i] = v[m].mean() - v[~m].mean()
        return out

    s = stats(x)
    hits = int((np.abs(s) >= abs(obs) - 1e-12).sum())
    p = hits / len(s)
    keep = []
    grid = np.linspace(obs - 3.0, obs + 3.0, 1201)
    for delta in grid:
        v = x.copy()
        v[g == trt] -= delta
        o2 = v[g == trt].mean() - v[g == ref].mean()
        s2 = stats(v)
        if np.mean(np.abs(s2) >= abs(o2) - 1e-12) > 0.05:
            keep.append(delta)
    lo, hi = checked_log_bounds(grid, keep)
    return float(np.exp(obs)), p, lo, hi, hits, len(s)


def floor_for(n_ref, n_trt):
    """Smallest attainable two-sided P: enumerate a perfectly separated dataset."""
    x = np.log(np.concatenate([np.full(n_ref, 1.0), np.full(n_trt, 10.0)]))
    g = np.array(["A"] * n_ref + ["B"] * n_trt)
    _, p, _, _, hits, tot = exact(x, g, "A", "B")
    return p, hits, tot


def main():
    rows = []
    for endpoint in ("proximity_contact", "complete", "any_overlap"):
        for m in MARKERS:
            v = per_animal(m, endpoint)
            if not {"LPS_72h", "LPS_60d"} <= set(v["cond"]):
                continue
            r, p, lo, hi, hits, tot = exact(v.x.values, v["cond"].values, "LPS_72h", "LPS_60d")
            rows.append(
                dict(
                    endpoint=endpoint,
                    marker=m,
                    comparison="LPS_60d vs LPS_72h",
                    ratio=r,
                    ci_low=lo,
                    ci_high=hi,
                    exact_p=p,
                    n_extreme=hits,
                    n_allocations=tot,
                )
            )
            print(
                f"{endpoint:18s} {m:7s} x{r:.4f} ({lo:.3f}-{hi:.3f})  P = {hits}/{tot} = {p:.5f}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    for ep, idx in df.groupby("endpoint").groups.items():
        sub = df.loc[idx]
        o = np.argsort(sub.exact_p.values)
        q = sub.exact_p.values[o] * len(sub) / np.arange(1, len(sub) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        out = np.empty(len(sub))
        out[o] = np.clip(q, 0, 1)
        df.loc[idx, "bh_q_within_endpoint"] = out
    df.to_csv(OUT / "timepoint_contrast.tsv", sep="\t", index=False)

    fl = [
        dict(n_reference=a, n_treatment=b, n_allocations=t, n_most_extreme=h, attainable_p=p)
        for a, b in [(5, 5), (5, 4), (4, 4)]
        for p, h, t in [floor_for(a, b)]
    ]
    F = pd.DataFrame(fl)
    F.to_csv(OUT / "exact_test_floors.tsv", sep="\t", index=False)
    print("\n=== SMALLEST ATTAINABLE TWO-SIDED P ===")
    print(F.to_string(index=False))
    print(f"\nwritten {OUT / 'timepoint_contrast.tsv'} and exact_test_floors.tsv")


if __name__ == "__main__":
    main()
