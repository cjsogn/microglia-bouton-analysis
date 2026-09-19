# -*- coding: utf-8 -*-
"""Paired TH-versus-comparator treatment interactions for Figure 3 and Table S6.

Counts are pooled within animal and region before forming enrichment. TH and
each comparator are paired within the same animal and region on the log scale.
Treatment comparisons use animal-label allocation tests and inverted confidence
sets, with the reported within-timepoint and broader multiplicity families."""

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
S = ROOT / "data/derived/lps/uniform_standard/torus_null/summaries"
COMPARATORS = ["ChaT", "gad65", "vgat", "vglut1", "vglut2"]


def load(marker, endpoint):
    stack = pd.read_csv(ROOT / "results/uniform_stack_summary.tsv", sep="\t")
    stack = stack[
        (stack.dataset == "LPS")
        & (stack.specification == "uniform_standard")
        & (stack.marker.astype(str).str.lower() == marker.lower())
    ]
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
    t = pd.DataFrame(rows).groupby(["animal", "cond", "region"]).sum().reset_index()
    # Counts are already pooled within animal and region by the groupby above, so the ratio is
    # formed on pooled counts and needs no continuity correction. This is the estimator used
    # everywhere else in the paper. A per-cell +0.5 shift would reintroduce the low-count bias
    # the pooling exists to avoid, and would hit the sparsest marker hardest.
    return t[t.e > 0]


def exact(v, g, ref, trt, grid_halfwidth=2.0, n_grid=801):
    from confidence_grid import checked_log_bounds

    v = np.asarray(v, float)
    g = np.asarray(g)
    k = np.isin(g, [ref, trt]) & np.isfinite(v)
    v, g = v[k], g[k]
    nt = int((g == trt).sum())
    n = len(v)
    combos = list(itertools.combinations(range(n), nt))

    def pval(vv):
        o = vv[g == trt].mean() - vv[g == ref].mean()
        s = np.array(
            [vv[list(c)].mean() - vv[[i for i in range(n) if i not in c]].mean() for c in combos]
        )
        return float(np.mean(np.abs(s) >= abs(o) - 1e-12)), o

    p, obs = pval(v)
    keep = []
    grid = np.linspace(obs - grid_halfwidth, obs + grid_halfwidth, n_grid)
    for d in grid:
        vv = v.copy()
        vv[g == trt] -= d
        if pval(vv)[0] > 0.05:
            keep.append(d)
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
    rows = []
    for endpoint in ("complete", "proximity_contact"):
        th = load("TH", endpoint)
        for comp in COMPARATORS:
            cm = load(comp, endpoint)
            m = th.merge(cm, on=["animal", "cond", "region"], suffixes=("_th", "_c"))
            # Region-pooled ratio per region for each marker, then the mean over the regions both
            # markers share, then the log difference. This is the estimator used throughout, keeps
            # the region pairing, and needs no continuity correction: a region with no events
            # contributes a ratio of zero to a mean that stays positive, rather than log(0).
            m["r_th"] = m.o_th / m.e_th
            m["r_c"] = m.o_c / m.e_c
            g = m.groupby(["animal", "cond"])[["r_th", "r_c"]].mean().reset_index()
            g = g[(g.r_th > 0) & (g.r_c > 0)]
            g["d"] = np.log(g.r_th) - np.log(g.r_c)
            pa = g[["animal", "cond", "d"]]
            base = float(np.exp(pa[pa["cond"] == "Control"]["d"].mean()))
            for trt in ("LPS_72h", "LPS_60d"):
                if trt not in set(pa["cond"]):
                    continue
                r, p, lo, hi = exact(pa["d"].values, pa["cond"].values, "Control", trt)
                rows.append(
                    dict(
                        endpoint=endpoint,
                        comparator=comp,
                        treatment=trt,
                        baseline_th_over_comparator=base,
                        n_regions_paired=int(m.region.nunique()),
                        interaction=r,
                        ci_low=lo,
                        ci_high=hi,
                        exact_p=p,
                    )
                )
                print(
                    f"{endpoint:18s} vs {comp:7s} {trt:8s} x{r:.3f} ({lo:.2f}-{hi:.2f}) P={p:.3f}",
                    flush=True,
                )
    df = pd.DataFrame(rows)
    for (ep, trt), idx in df.groupby(["endpoint", "treatment"]).groups.items():
        df.loc[idx, "bh_q_within_5"] = bh(df.loc[idx, "exact_p"].values)
    for ep, idx in df.groupby("endpoint").groups.items():
        df.loc[idx, "bh_q_within_10"] = bh(df.loc[idx, "exact_p"].values)
    df.to_csv(OUT / "unbiased_th_interactions.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
