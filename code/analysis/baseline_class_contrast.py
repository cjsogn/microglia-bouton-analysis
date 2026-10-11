# -*- coding: utf-8 -*-
"""Exploratory within-animal transmitter-class contrast.

For each animal, compare mean TH/ChAT enrichment with mean enrichment across
the other retained transmitter markers on the log-ratio scale. One-sample tests
and exhaustive animal-bootstrap intervals describe this data-selected contrast.
The module also exports the reported coverage and regional checks.

This calculation preserves within-animal pairing. It does not adjust for the
post hoc choice of transmitter grouping or remove marker-specific measurement
variation."""

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
from scipy import stats

ROOT = PIPELINE
V3 = output_dir()

MOD = ["TH", "ChaT"]  # modulatory
FAST = ["vglut1", "vglut2", "vgat", "gad65"]  # fast amino-acid
MARKERS = MOD + FAST
KEY = {"complete": "n_complete", "proximity_contact": "n_proximity_contact", "any_overlap": "n_any"}


def region_table(dataset, condition, endpoint):
    """One row per animal, region and marker: pooled observed and expected counts.

    Counts are pooled across the stacks of a region before any ratio is taken, which is the
    estimator used throughout the paper and is linear in the counts.
    """
    stack = pd.read_csv(ROOT / "results/uniform_stack_summary.tsv", sep="\t")
    stack = stack[
        (stack.dataset == dataset)
        & (stack.specification == "uniform_standard")
        & (stack.condition == condition)
    ]
    S = ROOT / "data/derived" / dataset.lower() / "uniform_standard/torus_null/summaries"
    rows = []
    for _, r in stack.iterrows():
        m = str(r.marker)
        key = {k.lower(): k for k in MARKERS}.get(m.lower())
        f = S / f"{r.file_id}.json"
        if key is None:
            continue
        require_file(f)
        ep = json.loads(f.read_text())[endpoint]
        rows.append(
            dict(
                animal=r.animal_id,
                region=r.region,
                marker=key,
                o=float(ep["observed"]),
                e=float(ep["null_mean"]),
            )
        )
    d = pd.DataFrame(rows)
    return d.groupby(["animal", "region", "marker"], as_index=False)[["o", "e"]].sum()


def per_animal_enrichment(rt):
    """Region-pooled ratio per region, then the mean over regions. Equal weight per region."""
    rt = rt[rt.e > 0].copy()
    rt["ratio"] = rt.o / rt.e
    return rt.groupby(["animal", "marker"], as_index=False).ratio.mean()


def contrast(pa, mod=MOD, fast=FAST):
    """One log contrast per animal: mean over modulatory markers against mean over the others."""
    w = pa.pivot_table(index="animal", columns="marker", values="ratio")
    a = w[[m for m in mod if m in w]].mean(axis=1)
    b = w[[m for m in fast if m in w]].mean(axis=1)
    return np.log(a / b).dropna()


def enumerate_bootstrap(x):
    """All n^n cluster resamples. At n = 5 this is 3,125, so the bootstrap is exact, not sampled."""
    x = np.asarray(x, float)
    n = len(x)
    means = np.array([np.mean(x[list(idx)]) for idx in itertools.product(range(n), repeat=n)])
    return means


def sign_p(k, n):
    from math import comb

    return min(
        1.0, sum(comb(n, j) for j in range(n + 1) if abs(j - n / 2) >= abs(k - n / 2)) / 2**n
    )


def summarise(x, label):
    x = np.asarray(x, float)
    n = len(x)
    t = stats.ttest_1samp(x, 0.0)
    bm = enumerate_bootstrap(x)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return dict(
        comparison=label,
        n_animals=n,
        ratio=float(np.exp(x.mean())),
        boot_lo=float(np.exp(lo)),
        boot_hi=float(np.exp(hi)),
        t_stat=float(t.statistic),
        t_p=float(t.pvalue),
        n_positive=int((x > 0).sum()),
        sign_p=sign_p(int((x > 0).sum()), n),
        boot_frac_le_zero=float((bm <= 0).mean()),
    )


def main():
    out, per_marker, per_region = [], [], []
    for endpoint in ("proximity_contact", "complete", "any_overlap"):
        rt = region_table("LPS", "Control", endpoint)
        pa = per_animal_enrichment(rt)
        x = contrast(pa)
        r = summarise(x.values, "TH+ChAT vs four fast markers")
        r["endpoint"] = endpoint
        out.append(r)

        # Two reported sensitivity variants, computed with a
        # deposited source: drop the one incompletely measured marker, and restrict to the
        # rats carrying all six markers.
        w_all = pa.pivot_table(index="animal", columns="marker", values="ratio")
        fast_no_v2 = [m for m in FAST if m != "vglut2"]
        xv = contrast(pa[pa.marker != "vglut2"], mod=MOD, fast=fast_no_v2)
        rv = summarise(xv.values, "TH+ChAT vs three fast markers, VGLUT2 dropped")
        rv["endpoint"] = endpoint
        out.append(rv)
        complete_rats = w_all.dropna().index
        xc = contrast(pa[pa.animal.isin(complete_rats)])
        if len(xc) >= 2:
            rc = summarise(xc.values, "TH+ChAT vs four fast markers, rats with all six markers")
            rc["endpoint"] = endpoint
            out.append(rc)

        # each marker on its own against the mean of the four fast markers
        w = pa.pivot_table(index="animal", columns="marker", values="ratio")
        base = w[[m for m in FAST if m in w]].mean(axis=1)
        for m in MARKERS:
            if m not in w:
                continue
            others = [c for c in FAST if c in w and c != m]
            ref = w[others].mean(axis=1) if m in FAST else base
            xm = np.log(w[m] / ref).dropna()
            rm = summarise(xm.values, m)
            rm["endpoint"] = endpoint
            per_marker.append(rm)

        # the contrast within each region, and a test of homogeneity across regions
        rt2 = rt[rt.e > 0].copy()
        rt2["ratio"] = rt2.o / rt2.e
        wr = rt2.pivot_table(index=["animal", "region"], columns="marker", values="ratio")
        a = wr[[m for m in MOD if m in wr]].mean(axis=1)
        b = wr[[m for m in FAST if m in wr]].mean(axis=1)
        xr = np.log(a / b).dropna().rename("x").reset_index()
        for reg, g in xr.groupby("region"):
            rr = summarise(g.x.values, reg)
            rr["endpoint"] = endpoint
            per_region.append(rr)
        # homogeneity: does the contrast differ across regions, with animal as a blocking factor?
        piv = xr.pivot_table(index="animal", columns="region", values="x")
        cc = piv.dropna()
        if cc.shape[0] >= 2 and cc.shape[1] >= 2:
            fr = stats.friedmanchisquare(*[cc[c].values for c in cc.columns])
            out[-1]["region_homogeneity_friedman_chi2"] = float(fr.statistic)
            out[-1]["region_homogeneity_p"] = float(fr.pvalue)
            out[-1]["n_regions"] = int(cc.shape[1])
            out[-1]["n_animals_complete_regions"] = int(cc.shape[0])
        print(
            f"{endpoint:18s} contrast x{r['ratio']:.3f} "
            f"({r['boot_lo']:.3f}-{r['boot_hi']:.3f})  t P={r['t_p']:.4f}  "
            f"{r['n_positive']}/{r['n_animals']} positive",
            flush=True,
        )

    A, B, C = pd.DataFrame(out), pd.DataFrame(per_marker), pd.DataFrame(per_region)
    A.to_csv(V3 / "baseline_class_contrast.tsv", sep="\t", index=False)
    B.to_csv(V3 / "baseline_contrast_per_marker.tsv", sep="\t", index=False)
    C.to_csv(V3 / "baseline_contrast_per_region.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print("\n=== EXPLORATORY ANIMAL-LEVEL CLASS CONTRAST ===")
    print(A.round(4).to_string(index=False))
    print("\n=== EACH MARKER AGAINST THE FAST AMINO-ACID MEAN ===")
    print(B.round(4).to_string(index=False))
    print("\n=== THE CONTRAST BY REGION ===")
    print(C.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
