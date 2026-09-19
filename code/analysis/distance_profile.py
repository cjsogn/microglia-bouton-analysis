# -*- coding: utf-8 -*-
"""Segmented-distance profiles for Figure S6.

Observed and shifted counts use the retained common edge-eligible population
across radii. Counts are pooled within animal and region before enrichment is
calculated. Baseline profiles, treatment ratios and the exploratory class
contrast are exported separately. The nested radii are not independent tests
of optically resolved membrane distances."""

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
)

import itertools

import numpy as np
import pandas as pd

ROOT = PIPELINE
OUT = output_dir()
MARKERS = ["TH", "ChaT", "vglut1", "vglut2", "vgat", "gad65"]
MOD, FAST = ["TH", "ChaT"], ["vglut1", "vglut2", "vgat", "gad65"]


def load():
    s = pd.read_csv(ROOT / "results/multiscale_proximity_stack.tsv", sep="\t")
    s = s[(s.dataset == "LPS") & (s.specification == "uniform_standard")]
    st = pd.read_csv(ROOT / "results/uniform_stack_summary.tsv", sep="\t")
    # The summary carries one row per stack PER DETECTOR SPECIFICATION, so joining it on file_id
    # alone matched each multiscale row twice and doubled the table. The duplicates were exact, so
    # the reported ratios and P values never moved, but the lineage was wrong and any later edit
    # that weighted rows would have broken silently. Filter to the same specification and assert
    # the join is many-to-one.
    st = st[st.specification == "uniform_standard"][
        ["file_id", "animal_id", "condition", "region", "marker"]
    ]
    d = s.merge(st, on="file_id", suffixes=("", "_y"), validate="many_to_one")
    key = {m.lower(): m for m in MARKERS}
    d["mk"] = d.marker.astype(str).str.lower().map(key)
    return d[d.mk.notna()]


def per_animal(sub):
    """Counts pooled within animal and region, region ratios averaged with equal weight."""
    t = sub.groupby(["animal_id", "condition", "region"], as_index=False)[
        ["observed", "null_mean"]
    ].sum()
    t = t[t.null_mean > 0]
    t["r"] = t.observed / t.null_mean
    return t.groupby(["animal_id", "condition"], as_index=False).r.mean()


def exact(x, g, ref, trt):
    x = np.asarray(x, float)
    g = np.asarray(g)
    k = np.isin(g, [ref, trt]) & np.isfinite(x)
    x, g = x[k], g[k]
    nt = int((g == trt).sum())
    n = len(x)
    obs = x[g == trt].mean() - x[g == ref].mean()
    s = []
    for idx in itertools.combinations(range(n), nt):
        m = np.zeros(n, bool)
        m[list(idx)] = True
        s.append(x[m].mean() - x[~m].mean())
    s = np.asarray(s)
    return float(np.exp(obs)), float(np.mean(np.abs(s) >= abs(obs) - 1e-12))


def main():
    d = load()
    radii = sorted(d.radius_um.unique())

    base, eff, contrast = [], [], []
    for rad in radii:
        pav = {}
        for m in MARKERS:
            pa = per_animal(d[(d.mk == m) & (d.radius_um == rad)])
            pav[m] = pa
            b = pa[pa.condition == "Control"]
            base.append(
                dict(radius_um=rad, marker=m, n_animals=len(b), enrichment=float(b.r.mean()))
            )
            if m == "TH":
                pa2 = pa.copy()
                pa2["x"] = np.log(pa2.r)
                for trt in ("LPS_72h", "LPS_60d"):
                    r, p = exact(pa2.x.values, pa2.condition.values, "Control", trt)
                    eff.append(dict(radius_um=rad, marker=m, treatment=trt, ratio=r, exact_p=p))
        # the transmitter-class contrast at this radius, per animal, control tissue only
        w = pd.concat([pav[m][pav[m].condition == "Control"].assign(marker=m) for m in MARKERS])
        w = w.pivot_table(index="animal_id", columns="marker", values="r")
        hi = w[[m for m in MOD if m in w]].mean(axis=1)
        lo = w[[m for m in FAST if m in w]].mean(axis=1)
        x = np.log(hi / lo).dropna()
        from scipy import stats as sps

        tt = sps.ttest_1samp(x.values, 0.0)
        contrast.append(
            dict(
                radius_um=rad,
                n_animals=len(x),
                ratio=float(np.exp(x.mean())),
                t_stat=float(tt.statistic),
                t_p=float(tt.pvalue),
                n_positive=int((x > 0).sum()),
            )
        )

    B, E, C = pd.DataFrame(base), pd.DataFrame(eff), pd.DataFrame(contrast)
    B.to_csv(OUT / "distance_profile_baseline.tsv", sep="\t", index=False)
    E.to_csv(OUT / "distance_profile_treatment.tsv", sep="\t", index=False)
    C.to_csv(OUT / "distance_profile_class_contrast.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print("=== BASELINE ENRICHMENT BY RADIUS ===")
    print(
        B.pivot_table(index="marker", columns="radius_um", values="enrichment").round(4).to_string()
    )
    print("\n=== TH TREATMENT EFFECT BY RADIUS ===")
    print(
        E.pivot_table(index="treatment", columns="radius_um", values=["ratio", "exact_p"])
        .round(4)
        .to_string()
    )
    print("\n=== TRANSMITTER-CLASS CONTRAST BY RADIUS (untreated) ===")
    print(C.round(4).to_string(index=False))
    print(f"\nwritten three tables to {OUT}")


if __name__ == "__main__":
    main()
