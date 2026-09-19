# -*- coding: utf-8 -*-
"""Reproduce detector and aggregation sensitivity analyses.

Two detector specifications compare how candidate acceptance changes baseline
enrichment. The late TH contrast is also recomputed with arithmetic or geometric
region aggregation and with pooled counts. These aggregation choices change
weighting or scale and therefore do not all target an identical estimand.

Results assess sensitivity to the listed choices. Agreement cannot establish
that all segmentation or sampling biases are absent."""

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


def region_counts(spec, endpoint, marker=None, condition=None):
    stack = pd.read_csv(ROOT / "results/uniform_stack_summary.tsv", sep="\t")
    stack = stack[(stack.dataset == "LPS") & (stack.specification == spec)]
    if condition is not None:
        stack = stack[stack.condition == condition]
    S = ROOT / f"data/derived/lps/{spec}/torus_null/summaries"
    key = {m.lower(): m for m in MARKERS}
    rows = []
    for _, r in stack.iterrows():
        m = key.get(str(r.marker).lower())
        f = S / f"{r.file_id}.json"
        if m is None:
            continue
        if marker is not None and m != marker:
            continue
        require_file(f)
        ep = json.loads(f.read_text())[endpoint]
        rows.append(
            dict(
                animal=r.animal_id,
                cond=r.condition,
                region=r.region,
                marker=m,
                o=float(ep["observed"]),
                e=float(ep["null_mean"]),
            )
        )
    d = pd.DataFrame(rows)
    return d.groupby(["animal", "cond", "region", "marker"], as_index=False)[["o", "e"]].sum()


def per_animal(rt, combine="arithmetic", region_weight="equal"):
    rt = rt[rt.e > 0].copy()
    if region_weight == "pooled":  # one ratio per animal, regions not weighted
        t = rt.groupby(["animal", "cond", "marker"], as_index=False)[["o", "e"]].sum()
        t["v"] = t.o / t.e
        return t[["animal", "cond", "marker", "v"]]
    rt["ratio"] = rt.o / rt.e
    g = rt.groupby(["animal", "cond", "marker"]).ratio
    v = (
        g.mean()
        if combine == "arithmetic"
        else g.apply(lambda x: float(np.exp(np.mean(np.log(x)))))
    )
    return v.rename("v").reset_index()


def exact(v, g, ref, trt):
    v = np.asarray(v, float)
    g = np.asarray(g)
    k = np.isin(g, [ref, trt]) & np.isfinite(v)
    v, g = v[k], g[k]
    nt = int((g == trt).sum())
    n = len(v)
    obs = v[g == trt].mean() - v[g == ref].mean()
    s = []
    for idx in itertools.combinations(range(n), nt):
        m = np.zeros(n, bool)
        m[list(idx)] = True
        s.append(v[m].mean() - v[~m].mean())
    s = np.asarray(s)
    return float(np.exp(obs)), float(np.mean(np.abs(s) >= abs(obs) - 1e-12)), len(s)


def main():
    # ---- baseline under two detector specifications ------------------------
    rows = []
    for spec, label in (("uniform_standard", "standard"), ("uniform_sensitive", "sensitive")):
        for endpoint in ("complete", "proximity_contact"):
            rt = region_counts(spec, endpoint, condition="Control")
            pa = per_animal(rt)
            for m, g in pa.groupby("marker"):
                rows.append(
                    dict(
                        specification=label,
                        endpoint=endpoint,
                        marker=m,
                        n_animals=int(g.animal.nunique()),
                        enrichment=float(g.v.mean()),
                    )
                )
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "detector_specification_baseline.tsv", sep="\t", index=False)
    print("=== BASELINE, two detector specifications, estimator used throughout ===")
    print(
        B.pivot_table(index="marker", columns=["endpoint", "specification"], values="enrichment")
        .round(3)
        .to_string()
    )

    # ---- the reported 60 d effect across analysis choices -------------------
    choices = [
        ("primary analysis", "uniform_standard", "arithmetic", "equal"),
        ("geometric region mean", "uniform_standard", "geometric", "equal"),
        ("regions pooled, not weighted", "uniform_standard", "arithmetic", "pooled"),
        ("sensitive detector", "uniform_sensitive", "arithmetic", "equal"),
    ]
    out = []
    for label, spec, comb, wt in choices:
        rt = region_counts(spec, "proximity_contact", marker="TH")
        pa = per_animal(rt, comb, wt)
        pa["x"] = np.log(pa.v)
        r, p, n = exact(pa.x.values, pa["cond"].values, "Control", "LPS_60d")
        out.append(
            dict(
                choice=label,
                specification=spec,
                region_combination=comb,
                region_weight=wt,
                ratio=r,
                exact_p=p,
                n_allocations=n,
            )
        )
        print(f"{label:30s} x{r:.3f}  exact P = {p:.4f}")
    E = pd.DataFrame(out)
    E.to_csv(OUT / "analysis_choice_sensitivity.tsv", sep="\t", index=False)
    print(
        f"\nwritten {OUT / 'analysis_choice_sensitivity.tsv'} and detector_specification_baseline.tsv"
    )


if __name__ == "__main__":
    main()
