# -*- coding: utf-8 -*-
"""Intensity-standardised raw-rate sensitivity.

The retained calculation accounts for differences in the distribution of
object intensity across conditions. It compares raw geometric rates, not
the primary spatial-null enrichment estimator. Intensity can reflect several
biological and measurement processes, so this adjustment is descriptive."""

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

import numpy as np
import pandas as pd

ROOT = PIPELINE
OUT = output_dir()
OBJ = ROOT / "data/derived/lps/uniform_standard/objects"

BINS = [0, 1.0, 1.25, 1.5, 2.0, np.inf]
LAB = ["<1.0", "1.0-1.25", "1.25-1.5", "1.5-2.0", ">2.0"]


def load(marker):
    st = pd.read_csv(ROOT / "results/uniform_stack_summary.tsv", sep="\t")
    st = st[
        (st.dataset == "LPS")
        & (st.specification == "uniform_standard")
        & (st.marker.astype(str).str.lower() == marker.lower())
    ]
    rows = []
    for _, r in st.iterrows():
        f = OBJ / f"{r.file_id}.parquet"
        require_file(f)
        o = pd.read_parquet(
            f,
            columns=[
                "mean_intensity",
                "min_distance_to_microglia_um",
                "contact_category",
                "complete_inclusion_eligible",
            ],
        )
        o["cond"] = r.condition
        o["animal"] = r.animal_id
        o["region"] = r.region
        o["rel"] = o.mean_intensity / float(r.object_noise_floor)
        rows.append(o)
    d = pd.concat(rows, ignore_index=True)
    d["contact"] = (d.min_distance_to_microglia_um <= 0.10).astype(float)
    d["complete"] = (d.contact_category == "complete_inclusion").astype(float)
    d["stratum"] = pd.cut(d.rel, BINS, labels=LAB)
    return d


def per_animal_stratified(d, endpoint):
    """Rate within each animal, standardised to the pooled intensity distribution.

    Each animal's rate is a weighted mean of its within-stratum rates, using one common set
    of stratum weights taken from the whole cohort. Groups are then compared at like
    intensity composition without any object being discarded.
    """
    sub = d if endpoint == "contact" else d[d.complete_inclusion_eligible]
    w = sub.groupby("stratum", observed=True).size()
    w = w / w.sum()
    rows = []
    for (a, c), g in sub.groupby(["animal", "cond"]):
        r = g.groupby("stratum", observed=True)[endpoint].mean()
        n = g.groupby("stratum", observed=True)[endpoint].size()
        common = [s for s in w.index if s in r.index and n.get(s, 0) >= 10]
        if not common:
            continue
        ww = w[common] / w[common].sum()
        rows.append(
            dict(
                animal=a,
                cond=c,
                crude=float(g[endpoint].mean()),
                standardised=float((r[common] * ww).sum()),
            )
        )
    return pd.DataFrame(rows)


def exact(v, g, ref, trt):
    v = np.asarray(v, float)
    g = np.asarray(g)
    k = np.isin(g, [ref, trt]) & np.isfinite(v)
    v, g = v[k], g[k]
    nt = int((g == trt).sum())
    n = len(v)
    obs = v[g == trt].mean() - v[g == ref].mean()
    c = t = 0
    for idx in itertools.combinations(range(n), nt):
        m = np.zeros(n, bool)
        m[list(idx)] = True
        if abs(v[m].mean() - v[~m].mean()) >= abs(obs) - 1e-12:
            c += 1
        t += 1
    return float(np.exp(obs)), c / t


def main():
    out = []
    for marker in ("TH", "vglut1"):
        d = load(marker)
        for endpoint in ("contact", "complete"):
            pa = per_animal_stratified(d, endpoint)
            for kind in ("crude", "standardised"):
                pa["x"] = np.log(pa[kind].clip(lower=1e-6))
                for trt in ("LPS_72h", "LPS_60d"):
                    if trt not in set(pa["cond"]):
                        continue
                    r, p = exact(pa["x"].values, pa["cond"].values, "Control", trt)
                    out.append(
                        dict(
                            marker=marker,
                            endpoint=endpoint,
                            adjustment=kind,
                            treatment=trt,
                            ratio=r,
                            exact_p=p,
                        )
                    )
                    print(
                        f"{marker:7s} {endpoint:9s} {kind:13s} {trt:8s} x{r:.3f}  P={p:.4f}",
                        flush=True,
                    )
    df = pd.DataFrame(out)
    df.to_csv(OUT / "intensity_stratified_effects.tsv", sep="\t", index=False)
    print(f"\nwritten {OUT / 'intensity_stratified_effects.tsv'}")
    print("\n=== TH, crude against intensity-standardised ===")
    t = df[df.marker == "TH"].pivot_table(
        index=["endpoint", "treatment"], columns="adjustment", values=["ratio", "exact_p"]
    )
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
