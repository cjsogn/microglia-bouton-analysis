# -*- coding: utf-8 -*-
"""Object intensity and detection-floor summaries supporting Table S8.

Raw marker intensity is expressed relative to the recorded stack acceptance
floor and compared across conditions and geometric categories. The summaries
describe measurement composition. They do not identify signal degradation,
place intensity on a causal pathway or establish antibody specificity."""

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

import numpy as np
import pandas as pd

ROOT = PIPELINE
OUT = output_dir()
OBJ = ROOT / "data/derived/lps/uniform_standard/objects"


def load(marker="TH"):
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
        o["floor"] = float(r.object_noise_floor)
        rows.append(o)
    d = pd.concat(rows, ignore_index=True)
    d["contact"] = (d.min_distance_to_microglia_um <= 0.10).astype(float)
    d["complete"] = (d.contact_category == "complete_inclusion").astype(float)
    d["rel"] = d.mean_intensity / d.floor
    return d


def main():
    d = load("TH")

    print("=" * 78)
    print("1. IS THE DETECTOR EQUALLY STRICT ACROSS CONDITIONS?")
    print("=" * 78)
    t = d.groupby("cond").agg(
        median_intensity=("mean_intensity", "median"),
        median_floor=("floor", "median"),
        median_rel_to_floor=("rel", "median"),
        pct_within_1p5x_floor=("rel", lambda x: 100 * (x < 1.5).mean()),
        n_objects=("rel", "size"),
    )
    print(t.round(3).to_string())
    # Deposited, not only printed: the Results quote pct_within_1p5x_floor for Control and 60 d,
    # and a number a reader cannot open is a number they cannot check.
    t.round(4).to_csv(OUT / "intensity_detection_floor_by_condition.tsv", sep="\t")
    print(f"written {OUT / 'intensity_detection_floor_by_condition.tsv'}")
    print("\nIf median_rel_to_floor is similar across conditions the detector is admitting")
    print("objects at the same distance above its own noise everywhere.")

    print()
    print("=" * 78)
    print("2. IS DIMNESS ASSOCIATED WITH ENGAGEMENT IN EVERY CONDITION?")
    print("=" * 78)
    bins = [0, 1.0, 1.25, 1.5, 2.0, 3.0, np.inf]
    lab = ["<1.0", "1.0-1.25", "1.25-1.5", "1.5-2.0", "2.0-3.0", ">3.0"]
    d["relbin"] = pd.cut(d.rel, bins, labels=lab)
    for ep in ("contact", "complete"):
        sub = d if ep == "contact" else d[d.complete_inclusion_eligible]
        g = sub.groupby(["relbin", "cond"], observed=True)[ep].mean().unstack() * 100
        n = sub.groupby(["relbin", "cond"], observed=True)[ep].size().unstack()
        print(f"\n{ep} rate (%), binned by intensity relative to that stack's own floor:")
        print(g.round(2).to_string())
        print("n objects:")
        print(n.to_string())
        # within-condition slope: does rate fall as intensity rises?
        print("\nSpearman correlation between object intensity and engagement, within condition:")
        for c in sorted(sub.cond.unique()):
            s = sub[sub.cond == c]
            r = np.corrcoef(s.rel.rank(), s[ep])[0, 1]
            print(f"   {c:9s} r = {r:+.4f}   (n = {len(s):,})")

    print()
    print("=" * 78)
    print("3. THE SAME QUESTION FOR AN ABUNDANT COMPARATOR (VGLUT1)")
    print("=" * 78)
    v = load("vglut1")
    for ep in ("contact", "complete"):
        sub = v if ep == "contact" else v[v.complete_inclusion_eligible]
        print(f"\n{ep}, Spearman intensity vs engagement, within condition:")
        for c in sorted(sub.cond.unique()):
            s = sub[sub.cond == c]
            r = np.corrcoef(s.rel.rank(), s[ep])[0, 1]
            print(f"   {c:9s} r = {r:+.4f}   (n = {len(s):,})")

    d.groupby(["cond", "relbin"], observed=True)[["contact", "complete"]].mean().to_csv(
        OUT / "intensity_engagement_diagnostic.tsv", sep="\t"
    )
    print(f"\nwritten {OUT / 'intensity_engagement_diagnostic.tsv'}")


if __name__ == "__main__":
    main()
