# -*- coding: utf-8 -*-
"""Paired marker interactions for the direct temporal contrast in Table S13.

TH is paired with each comparator within animal and anatomical region before
comparing the independent LPS time groups. Exact animal-label tests and inverted
confidence sets use the retained pooled-count estimator. Multiplicity is
adjusted within the stated comparator family for each endpoint."""

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
    output_dir,
)

import numpy as np
import pandas as pd

from estimate_interactions import COMPARATORS, load, exact, bh

OUT = output_dir()


def main():
    rows = []
    for endpoint in ("proximity_contact", "complete", "any_overlap"):
        th = load("TH", endpoint)
        if th.empty:
            continue
        for comp in COMPARATORS:
            cm = load(comp, endpoint)
            m = th.merge(cm, on=["animal", "cond", "region"], suffixes=("_th", "_c"))
            m["r_th"] = m.o_th / m.e_th
            m["r_c"] = m.o_c / m.e_c
            g = m.groupby(["animal", "cond"])[["r_th", "r_c"]].mean().reset_index()
            g = g[(g.r_th > 0) & (g.r_c > 0)]
            g["d"] = np.log(g.r_th) - np.log(g.r_c)
            pa = g[["animal", "cond", "d"]]
            if not {"LPS_72h", "LPS_60d"} <= set(pa["cond"]):
                continue
            r, p, lo, hi = exact(pa["d"].values, pa["cond"].values, "LPS_72h", "LPS_60d")
            rows.append(
                dict(
                    endpoint=endpoint,
                    comparator=comp,
                    contrast="LPS_60d/LPS_72h",
                    n_regions_paired=int(m.region.nunique()),
                    interaction=r,
                    ci_low=lo,
                    ci_high=hi,
                    exact_p=p,
                )
            )
            print(
                f"{endpoint:18s} TH vs {comp:7s} 60 d / 72 h  x{r:.3f} "
                f"({lo:.2f}-{hi:.2f})  P = {p:.4f}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    for ep, idx in df.groupby("endpoint").groups.items():
        df.loc[idx, "bh_q_within_5"] = bh(df.loc[idx, "exact_p"].values)
    df.to_csv(OUT / "timepoint_specificity.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print("\n" + df.round(4).to_string(index=False))
    print(f"\nwritten {OUT / 'timepoint_specificity.tsv'}")


if __name__ == "__main__":
    main()
