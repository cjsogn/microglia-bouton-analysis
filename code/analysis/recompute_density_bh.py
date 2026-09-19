# -*- coding: utf-8 -*-
"""Benjamini-Hochberg q values for the marker-availability family, over the six retained markers.

Absolute bouton density is the one family in the paper that is taken directly from the imaging
pipeline rather than recomputed by the scripts in this directory, because density needs no spatial
null and no estimator choice: it is a count per unit volume. What it does need is the right family.
Sections were also stained for serotonin, that channel showed Iba1 crosstalk and was withdrawn, and
the pipeline computed its q values over a family that still contained it. This rebuilds the family
over the six markers the paper reports.

Nothing else changes. The same exact two-sided P values enter and the same step-up procedure is
applied, so only the multiplicity denominator differs.
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
)

import numpy as np
import pandas as pd

RES = PIPELINE / "results"
OUT = output_dir()
OUT.mkdir(parents=True, exist_ok=True)

KEEP = ["TH", "ChaT", "gad65", "vgat", "vglut1", "vglut2"]


def bh(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.clip(ranked, 0, 1)
    return q


def main():
    de = pd.read_csv(RES / "absolute_density_treatment_effects.tsv", sep="\t")
    de = de[(de.specification == "uniform_standard") & (de.marker.isin(KEEP))].copy()
    de["bh_q"] = np.nan
    for _, idx in de.groupby("multiplicity_family").groups.items():
        de.loc[idx, "bh_q"] = bh(de.loc[idx, "exact_two_sided_p"].values)
    cols = [
        "dataset",
        "endpoint",
        "marker",
        "treatment",
        "density_ratio",
        "exact_inversion_ci_low_ratio",
        "exact_inversion_ci_high_ratio",
        "exact_two_sided_p",
        "bh_q",
        "multiplicity_family",
    ]
    out = de[cols]
    out.to_csv(OUT / "density_effects_bh.tsv", sep="\t", index=False)
    print(
        f"{len(out)} rows over {out.marker.nunique()} markers "
        f"in {out.multiplicity_family.nunique()} families"
    )
    print(f"written {OUT / 'density_effects_bh.tsv'}")
    print(out[out.endpoint == "all_bouton_candidates"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
