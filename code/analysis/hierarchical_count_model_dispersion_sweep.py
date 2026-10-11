# -*- coding: utf-8 -*-
"""Sensitivity of count-model estimates to fixed negative-binomial dispersion.

The primary count-model check fixes alpha at 1.0. This script refits each
comparison across the recorded dispersion grid and exports the point estimates.
Cluster counts are small, so the sweep is descriptive rather than a substitute
for the animal-level exact tests.
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


import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hierarchical_count_model as H

ALPHAS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def coef(m, term):
    return float(np.exp(m.params[term]))


def main():
    rows = []
    for endpoint in ("proximity_contact", "complete"):
        d = H.stack_table(endpoint)
        for a in ALPHAS:
            fam = sm.families.NegativeBinomial(alpha=a)
            b = d[d.cond == "Control"].copy()
            b["mclass"] = pd.Categorical(b.mclass, ["fast", "modulatory"])
            m = H.gee(b, "observed ~ mclass + C(region)", b.animal, fam)
            rows.append(
                dict(
                    endpoint=endpoint,
                    alpha=a,
                    comparison="modulatory vs fast, untreated",
                    ratio=coef(m, "mclass[T.modulatory]"),
                )
            )
            th = d[d.marker == "TH"].copy()
            th["cond"] = pd.Categorical(th.cond, ["Control", "LPS_72h", "LPS_60d"])
            m3 = H.gee(th, "observed ~ cond + C(region)", th.animal, fam)
            rows.append(
                dict(
                    endpoint=endpoint,
                    alpha=a,
                    comparison="TH 72 h vs control",
                    ratio=coef(m3, "cond[T.LPS_72h]"),
                )
            )
            rows.append(
                dict(
                    endpoint=endpoint,
                    alpha=a,
                    comparison="TH 60 d vs control",
                    ratio=coef(m3, "cond[T.LPS_60d]"),
                )
            )
            dd = d.copy()
            dd["cond"] = pd.Categorical(dd.cond, ["Control", "LPS_72h", "LPS_60d"])
            dd["isTH"] = pd.Categorical(np.where(dd.marker == "TH", "TH", "other"), ["other", "TH"])
            m5 = H.gee(dd, "observed ~ cond * isTH + C(region)", dd.animal, fam)
            rows.append(
                dict(
                    endpoint=endpoint,
                    alpha=a,
                    comparison="TH-specific 60 d interaction",
                    ratio=coef(m5, "cond[T.LPS_60d]:isTH[T.TH]"),
                )
            )

    out = pd.DataFrame(rows)
    out.to_csv(H.OUT / "hierarchical_count_model_dispersion_sweep.tsv", sep="\t", index=False)
    w = out.pivot_table(index=["endpoint", "comparison"], columns="alpha", values="ratio")
    print(w.round(3).to_string())
    print(
        "\nmax spread of any comparison across alpha:",
        float((w.max(1) - w.min(1)).max()).__round__(4),
    )


if __name__ == "__main__":
    main()
