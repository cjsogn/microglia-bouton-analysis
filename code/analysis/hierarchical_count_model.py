# -*- coding: utf-8 -*-
"""Retained exploratory stack-level count-model sensitivity.

The model uses the estimated spatial-null mean as an offset:

    log E[observed] = log(null mean) + linear predictor

It treats that estimated offset as fixed. Exponentiated coefficients describe
relative enrichment on the model's scale, with different weighting from the
primary animal-level estimator.

Poisson Pearson dispersion selects the retained working family. The negative-
binomial working family uses fixed alpha=1.0, not an estimated alpha. GEE fits
use animal clusters and an exchangeable working correlation. A separate baseline
class comparison uses a negative-binomial GLM with animal and region fixed effects.
That fixed-effect check does not estimate between-animal treatment effects.

Wald summaries are exploratory. The GEE summaries use a t reference with G-1
degrees of freedom, without claiming that this resolves small-cluster uncertainty.
They do not replace the primary imaging animal-label tests. The companion
hierarchical_count_model_dispersion_sweep.py varies the working dispersion.
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
    require_file,
)

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

ROOT = PIPELINE
OUT = output_dir()
MARKERS = ["TH", "ChaT", "vglut1", "vglut2", "vgat", "gad65"]
MOD = ["TH", "ChaT"]


def stack_table(endpoint):
    """One row per stack: observed events, torus expectation, animal, condition, region, marker."""
    st = pd.read_csv(ROOT / "results/uniform_stack_summary.tsv", sep="\t")
    st = st[(st.dataset == "LPS") & (st.specification == "uniform_standard")]
    S = ROOT / "data/derived/lps/uniform_standard/torus_null/summaries"
    key = {m.lower(): m for m in MARKERS}
    rows = []
    for _, r in st.iterrows():
        m = key.get(str(r.marker).lower())
        f = S / f"{r.file_id}.json"
        if m is None:
            continue
        require_file(f)
        ep = json.loads(f.read_text())[endpoint]
        rows.append(
            dict(
                file_id=r.file_id,
                animal=r.animal_id,
                cond=r.condition,
                region=r.region,
                marker=m,
                observed=float(ep["observed"]),
                expected=float(ep["null_mean"]),
            )
        )
    d = pd.DataFrame(rows)
    d = d[d.expected > 0].copy()
    d["observed"] = d.observed.round().astype(int)
    d["log_e"] = np.log(d.expected)
    d["mclass"] = np.where(d.marker.isin(MOD), "modulatory", "fast")
    return d


def dispersion(d):
    """Pearson dispersion of a Poisson fit with the offset and marker means only."""
    m = smf.glm("observed ~ C(marker)", data=d, family=sm.families.Poisson(), offset=d.log_e).fit()
    return float(m.pearson_chi2 / m.df_resid)


def gee(d, formula, groups, family):
    m = smf.gee(
        formula,
        groups=groups,
        data=d,
        family=family,
        offset=d.log_e,
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    return m


def report(m, term, n_clusters, label, out):
    if term not in m.params.index:
        raise ValueError(f"Required model term is missing: {term}")
    b, se = float(m.params[term]), float(m.bse[term])
    df = max(n_clusters - 1, 1)
    p = 2 * stats.t.sf(abs(b / se), df)
    lo, hi = b - stats.t.ppf(0.975, df) * se, b + stats.t.ppf(0.975, df) * se
    out.append(
        dict(
            comparison=label,
            term=term,
            ratio=float(np.exp(b)),
            ci_low=float(np.exp(lo)),
            ci_high=float(np.exp(hi)),
            wald_t_p=float(p),
            n_clusters=int(n_clusters),
            df=int(df),
        )
    )
    print(
        f"  {label:44s} x{np.exp(b):.4f} ({np.exp(lo):.3f}-{np.exp(hi):.3f})  P(t,{df}) = {p:.4f}"
    )


def main():
    out = []
    for endpoint in ("proximity_contact", "complete"):
        d = stack_table(endpoint)
        disp = dispersion(d)
        print(f"\n================ {endpoint} ================")
        print(
            f"Poisson dispersion = {disp:.1f} on {len(d)} stacks. "
            f"{'Retained negative-binomial working family.' if disp > 2 else 'Retained Poisson working family.'}"
        )

        fam = sm.families.NegativeBinomial(alpha=1.0) if disp > 2 else sm.families.Poisson()

        # --- baseline transmitter class, untreated tissue only -------------
        b = d[d.cond == "Control"].copy()
        b["mclass"] = pd.Categorical(b.mclass, ["fast", "modulatory"])
        m = gee(b, "observed ~ mclass + C(region)", b.animal, fam)
        print("BASELINE, untreated:")
        report(
            m,
            "mclass[T.modulatory]",
            b.animal.nunique(),
            "modulatory vs fast (GEE, animal cluster)",
            out,
        )
        # Each nonreference marker against the recorded GAD65 reference.
        b2 = b.copy()
        b2["marker"] = pd.Categorical(b2.marker, ["gad65"] + [x for x in MARKERS if x != "gad65"])
        m2 = gee(b2, "observed ~ C(marker) + C(region)", b2.animal, fam)
        for mk in MARKERS:
            if mk == "gad65":
                continue
            report(m2, f"C(marker)[T.{mk}]", b2.animal.nunique(), f"  {mk} vs GAD65", out)

        # --- treatment effects, TH only ------------------------------------
        th = d[d.marker == "TH"].copy()
        th["cond"] = pd.Categorical(th.cond, ["Control", "LPS_72h", "LPS_60d"])
        m3 = gee(th, "observed ~ cond + C(region)", th.animal, fam)
        print("TREATMENT, TH:")
        report(m3, "cond[T.LPS_72h]", th.animal.nunique(), "72 h vs control (GEE)", out)
        report(m3, "cond[T.LPS_60d]", th.animal.nunique(), "60 d vs control (GEE)", out)
        th2 = th[th.cond != "Control"].copy()
        th2["cond"] = pd.Categorical(th2.cond, ["LPS_72h", "LPS_60d"])
        m4 = gee(th2, "observed ~ cond + C(region)", th2.animal, fam)
        report(m4, "cond[T.LPS_60d]", th2.animal.nunique(), "60 d vs 72 h (GEE)", out)

        # --- marker by condition interaction, all six markers --------------
        dd = d.copy()
        dd["cond"] = pd.Categorical(dd.cond, ["Control", "LPS_72h", "LPS_60d"])
        dd["isTH"] = np.where(dd.marker == "TH", "TH", "other")
        dd["isTH"] = pd.Categorical(dd.isTH, ["other", "TH"])
        m5 = gee(dd, "observed ~ cond * isTH + C(region)", dd.animal, fam)
        print("INTERACTION, TH against the other five:")
        report(
            m5,
            "cond[T.LPS_60d]:isTH[T.TH]",
            dd.animal.nunique(),
            "TH-specific 60 d effect (GEE)",
            out,
        )
        report(
            m5,
            "cond[T.LPS_72h]:isTH[T.TH]",
            dd.animal.nunique(),
            "TH-specific 72 h effect (GEE)",
            out,
        )

        # --- fixed-effect negative binomial check --------------------------
        nb = smf.glm(
            "observed ~ mclass + C(animal) + C(region)",
            data=b,
            family=sm.families.NegativeBinomial(alpha=1.0),
            offset=b.log_e,
        ).fit()
        bb, se = float(nb.params["mclass[T.modulatory]"]), float(nb.bse["mclass[T.modulatory]"])
        print(
            f"  {'modulatory vs fast (NB GLM, animal fixed)':44s} x{np.exp(bb):.4f} "
            f"(SE {se:.4f} on the log scale)"
        )
        out.append(
            dict(
                comparison="modulatory vs fast (NB GLM, animal fixed effects)",
                term="mclass[T.modulatory]",
                ratio=float(np.exp(bb)),
                ci_low=float(np.exp(bb - 1.96 * se)),
                ci_high=float(np.exp(bb + 1.96 * se)),
                wald_t_p=np.nan,
                n_clusters=int(b.animal.nunique()),
                df=np.nan,
            )
        )
        for r in out[-1:]:
            r["endpoint"] = endpoint
        for r in out:
            r.setdefault("endpoint", endpoint)

    df = pd.DataFrame(out)
    df.to_csv(OUT / "hierarchical_count_model.tsv", sep="\t", index=False)
    print(f"\nwritten {OUT / 'hierarchical_count_model.tsv'}")


if __name__ == "__main__":
    main()
