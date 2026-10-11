# -*- coding: utf-8 -*-
"""Regional comparisons of the exploratory transmitter-class contrast.

The within-animal region tests assess the TH/ChAT-versus-other-marker contrast.
Condition labels are exchanged across animals for the condition-by-region
interaction in that class contrast. This is not a TH-only regional interaction
test. Region-specific TH effects are exported separately.

Permutation inference retains animals as the independent units and assumes the
stated label exchangeability. No permutation calibration corrects the post hoc
choice of transmitter grouping."""

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
    SOURCE_DATA,
    output_dir,
)

import itertools

import numpy as np
import pandas as pd
from scipy import stats

BASE = PKG
PIPE = PIPELINE
SD = SOURCE_DATA
OUT = output_dir()

ENDPOINT = "proximity_contact"
CASE = {
    "TH": "TH",
    "ChAT": "ChaT",
    "VGLUT1": "vglut1",
    "VGLUT2": "vglut2",
    "VGAT": "vgat",
    "GAD65": "gad65",
}
MODULATORY = ["TH", "ChaT"]
FAST = ["vglut1", "vglut2", "vgat", "gad65"]
ORDER = ["Control", "LPS_72h", "LPS_60d"]
N_PERM = 20000
SEED = 20260823


def per_animal_region():
    """Per animal, region and marker enrichment against the spatial null, pooled counts."""
    d = pd.read_csv(SD / "S4_per_animal_region_enrichment.tsv", sep="\t")
    d = d[(d.endpoint == ENDPOINT) & (d.specification == "uniform_standard")]
    g = (
        d.groupby(["dataset", "condition", "animal_id", "region", "marker"])[
            ["observed", "expected"]
        ]
        .sum()
        .reset_index()
    )
    g["enrichment"] = g.observed / g.expected
    g["marker"] = g.marker.map(CASE)
    return g[g.dataset == "LPS"]


def class_contrast(ar):
    """Modulatory over fast amino-acid, within animal and region, so scale cancels."""
    rows = []
    for (an, reg), g in ar.groupby(["animal_id", "region"]):
        m = g[g.marker.isin(MODULATORY)].enrichment
        f = g[g.marker.isin(FAST)].enrichment
        if len(m) and len(f):
            rows.append(
                dict(
                    animal=an,
                    region=reg,
                    condition=g.condition.iloc[0],
                    contrast=m.mean() / f.mean(),
                )
            )
    return pd.DataFrame(rows)


def f_type(mat):
    """Spread among column means, scaled by within-column variance. Sensitive to any difference."""
    mu = mat.mean(0)
    return float(((mu - mu.mean()) ** 2).sum() / max(mat.var(0, ddof=1).mean(), 1e-12))


def region_effect(wide, rng, n_perm=N_PERM):
    """Region labels permuted within each animal: each animal is its own control."""
    x = wide.to_numpy()
    obs_f = f_type(x)
    obs_r = float(x.mean(0).max() - x.mean(0).min())
    nf = nr = 0
    for _ in range(n_perm):
        sh = np.array([rng.permutation(row) for row in x])
        nf += f_type(sh) >= obs_f - 1e-12
        nr += (sh.mean(0).max() - sh.mean(0).min()) >= obs_r - 1e-12
    fr = stats.friedmanchisquare(*[x[:, k] for k in range(x.shape[1])])
    return dict(
        n_animals=x.shape[0],
        n_regions=x.shape[1],
        f_type=obs_f,
        f_type_p=(nf + 1) / (n_perm + 1),
        range=obs_r,
        range_p=(nr + 1) / (n_perm + 1),
        friedman_chi2=float(fr.statistic),
        friedman_p=float(fr.pvalue),
    )


def interaction(wide, cond):
    """Condition labels permuted over animals, every assignment enumerated."""
    x = wide.to_numpy()

    def stat(lab):
        cm = np.array([x[lab == c].mean(0) for c in ORDER])
        return float(((cm - cm.mean(0) - cm.mean(1)[:, None] + cm.mean()) ** 2).sum())

    obs = stat(cond)
    counts = [int((cond == c).sum()) for c in ORDER]
    hit = tot = 0
    for i in itertools.combinations(range(len(cond)), counts[0]):
        rest = [k for k in range(len(cond)) if k not in i]
        for j in itertools.combinations(rest, counts[1]):
            lab = np.array([ORDER[2]] * len(cond), dtype=object)
            lab[list(i)] = ORDER[0]
            lab[list(j)] = ORDER[1]
            hit += stat(lab) >= obs - 1e-12
            tot += 1
    return dict(statistic=obs, exact_p=hit / tot, n_assignments=tot)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ar = per_animal_region()
    ca = class_contrast(ar)
    # Ordered by the untreated gradient: the ordering is itself a claim about resting tissue,
    # so it must not be set by animals that have been treated.
    regions = (
        ca[ca.condition == "Control"]
        .groupby("region")
        .contrast.mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    rows = []
    for lab, sub in (("all_animals", ca), ("control_only", ca[ca.condition == "Control"])):
        w = sub.pivot_table(index="animal", columns="region", values="contrast")[regions]
        r = region_effect(w, np.random.default_rng(SEED))
        rows.append(dict(test="class_contrast_region", subset=lab, **r))
        print(
            f"class contrast, region effect [{lab}, n = {r['n_animals']}]: "
            f"F-type P = {r['f_type_p']:.4f}, range P = {r['range_p']:.4f}, "
            f"Friedman P = {r['friedman_p']:.4f}"
        )
        print("   " + "  ".join(f"{q} {w[q].mean():.2f}" for q in regions))

    w = ca.pivot_table(index="animal", columns="region", values="contrast")[regions]
    cond = ca.groupby("animal").condition.first().reindex(w.index).to_numpy()
    it = interaction(w, cond)
    rows.append(
        dict(
            test="class_contrast_region_x_treatment",
            subset="all_animals",
            n_animals=len(w),
            n_regions=len(regions),
            **it,
        )
    )
    print(
        f"class contrast, region x treatment: exact P = {it['exact_p']:.4f} "
        f"over {it['n_assignments']:,} assignments"
    )

    # TH enrichment, the endpoint panel c draws
    th = ar[ar.marker == "TH"]
    wt = th.pivot_table(index="animal_id", columns="region", values="enrichment")[regions]
    ct = th.groupby("animal_id").condition.first().reindex(wt.index).to_numpy()
    itt = interaction(wt, ct)
    rows.append(
        dict(
            test="TH_region_x_treatment",
            subset="all_animals",
            n_animals=len(wt),
            n_regions=len(regions),
            **itt,
        )
    )
    print(
        f"TH enrichment, region x treatment: exact P = {itt['exact_p']:.4f} "
        f"over {itt['n_assignments']:,} assignments"
    )

    pd.DataFrame(rows).to_csv(OUT / "regional_tests.tsv", sep="\t", index=False)

    # Use pooled counts and inverted exact-test intervals, as in the primary analysis.
    from regional_temporal_contrast import regional_control_effects

    te = regional_control_effects(ar)
    te.to_csv(OUT / "regional_th_per_region.tsv", sep="\t", index=False)
    print("\nper-region TH effects (60 d):")
    print(te[te.treatment == "LPS_60d"].to_string(index=False))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
