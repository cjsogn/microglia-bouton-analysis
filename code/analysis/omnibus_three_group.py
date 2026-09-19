# -*- coding: utf-8 -*-
"""Three-group animal-label omnibus tests for the LPS results.

The statistic is the between-group sum of squares of animal-level log
enrichment. The null enumerates assignments at the observed group sizes.
Contact and complete inclusion are exported separately. These are the tests
used in the retrospective analysis, not a claim of preregistration."""

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
GROUPS = ["Control", "LPS_72h", "LPS_60d"]


def per_animal(marker, endpoint):
    """Region-pooled ratio per region, averaged over regions, one value per animal."""
    stack = pd.read_csv(ROOT / "results/uniform_stack_summary.tsv", sep="\t")
    stack = stack[
        (stack.dataset == "LPS")
        & (stack.specification == "uniform_standard")
        & (stack.marker.astype(str).str.lower() == marker.lower())
    ]
    S = ROOT / "data/derived/lps/uniform_standard/torus_null/summaries"
    rows = []
    for _, r in stack.iterrows():
        f = S / f"{r.file_id}.json"
        require_file(f)
        ep = json.loads(f.read_text())[endpoint]
        rows.append(
            dict(
                animal=r.animal_id,
                cond=r.condition,
                region=r.region,
                o=float(ep["observed"]),
                e=float(ep["null_mean"]),
            )
        )
    d = pd.DataFrame(rows)
    d = d.groupby(["animal", "cond", "region"], as_index=False)[["o", "e"]].sum()
    d = d[d.e > 0]
    d["ratio"] = d.o / d.e
    v = d.groupby(["animal", "cond"], as_index=False).ratio.mean()
    v["x"] = np.log(v.ratio)
    return v


def between_group_ss(x, labels):
    """Sum of squared deviations of the group means from the grand mean, weighted by group size."""
    x = np.asarray(x, float)
    g = np.asarray(labels)
    gm = x.mean()
    return float(sum((x[g == k].mean() - gm) ** 2 * (g == k).sum() for k in np.unique(g)))


def exact_omnibus(v):
    """Enumerate every allocation of animals to groups of the observed sizes."""
    x = v.x.values
    labels = v["cond"].values
    sizes = [int((labels == k).sum()) for k in GROUPS]
    obs = between_group_ss(x, labels)
    n = len(x)
    idx = list(range(n))
    hits = total = 0
    for a in itertools.combinations(idx, sizes[0]):
        rest = [i for i in idx if i not in a]
        for b in itertools.combinations(rest, sizes[1]):
            lab = np.empty(n, dtype=object)
            lab[list(a)] = GROUPS[0]
            lab[list(b)] = GROUPS[1]
            lab[[i for i in rest if i not in b]] = GROUPS[2]
            if between_group_ss(x, lab) >= obs - 1e-12:
                hits += 1
            total += 1
    return obs, hits / total, total, sizes


def main():
    rows = []
    for endpoint in ("proximity_contact", "complete"):
        for m in MARKERS:
            v = per_animal(m, endpoint)
            if not set(GROUPS) <= set(v["cond"]):
                continue
            stat, p, n_alloc, sizes = exact_omnibus(v)
            rows.append(
                dict(
                    endpoint=endpoint,
                    marker=m,
                    statistic=stat,
                    exact_p=p,
                    n_allocations=n_alloc,
                    n_control=sizes[0],
                    n_72h=sizes[1],
                    n_60d=sizes[2],
                )
            )
            print(f"{endpoint:18s} {m:7s} P = {p:.4f}  ({n_alloc} allocations)", flush=True)
    df = pd.DataFrame(rows)
    # Every other family in the paper carries a BH q, so this one does too. The family is the six
    # markers within an endpoint, which is the same shape as the treatment families.
    df["bh_q"] = np.nan
    for ep, g in df.groupby("endpoint"):
        g = g.sort_values("exact_p")
        m = len(g)
        q = np.minimum.accumulate((g.exact_p.values * m / np.arange(1, m + 1))[::-1])[::-1]
        df.loc[g.index, "bh_q"] = np.minimum(q, 1.0)
    df.to_csv(OUT / "omnibus_three_group.tsv", sep="\t", index=False)
    print(f"\nwritten {OUT / 'omnibus_three_group.tsv'}")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
