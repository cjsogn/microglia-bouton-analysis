# -*- coding: utf-8 -*-
"""Exploratory within-animal comparisons of baseline marker association.

Omnibus, TH/ChAT-versus-other-marker, and TH-versus-comparator statistics use
within-animal ranks and log enrichment. Marker-label permutations require
exchangeability. Ranking does not remove marker-specific variation or guarantee
calibration. The separate variance-calibration scripts assess this limitation.

The transmitter grouping was chosen after inspection. There was no published
or registered study SAP. Neither permutation calibration nor the tests here
adjust for selecting the grouping from these data."""

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
    DERIVED,
    PIPELINE,
    OUTPUT,
    output_dir,
    derived_input,
    require_file,
)

import numpy as np
import pandas as pd

V3 = DERIVED
MARKERS = ["TH", "ChaT", "vglut1", "vglut2", "vgat", "gad65"]
HIGH = ["TH", "ChaT"]
NPERM = 200_000
SEED = 20260806


def blocks(endpoint, dataset="LPS", condition="Control"):
    """One row per animal, one column per marker, on the scale requested.

    Returns a list of (animal, markers_present, values) so that an animal missing a marker is
    permuted only over the markers it actually carries.
    """
    p = pd.read_csv(derived_input("unbiased_per_animal.tsv"), sep="\t")
    p = p[(p.dataset == dataset) & (p.condition == condition) & (p.endpoint == endpoint)]
    out = []
    for a, g in p.groupby("animal"):
        g = g[g.marker.isin(MARKERS)].dropna(subset=["enrichment"])
        out.append((a, list(g.marker.values), np.asarray(g.enrichment.values, float)))
    return out


def normalised_ranks(vals):
    """Within-block ranks mapped to [0, 1], using midranks for ties.

    Rank transformation does not itself establish marker exchangeability.
    """
    v = np.asarray(vals, float)
    order = v.argsort()
    r = np.empty(len(v), float)
    r[order] = np.arange(len(v), dtype=float)
    # midranks
    for u in np.unique(v):
        m = v == u
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r / (len(v) - 1)


def as_matrix(bl, scale):
    """Blocks -> dict marker -> array over animals (nan where absent), on the chosen scale."""
    d = {m: [] for m in MARKERS}
    for _, mk, v in bl:
        x = normalised_ranks(v) if scale == "rank" else np.log(v)
        s = dict(zip(mk, x))
        for m in MARKERS:
            d[m].append(s.get(m, np.nan))
    return {m: np.asarray(v, float) for m, v in d.items()}


def stat_friedman(mat):
    """Sum of squared deviations of marker means from the grand mean. Monotone in the Friedman
    statistic for complete blocks and well defined when one cell is missing."""
    mu = np.array([np.nanmean(mat[m]) for m in MARKERS])
    return float(np.nansum((mu - np.nanmean(mu)) ** 2))


def stat_contrast(mat):
    """Mean of the catecholaminergic and cholinergic markers minus mean of the other four."""
    hi = np.nanmean([np.nanmean(mat[m]) for m in HIGH])
    lo = np.nanmean([np.nanmean(mat[m]) for m in MARKERS if m not in HIGH])
    return float(hi - lo)


def stat_pair(mat, a, b):
    return float(np.nanmean(mat[a]) - np.nanmean(mat[b]))


def permute(bl, scale, statfun, nperm=NPERM, seed=SEED, two_sided=True):
    """Permute marker labels independently within each animal.

    One permutation per animal is applied to that animal's whole vector, which keeps the animal as
    the unit of inference. Regions are already collapsed inside the per-animal estimate, so no
    region is ever treated as an independent replicate.
    """
    rng = np.random.default_rng(seed)
    obs = statfun(as_matrix(bl, scale))
    hits = 0
    for _ in range(nperm):
        pb = [(a, list(rng.permutation(mk)), v) for a, mk, v in bl]
        s = statfun(as_matrix(pb, scale))
        if (abs(s) >= abs(obs) - 1e-12) if two_sided else (s >= obs - 1e-12):
            hits += 1
    return obs, (hits + 1) / (nperm + 1)


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    r = p[o] * m / np.arange(1, m + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    q = np.empty(m)
    q[o] = np.clip(r, 0, 1)
    return q


def sparsity():
    """Events per stack for each marker, to test the obvious alternative explanation.

    TH is the sparsest marker, so a sceptic will ask whether its position reflects nothing but a
    noisier estimate. The internal control is ChAT and VGLUT2, which have comparable event rates
    and sit at opposite ends of the ordering.
    """
    root = PIPELINE
    import json

    st = pd.read_csv(root / "results/uniform_stack_summary.tsv", sep="\t")
    st = st[(st.dataset == "LPS") & (st.specification == "uniform_standard")]
    S = root / "data/derived/lps/uniform_standard/torus_null/summaries"
    rows = []
    for _, r in st.iterrows():
        f = S / f"{r.file_id}.json"
        if str(r.condition) != "Control":
            continue
        require_file(f)
        d = json.loads(f.read_text())
        rows.append(
            dict(
                marker=str(r.marker).lower(),
                complete=float(d["complete"]["observed"]),
                contact=float(d["proximity_contact"]["observed"]),
            )
        )
    d = pd.DataFrame(rows)
    key = {m.lower(): m for m in MARKERS}
    d["marker"] = d.marker.map(key)
    return d.groupby("marker")[["complete", "contact"]].median()


def main():
    rows = []
    for endpoint in ("proximity_contact", "complete"):
        bl = blocks(endpoint)
        n_an = len(bl)
        # sensitivity: is the omnibus carried by VGLUT2 alone, which is last in every animal?
        for drop in ("vglut2", "TH", "ChaT"):
            sub = [
                (
                    a,
                    [m for m in mk if m != drop],
                    np.asarray([v for m, v in zip(mk, vv) if m != drop], float),
                )
                for a, mk, vv in bl
            ]
            g = globals()
            old = g["MARKERS"]
            g["MARKERS"] = [m for m in old if m != drop]
            try:
                o, p = permute(sub, "rank", stat_friedman, nperm=50_000, two_sided=False)
                o2, p2 = permute(sub, "log", stat_friedman, nperm=50_000, two_sided=False)
            finally:
                g["MARKERS"] = old
            rows.append(
                dict(
                    endpoint=endpoint,
                    scale="rank",
                    test="omnibus_leave_one_out",
                    comparison=f"without {drop}",
                    statistic=o,
                    perm_p=p,
                    n_animals=n_an,
                )
            )
            rows.append(
                dict(
                    endpoint=endpoint,
                    scale="log",
                    test="omnibus_leave_one_out",
                    comparison=f"without {drop}",
                    statistic=o2,
                    perm_p=p2,
                    n_animals=n_an,
                )
            )
        for scale in ("rank", "log"):
            as_matrix(bl, scale)  # Retain the original shape validation.
            o, p = permute(bl, scale, stat_friedman, two_sided=False)
            rows.append(
                dict(
                    endpoint=endpoint,
                    scale=scale,
                    test="omnibus_friedman",
                    comparison="six markers",
                    statistic=o,
                    perm_p=p,
                    n_animals=n_an,
                )
            )
            o, p = permute(bl, scale, stat_contrast)
            rows.append(
                dict(
                    endpoint=endpoint,
                    scale=scale,
                    test="focused_contrast",
                    comparison="TH+ChAT vs four others",
                    statistic=o,
                    perm_p=p,
                    n_animals=n_an,
                )
            )
            ps, comps, sts = [], [], []
            for c in [m for m in MARKERS if m != "TH"]:
                o, p = permute(bl, scale, lambda mm, c=c: stat_pair(mm, "TH", c))
                comps.append(c)
                sts.append(o)
                ps.append(p)
            for c, o, p, q in zip(comps, sts, ps, bh(ps)):
                rows.append(
                    dict(
                        endpoint=endpoint,
                        scale=scale,
                        test="pairwise_TH_vs",
                        comparison=c,
                        statistic=o,
                        perm_p=p,
                        bh_q=q,
                        n_animals=n_an,
                    )
                )
            print(f"{endpoint:18s} {scale:5s} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir() / "baseline_ordering_tests.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print("\n=== BASELINE ORDERING, within-animal marker-label permutation ===")
    print(df.round(4).to_string(index=False))
    print(f"\nwritten {OUTPUT / 'baseline_ordering_tests.tsv'}")

    sp = sparsity()
    print("\n=== MEDIAN EVENTS PER STACK, untreated tissue ===")
    print(sp.round(1).to_string())
    print("\nRank position of each marker (contact, mean normalised rank across animals):")
    m = as_matrix(blocks("proximity_contact"), "rank")
    for k in sorted(MARKERS, key=lambda x: -float(np.nanmean(m[x]))):
        print(
            f"   {k:7s} mean rank {np.nanmean(m[k]):.2f}   median complete events/stack "
            f"{sp.loc[k, 'complete']:.0f}"
        )
    sp.to_csv(output_dir() / "baseline_marker_sparsity.tsv", sep="\t")


if __name__ == "__main__":
    main()
