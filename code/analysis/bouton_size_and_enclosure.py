# -*- coding: utf-8 -*-
"""Object-size and intensity summaries for Figure S6 and Table S15.

The analysis compares detected, contacted, completely included and distant
objects within the recorded marker and field strata. Volume-stratified intensity
ratios describe sensitivity to object-size composition. These observational
summaries do not establish degradation, a causal mediator or freedom from
detection bias."""

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

import pandas as pd
from scipy.stats import spearmanr

ROOT = PIPELINE
OUT = output_dir()
OBJ = ROOT / "data/derived/lps/uniform_standard/objects"
COLS = ["mean_intensity", "volume_um3", "contact_category", "complete_inclusion_eligible"]
MARKERS = ["TH", "ChaT", "vglut1", "vglut2", "vgat", "gad65"]
MOD = ["TH", "ChaT"]


def load(marker, condition=None, eligible_only=True):
    st = pd.read_csv(ROOT / "results/uniform_stack_summary.tsv", sep="\t")
    st = st[
        (st.dataset == "LPS")
        & (st.specification == "uniform_standard")
        & (st.marker.astype(str).str.lower() == marker.lower())
    ]
    if condition is not None:
        st = st[st.condition == condition]
    frames = []
    for _, r in st.iterrows():
        f = OBJ / f"{r.file_id}.parquet"
        require_file(f)
        o = pd.read_parquet(f, columns=COLS)
        o["file_id"] = r.file_id
        o["condition"] = r.condition
        frames.append(o)
    d = pd.concat(frames, ignore_index=True)
    if eligible_only:
        # the complete-inclusion analyses are restricted to boundary-clear objects; the contact
        # endpoint is not, so that filter must not be applied when asking what contact selects
        d = d[d.complete_inclusion_eligible].copy()
    d["enclosed"] = d.contact_category == "complete_inclusion"
    return d


def intensity_test(d):
    """Enclosed against free intensity, raw and within stack by volume quintile."""
    med = d.groupby("enclosed").mean_intensity.median()
    raw = float(med[True] / med[False])
    d = d.copy()
    d["vq"] = d.groupby("file_id").volume_um3.transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
    )
    g = (
        d.groupby(["file_id", "vq"])
        .apply(
            lambda x: pd.Series(
                {
                    "e": x.loc[x.enclosed, "mean_intensity"].median(),
                    "f": x.loc[~x.enclosed, "mean_intensity"].median(),
                    "n": int(x.enclosed.sum()),
                }
            ),
            include_groups=False,
        )
        .dropna()
    )
    g = g[g.n >= 3]
    ratios = g.e / g.f
    return dict(
        raw_ratio=raw,
        n_objects=int(len(d)),
        stratified_ratio=float(ratios.median()),
        stratified_iqr_low=float(ratios.quantile(0.25)),
        stratified_iqr_high=float(ratios.quantile(0.75)),
        n_strata=int(len(g)),
        median_volume_enclosed=float(d.loc[d.enclosed, "volume_um3"].median()),
        median_volume_free=float(d.loc[~d.enclosed, "volume_um3"].median()),
    )


def contact_selection(marker):
    """Does the contact endpoint select large or small objects, and bright or dim ones?

    The mediation argument for complete inclusion, that a brightness threshold conditions on size
    because enclosure selects small objects, does not automatically carry to the primary endpoint.
    Contact selects the other way, and this quantifies it so the claim traces to a file.
    """
    d = load(marker, condition="Control", eligible_only=False)
    con = d.contact_category != "no_contact"
    return dict(
        marker=marker,
        n_objects=int(len(d)),
        volume_ratio_contacted_over_not=float(
            d.loc[con, "volume_um3"].median() / d.loc[~con, "volume_um3"].median()
        ),
        intensity_ratio_contacted_over_not=float(
            d.loc[con, "mean_intensity"].median() / d.loc[~con, "mean_intensity"].median()
        ),
        n_contacted=int(con.sum()),
    )


def main():
    rows = []
    for mk in ("TH", "vglut1"):
        d = load(mk)
        r = intensity_test(d)
        r["marker"] = mk
        rows.append(r)
        print(
            f"{mk:7s} intensity enclosed/free: raw {r['raw_ratio']:.3f}, "
            f"within stack x volume quintile {r['stratified_ratio']:.3f} "
            f"({r['stratified_iqr_low']:.3f}-{r['stratified_iqr_high']:.3f}, "
            f"{r['n_strata']} strata)",
            flush=True,
        )
    intensity_summary = pd.DataFrame(rows)[
        [
            "marker",
            "n_objects",
            "raw_ratio",
            "stratified_ratio",
            "stratified_iqr_low",
            "stratified_iqr_high",
            "n_strata",
            "median_volume_enclosed",
            "median_volume_free",
        ]
    ]
    intensity_summary.to_csv(OUT / "enclosure_intensity_size.tsv", sep="\t", index=False)

    size = []
    for mk in MARKERS:
        d = load(mk, condition="Control")
        enc = d[d.enclosed]
        size.append(
            dict(
                marker=mk,
                transmitter_class="modulatory" if mk in MOD else "fast",
                n_objects=int(len(d)),
                median_volume_um3=float(d.volume_um3.median()),
                median_volume_enclosed_um3=float(enc.volume_um3.median()),
                percent_enclosed=float(100 * len(enc) / len(d)),
            )
        )
    S = pd.DataFrame(size)
    rho, p = spearmanr(S.median_volume_um3, S.percent_enclosed)
    S.to_csv(OUT / "bouton_size_by_marker.tsv", sep="\t", index=False)
    print("\n=== BOUTON SIZE BY MARKER, UNTREATED ===")
    print(S.round(4).to_string(index=False))
    print(
        f"\nmodulatory median volume {S[S.transmitter_class == 'modulatory'].median_volume_um3.mean():.4f} µm³"
        f"   fast {S[S.transmitter_class == 'fast'].median_volume_um3.mean():.4f} µm³"
    )
    print(f"volume against enclosure rate across the six markers: rho = {rho:+.3f}, P = {p:.3f}")
    C = pd.DataFrame([contact_selection(m) for m in ("TH", "vglut1")])
    C.to_csv(OUT / "contact_selection_size_intensity.tsv", sep="\t", index=False)
    print("\n=== WHAT THE CONTACT ENDPOINT SELECTS, UNTREATED ===")
    print(C.round(4).to_string(index=False))
    print(
        f"\nwritten {OUT / 'enclosure_intensity_size.tsv'}, bouton_size_by_marker.tsv "
        f"and contact_selection_size_intensity.tsv"
    )


if __name__ == "__main__":
    main()
