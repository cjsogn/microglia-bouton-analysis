# -*- coding: utf-8 -*-
"""Calibrated-null type-I error of the within-animal marker-label permutation test.

Step 1. For every (endpoint, marker, animal) in the untreated LPS cohort, build the exact
sampling distribution of the per-animal enrichment under the study's own torus null, by
drawing each stack's observed count from that stack's own 1,000 torus shifts and running the
SAME region-pooled estimator as estimate_enrichment.estimate()/per_animal().

Step 2. Simulate the full 5-animal x 6-marker table from those banks (so marker identity
carries no signal, but each marker keeps its real sampling variance) and run the SAME
permutation test as baseline_ordering_test.py. Report actual type-I error at 0.05 and 0.01.
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
import sys
import pickle
import numpy as np
import pandas as pd

ROOT = PIPELINE
V3 = output_dir()
# The simulation cache is regenerated output, not deposited data, so it lives beside the
# other regenerated tables and never mixes with the reported values.
SCRATCH = output_dir() / "_cache"
SCRATCH.mkdir(parents=True, exist_ok=True)

KEY = {"complete": "n_complete", "proximity_contact": "n_proximity_contact"}
MARKERS = ["TH", "ChaT", "vglut1", "vglut2", "vgat", "gad65"]  # order used by the test script
HIGH = ["TH", "ChaT"]
ANIMALS = ["kiv1", "kiv2", "kiv3", "kiv4", "kiv5"]
BANK = 60_000
NPERM = 10_000
NSIM = 20_000
SEED = 424242


# ---------------------------------------------------------------- step 1: banks
def stack_table():
    st = pd.read_csv(ROOT / "results/uniform_stack_summary.tsv", sep="\t")
    return st[
        (st.dataset == "LPS")
        & (st.specification == "uniform_standard")
        & (st.condition == "Control")
    ]


def build_banks(rng):
    """bank[endpoint][marker][animal] = array of BANK draws of the per-animal enrichment
    under the torus null (true enrichment exactly 1)."""
    st = stack_table()
    S = ROOT / "data/derived/lps/uniform_standard/torus_null/summaries"
    iterations = ROOT / "data/derived/lps/uniform_standard/torus_null/iterations"
    banks = {ep: {m: {} for m in MARKERS} for ep in KEY}
    obsr = {ep: {m: {} for m in MARKERS} for ep in KEY}  # real observed per-animal ratio
    nev = {ep: {m: [] for m in MARKERS} for ep in KEY}  # observed events per stack
    for m in MARKERS:
        sub = st[st.marker.astype(str).str.lower() == m.lower()]
        for a in ANIMALS:
            g = sub[sub.animal_id == a]
            regions, e_list, o_list, nulls = [], [], [], {}
            for _, r in g.iterrows():
                fs, fi = S / f"{r.file_id}.json", iterations / f"{r.file_id}.npz"
                require_file(fs)
                require_file(fi)
                j = json.loads(fs.read_text())
                z = np.load(fi)
                regions.append(r.region)
                e_list.append({ep: float(j[ep]["null_mean"]) for ep in KEY})
                o_list.append({ep: float(j[ep]["observed"]) for ep in KEY})
                nulls[len(regions) - 1] = {ep: z[KEY[ep]].astype(np.float64) for ep in KEY}
            if not regions:
                continue
            regions = np.asarray(regions)
            for ep in KEY:
                e = np.array([d[ep] for d in e_list])
                o = np.array([d[ep] for d in o_list])
                nev[ep][m].extend(o.tolist())
                ureg = np.unique(regions)
                R = len(ureg)
                # per-animal ratio = sum_s w_s * o_s  with w_s = 1/(R * E_region(s))
                w = np.zeros(len(e))
                for u in ureg:
                    k = regions == u
                    w[k] = 1.0 / (R * e[k].sum())
                obsr[ep][m][a] = float(w @ o)
                acc = np.zeros(BANK)
                for s in range(len(e)):
                    arr = nulls[s][ep]
                    acc += w[s] * arr[rng.integers(0, len(arr), BANK)]
                banks[ep][m][a] = acc.astype(np.float32)
    return banks, obsr, nev


# ---------------------------------------------------------------- permutation machinery
COL = {m: i for i, m in enumerate(MARKERS)}
HI_IDX = [COL[m] for m in HIGH]
LO_IDX = [COL[m] for m in MARKERS if m not in HIGH]


def perm_index(nperm, present, rng):
    """flat index array (nperm, 5, 6) into a flattened 5x6 value matrix.

    present[a] = boolean array over the 6 marker columns. Marker column c of animal a receives
    the value that sits in one of that animal's present slots, under a random bijection. Absent
    columns always point at a slot holding 0 (they contribute nothing and are not counted).
    """
    P = np.empty((nperm, 5, 6), np.int64)
    for a in range(5):
        cols = np.where(present[a])[0]
        base = a * 6
        # absent columns point at their own (zero-filled) slot
        for c in range(6):
            P[:, a, c] = base + c
        for b in range(nperm):
            P[b, a, cols] = base + rng.permutation(cols)
    return P.reshape(nperm, 30)


def stats_from_matrix(X, present):
    """observed (friedman, contrast) for a 5x6 matrix with nan in absent cells."""
    n = present.sum(0).astype(float)
    Z = np.where(present, np.nan_to_num(X, nan=0.0), 0.0)
    mu = Z.sum(0) / n
    fr = float(((mu - mu.mean()) ** 2).sum())
    ct = float(mu[HI_IDX].mean() - mu[LO_IDX].mean())
    return fr, ct, mu


def null_stats(X, present, P):
    """permutation null of both statistics. P is the flat index array."""
    n = present.sum(0).astype(float)
    flat = np.where(present, np.nan_to_num(X, nan=0.0), 0.0).ravel()
    V = flat[P].reshape(len(P), 5, 6)
    mu = V.sum(1) / n  # (nperm, 6)
    fr = ((mu - mu.mean(1, keepdims=True)) ** 2).sum(1)
    ct = mu[:, HI_IDX].mean(1) - mu[:, LO_IDX].mean(1)
    return fr, ct


def normalised_ranks(v):
    v = np.asarray(v, float)
    order = v.argsort()
    r = np.empty(len(v), float)
    r[order] = np.arange(len(v), dtype=float)
    for u in np.unique(v):
        m = v == u
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r / (len(v) - 1)


def rank_matrix(X, present):
    R = np.full((5, 6), np.nan)
    for a in range(5):
        cols = np.where(present[a])[0]
        R[a, cols] = normalised_ranks(X[a, cols])
    return R


def run_test(X, present, P, eps=1e-12):
    """returns dict of the four P values, mirroring baseline_ordering_test.permute()."""
    out = {}
    for scale, M in (("rank", rank_matrix(X, present)), ("log", np.log(X))):
        fr_o, ct_o, _ = stats_from_matrix(M, present)
        fr_n, ct_n = null_stats(M, present, P)
        nb = len(P)
        out[f"omnibus_{scale}"] = (np.sum(fr_n >= fr_o - eps) + 1) / (nb + 1)
        out[f"contrast_{scale}"] = (np.sum(np.abs(ct_n) >= abs(ct_o) - eps) + 1) / (nb + 1)
    return out


# ---------------------------------------------------------------- worker
def worker(args):
    ep, nsim, seed, banks_path, animal_sd = args
    with open(banks_path, "rb") as fh:
        banks, present = pickle.load(fh)
    B = banks[ep]
    rng = np.random.default_rng(seed)
    P = perm_index(NPERM, present, np.random.default_rng(seed + 7))
    keys = ["omnibus_rank", "contrast_rank", "omnibus_log", "contrast_log"]
    res = {k: np.empty(nsim) for k in keys}
    nbad = 0
    for i in range(nsim):
        X = np.full((5, 6), np.nan)
        for ai, a in enumerate(ANIMALS):
            off = rng.normal(0.0, animal_sd) if animal_sd else 0.0
            for m in MARKERS:
                if not present[ai, COL[m]]:
                    continue
                v = B[m][a][rng.integers(0, BANK)]
                X[ai, COL[m]] = v * np.exp(off)
        if not np.all(np.isfinite(np.log(np.where(present, X, 1.0)))):
            nbad += 1
            X = np.where(present & (X <= 0), 1e-6, X)
        r = run_test(X, present, P)
        for k in keys:
            res[k][i] = r[k]
    return ep, res, nbad


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    animal_sd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    rng = np.random.default_rng(SEED)
    bp = SCRATCH / "banks.pkl"
    if bp.exists():
        with open(bp, "rb") as fh:
            banks, present = pickle.load(fh)
        with open(SCRATCH / "banks_meta.pkl", "rb") as fh:
            obsr, nev = pickle.load(fh)
    else:
        banks, obsr, nev = build_banks(rng)
        present = np.ones((5, 6), bool)
        present[ANIMALS.index("kiv1"), COL["vglut2"]] = False
        with open(bp, "wb") as fh:
            pickle.dump((banks, present), fh, protocol=4)
        with open(SCRATCH / "banks_meta.pkl", "wb") as fh:
            pickle.dump((obsr, nev), fh, protocol=4)

    # ---- variance report
    rows = []
    for ep in KEY:
        for m in MARKERS:
            sds, mns = [], []
            for a in ANIMALS:
                if a in banks[ep][m]:
                    v = banks[ep][m][a]
                    mns.append(v.mean())
                    sds.append(np.log(v[v > 0]).std())
            real = np.array([obsr[ep][m][a] for a in ANIMALS if a in obsr[ep][m]])
            rows.append(
                dict(
                    endpoint=ep,
                    marker=m,
                    median_events_per_stack=float(np.median(nev[ep][m])),
                    null_mean_enrichment=float(np.mean(mns)),
                    null_sd_log=float(np.mean(sds)),
                    observed_between_animal_sd_log=float(np.std(np.log(real), ddof=1)),
                    observed_mean_enrichment=float(np.mean(real)),
                )
            )
    V = pd.DataFrame(rows)
    V.to_csv(SCRATCH / "marker_null_variance.tsv", sep="\t", index=False)
    print("=== per-animal sampling variance under the torus null ===")
    print(V.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- simulation, 14 cores
    from multiprocessing import Pool

    NCORE = 14
    per = NSIM // NCORE
    jobs = []
    for ep in KEY:
        for c in range(NCORE):
            jobs.append(
                (ep, per, SEED + 1000 * c + (0 if ep == "complete" else 500), str(bp), animal_sd)
            )
    with Pool(NCORE) as pool:
        out = pool.map(worker, jobs)
    agg = {}
    bad = 0
    for ep, res, nbad in out:
        bad += nbad
        for k, v in res.items():
            agg.setdefault((ep, k), []).append(v)
    rows = []
    for (ep, k), lst in agg.items():
        p = np.concatenate(lst)
        rows.append(
            dict(
                endpoint=ep,
                test=k,
                n_sim=len(p),
                typeI_0p05=float(np.mean(p <= 0.05)),
                typeI_0p01=float(np.mean(p <= 0.01)),
                typeI_0p10=float(np.mean(p <= 0.10)),
                typeI_0p001=float(np.mean(p <= 0.001)),
                median_p=float(np.median(p)),
            )
        )
    T = pd.DataFrame(rows).sort_values(["endpoint", "test"])
    T.to_csv(SCRATCH / f"typeI_{tag}.tsv", sep="\t", index=False)
    print(f"\n=== realised type-I error under the calibrated null (animal_sd={animal_sd}) ===")
    print(T.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nnonpositive-ratio simulations repaired: {bad}")
    # store the raw p-values for the honest-P mapping
    np.savez_compressed(
        SCRATCH / f"pvals_{tag}.npz",
        **{f"{ep}__{k}": np.concatenate(lst) for (ep, k), lst in agg.items()},
    )


if __name__ == "__main__":
    main()
