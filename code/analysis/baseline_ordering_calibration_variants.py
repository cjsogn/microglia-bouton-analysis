# -*- coding: utf-8 -*-
"""Variants of the calibrated-null simulation.

homog   every cell drawn from the SAME marker's bank (gad65) for that animal, so the six
        columns are exactly exchangeable. Machinery control: type-I MUST come out nominal.
matched each marker's total between-animal log SD is inflated to the value actually observed
        in the untreated data by adding an independent biological component. Reality check:
        the real data are more variance-homogeneous than pure torus sampling implies.
sampling identical to the primary run, repeated here so all three use one code path.
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
    output_dir,
)

import sys
import pickle
import numpy as np
import pandas as pd

# The simulation cache is regenerated output, not deposited data, so it lives beside the
# other regenerated tables and never mixes with the reported values.
SCRATCH = output_dir() / "_cache"
SCRATCH.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SCRATCH))
import exch_sim as E  # noqa: E402

NPERM = 10_000
import os

NSIM = int(os.environ.get("NSIM", 20_000))
SEED = int(os.environ.get("SEED2", 909090))
OBS_P = {  # the real permutation P values reported by baseline_ordering_test.py
    ("proximity_contact", "omnibus_rank"): 0.0040,
    ("proximity_contact", "contrast_rank"): 0.0022,
    ("proximity_contact", "omnibus_log"): 0.0008,
    ("proximity_contact", "contrast_log"): 0.0002,
    ("complete", "omnibus_rank"): 0.0315,
    ("complete", "contrast_rank"): 0.0045,
    ("complete", "omnibus_log"): 0.0108,
    ("complete", "contrast_log"): 0.0001,
}


def load():
    with open(SCRATCH / "banks.pkl", "rb") as fh:
        banks, present = pickle.load(fh)
    V = pd.read_csv(SCRATCH / "marker_null_variance.tsv", sep="\t")
    return banks, present, V


def prep(banks, V, mode):
    """returns z[ep][m][a] (standardised log draws) and target log SD per (ep, m)."""
    z, sd = {}, {}
    for ep in E.KEY:
        z[ep], sd[ep] = {}, {}
        for m in E.MARKERS:
            z[ep][m] = {}
            r = V[(V.endpoint == ep) & (V.marker == m)].iloc[0]
            src = "gad65" if mode == "homog" else m
            for a in E.ANIMALS:
                if a not in banks[ep][src]:
                    continue
                lv = np.log(banks[ep][src][a].astype(np.float64))
                z[ep][m][a] = ((lv - lv.mean()) / lv.std()).astype(np.float32)
            if mode == "sampling":
                sd[ep][m] = float(r.null_sd_log)
            elif mode == "homog":
                sd[ep][m] = float(V[(V.endpoint == ep) & (V.marker == "gad65")].iloc[0].null_sd_log)
            elif mode == "observed":
                sd[ep][m] = float(r.observed_between_animal_sd_log)
            else:  # matched
                sd[ep][m] = float(max(r.null_sd_log, r.observed_between_animal_sd_log))
    return z, sd


def worker(args):
    ep, mode, nsim, seed = args
    banks, present, V = load()
    z, sd = prep(banks, V, mode)
    rng = np.random.default_rng(seed)
    P = E.perm_index(NPERM, present, np.random.default_rng(seed + 13))
    keys = ["omnibus_rank", "contrast_rank", "omnibus_log", "contrast_log"]
    res = {k: np.empty(nsim) for k in keys}
    nb = len(z[ep]["gad65"]["kiv2"])
    for i in range(nsim):
        X = np.full((5, 6), np.nan)
        for ai, a in enumerate(E.ANIMALS):
            for m in E.MARKERS:
                c = E.COL[m]
                if not present[ai, c]:
                    continue
                s = sd[ep][m]
                zz = z[ep][m][a][rng.integers(0, nb)]
                # keep the shape of the real sampling distribution, set the scale to s,
                # and carry the Jensen offset that an unbiased ratio estimator produces
                X[ai, c] = np.exp(s * zz - 0.5 * s * s)
        r = E.run_test(X, present, P)
        for k in keys:
            res[k][i] = r[k]
    return ep, mode, res


def _require_banks():
    """The torus banks are built once by baseline_ordering_calibration.py and cached. Both variant
    scripts read that cache, so fail with an instruction rather than an IndexError if it is absent."""
    bp = SCRATCH / "banks.pkl"
    if not bp.exists():
        raise SystemExit(
            "Missing torus bank cache at " + str(bp) + ".\n"
            "Run baseline_ordering_calibration.py first; it builds the cache both variant scripts "
            "read. It takes several minutes."
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("homog", "sampling", "observed", "matched"))
    mode = parser.parse_args().mode
    _require_banks()
    if NSIM < 14:
        raise ValueError("NSIM must support all 14 recorded worker batches")
    from multiprocessing import Pool

    NCORE = 14
    per = NSIM // NCORE
    jobs = [
        (ep, mode, per, SEED + 1000 * c + (0 if ep == "complete" else 500))
        for ep in E.KEY
        for c in range(NCORE)
    ]
    with Pool(NCORE) as pool:
        out = pool.map(worker, jobs)
    agg = {}
    for ep, _, res in out:
        for k, v in res.items():
            agg.setdefault((ep, k), []).append(v)
    rows = []
    for (ep, k), lst in agg.items():
        p = np.concatenate(lst)
        obs = OBS_P[(ep, k)]
        rows.append(
            dict(
                endpoint=ep,
                test=k,
                n_sim=len(p),
                typeI_0p05=float(np.mean(p <= 0.05)),
                typeI_0p01=float(np.mean(p <= 0.01)),
                typeI_0p001=float(np.mean(p <= 0.001)),
                nominal_p=obs,
                calibrated_p=float((np.sum(p <= obs + 1e-12) + 1) / (len(p) + 1)),
            )
        )
    T = pd.DataFrame(rows).sort_values(["endpoint", "test"])
    T.to_csv(SCRATCH / f"typeI_{mode}.tsv", sep="\t", index=False)
    print(f"=== mode = {mode} ===")
    print(T.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    np.savez_compressed(
        SCRATCH / f"pvals_{mode}.npz",
        **{f"{ep}__{k}": np.concatenate(lst) for (ep, k), lst in agg.items()},
    )


if __name__ == "__main__":
    main()
