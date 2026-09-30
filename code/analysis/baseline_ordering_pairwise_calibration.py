# -*- coding: utf-8 -*-
"""(1) calibrated P for the primary pure-sampling run, (2) the mechanism, (3) pairwise tests."""

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
import numpy as np
import pandas as pd

# The simulation cache is regenerated output, not deposited data, so it lives beside the
# other regenerated tables and never mixes with the reported values.
SCRATCH = output_dir() / "_cache"
SCRATCH.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SCRATCH))
import exch_sim as E  # noqa: E402
from exch_variants import load, prep, NPERM  # noqa: E402

NSIM = 20_000
SEED = 771177
PAIRS = [m for m in E.MARKERS if m != "TH"]
OBS_PAIR = {  # from data/derived/baseline_ordering_tests.tsv
    ("proximity_contact", "rank"): dict(
        ChaT=0.7023864880675597,
        vglut1=0.31841840790796044,
        vglut2=0.0019099904500477498,
        vgat=0.1370643146784266,
        gad65=0.4457527712361438,
    ),
    ("proximity_contact", "log"): dict(
        ChaT=0.8827555862220688,
        vglut1=0.1443442782786086,
        vglut2=0.00035999820000899997,
        vgat=0.043964780176099116,
        gad65=0.0859395703021485,
    ),
    ("complete", "rank"): dict(
        ChaT=0.9561452192739036,
        vglut1=0.4471477642611787,
        vglut2=0.013589932050339749,
        vgat=0.033119834400828,
        gad65=0.24506377468112658,
    ),
    ("complete", "log"): dict(
        ChaT=0.9380903095484523,
        vglut1=0.10215448922755387,
        vglut2=0.037159814200928995,
        vgat=0.003924980375098124,
        gad65=0.060844695776521115,
    ),
}


def run_pairs(X, present, P, eps=1e-12):
    out = {}
    for scale, M in (("rank", E.rank_matrix(X, present)), ("log", np.log(X))):
        n = present.sum(0).astype(float)
        flat = np.where(present, np.nan_to_num(M, nan=0.0), 0.0)
        mu_o = flat.sum(0) / n
        V = flat.ravel()[P].reshape(len(P), 5, 6)
        mu = V.sum(1) / n
        for c in PAIRS:
            o = mu_o[E.COL["TH"]] - mu_o[E.COL[c]]
            s = mu[:, E.COL["TH"]] - mu[:, E.COL[c]]
            out[f"TH_vs_{c}_{scale}"] = (np.sum(np.abs(s) >= abs(o) - eps) + 1) / (len(P) + 1)
    return out


def worker(args):
    ep, mode, nsim, seed = args
    banks, present, V = load()
    z, sd = prep(banks, V, mode)
    rng = np.random.default_rng(seed)
    P = E.perm_index(NPERM, present, np.random.default_rng(seed + 13))
    keys = [f"TH_vs_{c}_{s}" for c in PAIRS for s in ("rank", "log")]
    res = {k: np.empty(nsim) for k in keys}
    rank_acc, nb = [], len(z[ep]["gad65"]["kiv2"])
    for i in range(nsim):
        X = np.full((5, 6), np.nan)
        for ai, a in enumerate(E.ANIMALS):
            for m in E.MARKERS:
                c = E.COL[m]
                if not present[ai, c]:
                    continue
                s = sd[ep][m]
                X[ai, c] = np.exp(s * z[ep][m][a][rng.integers(0, nb)] - 0.5 * s * s)
        R = E.rank_matrix(X, present)
        rank_acc.append(np.where(present, R, np.nan).sum(0) / present.sum(0))
        r = run_pairs(X, present, P)
        for k in keys:
            res[k][i] = r[k]
    return ep, res, np.asarray(rank_acc)


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
    agg, rk = {}, {}
    for ep, res, ra in out:
        rk.setdefault(ep, []).append(ra)
        for k, v in res.items():
            agg.setdefault((ep, k), []).append(v)
    rows = []
    for (ep, k), lst in agg.items():
        p = np.concatenate(lst)
        c, scale = k.replace("TH_vs_", "").rsplit("_", 1)
        obs = OBS_PAIR[(ep, scale)][c]
        rows.append(
            dict(
                endpoint=ep,
                scale=scale,
                comparison=c,
                n_sim=len(p),
                typeI_0p05=float(np.mean(p <= 0.05)),
                typeI_0p01=float(np.mean(p <= 0.01)),
                nominal_p=obs,
                calibrated_p=float((np.sum(p <= obs + 1e-12) + 1) / (len(p) + 1)),
            )
        )
    T = pd.DataFrame(rows).sort_values(["endpoint", "scale", "comparison"])
    T.to_csv(SCRATCH / f"typeI_pairs_{mode}.tsv", sep="\t", index=False)
    print(f"=== pairwise TH vs each comparator, mode = {mode} ===")
    print(T.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== mechanism: distribution of each marker's MEAN normalised rank ===")
    banks, present, _ = load()
    Pn = E.perm_index(200_000, present, np.random.default_rng(3))
    for ep in E.KEY:
        A = np.concatenate(rk[ep])
        # permutation reference: mean rank distribution when labels are exchangeable
        canon = np.full((5, 6), np.nan)
        for a in range(5):
            cols = np.where(present[a])[0]
            canon[a, cols] = np.arange(len(cols)) / (len(cols) - 1)
        flat = np.where(present, np.nan_to_num(canon, nan=0.0), 0.0).ravel()
        Vp = flat[Pn].reshape(len(Pn), 5, 6).sum(1) / present.sum(0)
        print(f"\n  {ep}")
        print(
            f"    {'marker':8s} {'null mean':>10s} {'null SD':>9s} {'perm SD':>9s} {'SD ratio':>9s}"
        )
        for m in E.MARKERS:
            c = E.COL[m]
            print(
                f"    {m:8s} {A[:, c].mean():10.4f} {A[:, c].std():9.4f} "
                f"{Vp[:, c].std():9.4f} {A[:, c].std() / Vp[:, c].std():9.2f}"
            )


if __name__ == "__main__":
    main()
