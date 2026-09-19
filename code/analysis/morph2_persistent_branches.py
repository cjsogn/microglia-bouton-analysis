# -*- coding: utf-8 -*-
"""Analyse the reported terminal-branch outcome using the final eligible cells.

The threshold is the 90th percentile of control branch persistence. The primary
animal-label test recomputes this threshold in every allocation. Fixed-threshold
sweeps are descriptive sensitivity analyses. A control-derived threshold does
not establish prospective selection or independently validate branch identity.

Cells are averaged within field and fields within animal. The loader checks
that every eligible cell has barcode records. A cell with records but no branch
above threshold contributes zero. Missing barcode records are an error, not zero."""

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
    OUTPUT,
    MORPHOLOGY,
    output_dir,
    require_file,
)

import itertools

import numpy as np
import pandas as pd

# Raw stacks, segmentation masks and the derived per-cell tables live outside the
# code tree because of their size. One constant points at them.
MORPH = MORPHOLOGY
OUT = output_dir("morphology", "region_analysis")
OUT.mkdir(parents=True, exist_ok=True)
COND = {"kiv": "Control", "liv": "LPS_72h", "livl": "LPS_60d"}
ORDER = ["Control", "LPS_72h", "LPS_60d"]
RLAB = {"ca1": "CA1", "dg": "DG", "cortex2": "Ctx L2/3", "cortex4": "Ctx L4/5", "SLM": "SLM"}
CONTROL_PERCENTILE = 90


def load():
    barcode_path = OUTPUT / "morphology" / "topology_barcodes_v6.csv"
    if not barcode_path.is_file():
        barcode_path = MORPH / "output" / "topology_barcodes_v6.csv"
    b = pd.read_csv(require_file(barcode_path))
    f = pd.read_csv(MORPH / "output" / "features_3d_full_morphomics_v6_final.csv")
    if f.duplicated(["fov", "cell_id"]).any():
        raise ValueError("Duplicate eligible morphology cell keys")
    if f[["fov", "cell_id", "animal", "group", "region"]].isna().any().any():
        raise ValueError("Missing eligible-cell metadata")
    if not set(f.group).issubset(COND):
        raise ValueError("Unknown morphology condition")
    required = set(zip(f.fov, f.cell_id))
    present = set(zip(b.fov, b.label))
    if missing := required - present:
        raise ValueError(f"Eligible cells missing barcode records: {sorted(missing)[:10]}")
    if not np.isfinite(b.persistence).all() or (b.persistence <= 0).any():
        raise ValueError("Barcodes must contain finite positive persistence values")
    m = b.merge(
        f[["fov", "cell_id", "animal", "group", "region"]],
        left_on=["fov", "label"],
        right_on=["fov", "cell_id"],
        validate="many_to_one",
    )
    m["cond"] = m.group.map(COND)
    return m


def per_animal(m, thr):
    """Bars above threshold per cell, averaged within field then within animal."""
    cell = (
        m.assign(hit=m.persistence > thr)
        .groupby(["fov", "label", "animal", "cond", "region"])
        .hit.sum()
        .reset_index()
    )
    fov = cell.groupby(["animal", "cond", "region", "fov"]).hit.mean().reset_index()
    return cell, fov.groupby(["animal", "cond"]).hit.mean().reset_index()


def exact_contrast_full_procedure(m, group, reference="Control", pct=None):
    """Exact test that repeats the WHOLE procedure inside every allocation.

    The threshold is the 90th percentile of the animals labelled control, so it is itself a
    function of the labels. Permuting labels while holding the threshold at its observed value
    gives a P value that is exact only conditional on that threshold, not for the procedure as
    performed. This recomputes the percentile from each allocation's pseudo-control set, rebins
    every branch against it, and re-derives the statistic, which is the exact test of the sharp
    null for the full label-dependent pipeline.

    It also changes the attainable floor. Under a fixed threshold the statistic is antisymmetric
    under swapping the labels, so a split and its complement are always equally extreme and the
    floor at five against five is 2/252. Recomputing breaks that symmetry, because the complement
    allocation derives a different threshold, so 1/252 becomes attainable.
    """
    pct = CONTROL_PERCENTILE if pct is None else pct
    sub = m[m.cond.isin([reference, group])]
    animals = sorted(sub.animal.unique())
    lab = {a: sub[sub.animal == a].cond.iloc[0] for a in animals}
    ctrl = [a for a in animals if lab[a] == reference]
    trt = [a for a in animals if lab[a] == group]

    obs_thr = float(np.percentile(sub[sub.animal.isin(ctrl)].persistence, pct))
    counts = per_animal(sub, obs_thr)[1].set_index("animal").hit
    obs = counts[trt].mean() - counts[ctrl].mean()

    hit = tot = 0
    for combo in itertools.combinations(animals, len(ctrl)):
        pc = list(combo)
        pt = [a for a in animals if a not in combo]
        thr = float(np.percentile(sub[sub.animal.isin(pc)].persistence, pct))
        c = per_animal(sub, thr)[1].set_index("animal").hit
        if abs(c[pt].mean() - c[pc].mean()) >= abs(obs) - 1e-12:
            hit += 1
        tot += 1
    return obs, hit / tot, 1 / tot


def exact_contrast(values, labels, group, reference="Control"):
    x = values[labels == group]
    y = values[labels == reference]
    pool = np.concatenate([x, y])
    obs = x.mean() - y.mean()
    hit = tot = 0
    for idx in itertools.combinations(range(len(pool)), len(x)):
        mask = np.zeros(len(pool), bool)
        mask[list(idx)] = True
        if abs(pool[mask].mean() - pool[~mask].mean()) >= abs(obs) - 1e-12:
            hit += 1
        tot += 1
    # Smallest attainable P. With equal group sizes the observed split and its complement are both
    # enumerated, so the floor is 2/tot; with unequal sizes only the observed split appears and the
    # floor is 1/tot. Getting this wrong makes a result look capped when it is not.
    floor = (2 if len(x) == len(y) else 1) / tot
    return obs / np.std(pool, ddof=1), hit / tot, floor


def region_tests(fov, rng, n_perm=20000):
    """Region main effect and region-by-treatment interaction on persistent branch counts.

    Region is measured within every animal, so the null of no region effect is tested by permuting
    region labels inside each animal; the animal is never broken apart. Two statistics are reported
    because they answer slightly different questions: an F-type statistic, which is sensitive to any
    spread among the five region means, and the range, which is sensitive to one region standing
    apart. Treatment is a between-animal factor, so the interaction is tested instead by permuting
    condition labels over animals, enumerated exactly.
    """
    per = fov.groupby(["animal", "cond", "region"]).hit.mean().reset_index()
    w = per.pivot_table(index="animal", columns="region", values="hit")
    x = w.to_numpy()

    def f_type(mat):
        mu = mat.mean(0)
        return float(((mu - mu.mean()) ** 2).sum() / mat.var(0, ddof=1).mean())

    obs_f, obs_r = f_type(x), float(x.mean(0).max() - x.mean(0).min())
    nf = nr = 0
    for _ in range(n_perm):
        sh = np.array([rng.permutation(row) for row in x])
        nf += f_type(sh) >= obs_f - 1e-12
        nr += (sh.mean(0).max() - sh.mean(0).min()) >= obs_r - 1e-12
    p_f, p_r = (nf + 1) / (n_perm + 1), (nr + 1) / (n_perm + 1)

    # interaction: residual from the additive region + condition model, condition labels exchanged
    cond = w.index.map(per.set_index("animal").cond.groupby(level=0).first()).to_numpy()

    def inter(lab):
        cm = np.array([x[lab == c].mean(0) for c in ORDER])
        return float(((cm - cm.mean(0) - cm.mean(1)[:, None] + cm.mean()) ** 2).sum())

    obs_i = inter(cond)
    counts = [int((cond == c).sum()) for c in ORDER]
    hit = tot = 0
    for i in itertools.combinations(range(len(cond)), counts[0]):
        rest = [k for k in range(len(cond)) if k not in i]
        for j in itertools.combinations(rest, counts[1]):
            lab = np.array([ORDER[2]] * len(cond), dtype=object)
            lab[list(i)] = ORDER[0]
            lab[list(j)] = ORDER[1]
            hit += inter(lab) >= obs_i - 1e-9
            tot += 1
    return dict(
        region_f=obs_f,
        region_f_p=p_f,
        region_range=obs_r,
        region_range_p=p_r,
        region_mean=float(x.mean()),
        interaction=obs_i,
        interaction_p=hit / tot,
        interaction_assignments=tot,
    )


def main():
    m = load()
    ctrl = m[m.cond == "Control"].persistence
    thr = float(np.percentile(ctrl, CONTROL_PERCENTILE))
    print(
        f"{len(m):,} branches from {m.groupby(['fov', 'label']).ngroups} cells, "
        f"{m.animal.nunique()} animals"
    )
    print(f"threshold = {CONTROL_PERCENTILE}th percentile of CONTROL branches = {thr:.2f} um\n")

    cell, a = per_animal(m, thr)
    print(
        f"persistent branches per cell: {cell.hit.mean():.1f} "
        f"(control {cell[cell.cond == 'Control'].hit.mean():.1f})"
    )
    rows = []
    for g in ("LPS_72h", "LPS_60d"):
        d, p_cond, floor_cond = exact_contrast(a.hit.values, a.cond.values, g)
        raw, p_full, floor_full = exact_contrast_full_procedure(m, g)
        rows.append(
            dict(
                contrast=f"{g} vs Control",
                delta_sd=d,
                raw_diff_branches_per_cell=raw,
                exact_p=p_full,
                floor=floor_full,
                exact_p_conditional_on_observed_threshold=p_cond,
                floor_conditional=floor_cond,
            )
        )
        print(
            f"  {g:8s} {raw:+.2f} branches/cell ({d:+.2f} std. diff.), "
            f"exact P = {p_full:.4f} (floor {floor_full:.4f}); "
            f"conditional-on-threshold P = {p_cond:.4f}"
        )
    pd.DataFrame(rows).to_csv(OUT / "persistent_branches_treatment.tsv", sep="\t", index=False)

    # ---- is it crowding? cells per field rises 28 % at 72 h ------------------------------
    cpf = cell.groupby(["animal", "cond", "fov"]).size().rename("cpf").reset_index()
    fov = (
        cell.groupby(["animal", "cond", "fov"])
        .hit.mean()
        .reset_index()
        .merge(cpf, on=["animal", "cond", "fov"])
    )
    slope = np.polyfit(fov.cpf, fov.hit, 1)
    fov["adj"] = fov.hit - np.polyval(slope, fov.cpf)
    aa = fov.groupby(["animal", "cond"]).adj.mean().reset_index()
    print("\nafter adjusting for cells per field:")
    rows = []
    for g in ("LPS_72h", "LPS_60d"):
        d, p, _ = exact_contrast(aa.adj.values, aa.cond.values, g)
        rows.append(dict(contrast=f"{g} vs Control", delta_sd=d, exact_p=p))
        print(f"  {g:8s} {d:+.2f} SD, exact P = {p:.4f}")
    pd.DataFrame(rows).to_csv(
        OUT / "persistent_branches_density_adjusted.tsv", sep="\t", index=False
    )

    # ---- region and region x treatment ---------------------------------------------------
    fovr = cell.groupby(["animal", "cond", "region", "fov"]).hit.mean().reset_index()
    pr = fovr.groupby(["animal", "cond", "region"]).hit.mean().reset_index()
    rt = region_tests(fovr, np.random.default_rng(20260823))
    w = pr.pivot_table(index="animal", columns="region", values="hit")
    print("\nregion, region labels permuted within animal:")
    print(
        f"  F-type {rt['region_f']:.2f}, P = {rt['region_f_p']:.4f}"
        f"   range {rt['region_range']:.2f} branches, P = {rt['region_range_p']:.4f}"
        f"   (grand mean {rt['region_mean']:.1f})"
    )
    print("  " + "  ".join(f"{RLAB[r]} {w[r].mean():.1f}" for r in w.columns))
    print(
        f"region x treatment, condition labels permuted over animals: "
        f"P = {rt['interaction_p']:.4f} over {rt['interaction_assignments']:,} assignments"
    )
    pr.to_csv(OUT / "persistent_branches_by_region.tsv", sep="\t", index=False)
    pd.DataFrame([rt]).to_csv(OUT / "persistent_branches_region_tests.tsv", sep="\t", index=False)

    # ---- the sweep is the robustness evidence, so it ships with the result ---------------
    sweep = []
    for t in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]:
        _, at = per_animal(m, t)
        d72, p72, _ = exact_contrast(at.hit.values, at.cond.values, "LPS_72h")
        d60, p60, _ = exact_contrast(at.hit.values, at.cond.values, "LPS_60d")
        sweep.append(dict(threshold_um=t, delta_72h=d72, p_72h=p72, delta_60d=d60, p_60d=p60))
    pd.DataFrame(sweep).to_csv(
        OUT / "persistent_branches_threshold_sweep.tsv", sep="\t", index=False
    )
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
