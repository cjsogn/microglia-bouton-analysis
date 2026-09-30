"""Reproduce the retained exploratory morphology likelihood-ratio screens.

Each model uses cell measurements, fixed condition/region terms and an animal
random intercept. Maximum likelihood, scaling and formulas match the retained
regional analysis. These asymptotic screens are distinct from the animal-level
exact tests. Fit warnings and convergence flags are exported for review.

Run with BLAS threads limited to one when using multiple worker processes.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mg_paths import MORPHOLOGY, output_dir

META = {"cell_id", "animal", "group", "region", "zstack", "batch", "fov"}
GROUP_LABEL = {"kiv": "Control", "liv": "LPS_72h", "livl": "LPS_60d"}
MODELS = {
    "region": ("y ~ C(condition) + C(region)", "y ~ C(condition)", 4),
    "treatment": ("y ~ C(condition) + C(region)", "y ~ C(region)", 2),
    "interaction": ("y ~ C(condition) * C(region)", "y ~ C(condition) + C(region)", 8),
    "region_intensity_only": ("y ~ C(condition) + C(region)", "y ~ C(condition)", 4),
}


def fit_one(job):
    family, feature, data, repair = job
    full_formula, reduced_formula, degrees = MODELS[family]
    data = data.dropna().rename(columns={feature: "y"})
    data["y"] = (data.y - data.y.mean()) / data.y.std()
    record = dict(
        family=family,
        feature=feature,
        n_cells=len(data),
        n_animals=data.animal.nunique(),
        full_formula=full_formula,
        reduced_formula=reduced_formula,
        df_difference=degrees,
    )
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        try:
            models = [
                smf.mixedlm(formula, data, groups=data.animal)
                for formula in (full_formula, reduced_formula)
            ]
            results = [model.fit(reml=False) for model in models]
            record["original_full_converged"] = bool(results[0].converged)
            record["original_reduced_converged"] = bool(results[1].converged)
            record["original_p"] = float(chi2.sf(2 * (results[0].llf - results[1].llf), degrees))
            record["optimizer_check"] = ""
            if repair and not all(result.converged for result in results):
                # Same likelihood and model, with explicit numerical convergence
                # checks. This is not a new covariance structure or test.
                repaired = []
                for model, original in zip(models, results):
                    candidates = [original] if original.converged else []
                    for method in ("powell", "bfgs"):
                        candidate = model.fit(reml=False, method=method, maxiter=2000)
                        if candidate.converged and np.isfinite(candidate.llf):
                            candidates.append(candidate)
                    if not candidates:
                        raise RuntimeError("No converged maximum-likelihood fit")
                    repaired.append(max(candidates, key=lambda result: result.llf))
                results = repaired
                record["optimizer_check"] = (
                    "maximum converged likelihood: default, powell, bfgs; maxiter=2000 for retries"
                )
            full, reduced = results
            statistic = 2 * (full.llf - reduced.llf)
            record.update(
                p=float(chi2.sf(statistic, degrees)),
                lr_statistic=statistic,
                full_log_likelihood=full.llf,
                reduced_log_likelihood=reduced.llf,
                full_converged=bool(full.converged),
                reduced_converged=bool(reduced.converged),
                error="",
            )
        except Exception as error:
            record.update(
                p=np.nan,
                full_converged=False,
                reduced_converged=False,
                error=f"{type(error).__name__}: {error}",
            )
    record["warnings"] = " | ".join(dict.fromkeys(str(w.message) for w in messages))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("MG_WORKERS", "14")))
    parser.add_argument(
        "--check-optimizers",
        action="store_true",
        help="Refit failed comparisons under the same model with additional optimizers",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    source = MORPHOLOGY / "output/features_3d_full_morphomics_v6_final.csv"
    data = pd.read_csv(source)
    data["condition"] = data.group.map(GROUP_LABEL)
    if data.condition.isna().any():
        raise ValueError("Unknown condition in feature input")
    intensity = [c for c in data if c.startswith("iba1_") or "intensity" in c]
    shape = [
        c
        for c in data
        if c not in META and c not in intensity and c != "condition" and data[c].std() > 0
    ]
    jobs = [
        (family, feature, data[["condition", "region", "animal", feature]], args.check_optimizers)
        for family in MODELS
        for feature in (intensity if family == "region_intensity_only" else shape)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(fit_one, jobs))
    fits = pd.DataFrame(records)
    out = output_dir("morphology", "exploratory_mixed_models")
    fits.to_csv(out / "fit_diagnostics.tsv", sep="\t", index=False)
    # No failed feature may silently disappear from a multiple-testing family.
    if fits.p.isna().any() or (~np.isfinite(fits.lr_statistic)).any():
        raise RuntimeError(f"Incomplete fits. Inspect {out / 'fit_diagnostics.tsv'}")
    for family in MODELS:
        selected = fits.loc[fits.family.eq(family), ["feature", "p"]].copy()
        selected["q"] = multipletests(selected.p, method="fdr_bh")[1]
        selected.sort_values("q").to_csv(out / f"lmm_{family}.tsv", sep="\t", index=False)
    bad = ~fits.full_converged | ~fits.reduced_converged | fits.lr_statistic.lt(-1e-8)
    if bad.any():
        raise RuntimeError(
            f"{bad.sum()} comparisons need convergence/nesting review. "
            f"Numerical outputs are provisional. Inspect {out / 'fit_diagnostics.tsv'}"
        )
    print(f"Completed {len(fits)} mixed-model comparisons with fit diagnostics.")


if __name__ == "__main__":
    main()
