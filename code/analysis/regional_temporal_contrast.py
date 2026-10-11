"""Regional TH temporal estimates and profile test.

Independent animals, not longitudinal pairs. Per-region tests call the existing
exact allocation/confidence-set implementation without changing its grid or
statistic. The profile test permutes whole region vectors after within-animal
centring on the log scale, removing any common multiplicative time effect.
All 126 allocations are enumerated. This is exploratory, not preregistered.
"""

from pathlib import Path
import sys
import itertools
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mg_paths import SOURCE_DATA, PIPELINE, output_dir
from timepoint_contrast import exact
from regional_tests import per_animal_region, class_contrast

REGIONS = ["cortex_L2", "cortex_L4", "CA1", "SLM", "DG"]
MARKERS = ["TH", "ChaT", "vglut1", "vglut2", "vgat", "gad65"]


def temporal_profile_test(log_values, is_late):
    """Exact whole-animal allocation test of the centred regional log profile.

    The null is exchangeability of centred profiles between the two groups.
    Subtracting each animal's regional mean removes a common time main effect.
    The statistic is the sum of squared differences between group mean profiles.
    """
    x = np.asarray(log_values, dtype=float)
    g = np.asarray(is_late, dtype=bool)
    if x.ndim != 2 or len(x) != len(g) or not np.isfinite(x).all():
        raise ValueError("Complete finite animal-by-region matrix required")
    if not 0 < g.sum() < len(g):
        raise ValueError("Both independent time groups required")
    centred = x - x.mean(axis=1, keepdims=True)

    def statistic(mask):
        return float(np.sum((centred[mask].mean(0) - centred[~mask].mean(0)) ** 2))

    observed = statistic(g)
    values = []
    for ids in itertools.combinations(range(len(g)), int(g.sum())):
        mask = np.zeros(len(g), dtype=bool)
        mask[list(ids)] = True
        values.append(statistic(mask))
    hits = int(np.sum(np.asarray(values) >= observed - 1e-12))
    return dict(
        statistic=observed,
        exact_p=hits / len(values),
        n_extreme=hits,
        n_allocations=len(values),
        n_72h=int((~g).sum()),
        n_60d=int(g.sum()),
        n_regions=x.shape[1],
    )


def bootstrap_summary(values, log=False):
    """Exhaustive percentile bootstrap of independent animal summaries."""
    v = np.asarray(values, float)
    if v.ndim != 1 or len(v) < 2 or not np.isfinite(v).all():
        raise ValueError("At least two finite animal values are required")
    if log and (v <= 0).any():
        raise ValueError("Log bootstrap requires positive animal values")
    if log:
        v = np.log(v)
    indices = np.asarray(list(itertools.product(range(len(v)), repeat=len(v))))
    means = v[indices].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    result = np.asarray([v.mean(), lo, hi])
    if log:
        result = np.exp(result)
    return dict(
        estimate=result[0],
        ci_low=result[1],
        ci_high=result[2],
        n_animals=len(v),
        n_bootstrap_resamples=len(indices),
    )


def regional_marker_summaries(animals):
    """Descriptive arithmetic means and exhaustive intervals for every condition.

    Input rows are pooled observed/expected counts for one animal, marker and
    region. Animals are resampled within each condition-region-marker cell.
    These display intervals are not treatment tests or simultaneous intervals.
    """
    keys = ["condition", "region", "marker"]
    if animals.duplicated(["animal_id", *keys]).any():
        raise ValueError("One row per animal, condition, region and marker is required")
    return pd.DataFrame(
        [
            dict(zip(keys, key), **bootstrap_summary(group.enrichment))
            for key, group in animals.groupby(keys, sort=True)
        ]
    )


def regional_control_effects(ar):
    """Treatment/control effects using the primary pooled estimator."""
    rows = []
    for region in REGIONS:
        g = ar.loc[ar.marker.eq("TH") & ar.region.eq(region)]
        for treatment in ["LPS_72h", "LPS_60d"]:
            ratio, p, lo, hi, hits, n = exact(
                np.log(g.enrichment), g.condition, "Control", treatment
            )
            rows.append(
                dict(
                    region=region,
                    treatment=treatment,
                    effect_ratio=ratio,
                    ci_low_ratio=lo,
                    ci_high_ratio=hi,
                    exact_two_sided_p=p,
                    n_extreme=hits,
                    n_allocations=n,
                    n_reference=int(g.condition.eq("Control").sum()),
                    n_treatment=int(g.condition.eq(treatment).sum()),
                )
            )
    effects = pd.DataFrame(rows)
    effects["bh_q_within_TH_region_family"] = multipletests(
        effects.exact_two_sided_p, method="fdr_bh"
    )[1]
    return effects


def main():
    out = output_dir()
    ar = per_animal_region()
    assert not ar.duplicated(["animal_id", "region", "marker"]).any()
    assert (ar.expected > 0).all() and (ar.enrichment > 0).all()
    assert set(ar.region) == set(REGIONS)
    ar.to_csv(out / "regional_marker_animal_values.tsv", sep="\t", index=False)
    regional_marker_summaries(ar).to_csv(out / "regional_marker_summary.tsv", sep="\t", index=False)
    th = ar[ar.marker.eq("TH")].copy()
    assert th.groupby("condition").animal_id.nunique().to_dict() == {
        "Control": 5,
        "LPS_72h": 5,
        "LPS_60d": 4,
    }
    wide = th.pivot(index=["animal_id", "condition"], columns="region", values="enrichment")[
        REGIONS
    ]
    assert wide.notna().all().all() and len(wide) == 14
    rows = []
    for reg in REGIONS:
        g = th[th.region.eq(reg)]
        ratio, p, lo, hi, hits, n = exact(np.log(g.enrichment), g.condition, "LPS_72h", "LPS_60d")
        rows.append(
            dict(
                region=reg,
                comparison="LPS_60d vs LPS_72h",
                ratio=ratio,
                ci_low=lo,
                ci_high=hi,
                exact_p=p,
                n_extreme=hits,
                n_allocations=n,
                n_72h=5,
                n_60d=4,
            )
        )
    temporal = pd.DataFrame(rows)
    temporal["bh_q_five_regions"] = multipletests(temporal.exact_p, method="fdr_bh")[1]
    temporal.to_csv(out / "regional_th_temporal.tsv", sep="\t", index=False)
    selected = wide.loc[wide.index.get_level_values("condition") != "Control"]
    profile = temporal_profile_test(
        np.log(selected.to_numpy()), selected.index.get_level_values("condition") == "LPS_60d"
    )
    pd.DataFrame([profile]).to_csv(
        out / "regional_th_temporal_interaction.tsv", sep="\t", index=False
    )
    th.to_csv(out / "regional_th_animal_values.tsv", sep="\t", index=False)

    effects = regional_control_effects(ar)
    effects.to_csv(out / "regional_th_per_region.tsv", sep="\t", index=False)
    # Verify the treatment ratios directly against the animal-level values.
    for row in effects.itertuples():
        g = th[th.region.eq(row.region)]
        ratio = np.exp(
            np.log(g.loc[g.condition.eq(row.treatment), "enrichment"]).mean()
            - np.log(g.loc[g.condition.eq("Control"), "enrichment"]).mean()
        )
        np.testing.assert_allclose(ratio, row.effect_ratio, rtol=1e-12)

    control = ar.loc[ar.condition.eq("Control")].copy()
    control.to_csv(out / "regional_baseline_animal_values.tsv", sep="\t", index=False)
    baseline = [
        dict(region=reg, marker=marker, **bootstrap_summary(g.enrichment))
        for (reg, marker), g in control.groupby(["region", "marker"])
    ]
    pd.DataFrame(baseline).to_csv(
        out / "regional_baseline_marker_summary.tsv", sep="\t", index=False
    )
    cc = class_contrast(ar).query("condition == 'Control'")
    cc.to_csv(out / "regional_baseline_group_animal_values.tsv", sep="\t", index=False)
    summaries = [
        dict(region=reg, **bootstrap_summary(g.contrast, log=True))
        for reg, g in cc.groupby("region")
    ]
    pd.DataFrame(summaries).to_csv(
        out / "regional_baseline_group_summary.tsv", sep="\t", index=False
    )

    # Retain the original aggregate correlation definition, using only the six
    # retained staining panels and one specification per physical stack.
    # This is not an animal-level test or adjustment for anatomy.
    stacks = pd.read_csv(PIPELINE / "results/uniform_stack_summary.tsv", sep="\t")
    stacks = stacks.loc[
        stacks.dataset.eq("LPS")
        & stacks.condition.eq("Control")
        & stacks.specification.eq("uniform_standard")
        & stacks.marker.isin(MARKERS)
    ]
    assert not stacks.file_id.duplicated().any()
    mask = stacks.groupby("region").microglia_volume_fraction.mean()
    correlation = pd.DataFrame(
        {
            "iba1_volume_fraction": mask,
            "group_contrast_arithmetic_mean": cc.groupby("region").contrast.mean(),
        }
    ).loc[REGIONS]
    r, p = pearsonr(correlation.iba1_volume_fraction, correlation.group_contrast_arithmetic_mean)
    correlation.reset_index().to_csv(
        out / "regional_iba1_correlation_values.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        [
            dict(
                pearson_r=r,
                nominal_p=p,
                n_regions=len(correlation),
                aggregation="unweighted stack mean mask fraction, arithmetic animal mean group contrast",
            )
        ]
    ).to_csv(out / "regional_iba1_correlation_summary.tsv", sep="\t", index=False)
    inputs = [
        SOURCE_DATA / "S4_per_animal_region_enrichment.tsv",
        PIPELINE / "results/uniform_stack_summary.tsv",
    ]
    provenance = dict(
        analysis="exploratory regional temporal contrasts",
        seed=None,
        randomisation="exhaustive enumeration; no Monte Carlo sampling",
        interval="existing exact-test inversion on 1201 log shifts, effect +/-3, step 0.005",
        interaction="sum of squared group mean differences in animal-centred regional log enrichment",
        fdr="five direct regional temporal tests; existing ten TH treatment/control tests kept separate",
        descriptive_summary="Arithmetic mean enrichment and 95% exhaustive animal-bootstrap percentile intervals within each condition, region and marker. No additional hypothesis tests.",
        estimator="Observed and expected counts pooled within animal, marker and region before forming enrichment ratios.",
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    )
    (out / "regional_temporal_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        temporal.to_string(index=False),
        "\nInteraction:",
        profile,
        "\nTreatment/control:\n",
        effects.to_string(index=False),
        "\nIba1:",
        r,
        p,
        flush=True,
    )


if __name__ == "__main__":
    main()
