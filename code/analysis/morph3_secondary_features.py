"""Reproduce the retained secondary morphology tables from eligible-cell features.

Cells are averaged within field, then fields within animal. Exact two-sided
animal-label tests use the retained permutation helper. Secondary display effects
are divided by the sample SD across all animals, as in the deposited tables.
This differs from the pairwise SD used for the primary branch-count display.

General shape measurements come from the retained feature table. Topology
measurements may come from the retained table or a new topology extraction.
Unadjusted secondary P values are exported for audit, not as a new
confirmatory family. The fixed 5-um topology descriptor is not the reported
control-percentile terminal-branch outcome.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mg_paths import MORPHOLOGY, OUTPUT, output_dir, require_file
from morph2_persistent_branches import COND, exact_contrast

SHAPE_FEATURES = """vol_um3 surface_area_um2 sphericity sa_vol_ratio extent_3d
solidity_3d euler_number_3d max_dt_um mean_dt_um soma_vol_um3 soma_vol_fraction
skel_total_length_um skel_n_branches skel_n_endpoints skel_n_junctions
skel_n_triple_pts skel_n_quadruple_pts skel_slab_voxels skel_avg_branch_len
skel_max_branch_len process_mean_thickness_um process_std_thickness_um
process_max_thickness_um process_thin_fraction sholl_max_inters
sholl_critical_radius sholl_sum_inters sholl_mean_inters sholl_enclosing_radius
sholl_auc sholl_primary_branches sholl_rami_index fractal_dim_3d hull_vol_um3
hull_surface_um2 hull_compactness vol_hull_ratio elongation_3d flatness_3d
anisotropy_3d""".split()

TOPOLOGY_FEATURES = """tmd_bars_gt5um tmd_total_persistence tmd_mean_persistence
tmd_entropy tmd_n_bars bifurcation_angle_mean path_length_max_um
bifurcation_angle_sd tortuosity_mean strahler_max taper_mean tortuosity_max
tmd_max_persistence""".split()


def summarize(data, features):
    """Return animal estimates and exact pairwise secondary results."""
    data = data.copy()
    data["cond"] = data.group.map(COND)
    if data[["animal", "cond", "fov"]].isna().any().any():
        raise ValueError("Incomplete morphology metadata")
    if not np.isfinite(data[features].to_numpy()).all():
        raise ValueError("Missing or nonfinite secondary feature measurement")
    fields = data.groupby(["animal", "cond", "fov"])[features].mean().reset_index()
    animals = fields.groupby(["animal", "cond"])[features].mean().reset_index()
    if animals.animal.duplicated().any():
        raise ValueError("An animal has more than one treatment label")
    rows = []
    for feature in features:
        overall_sd = animals[feature].std(ddof=1)
        if not np.isfinite(overall_sd) or overall_sd <= 0:
            raise ValueError(f"Cannot standardize feature: {feature}")
        control_mean = animals.loc[animals.cond.eq("Control"), feature].mean()
        row = {"feature": feature}
        for condition, suffix in [("LPS_72h", "72"), ("LPS_60d", "60")]:
            _, p_value, _ = exact_contrast(
                animals[feature].to_numpy(), animals.cond.to_numpy(), condition
            )
            row["d" + suffix] = (
                animals.loc[animals.cond.eq(condition), feature].mean() - control_mean
            ) / overall_sd
            row["p" + suffix] = p_value
        rows.append(row)
    return animals, pd.DataFrame(rows)


def select_topology_input(mode="auto", explicit=None):
    """Resolve one topology table without falling back from an explicit choice."""
    if explicit is not None:
        if mode != "auto":
            raise ValueError("Use either --topology-features or --topology-source")
        return require_file(explicit.expanduser().resolve()), "explicit"
    if mode not in {"auto", "deposited", "regenerated"}:
        raise ValueError(f"Unknown topology source: {mode}")
    regenerated = OUTPUT / "morphology/topology_features_v6.csv"
    if mode == "regenerated" or (mode == "auto" and regenerated.exists()):
        return require_file(regenerated), "regenerated"
    return require_file(MORPHOLOGY / "output/topology_features_v6.csv"), "deposited"


def input_record(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology-source",
        choices=("auto", "deposited", "regenerated"),
        default="auto",
        help="Auto prefers topology measurements in the active run, then deposited measurements.",
    )
    parser.add_argument(
        "--topology-features", type=Path, help="Explicit topology measurement table."
    )
    args = parser.parse_args()
    source = MORPHOLOGY / "output"
    features_path = require_file(source / "features_3d_full_morphomics_v6_final.csv")
    topology_path, topology_source = select_topology_input(
        args.topology_source, args.topology_features
    )
    features = pd.read_csv(features_path)
    if features.duplicated(["fov", "cell_id"]).any():
        raise ValueError("Duplicate eligible-cell keys")
    topology = pd.read_csv(topology_path)
    print(f"Topology input ({topology_source}): {topology_path}", flush=True)
    topology = features[["fov", "cell_id", "animal", "group", "region"]].merge(
        topology,
        left_on=["fov", "cell_id"],
        right_on=["fov", "label"],
        how="left",
        validate="one_to_one",
    )
    out = output_dir("morphology", "region_analysis")
    for name, data, columns in [
        ("all40_exact", features, SHAPE_FEATURES),
        ("topology_exact", topology, TOPOLOGY_FEATURES),
    ]:
        animals, result = summarize(data, columns)
        result.to_csv(out / f"{name}.tsv", sep="\t", index=False)
        animals.to_csv(out / f"{name}_animal_values.tsv", sep="\t", index=False)
    (out / "secondary_feature_provenance.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "shape": input_record(features_path),
                    "topology": input_record(topology_path),
                },
                "topology_source": topology_source,
                "aggregation": "cell mean within field, field mean within animal",
                "display_standardization": "sample SD across all retained animals",
                "test": "exact two-sided animal-label permutation, unadjusted secondary P values",
                "seed": "not applicable: exhaustive enumeration",
                "shape_measurements": "deposited",
                "feature_extraction_run_by_this_script": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(out)


if __name__ == "__main__":
    main()
