#!/usr/bin/env python3
"""Aggregate checkpointed stack summaries and enforce completeness."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from common import MANIFEST_PATH, require_file
from common import (
    ROOT,
    PIPELINE_VERSION,
    MICROGLIA_MASK_VERSION,
    canonical_sha256,
    load_config,
    stack_detector_contract_is_current,
)


def same_optional_number(left, right):
    if left is None or right is None:
        return left is None and right is None
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    manifest = pd.read_csv(require_file(MANIFEST_PATH), sep="\t")
    all_rows = []
    status_rows = []
    for dataset in ["LPS", "MPTP"]:
        expected = manifest[manifest.dataset.eq(dataset)]
        for specification in ["uniform_standard", "uniform_sensitive"]:
            folder = ROOT / "data" / "derived" / dataset.lower() / specification / "stack_summaries"
            rows = []
            for path in sorted(folder.glob("*.json")):
                rows.append(json.loads(path.read_text()))
            observed = {row["file_id"] for row in rows}
            expected_ids = set(expected.file_id)
            missing = sorted(expected_ids - observed)
            unexpected = sorted(observed - expected_ids)
            duplicate_ids = sorted(
                pd.Series([row["file_id"] for row in rows])
                .loc[lambda x: x.duplicated(keep=False)]
                .unique()
            )
            stale_pipeline = sorted(
                {row["file_id"] for row in rows if row.get("pipeline_version") != PIPELINE_VERSION}
            )
            stale_mask = sorted(
                {
                    row["file_id"]
                    for row in rows
                    if row.get("microglia_mask_version") != MICROGLIA_MASK_VERSION
                }
            )
            parameter_integrity_mismatch = sorted(
                {
                    row["file_id"]
                    for row in rows
                    if not isinstance(row.get("parameters"), dict)
                    or row.get("parameter_fingerprint") != canonical_sha256(row["parameters"])
                }
            )
            current_parameter_contract_mismatch = sorted(
                {
                    row["file_id"]
                    for row in rows
                    if not stack_detector_contract_is_current(
                        row,
                        specification,
                        cfg,
                    )
                }
            )
            specification_keys = [
                "min_volume_factor",
                "max_volume_factor",
                "foreground_noise_k_factor",
                "global_intensity_factor",
            ]
            expected_specification = cfg["specifications"][specification]
            specification_config_mismatch = sorted(
                {
                    row["file_id"]
                    for row in rows
                    if not isinstance(row.get("specification_config"), dict)
                    or any(
                        row["specification_config"].get(key) != expected_specification.get(key)
                        for key in specification_keys
                    )
                }
            )
            manifest_fingerprint = expected.set_index("file_id")["boundary_sha256"].astype(str)
            fingerprint_mismatch = sorted(
                {
                    row["file_id"]
                    for row in rows
                    if row["file_id"] in manifest_fingerprint.index
                    and str(row.get("raw_boundary_sha256", ""))
                    != manifest_fingerprint.loc[row["file_id"]]
                }
            )
            sensitive_reference_mismatch = []
            if specification == "uniform_sensitive":
                standard_folder = (
                    ROOT
                    / "data"
                    / "derived"
                    / dataset.lower()
                    / "uniform_standard"
                    / "stack_summaries"
                )
                standard_records = {}
                for standard_path in standard_folder.glob("*.json"):
                    standard_record = json.loads(standard_path.read_text())
                    standard_records[standard_record["file_id"]] = standard_record
                for row in rows:
                    standard = standard_records.get(row["file_id"])
                    if (
                        standard is None
                        or not stack_detector_contract_is_current(
                            standard,
                            "uniform_standard",
                            cfg,
                        )
                        or row.get("raw_boundary_sha256") != standard.get("raw_boundary_sha256")
                        or not same_optional_number(
                            row.get("intensity_reference_object_noise_floor"),
                            standard.get("object_noise_floor"),
                        )
                        or not same_optional_number(
                            row.get("intensity_reference_rescue_base_floor"),
                            standard.get("rescue_base_floor"),
                        )
                        or row.get("intensity_reference_candidate_count")
                        != standard.get("n_synapses")
                    ):
                        sensitive_reference_mismatch.append(row["file_id"])
                sensitive_reference_mismatch = sorted(set(sensitive_reference_mismatch))
            status_rows.append(
                {
                    "dataset": dataset,
                    "specification": specification,
                    "expected": len(expected_ids),
                    "observed": len(observed),
                    "missing": len(missing),
                    "unexpected": len(unexpected),
                    "duplicate_ids": len(duplicate_ids),
                    "stale_pipeline_version": len(stale_pipeline),
                    "stale_microglia_mask_version": len(stale_mask),
                    "parameter_integrity_mismatch": len(parameter_integrity_mismatch),
                    "current_parameter_contract_mismatch": len(current_parameter_contract_mismatch),
                    "specification_config_mismatch": len(specification_config_mismatch),
                    "raw_fingerprint_mismatch": len(fingerprint_mismatch),
                    "sensitive_standard_reference_mismatch": len(sensitive_reference_mismatch),
                    "missing_ids": "|".join(missing),
                    "unexpected_ids": "|".join(unexpected),
                    "duplicate_file_ids": "|".join(duplicate_ids),
                    "stale_pipeline_ids": "|".join(stale_pipeline),
                    "stale_mask_ids": "|".join(stale_mask),
                    "parameter_integrity_mismatch_ids": "|".join(parameter_integrity_mismatch),
                    "current_parameter_contract_mismatch_ids": "|".join(
                        current_parameter_contract_mismatch
                    ),
                    "specification_config_mismatch_ids": "|".join(specification_config_mismatch),
                    "fingerprint_mismatch_ids": "|".join(fingerprint_mismatch),
                    "sensitive_standard_reference_mismatch_ids": "|".join(
                        sensitive_reference_mismatch
                    ),
                    "required_pipeline_version": PIPELINE_VERSION,
                    "required_microglia_mask_version": MICROGLIA_MASK_VERSION,
                }
            )
            all_rows.extend(rows)
            invalid = (
                missing
                or unexpected
                or duplicate_ids
                or stale_pipeline
                or stale_mask
                or parameter_integrity_mismatch
                or current_parameter_contract_mismatch
                or specification_config_mismatch
                or fingerprint_mismatch
                or sensitive_reference_mismatch
            )
            if invalid and not args.allow_incomplete:
                raise RuntimeError(
                    f"Incomplete {dataset}/{specification}: missing={len(missing)}, "
                    f"unexpected={len(unexpected)}, duplicate={len(duplicate_ids)}, "
                    f"stale_pipeline={len(stale_pipeline)}, stale_mask={len(stale_mask)}, "
                    f"parameter_integrity={len(parameter_integrity_mismatch)}, "
                    "current_parameter_contract="
                    f"{len(current_parameter_contract_mismatch)}, "
                    f"specification_config={len(specification_config_mismatch)}, "
                    f"fingerprint_mismatch={len(fingerprint_mismatch)}"
                    ", sensitive_standard_reference_mismatch="
                    f"{len(sensitive_reference_mismatch)}"
                )

    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(status_rows).to_csv(
        results / "uniform_pipeline_completeness.tsv", sep="\t", index=False
    )
    combined = pd.DataFrame(all_rows)
    combined.to_parquet(results / "uniform_stack_summary.parquet", index=False, compression="zstd")
    combined.to_csv(results / "uniform_stack_summary.tsv", sep="\t", index=False)
    print(pd.DataFrame(status_rows).to_string(index=False))
    if len(combined):
        print(
            combined.groupby(["dataset", "specification", "marker", "condition"])["n_synapses"]
            .agg(["count", "sum", "median"])
            .to_string()
        )


if __name__ == "__main__":
    main()
