#!/usr/bin/env python3
"""Restartable raw-to-object pipeline with Iba1 withheld from bouton detection."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from common import CODE_ROOT, MANIFEST_PATH, require_file
from common import (
    ROOT,
    PIPELINE_VERSION,
    MICROGLIA_MASK_VERSION,
    atomic_write_json,
    boundary_fingerprint,
    canonical_sha256,
    current_marker_parameters,
    load_config,
    sensitive_intensity_reference_is_exact,
)


VENDOR = CODE_ROOT / "vendor"
_MODULES = None


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, VENDOR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if filename == "01_preprocessing.py" and module.TOPHAT_BACKEND != "opencv_exact_flat_disk":
        raise RuntimeError(
            "The retained preprocessing uses OpenCV. Install requirements-raw-image.txt. "
            "A different top-hat backend is not an approved substitute."
        )
    return module


def modules():
    global _MODULES
    if _MODULES is None:
        _MODULES = {
            "pre": load_module("uniform_preprocessing", "01_preprocessing.py"),
            "mg": load_module("uniform_microglia", "02_microglia_segmentation.py"),
            "syn": load_module("uniform_synapse", "03_synapse_segmentation.py"),
            "contact": load_module("uniform_geometry", "04_contact_geometry.py"),
        }
    return _MODULES


def output_paths(dataset: str, specification: str, file_id: str):
    base = ROOT / "data" / "derived" / dataset.lower()
    shared = base / "shared"
    spec = base / specification
    return {
        "mg_mask": shared / "microglia_masks" / f"{file_id}.npz",
        "labels": spec / "synapse_labels" / f"{file_id}.npz",
        "objects": spec / "objects" / f"{file_id}.parquet",
        "summary": spec / "stack_summaries" / f"{file_id}.json",
        "failure": spec / "failures" / f"{file_id}.json",
    }


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + ".", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def marker_params(dataset: str, marker: str, specification: str, cfg: dict):
    return current_marker_parameters(dataset, marker, specification, cfg)


def load_or_segment_microglia(path: Path, mg_preprocessed, voxel_xy, voxel_z):
    mg_mod = modules()["mg"]
    if path.exists():
        loaded = np.load(path)
        stored_version = (
            str(loaded["mask_version"].item()) if "mask_version" in loaded.files else ""
        )
        if stored_version == MICROGLIA_MASK_VERSION:
            return {
                "binary": loaded["binary"].astype(bool, copy=False),
                "labels": loaded["labels"].astype(np.int32, copy=False),
                "n_objects": int(loaded["n_objects"]),
                "volume_fraction": float(loaded["volume_fraction"]),
                "source": "shared_checkpoint",
            }
    result = mg_mod.segment_microglia(
        mg_preprocessed,
        voxel_xy=voxel_xy,
        voxel_z=voxel_z,
    )
    atomic_npz(
        path,
        binary=result["binary"].astype(np.uint8),
        labels=result["labels"].astype(np.int32),
        n_objects=np.array(result["n_objects"], dtype=np.int32),
        volume_fraction=np.array(result["volume_fraction"], dtype=np.float64),
        mask_version=np.array(MICROGLIA_MASK_VERSION),
    )
    result["source"] = "new_segmentation"
    return result


def summarize_objects(objects: pd.DataFrame, mg_binary: np.ndarray, shape, voxel_xy, voxel_z):
    voxel_volume = float(voxel_xy * voxel_xy * voxel_z)
    total_volume = float(np.prod(shape) * voxel_volume)
    n = len(objects)
    if n:
        overlap = objects["overlap_fraction"].to_numpy()
        complete_eligible = objects["complete_inclusion_eligible"].fillna(False).to_numpy(bool)
        n_any = int(np.sum(overlap > 0))
        n_complete = int(np.sum((overlap >= 1.0) & complete_eligible))
        n_partial = int(np.sum((overlap > 0) & (overlap < 1.0)))
        n_boundary_truncated = int(
            np.sum(
                (overlap >= 1.0) & ~complete_eligible,
            )
        )
        n_complete_eligible = int(np.sum(complete_eligible))
    else:
        n_any = n_complete = n_partial = n_boundary_truncated = 0
        n_complete_eligible = 0
    mg_volume = float(mg_binary.sum() * voxel_volume)
    return {
        "n_synapses": int(n),
        "n_overlap_any": n_any,
        "n_complete": n_complete,
        "n_partial": n_partial,
        "n_boundary_truncated_overlap": n_boundary_truncated,
        "n_complete_inclusion_eligible": n_complete_eligible,
        "n_no_overlap": int(n - n_any),
        "overlap_any_rate": n_any / n if n else 0.0,
        "complete_rate": n_complete / n_complete_eligible if n_complete_eligible else 0.0,
        "complete_rate_all_candidates": n_complete / n if n else 0.0,
        "total_volume_um3": total_volume,
        "microglia_volume_um3": mg_volume,
        "microglia_volume_fraction": mg_volume / total_volume if total_volume else np.nan,
        "synapse_density_per_um3": n / total_volume if total_volume else np.nan,
        "mean_synapse_volume_um3": float(objects.volume_um3.mean()) if n else np.nan,
        "median_synapse_volume_um3": float(objects.volume_um3.median()) if n else np.nan,
        "mean_synapse_intensity": float(objects.mean_intensity.mean()) if n else np.nan,
    }


def _checkpoint_is_current(
    paths,
    force,
    expected_parameter_fingerprint=None,
    expected_raw_fingerprint=None,
    expected_intensity_reference_parameter_fingerprint=None,
):
    if force or not all(paths[k].exists() for k in ["labels", "objects", "summary"]):
        return False
    try:
        previous = json.loads(paths["summary"].read_text())
    except (OSError, json.JSONDecodeError):
        return False
    current = (
        previous.get("pipeline_version") == PIPELINE_VERSION
        and previous.get("microglia_mask_version") == MICROGLIA_MASK_VERSION
    )
    if expected_parameter_fingerprint is not None:
        current = current and (
            previous.get("parameter_fingerprint") == expected_parameter_fingerprint
        )
    if expected_raw_fingerprint is not None:
        current = current and (previous.get("raw_boundary_sha256") == expected_raw_fingerprint)
    if expected_intensity_reference_parameter_fingerprint is not None:
        current = current and (
            previous.get("intensity_reference_parameter_fingerprint")
            == expected_intensity_reference_parameter_fingerprint
            and previous.get("raw_intensity_acceptance_policy")
            == "exact standard-stack marker-only floors"
            and sensitive_intensity_reference_is_exact(previous)
        )
    return current


def process_one(row: dict, specifications, cfg: dict, force: bool = False):
    """Process one raw stack for one or both marker-only specifications.

    When both specifications are requested, raw loading and marker
    preprocessing occur once.  Both candidate label fields are frozen before
    the Iba1 channel is preprocessed or its mask is loaded.
    """
    file_id = row["file_id"]
    dataset = row["dataset"]
    if isinstance(specifications, str):
        specifications = [specifications]
    paths_by_spec = {
        specification: output_paths(dataset, specification, file_id)
        for specification in specifications
    }
    expected_parameters = {
        specification: marker_params(
            dataset,
            row["marker"],
            specification,
            cfg,
        )
        for specification in specifications
    }
    expected_parameter_fingerprints = {
        specification: canonical_sha256(parameters)
        for specification, parameters in expected_parameters.items()
    }
    standard_reference_parameters = marker_params(
        dataset,
        row["marker"],
        "uniform_standard",
        cfg,
    )
    standard_reference_parameter_fingerprint = canonical_sha256(standard_reference_parameters)
    expected_raw_fingerprint = str(row.get("boundary_sha256", "")).strip()
    if not expected_raw_fingerprint or expected_raw_fingerprint.lower() == "nan":
        expected_raw_fingerprint = None
    needed = [
        specification
        for specification in specifications
        if not _checkpoint_is_current(
            paths_by_spec[specification],
            force,
            expected_parameter_fingerprints[specification],
            expected_raw_fingerprint,
            (
                standard_reference_parameter_fingerprint
                if specification == "uniform_sensitive"
                else None
            ),
        )
    ]
    if not needed:
        return {
            "file_id": file_id,
            "dataset": dataset,
            "status": "skipped",
            "specifications": list(specifications),
            "pipeline_version": PIPELINE_VERSION,
            "microglia_mask_version": MICROGLIA_MASK_VERSION,
        }

    started = time.time()
    shared_timing = {}
    try:
        mods = modules()
        raw_path = Path(row["raw_path"])
        observed_boundary_fingerprint = boundary_fingerprint(raw_path)
        expected_boundary_fingerprint = expected_raw_fingerprint or ""
        if (
            expected_boundary_fingerprint
            and observed_boundary_fingerprint != expected_boundary_fingerprint
        ):
            raise RuntimeError(f"raw source fingerprint differs from frozen manifest: {raw_path}")

        t = time.time()
        loaded = mods["pre"].load_lsm(raw_path)
        synapse_raw, microglia_raw = mods["pre"].extract_channels(loaded["data"])
        # Only the marker channel is processed before candidate labels freeze.
        syn_prep = mods["pre"].preprocess_synapse_channel(synapse_raw)
        voxel_xy, voxel_z = loaded["voxel_xy"], loaded["voxel_z"]
        shape = synapse_raw.shape
        shared_timing["load_and_marker_preprocess_s"] = time.time() - t

        # Freeze every requested marker-only label field before touching Iba1.
        syn_results = {}
        params_by_spec = {
            specification: expected_parameters[specification] for specification in needed
        }
        segmentation_times = {}
        t = time.time()
        shared_detection = mods["syn"].prepare_shared_detection(
            syn_prep["preprocessed"],
            row["marker"],
            voxel_xy=voxel_xy,
            voxel_z=voxel_z,
            params=standard_reference_parameters,
        )
        shared_timing["marker_log_and_seed_detection_s"] = time.time() - t
        standard_reference_result = None
        processing_order = sorted(
            needed,
            key=lambda value: value != "uniform_standard",
        )
        # If only the sensitive checkpoint is stale, execute the standard
        # marker-only segmentation in memory to obtain its exact object and
        # rescue floors.  Existing standard artifacts remain untouched.
        if "uniform_sensitive" in needed and "uniform_standard" not in needed:
            t = time.time()
            standard_reference_result = mods["syn"].segment_synapses(
                syn_prep["preprocessed"],
                synapse_raw,
                row["marker"],
                microglia_mask=None,
                voxel_xy=voxel_xy,
                voxel_z=voxel_z,
                params=standard_reference_parameters,
                shared_detection=shared_detection,
            )
            shared_timing["standard_intensity_reference_segmentation_s"] = time.time() - t
        for specification in processing_order:
            params = params_by_spec[specification]
            runtime_params = dict(params)
            if specification == "uniform_sensitive":
                if standard_reference_result is None:
                    raise RuntimeError("sensitive detector lacks its standard intensity reference")
                runtime_params["precomputed_noise_floor"] = standard_reference_result["noise_floor"]
                runtime_params["precomputed_rescue_base_floor"] = standard_reference_result.get(
                    "rescue_base_floor"
                )
            t = time.time()
            syn_results[specification] = mods["syn"].segment_synapses(
                syn_prep["preprocessed"],
                synapse_raw,
                row["marker"],
                microglia_mask=None,
                voxel_xy=voxel_xy,
                voxel_z=voxel_z,
                params=runtime_params,
                shared_detection=shared_detection,
            )
            segmentation_times[specification] = time.time() - t
            if specification == "uniform_standard":
                standard_reference_result = syn_results[specification]
            else:
                if not np.isclose(
                    syn_results[specification]["noise_floor"],
                    standard_reference_result["noise_floor"],
                    rtol=0,
                    atol=1e-12,
                ):
                    raise RuntimeError("sensitive object floor differs from standard reference")
                sensitive_rescue = syn_results[specification].get("rescue_base_floor")
                standard_rescue = standard_reference_result.get("rescue_base_floor")
                if (
                    sensitive_rescue is not None
                    and standard_rescue is not None
                    and not np.isclose(
                        sensitive_rescue,
                        standard_rescue,
                        rtol=0,
                        atol=1e-12,
                    )
                ):
                    raise RuntimeError("sensitive rescue floor differs from standard reference")

        # Iba1 is now revealed for geometry measurement only.
        t = time.time()
        mg_prep = mods["pre"].preprocess_microglia_channel(microglia_raw)
        representative_paths = paths_by_spec[needed[0]]
        mg_result = load_or_segment_microglia(
            representative_paths["mg_mask"],
            mg_prep["preprocessed"],
            voxel_xy,
            voxel_z,
        )
        shared_timing["iba1_preprocess_and_segmentation_s"] = time.time() - t
        t = time.time()
        distance = mods["contact"].compute_distance_to_microglia(
            mg_result["binary"],
            voxel_xy=voxel_xy,
            voxel_z=voxel_z,
        )
        shared_timing["distance_transform_s"] = time.time() - t

        output_counts = {}
        for specification in needed:
            paths = paths_by_spec[specification]
            syn_result = syn_results[specification]
            params = params_by_spec[specification]
            timing = dict(shared_timing)
            timing["marker_segmentation_s"] = segmentation_times[specification]

            t = time.time()
            measured = mods["contact"].measure_synapses(
                syn_result["labels"],
                synapse_raw,
                distance,
                row["marker"],
                voxel_xy=voxel_xy,
                voxel_z=voxel_z,
            )
            contacts = mods["contact"].classify_contacts(
                syn_result["labels"],
                mg_result["binary"],
                distance,
                contact_distance_um=cfg["contact_distance_um"],
                full_threshold=cfg["full_overlap_threshold"],
            )
            objects = measured.merge(
                contacts,
                on="label_id",
                how="outer",
                validate="one_to_one",
            )
            for key in [
                "dataset",
                "file_id",
                "marker",
                "condition",
                "animal_id",
                "region",
                "stack_num",
                "raw_path",
            ]:
                objects[key] = row.get(key)
            objects["analysis_category"] = objects["contact_category"].replace(
                {
                    "complete_inclusion": "complete",
                    "proximity_without_overlap": "no_overlap",
                    "no_contact": "no_overlap",
                }
            )
            timing["geometry_measurement_s"] = time.time() - t

            t = time.time()
            atomic_npz(
                paths["labels"],
                labels=syn_result["labels"].astype(np.int32),
            )
            atomic_parquet(objects, paths["objects"])
            summary = summarize_objects(
                objects,
                mg_result["binary"],
                shape,
                voxel_xy,
                voxel_z,
            )
            timing["save_s"] = time.time() - t
            summary.update(
                {
                    "dataset": dataset,
                    "specification": specification,
                    "file_id": file_id,
                    "marker": row["marker"],
                    "condition": row["condition"],
                    "animal_id": row["animal_id"],
                    "region": row["region"],
                    "stack_num": int(row["stack_num"]),
                    "raw_path": str(raw_path),
                    "raw_boundary_sha256": observed_boundary_fingerprint,
                    "shape": "x".join(map(str, shape)),
                    "voxel_xy_um": float(voxel_xy),
                    "voxel_z_um": float(voxel_z),
                    "iba1_used_for_bouton_detection": False,
                    "candidate_labels_frozen_before_iba1_preprocessing": True,
                    "data_derived_bleedthrough_correction": False,
                    "geometry_implementation": ("physical_laplacian_and_anisotropic_morphology_v2"),
                    "marker_background_backend": mods["pre"].TOPHAT_BACKEND,
                    "pipeline_version": PIPELINE_VERSION,
                    "microglia_mask_version": MICROGLIA_MASK_VERSION,
                    "microglia_source": mg_result["source"],
                    "n_microglia_objects": int(mg_result["n_objects"]),
                    "n_seeds": int(syn_result["n_seeds"]),
                    "foreground_threshold": float(syn_result["foreground_threshold"]),
                    "foreground_method": syn_result["foreground_method"],
                    "object_noise_floor": float(syn_result["noise_floor"]),
                    "rescue_base_floor": syn_result.get("rescue_base_floor"),
                    "raw_intensity_acceptance_policy": (
                        "exact standard-stack marker-only floors"
                        if specification == "uniform_sensitive"
                        else "standard marker-only adaptive floors"
                    ),
                    "intensity_reference_specification": (
                        "uniform_standard" if specification == "uniform_sensitive" else None
                    ),
                    "intensity_reference_parameter_fingerprint": (
                        standard_reference_parameter_fingerprint
                        if specification == "uniform_sensitive"
                        else None
                    ),
                    "intensity_reference_object_noise_floor": (
                        float(standard_reference_result["noise_floor"])
                        if specification == "uniform_sensitive"
                        else None
                    ),
                    "intensity_reference_rescue_base_floor": (
                        standard_reference_result.get("rescue_base_floor")
                        if specification == "uniform_sensitive"
                        else None
                    ),
                    "intensity_reference_candidate_count": (
                        int(standard_reference_result["n_boutons"])
                        if specification == "uniform_sensitive"
                        else None
                    ),
                    "removed_counts": syn_result.get("removed_counts", {}),
                    "parameters": params,
                    "parameter_fingerprint": (expected_parameter_fingerprints[specification]),
                    "specification_config": cfg["specifications"][specification],
                    "timing": timing,
                    "elapsed_s": time.time() - started,
                }
            )
            atomic_write_json(paths["summary"], summary)
            if paths["failure"].exists():
                paths["failure"].unlink()
            output_counts[specification] = summary["n_synapses"]

        # Release references before the next large stack.
        del loaded, synapse_raw, microglia_raw, syn_prep, mg_prep
        del mg_result, syn_results, shared_detection, distance, measured, contacts, objects
        gc.collect()
        return {
            "file_id": file_id,
            "dataset": dataset,
            "status": "ok",
            "specifications": needed,
            "n_candidates": output_counts,
            "pipeline_version": PIPELINE_VERSION,
            "microglia_mask_version": MICROGLIA_MASK_VERSION,
            "elapsed_s": time.time() - started,
        }
    except Exception as exc:
        for specification in needed:
            failure = {
                "dataset": dataset,
                "specification": specification,
                "file_id": file_id,
                "raw_path": row.get("raw_path"),
                "pipeline_version": PIPELINE_VERSION,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_s": time.time() - started,
            }
            atomic_write_json(paths_by_spec[specification]["failure"], failure)
        gc.collect()
        return {
            "file_id": file_id,
            "dataset": dataset,
            "status": "failed",
            "pipeline_version": PIPELINE_VERSION,
            "microglia_mask_version": MICROGLIA_MASK_VERSION,
            "error": repr(exc),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--specification",
        choices=["uniform_standard", "uniform_sensitive", "ALL"],
        required=True,
    )
    parser.add_argument("--dataset", choices=["LPS", "MPTP", "ALL"], default="ALL")
    parser.add_argument(
        "--marker",
        action="append",
        help=(
            "Restrict to one marker (repeatable). Intended for disjoint, "
            "checkpoint-safe parallel batches."
        ),
    )
    parser.add_argument("--file-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    workers = args.workers or int(cfg["workers"])
    specifications = (
        ["uniform_standard", "uniform_sensitive"]
        if args.specification == "ALL"
        else [args.specification]
    )
    manifest = pd.read_csv(require_file(MANIFEST_PATH), sep="\t")
    if args.dataset != "ALL":
        manifest = manifest[manifest.dataset.eq(args.dataset)]
    if args.marker:
        manifest = manifest[manifest.marker.isin(args.marker)]
    if args.file_id:
        manifest = manifest[manifest.file_id.eq(args.file_id)]
    if args.limit:
        manifest = manifest.head(args.limit)
    rows = manifest.to_dict("records")
    if not rows:
        raise SystemExit("No manifest rows selected")

    marker_suffix = "_" + "-".join(str(marker) for marker in args.marker) if args.marker else ""
    log_path = (
        ROOT
        / "logs"
        / (f"pipeline_{args.specification}_{args.dataset.lower()}{marker_suffix}.jsonl")
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Queued {len(rows)} stacks: spec={','.join(specifications)}, "
        f"markers={','.join(args.marker) if args.marker else 'ALL'}, "
        f"workers={workers}",
        flush=True,
    )
    started = time.time()
    completed = 0
    counts = {"ok": 0, "skipped": 0, "failed": 0}

    if workers == 1:
        iterator = (process_one(row, specifications, cfg, args.force) for row in rows)
        for result in iterator:
            completed += 1
            counts[result["status"]] += 1
            with log_path.open("a") as handle:
                handle.write(json.dumps(result, default=str) + "\n")
            print(f"[{completed}/{len(rows)}] {result}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_one, row, specifications, cfg, args.force): row["file_id"]
                for row in rows
            }
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                counts[result["status"]] += 1
                with log_path.open("a") as handle:
                    handle.write(json.dumps(result, default=str) + "\n")
                elapsed = time.time() - started
                print(
                    f"[{completed}/{len(rows)}] {result['status']} {result['file_id']} "
                    f"elapsed={elapsed / 60:.1f}m counts={counts}",
                    flush=True,
                )

    print(f"Finished in {(time.time() - started) / 60:.1f} min: {counts}")
    if counts["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
