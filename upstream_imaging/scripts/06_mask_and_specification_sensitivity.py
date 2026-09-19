#!/usr/bin/env python3
"""Paired Iba1-mask and bouton-specification sensitivity analyses.

For every frozen bouton label image, the observed overlap and its torus null
are recomputed with the *same* eroded, primary, or dilated Iba1 mask.  This is
critical: changing only the observed mask would change the geometric baseline
and can manufacture apparent sensitivity.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

from common import (
    CODE_ROOT,
    MANIFEST_PATH,
    ROOT,
    PIPELINE_VERSION,
    MICROGLIA_MASK_VERSION,
    TORUS_VERSION,
    atomic_write_json,
    boundary_fingerprint,
    canonical_sha256,
    load_config,
    stable_seed,
    stack_detector_contract_is_current,
    torus_configuration_fingerprint,
)


VARIANTS = {
    # The 0.1-um variants are the primary physically scaled boundary check.
    # One acquired voxel in all three axes is retained as a deliberately
    # stringent stress test because the axial step is much larger (0.38 um).
    "eroded_0p1um": ("physical", -1, 0.1),
    "primary": ("none", 0, 0.0),
    "dilated_0p1um": ("physical", 1, 0.1),
    "eroded_1voxel_3d": ("voxel", -1, 1.0),
    "dilated_1voxel_3d": ("voxel", 1, 1.0),
    # Connected-component sensitivity retains the marker-blind primary mask
    # geometry while progressively reinstating the historical whole-cell-like
    # cutoff.  This directly tests whether discarding disconnected processes
    # changes the spatial conclusion.
    "component_min_0p3um3": ("component_min", 0, 0.3),
    "component_min_1um3": ("component_min", 0, 1.0),
    "component_min_5um3_legacy": ("component_min", 0, 5.0),
    # The primary mask preserves enclosed voids.  These variants test whether
    # filling only tiny threshold holes, or reproducing the historical
    # unrestricted fill-all operation, changes the spatial conclusion.
    "hole_fill_max_0p015um3": ("hole_fill_max", 0, 0.015),
    "hole_fill_max_0p05um3": ("hole_fill_max", 0, 0.05),
    "hole_fill_all_legacy": ("hole_fill_all", 0, 0.0),
    # These masks are rebuilt from raw Iba1 with the same outcome-blind
    # pipeline while perturbing only the local-threshold multiplier.
    "threshold_factor_0p70": ("prebuilt_threshold", 0, 0.70),
    "threshold_factor_0p90": ("prebuilt_threshold", 0, 0.90),
    # Raw-Iba1 rebuilds isolate post-threshold processing choices that can
    # preferentially remove or connect thin process fragments.
    "fragment_min_0um3": ("prebuilt_threshold", 0, 0.0),
    "closing_radius_0um": ("prebuilt_threshold", 0, 0.0),
    "closing_radius_0p15um": ("prebuilt_threshold", 0, 0.15),
}
MASK_SENSITIVITY_VERSION = "mask_sensitivity_v2_9_boundary_truncation_safe_partial_overlap"
THRESHOLD_MASK_VERSION = "iba1_threshold_fragment_closing_rebuilds_v2"
VENDOR = CODE_ROOT / "vendor"
_MODULES = None


def mask_sensitivity_configuration_fingerprint(
    cfg,
    max_shifts=None,
    sampling_scheme="all_stacks",
):
    return canonical_sha256(
        {
            "mask_sensitivity_version": MASK_SENSITIVITY_VERSION,
            "threshold_mask_version": THRESHOLD_MASK_VERSION,
            "variants": VARIANTS,
            "contact_distance_um": float(cfg["contact_distance_um"]),
            "full_overlap_threshold": float(cfg["torus_null"]["full_overlap_threshold"]),
            "upstream_torus_configuration_fingerprint": (
                torus_configuration_fingerprint(cfg, int(cfg["torus_null"]["n_shifts"]))
            ),
            "requested_max_shifts": (None if max_shifts is None else int(max_shifts)),
            "sampling_scheme": str(sampling_scheme),
            "primary_acquisition_edge_correction": True,
            "observed_and_null_recomputed_together": True,
        }
    )


def validated_stack_summary(path: Path, cfg=None):
    cfg = cfg or load_config()
    record = json.loads(path.read_text())
    specification = record.get("specification")
    if (
        record.get("pipeline_version") != PIPELINE_VERSION
        or record.get("microglia_mask_version") != MICROGLIA_MASK_VERSION
        or specification not in {"uniform_standard", "uniform_sensitive"}
        or not stack_detector_contract_is_current(
            record,
            specification,
            cfg,
        )
    ):
        raise RuntimeError(f"stale upstream stack summary: {path}")
    return record


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
            "pre": load_module("mask_sensitivity_pre", "01_preprocessing.py"),
            "mg": load_module("mask_sensitivity_mg", "02_microglia_segmentation.py"),
        }
    return _MODULES


def paths_for(dataset: str, specification: str, file_id: str):
    base = ROOT / "data" / "derived" / dataset.lower()
    spec = base / specification
    return {
        "mask": base / "shared" / "microglia_masks" / f"{file_id}.npz",
        "labels": spec / "synapse_labels" / f"{file_id}.npz",
        "stack_summary": spec / "stack_summaries" / f"{file_id}.json",
        "shifts": spec / "torus_null" / "iterations" / f"{file_id}.npz",
        "primary_summary": spec / "torus_null" / "summaries" / f"{file_id}.json",
        "threshold_masks": base / "shared" / "microglia_threshold_sensitivity" / f"{file_id}.npz",
        "threshold_failure": base
        / "shared"
        / "microglia_threshold_sensitivity"
        / "failures"
        / f"{file_id}.json",
        "summary": spec / "mask_sensitivity" / "summaries" / f"{file_id}.json",
        "failure": spec / "mask_sensitivity" / "failures" / f"{file_id}.json",
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


def build_threshold_masks(row, force=False):
    """Rebuild two Iba1 threshold-factor variants once per selected stack."""
    paths = paths_for(row["dataset"], "uniform_standard", row["file_id"])
    target = paths["threshold_masks"]
    if not force and target.exists():
        previous = np.load(target)
        version = (
            str(previous["threshold_mask_version"].item())
            if "threshold_mask_version" in previous.files
            else ""
        )
        source_hash = (
            str(previous["raw_boundary_sha256"].item())
            if "raw_boundary_sha256" in previous.files
            else ""
        )
        pipeline_version = (
            str(previous["pipeline_version"].item()) if "pipeline_version" in previous.files else ""
        )
        mask_version = (
            str(previous["microglia_mask_version"].item())
            if "microglia_mask_version" in previous.files
            else ""
        )
        if (
            version == THRESHOLD_MASK_VERSION
            and source_hash == str(row.get("boundary_sha256", ""))
            and pipeline_version == PIPELINE_VERSION
            and mask_version == MICROGLIA_MASK_VERSION
            and {
                "factor_0p70",
                "primary_factor_0p80",
                "factor_0p90",
                "fragment_min_0um3",
                "closing_radius_0um",
                "closing_radius_0p15um",
            }.issubset(previous.files)
        ):
            return {"status": "skipped", "file_id": row["file_id"]}
    started = time.time()
    try:
        mods = modules()
        raw_path = Path(row["raw_path"])
        observed_hash = boundary_fingerprint(raw_path)
        expected_hash = str(row.get("boundary_sha256", ""))
        if expected_hash and expected_hash.lower() != "nan" and observed_hash != expected_hash:
            raise RuntimeError("raw source fingerprint differs from frozen manifest")
        loaded = mods["pre"].load_lsm(raw_path)
        _, iba1_raw = mods["pre"].extract_channels(loaded["data"])
        iba1_pre = mods["pre"].preprocess_microglia_channel(
            iba1_raw,
        )["preprocessed"]
        voxel_xy = float(loaded["voxel_xy"])
        voxel_z = float(loaded["voxel_z"])
        mg_mod = mods["mg"]
        smoothed = mg_mod.smooth_microglia(iba1_pre)
        thresholded = {
            factor: mg_mod.adaptive_threshold_3d(smoothed, factor=factor)
            for factor in [0.70, 0.80, 0.90]
        }

        def finalize(binary, closing_radius_um=None, min_fragment_voxels=None, min_volume_um3=None):
            refined = mg_mod.morphological_refinement(
                binary,
                closing_radius_um=closing_radius_um,
                min_fragment_voxels=min_fragment_voxels,
                voxel_xy=voxel_xy,
                voxel_z=voxel_z,
                max_hole_volume_um3=0.0,
            )
            filtered, _, _ = mg_mod.filter_by_volume(
                refined,
                min_volume_um3=min_volume_um3,
                voxel_xy=voxel_xy,
                voxel_z=voxel_z,
            )
            return filtered.astype(np.uint8)

        masks = {
            "factor_0p70": finalize(thresholded[0.70]),
            "primary_factor_0p80": finalize(thresholded[0.80]),
            "factor_0p90": finalize(thresholded[0.90]),
            "fragment_min_0um3": finalize(
                thresholded[0.80],
                min_fragment_voxels=1,
                min_volume_um3=0.0,
            ),
            "closing_radius_0um": finalize(
                thresholded[0.80],
                closing_radius_um=0.0,
            ),
            "closing_radius_0p15um": finalize(
                thresholded[0.80],
                closing_radius_um=0.15,
            ),
        }
        atomic_npz(
            target,
            **masks,
            threshold_mask_version=np.array(THRESHOLD_MASK_VERSION),
            pipeline_version=np.array(PIPELINE_VERSION),
            microglia_mask_version=np.array(MICROGLIA_MASK_VERSION),
            raw_boundary_sha256=np.array(observed_hash),
            voxel_xy_um=np.array(float(loaded["voxel_xy"])),
            voxel_z_um=np.array(float(loaded["voxel_z"])),
        )
        if paths["threshold_failure"].exists():
            paths["threshold_failure"].unlink()
        return {
            "status": "ok",
            "file_id": row["file_id"],
            "elapsed_s": time.time() - started,
        }
    except Exception as exc:
        atomic_write_json(
            paths["threshold_failure"],
            {
                "file_id": row["file_id"],
                "dataset": row["dataset"],
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_s": time.time() - started,
            },
        )
        return {
            "status": "failed",
            "file_id": row["file_id"],
            "error": repr(exc),
        }


def count_overlap(labels, mask, totals, full_threshold, complete_eligible=None):
    n = len(totals)
    if n == 0:
        return 0, 0, 0
    ids = labels[mask]
    ids = ids[ids > 0]
    inside = np.bincount(ids, minlength=n + 1)[1:] if len(ids) else np.zeros(n, dtype=np.int64)
    frac = np.divide(inside, totals, out=np.zeros(n, float), where=totals > 0)
    n_any = int(np.sum(inside > 0))
    if complete_eligible is None:
        complete_eligible = np.ones(n, dtype=bool)
    fully_overlapped = frac >= full_threshold
    n_complete = int(np.sum(fully_overlapped & complete_eligible))
    # Boundary-truncated candidates that otherwise meet the complete-inclusion
    # threshold are neither eligible complete events nor partial overlaps. This
    # matches the released torus implementation and keeps acquisition-boundary
    # truncation from being relabelled as biology.
    n_partial = int(np.sum((inside > 0) & ~fully_overlapped))
    return n_any, n_complete, n_partial


def candidate_voxels(labels):
    z, y, x = np.nonzero(labels)
    ids = labels[z, y, x].astype(np.int32, copy=False)
    return z, y, x, ids


def proximity_edge_eligible(labels, radius_um, voxel_xy, voxel_z):
    """Flag labels whose voxel centres have a full physical FOV margin."""
    n = int(labels.max())
    eligible = np.zeros(n, dtype=bool)
    spacing = (float(voxel_z), float(voxel_xy), float(voxel_xy))
    shape = labels.shape
    for label_id, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        lower = [(sl.start + 0.5) * step for sl, step in zip(slices, spacing)]
        upper = [(dim - sl.stop + 0.5) * step for sl, step, dim in zip(slices, spacing, shape)]
        eligible[label_id - 1] = min(lower + upper) >= radius_um - 1e-12
    return eligible


def count_any_eligible_sampled(label_ids, sampled_inside, eligible):
    if not len(eligible):
        return 0
    inside_ids = np.unique(
        label_ids[np.asarray(sampled_inside, dtype=bool)],
    )
    inside_ids = inside_ids[inside_ids > 0]
    return int(np.sum(eligible[inside_ids - 1]))


def count_overlap_sampled(
    label_ids, sampled_inside, totals, full_threshold, complete_eligible=None
):
    n = len(totals)
    if n == 0:
        return 0, 0, 0
    inside_ids = label_ids[np.asarray(sampled_inside, dtype=bool)]
    inside = (
        np.bincount(inside_ids, minlength=n + 1)[1:]
        if len(inside_ids)
        else np.zeros(n, dtype=np.int64)
    )
    frac = np.divide(
        inside,
        totals,
        out=np.zeros(n, dtype=float),
        where=totals > 0,
    )
    if complete_eligible is None:
        complete_eligible = np.ones(n, dtype=bool)
    n_any = int(np.sum(inside > 0))
    fully_overlapped = frac >= full_threshold
    n_complete = int(np.sum(fully_overlapped & complete_eligible))
    n_partial = int(np.sum((inside > 0) & ~fully_overlapped))
    return n_any, n_complete, n_partial


def sample_torus(mask, z, y, x, shift_y=0, shift_x=0):
    return mask[
        z,
        (y + int(shift_y)) % mask.shape[1],
        (x + int(shift_x)) % mask.shape[2],
    ]


def endpoint_summary(observed, null):
    null = np.asarray(null)
    mean = float(null.mean())
    return {
        "observed": int(observed),
        "null_mean": mean,
        "null_sd": float(null.std(ddof=1)) if len(null) > 1 else 0.0,
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "enrichment": float(observed / mean) if mean > 0 else None,
        "p_upper_finite": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
        "p_lower_finite": float((1 + np.sum(null <= observed)) / (len(null) + 1)),
    }


def exact_two_group(values, groups, reference, treatment):
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    keep = np.isfinite(values) & np.isin(groups, [reference, treatment])
    values, groups = values[keep], groups[keep]
    n_treatment = int(np.sum(groups == treatment))
    if n_treatment == 0 or n_treatment == len(values):
        return np.nan, np.nan, 0, np.array([]), np.array([])
    observed = float(values[groups == treatment].mean() - values[groups == reference].mean())
    null = []
    for selected in itertools.combinations(range(len(values)), n_treatment):
        mask = np.zeros(len(values), dtype=bool)
        mask[list(selected)] = True
        null.append(values[mask].mean() - values[~mask].mean())
    null = np.asarray(null)
    p_value = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return (
        observed,
        p_value,
        int(len(null)),
        values[groups == reference],
        values[groups == treatment],
    )


def exact_sign_flip(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, 0
    signs = np.asarray(
        list(itertools.product([-1, 1], repeat=len(values))),
        dtype=np.int8,
    )
    null = np.abs((signs * values).mean(axis=1))
    observed = abs(float(values.mean()))
    return float(np.mean(null >= observed - 1e-12)), int(len(null))


def bootstrap_difference(reference, treatment, seed, draws=10000):
    rng = np.random.default_rng(seed)
    reference = np.asarray(reference, float)
    treatment = np.asarray(treatment, float)
    sampled_reference = reference[
        rng.integers(0, len(reference), size=(draws, len(reference)))
    ].mean(axis=1)
    sampled_treatment = treatment[
        rng.integers(0, len(treatment), size=(draws, len(treatment)))
    ].mean(axis=1)
    values = sampled_treatment - sampled_reference
    return np.quantile(values, [0.025, 0.975])


def bootstrap_mean(values, seed, draws=10000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def bh(values):
    values = np.asarray(values, float)
    output = np.full(len(values), np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return output
    p = values[finite]
    order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    output[finite] = restored
    return output


def physical_ball(radius_um, voxel_xy, voxel_z):
    rz = int(np.ceil(radius_um / voxel_z))
    rxy = int(np.ceil(radius_um / voxel_xy))
    zz, yy, xx = np.ogrid[-rz : rz + 1, -rxy : rxy + 1, -rxy : rxy + 1]
    return (zz * voxel_z) ** 2 + (yy * voxel_xy) ** 2 + (xx * voxel_xy) ** 2 <= radius_um**2 + 1e-12


def periodic_xy_dilation(mask, radius_um, voxel_xy, voxel_z):
    structure = physical_ball(radius_um, voxel_xy, voxel_z)
    rz = structure.shape[0] // 2
    ry = structure.shape[1] // 2
    rx = structure.shape[2] // 2
    padded = np.pad(mask, ((0, 0), (ry, ry), (rx, rx)), mode="wrap")
    if rz:
        padded = np.pad(
            padded,
            ((rz, rz), (0, 0), (0, 0)),
            mode="constant",
            constant_values=False,
        )
    dilated = binary_dilation(padded, structure=structure)
    return dilated[
        slice(rz, -rz) if rz else slice(None),
        slice(ry, -ry) if ry else slice(None),
        slice(rx, -rx) if rx else slice(None),
    ]


def fill_holes_by_volume(mask, max_volume_um3, voxel_xy, voxel_z):
    filled_all = ndimage.binary_fill_holes(mask)
    holes = filled_all & ~mask
    labels, n_labels = ndimage.label(holes)
    if not n_labels:
        return mask.copy()
    max_voxels = int(
        np.floor(
            max_volume_um3 / (voxel_xy * voxel_xy * voxel_z) + 1e-12,
        )
    )
    if max_voxels < 1:
        return mask.copy()
    sizes = np.bincount(labels.ravel(), minlength=n_labels + 1)
    keep = sizes <= max_voxels
    keep[0] = False
    return mask | keep[labels]


def variant_mask(mask, definition, voxel_xy, voxel_z):
    kind, direction, amount = definition
    if kind == "physical":
        structure = physical_ball(amount, voxel_xy, voxel_z)
    elif kind == "voxel":
        structure = generate_binary_structure(3, 1)
    elif kind == "component_min":
        labels, n_labels = ndimage.label(mask)
        if not n_labels:
            return mask.copy()
        min_voxels = int(np.ceil(amount / (voxel_xy * voxel_xy * voxel_z)))
        sizes = np.bincount(labels.ravel(), minlength=n_labels + 1)
        keep = sizes >= min_voxels
        keep[0] = False
        return keep[labels]
    elif kind == "hole_fill_max":
        return fill_holes_by_volume(mask, amount, voxel_xy, voxel_z)
    elif kind == "hole_fill_all":
        return ndimage.binary_fill_holes(mask)
    else:
        return mask
    if direction < 0:
        return binary_erosion(mask, structure=structure, iterations=abs(direction))
    if direction > 0:
        return binary_dilation(mask, structure=structure, iterations=direction)
    return mask


def process_one(
    row, specification, cfg, max_shifts=None, sampling_scheme="all_stacks", force=False
):
    dataset, file_id = row["dataset"], row["file_id"]
    paths = paths_for(dataset, specification, file_id)
    configuration_fingerprint = mask_sensitivity_configuration_fingerprint(
        cfg,
        max_shifts,
        sampling_scheme,
    )
    expected_torus_fingerprint = torus_configuration_fingerprint(
        cfg,
        int(cfg["torus_null"]["n_shifts"]),
    )
    if (
        not force
        and paths["summary"].exists()
        and paths["stack_summary"].exists()
        and paths["primary_summary"].exists()
        and paths["mask"].exists()
        and paths["shifts"].exists()
        and paths["threshold_masks"].exists()
    ):
        try:
            previous = json.loads(paths["summary"].read_text())
            stack_upstream = validated_stack_summary(paths["stack_summary"])
            torus_upstream = json.loads(paths["primary_summary"].read_text())
            primary_mask_fingerprint = boundary_fingerprint(paths["mask"])
            primary_shift_fingerprint = boundary_fingerprint(paths["shifts"])
            threshold_mask_fingerprint = boundary_fingerprint(paths["threshold_masks"])
        except (OSError, json.JSONDecodeError, RuntimeError):
            previous = {}
            stack_upstream = torus_upstream = {}
            primary_mask_fingerprint = None
            primary_shift_fingerprint = None
            threshold_mask_fingerprint = None
        if (
            "proximity_contact" in next(iter(previous.get("variants", {}).values()), {})
            and set(previous.get("variants", {})) == set(VARIANTS)
            and previous.get("mask_sensitivity_version") == MASK_SENSITIVITY_VERSION
            and previous.get("pipeline_version") == PIPELINE_VERSION
            and previous.get("microglia_mask_version") == MICROGLIA_MASK_VERSION
            and previous.get("torus_version") == TORUS_VERSION
            and previous.get("threshold_mask_version") == THRESHOLD_MASK_VERSION
            and previous.get("requested_max_shifts") == max_shifts
            and previous.get("stack_sampling_scheme") == sampling_scheme
            and previous.get("mask_sensitivity_configuration_fingerprint")
            == configuration_fingerprint
            and previous.get("upstream_parameter_fingerprint")
            == stack_upstream.get("parameter_fingerprint")
            and previous.get("upstream_raw_boundary_sha256")
            == stack_upstream.get("raw_boundary_sha256")
            and previous.get("primary_microglia_mask_boundary_fingerprint")
            == primary_mask_fingerprint
            and previous.get("primary_shift_boundary_fingerprint") == primary_shift_fingerprint
            and previous.get("threshold_masks_boundary_fingerprint") == threshold_mask_fingerprint
            and torus_upstream.get("torus_configuration_fingerprint") == expected_torus_fingerprint
            and torus_upstream.get("upstream_parameter_fingerprint")
            == stack_upstream.get("parameter_fingerprint")
            and torus_upstream.get("upstream_microglia_mask_boundary_fingerprint")
            == primary_mask_fingerprint
            and torus_upstream.get("iterations_boundary_fingerprint") == primary_shift_fingerprint
        ):
            return {"dataset": dataset, "file_id": file_id, "status": "skipped"}
    started = time.time()
    try:
        required = [
            "mask",
            "labels",
            "shifts",
            "primary_summary",
            "stack_summary",
            "threshold_masks",
        ]
        missing = [k for k in required if not paths[k].exists()]
        if missing:
            raise FileNotFoundError(f"missing upstream checkpoints: {missing}")
        mask_file = np.load(paths["mask"])
        primary_mask_fingerprint = boundary_fingerprint(paths["mask"])
        primary_shift_fingerprint = boundary_fingerprint(paths["shifts"])
        threshold_mask_fingerprint = boundary_fingerprint(paths["threshold_masks"])
        stored_mask_version = (
            str(mask_file["mask_version"].item()) if "mask_version" in mask_file.files else ""
        )
        primary_summary = json.loads(paths["primary_summary"].read_text())
        stack_summary = validated_stack_summary(paths["stack_summary"])
        if (
            stored_mask_version != MICROGLIA_MASK_VERSION
            or primary_summary.get("torus_version") != TORUS_VERSION
            or primary_summary.get("pipeline_version") != PIPELINE_VERSION
            or primary_summary.get("microglia_mask_version") != MICROGLIA_MASK_VERSION
            or primary_summary.get("torus_configuration_fingerprint") != expected_torus_fingerprint
            or primary_summary.get("upstream_parameter_fingerprint")
            != stack_summary.get("parameter_fingerprint")
            or primary_summary.get("upstream_microglia_mask_boundary_fingerprint")
            != primary_mask_fingerprint
            or primary_summary.get("iterations_boundary_fingerprint") != primary_shift_fingerprint
        ):
            raise RuntimeError("stale upstream result for mask sensitivity")
        mask = mask_file["binary"].astype(bool, copy=False)
        threshold_file = np.load(paths["threshold_masks"])
        stored_threshold_version = (
            str(threshold_file["threshold_mask_version"].item())
            if "threshold_mask_version" in threshold_file.files
            else ""
        )
        if stored_threshold_version != THRESHOLD_MASK_VERSION:
            raise RuntimeError("stale Iba1 threshold-sensitivity masks")
        if (
            str(threshold_file["pipeline_version"].item()) != PIPELINE_VERSION
            or str(threshold_file["microglia_mask_version"].item()) != MICROGLIA_MASK_VERSION
            or str(threshold_file["raw_boundary_sha256"].item())
            != str(row.get("boundary_sha256", ""))
        ):
            raise RuntimeError("threshold masks use a stale primary pipeline")
        threshold_masks = {
            "threshold_factor_0p70": threshold_file["factor_0p70"].astype(
                bool,
                copy=False,
            ),
            "threshold_factor_0p90": threshold_file["factor_0p90"].astype(
                bool,
                copy=False,
            ),
            "fragment_min_0um3": threshold_file["fragment_min_0um3"].astype(bool, copy=False),
            "closing_radius_0um": threshold_file["closing_radius_0um"].astype(bool, copy=False),
            "closing_radius_0p15um": threshold_file["closing_radius_0p15um"].astype(
                bool, copy=False
            ),
        }
        rebuilt_primary = threshold_file["primary_factor_0p80"].astype(
            bool,
            copy=False,
        )
        if rebuilt_primary.shape != mask.shape or not np.array_equal(
            rebuilt_primary,
            mask,
        ):
            raise RuntimeError("raw-Iba1 factor-0.80 rebuild does not reproduce primary mask")
        labels = np.load(paths["labels"])["labels"].astype(np.int32, copy=False)
        shifts = np.load(paths["shifts"])
        shifts_y, shifts_x = shifts["shift_y"], shifts["shift_x"]
        n_shifts_available = int(len(shifts_y))
        if max_shifts is not None and max_shifts < n_shifts_available:
            # Stored shifts were generated in a fixed random order.  Taking a
            # frozen prefix preserves paired comparisons across mask variants
            # and avoids outcome-dependent shift selection.
            shifts_y = shifts_y[:max_shifts]
            shifts_x = shifts_x[:max_shifts]
        if labels.shape != mask.shape:
            raise ValueError(f"mask/label shape mismatch: {mask.shape} vs {labels.shape}")
        n = int(labels.max())
        totals = np.bincount(labels.ravel(), minlength=n + 1)[1:]
        candidate_z, candidate_y, candidate_x, candidate_ids = candidate_voxels(labels)
        boundary_ids = np.unique(
            np.concatenate(
                [
                    labels[0].ravel(),
                    labels[-1].ravel(),
                    labels[:, 0, :].ravel(),
                    labels[:, -1, :].ravel(),
                    labels[:, :, 0].ravel(),
                    labels[:, :, -1].ravel(),
                ]
            )
        )
        boundary_ids = boundary_ids[boundary_ids > 0]
        complete_eligible = np.ones(n, dtype=bool)
        complete_eligible[boundary_ids - 1] = False
        full = float(cfg["torus_null"]["full_overlap_threshold"])
        voxel_xy = float(row["voxel_x_um"])
        voxel_z = float(row["voxel_z_um"])
        contact_distance = float(cfg["contact_distance_um"])
        primary_proximity_eligible = proximity_edge_eligible(
            labels,
            contact_distance,
            voxel_xy,
            voxel_z,
        )
        variants = {}
        for name, definition in VARIANTS.items():
            kind, direction, amount = definition
            perturbed = (
                threshold_masks[name]
                if kind == "prebuilt_threshold"
                else variant_mask(mask, definition, voxel_xy, voxel_z)
            )
            if perturbed.shape != mask.shape:
                raise ValueError(
                    f"threshold-mask shape mismatch for {name}: {perturbed.shape} vs {mask.shape}"
                )
            observed_contact_mask = binary_dilation(
                perturbed,
                structure=physical_ball(contact_distance, voxel_xy, voxel_z),
            )
            null_contact_mask = periodic_xy_dilation(
                perturbed,
                contact_distance,
                voxel_xy,
                voxel_z,
            )
            observed = count_overlap_sampled(
                candidate_ids,
                sample_torus(
                    perturbed,
                    candidate_z,
                    candidate_y,
                    candidate_x,
                ),
                totals,
                full,
                complete_eligible,
            )
            null = np.zeros((len(shifts_y), 3), dtype=np.int32)
            observed_contact = count_any_eligible_sampled(
                candidate_ids,
                sample_torus(
                    observed_contact_mask,
                    candidate_z,
                    candidate_y,
                    candidate_x,
                ),
                primary_proximity_eligible,
            )
            null_contact = np.zeros(len(shifts_y), dtype=np.int32)
            for i, (dy, dx) in enumerate(zip(shifts_y, shifts_x)):
                null[i] = count_overlap_sampled(
                    candidate_ids,
                    sample_torus(
                        perturbed,
                        candidate_z,
                        candidate_y,
                        candidate_x,
                        dy,
                        dx,
                    ),
                    totals,
                    full,
                    complete_eligible,
                )
                null_contact[i] = count_any_eligible_sampled(
                    candidate_ids,
                    sample_torus(
                        null_contact_mask,
                        candidate_z,
                        candidate_y,
                        candidate_x,
                        dy,
                        dx,
                    ),
                    primary_proximity_eligible,
                )
            variants[name] = {
                "perturbation_kind": kind,
                "direction_voxels": direction,
                "amount": amount,
                "microglia_voxels": int(perturbed.sum()),
                "microglia_volume_fraction": float(perturbed.mean()),
                "proximity_contact": endpoint_summary(
                    observed_contact,
                    null_contact,
                ),
                "any_overlap": endpoint_summary(observed[0], null[:, 0]),
                "complete": endpoint_summary(observed[1], null[:, 1]),
                "partial": endpoint_summary(observed[2], null[:, 2]),
            }
        for endpoint in ["proximity_contact", "any_overlap", "complete", "partial"]:
            if variants["primary"][endpoint]["observed"] != primary_summary[endpoint]["observed"]:
                raise RuntimeError(
                    f"primary-mask observed {endpoint} does not reproduce the released torus result"
                )
            if len(shifts_y) == int(primary_summary.get("n_shifts", -1)):
                if not np.isclose(
                    variants["primary"][endpoint]["null_mean"],
                    primary_summary[endpoint]["null_mean"],
                    rtol=0,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        f"primary-mask null {endpoint} does not reproduce the released torus result"
                    )
        summary = {
            "dataset": dataset,
            "specification": specification,
            "file_id": file_id,
            "marker": row["marker"],
            "condition": row["condition"],
            "animal_id": row["animal_id"],
            "region": row["region"],
            "stack_num": int(row["stack_num"]),
            "n_synapses": n,
            "n_complete_inclusion_eligible": int(complete_eligible.sum()),
            "n_primary_proximity_eligible": int(primary_proximity_eligible.sum()),
            "primary_proximity_acquisition_edge_corrected": True,
            "boundary_touching_candidates_excluded_from_complete_only": True,
            "mask_sensitivity_version": MASK_SENSITIVITY_VERSION,
            "mask_sensitivity_configuration_fingerprint": (configuration_fingerprint),
            "pipeline_version": PIPELINE_VERSION,
            "microglia_mask_version": MICROGLIA_MASK_VERSION,
            "torus_version": TORUS_VERSION,
            "threshold_mask_version": THRESHOLD_MASK_VERSION,
            "upstream_parameter_fingerprint": stack_summary["parameter_fingerprint"],
            "upstream_raw_boundary_sha256": stack_summary.get("raw_boundary_sha256"),
            "primary_microglia_mask_boundary_fingerprint": (primary_mask_fingerprint),
            "primary_shift_boundary_fingerprint": primary_shift_fingerprint,
            "threshold_masks_boundary_fingerprint": (threshold_mask_fingerprint),
            "upstream_torus_configuration_fingerprint": (expected_torus_fingerprint),
            "n_shifts": int(len(shifts_y)),
            "n_shifts_available": n_shifts_available,
            "requested_max_shifts": max_shifts,
            "shift_subset_rule": "frozen prefix of stored torus shifts",
            "stack_sampling_scheme": sampling_scheme,
            "contact_distance_um": contact_distance,
            "variant_definition": (
                "primary physical boundary sensitivity at +/-0.1 um; "
                "face-connected +/-one-acquired-voxel 3D stress test; "
                "Iba1 component cutoffs 0.3, 1, and legacy 5 um3; "
                "raw-Iba1 no-fragment-cutoff and closing-radius 0/0.15 um "
                "rebuilds around the primary 0.30-um closing; "
                "physically capped 0.015/0.05 um3 hole filling and legacy "
                "unrestricted 3-D hole filling; raw-Iba1 local-threshold "
                "factor perturbations 0.70 and 0.90 around primary 0.80"
            ),
            "observed_and_null_recomputed_together": True,
            "torus_implementation": (
                "exact_candidate_voxel_mask_sampling; periodic XY dilation "
                "for proximity null; identical 0.1-um edge-eligible candidate "
                "set for observed and null"
            ),
            "variants": variants,
            "elapsed_s": time.time() - started,
        }
        atomic_write_json(paths["summary"], summary)
        if paths["failure"].exists():
            paths["failure"].unlink()
        return {"dataset": dataset, "file_id": file_id, "status": "ok"}
    except Exception as exc:
        atomic_write_json(
            paths["failure"],
            {
                "dataset": dataset,
                "specification": specification,
                "file_id": file_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_s": time.time() - started,
            },
        )
        return {
            "dataset": dataset,
            "file_id": file_id,
            "status": "failed",
            "error": repr(exc),
        }


def add_named_family_q(frame, p_column, family_column, output_column):
    frame = frame.copy()
    frame[output_column] = np.nan
    for _, index in frame.groupby(family_column, sort=False).groups.items():
        frame.loc[index, output_column] = bh(frame.loc[index, p_column])
    return frame


def direct_marker_animal_values(region):
    """Construct TH-minus-comparator values from matched animal regions."""
    rows = []
    for keys, sub in region.groupby(
        ["dataset", "specification", "endpoint", "variant"],
    ):
        dataset, specification, endpoint, variant = keys
        for comparator in sorted(set(sub.marker) - {"TH"}):
            wide = (
                sub[sub.marker.isin(["TH", comparator])]
                .pivot_table(
                    index=["animal_id", "condition", "region"],
                    columns="marker",
                    values="log_enrichment",
                )
                .dropna(subset=["TH", comparator])
                .reset_index()
            )
            if wide.empty:
                continue
            wide["th_minus_comparator"] = wide["TH"] - wide[comparator]
            animal = wide.groupby(
                ["animal_id", "condition"],
                as_index=False,
            ).agg(
                th_minus_comparator=("th_minus_comparator", "mean"),
                n_paired_regions=("region", "nunique"),
            )
            animal["dataset"] = dataset
            animal["specification"] = specification
            animal["endpoint"] = endpoint
            animal["variant"] = variant
            animal["focal_marker"] = "TH"
            animal["comparator"] = comparator
            rows.append(animal)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def infer_direct_marker_mask_sensitivity(region, cfg, results):
    """Test direct marker contrasts and their robustness to Iba1 masks."""
    marker_animal = direct_marker_animal_values(region)
    if marker_animal.empty:
        return
    marker_animal.to_csv(
        results / "mask_sensitivity_direct_th_animal_values.tsv",
        sep="\t",
        index=False,
    )
    contrasts = {
        "LPS": [("Control", "LPS_72h"), ("Control", "LPS_60d")],
        "MPTP": [("Sham", "MPTP")],
    }
    baseline_conditions = {"LPS": "Control", "MPTP": "Sham"}
    baseline_rows = []
    interaction_rows = []
    group_columns = [
        "dataset",
        "specification",
        "endpoint",
        "variant",
        "focal_marker",
        "comparator",
    ]
    for keys, sub in marker_animal.groupby(group_columns):
        key_values = dict(zip(group_columns, keys))
        dataset = key_values["dataset"]
        baseline_values = sub.loc[
            sub.condition.eq(baseline_conditions[dataset]),
            "th_minus_comparator",
        ].to_numpy(float)
        if len(baseline_values) >= 2:
            p_value, assignments = exact_sign_flip(baseline_values)
            interval = bootstrap_mean(
                baseline_values,
                stable_seed(
                    cfg["random_seed"],
                    "mask_direct_marker_baseline",
                    *keys,
                ),
            )
            effect = float(baseline_values.mean())
            baseline_rows.append(
                {
                    **key_values,
                    "condition": baseline_conditions[dataset],
                    "n_paired_animals": len(baseline_values),
                    "effect_log_ratio": effect,
                    "th_to_comparator_enrichment_ratio": float(np.exp(effect)),
                    "ci_low_ratio": float(np.exp(interval[0])),
                    "ci_high_ratio": float(np.exp(interval[1])),
                    "exact_sign_flip_p": p_value,
                    "n_exact_sign_assignments": assignments,
                    "multiplicity_family": (
                        f"{dataset.lower()}:{key_values['specification']}:"
                        f"{key_values['endpoint']}:mask_variant_direct_th_baseline"
                    ),
                }
            )
        for reference, treatment in contrasts[dataset]:
            effect, p_value, assignments, ref, trt = exact_two_group(
                sub.th_minus_comparator,
                sub.condition,
                reference,
                treatment,
            )
            if not np.isfinite(effect):
                continue
            interval = bootstrap_difference(
                ref,
                trt,
                stable_seed(
                    cfg["random_seed"],
                    "mask_direct_marker_interaction",
                    *keys,
                    reference,
                    treatment,
                ),
            )
            interaction_rows.append(
                {
                    **key_values,
                    "reference": reference,
                    "treatment": treatment,
                    "n_reference": len(ref),
                    "n_treatment": len(trt),
                    "interaction_log_ratio": effect,
                    "interaction_ratio": float(np.exp(effect)),
                    "ci_low_ratio": float(np.exp(interval[0])),
                    "ci_high_ratio": float(np.exp(interval[1])),
                    "exact_two_sided_p": p_value,
                    "n_exact_assignments": assignments,
                    "multiplicity_family": (
                        f"{dataset.lower()}:{key_values['specification']}:"
                        f"{key_values['endpoint']}:"
                        "mask_variant_direct_th_treatment_interaction"
                    ),
                }
            )

    baseline = pd.DataFrame(baseline_rows)
    if len(baseline):
        baseline = add_named_family_q(
            baseline,
            "exact_sign_flip_p",
            "multiplicity_family",
            "bh_q_within_named_sensitivity_family",
        )
        baseline.to_csv(
            results / "mask_sensitivity_direct_th_baseline_contrasts.tsv",
            sep="\t",
            index=False,
        )
    interactions = pd.DataFrame(interaction_rows)
    if len(interactions):
        interactions = add_named_family_q(
            interactions,
            "exact_two_sided_p",
            "multiplicity_family",
            "bh_q_within_named_sensitivity_family",
        )
        interactions.to_csv(
            results / "mask_sensitivity_direct_th_treatment_interactions.tsv",
            sep="\t",
            index=False,
        )

    # Direct robustness estimand: does a mask variant change the paired
    # TH-versus-comparator baseline contrast or treatment interaction?
    index = [
        "dataset",
        "specification",
        "endpoint",
        "focal_marker",
        "comparator",
        "condition",
        "animal_id",
    ]
    wide = marker_animal.pivot_table(
        index=index,
        columns="variant",
        values="th_minus_comparator",
    ).reset_index()
    baseline_delta_rows = []
    interaction_delta_rows = []
    for variant in [name for name in VARIANTS if name != "primary"]:
        if variant not in wide or "primary" not in wide:
            continue
        working = wide.dropna(subset=[variant, "primary"]).copy()
        working["delta_variant_minus_primary"] = working[variant] - working["primary"]
        for keys, sub in working.groupby(
            [
                "dataset",
                "specification",
                "endpoint",
                "focal_marker",
                "comparator",
            ]
        ):
            dataset, specification, endpoint, focal_marker, comparator = keys
            baseline_values = sub.loc[
                sub.condition.eq(baseline_conditions[dataset]),
                "delta_variant_minus_primary",
            ].to_numpy(float)
            if len(baseline_values) >= 2:
                p_value, assignments = exact_sign_flip(baseline_values)
                interval = bootstrap_mean(
                    baseline_values,
                    stable_seed(
                        cfg["random_seed"],
                        "mask_direct_baseline_delta",
                        *keys,
                        variant,
                    ),
                )
                effect = float(baseline_values.mean())
                baseline_delta_rows.append(
                    {
                        "dataset": dataset,
                        "specification": specification,
                        "endpoint": endpoint,
                        "focal_marker": focal_marker,
                        "comparator": comparator,
                        "variant": variant,
                        "condition": baseline_conditions[dataset],
                        "n_paired_animals": len(baseline_values),
                        "change_in_paired_log_ratio_vs_primary": effect,
                        "ratio_of_paired_ratios": float(np.exp(effect)),
                        "ci_low_ratio_of_ratios": float(np.exp(interval[0])),
                        "ci_high_ratio_of_ratios": float(np.exp(interval[1])),
                        "exact_sign_flip_p": p_value,
                        "n_exact_sign_assignments": assignments,
                        "multiplicity_family": (
                            f"{dataset.lower()}:{specification}:{endpoint}:"
                            "direct_th_baseline_mask_stability"
                        ),
                    }
                )
            for reference, treatment in contrasts[dataset]:
                effect, p_value, assignments, ref, trt = exact_two_group(
                    sub.delta_variant_minus_primary,
                    sub.condition,
                    reference,
                    treatment,
                )
                if not np.isfinite(effect):
                    continue
                interval = bootstrap_difference(
                    ref,
                    trt,
                    stable_seed(
                        cfg["random_seed"],
                        "mask_direct_interaction_delta",
                        *keys,
                        variant,
                        reference,
                        treatment,
                    ),
                )
                interaction_delta_rows.append(
                    {
                        "dataset": dataset,
                        "specification": specification,
                        "endpoint": endpoint,
                        "focal_marker": focal_marker,
                        "comparator": comparator,
                        "variant": variant,
                        "reference": reference,
                        "treatment": treatment,
                        "n_reference": len(ref),
                        "n_treatment": len(trt),
                        "change_in_interaction_log_ratio_vs_primary": effect,
                        "ratio_of_interaction_ratios": float(np.exp(effect)),
                        "ci_low_ratio_of_ratios": float(np.exp(interval[0])),
                        "ci_high_ratio_of_ratios": float(np.exp(interval[1])),
                        "exact_two_sided_p": p_value,
                        "n_exact_assignments": assignments,
                        "multiplicity_family": (
                            f"{dataset.lower()}:{specification}:{endpoint}:"
                            "direct_th_interaction_mask_stability"
                        ),
                    }
                )

    baseline_delta = pd.DataFrame(baseline_delta_rows)
    if len(baseline_delta):
        baseline_delta = add_named_family_q(
            baseline_delta,
            "exact_sign_flip_p",
            "multiplicity_family",
            "bh_q_within_named_sensitivity_family",
        )
        baseline_delta.to_csv(
            results / "mask_sensitivity_direct_th_baseline_delta_vs_primary.tsv",
            sep="\t",
            index=False,
        )
    interaction_delta = pd.DataFrame(interaction_delta_rows)
    if len(interaction_delta):
        interaction_delta = add_named_family_q(
            interaction_delta,
            "exact_two_sided_p",
            "multiplicity_family",
            "bh_q_within_named_sensitivity_family",
        )
        interaction_delta.to_csv(
            results / "mask_sensitivity_direct_th_interaction_delta_vs_primary.tsv",
            sep="\t",
            index=False,
        )


def infer_mask_sensitivity(animal, cfg, results):
    contrasts = {
        "LPS": [("Control", "LPS_72h"), ("Control", "LPS_60d")],
        "MPTP": [("Sham", "MPTP")],
    }
    treatment_rows = []
    for keys, sub in animal.groupby(
        ["dataset", "specification", "marker", "endpoint", "variant"],
    ):
        dataset, specification, marker, endpoint, variant = keys
        for reference, treatment in contrasts[dataset]:
            effect, p_value, assignments, ref, trt = exact_two_group(
                sub.log_enrichment,
                sub.condition,
                reference,
                treatment,
            )
            if not np.isfinite(effect):
                continue
            interval = bootstrap_difference(
                ref,
                trt,
                stable_seed(
                    cfg["random_seed"],
                    "mask_sensitivity_treatment",
                    *keys,
                    reference,
                    treatment,
                ),
            )
            treatment_rows.append(
                {
                    "dataset": dataset,
                    "specification": specification,
                    "marker": marker,
                    "endpoint": endpoint,
                    "variant": variant,
                    "analysis_role": (
                        "primary_reproduction" if variant == "primary" else "mask_sensitivity"
                    ),
                    "reference": reference,
                    "treatment": treatment,
                    "n_reference": len(ref),
                    "n_treatment": len(trt),
                    "effect_log_enrichment_ratio": effect,
                    "effect_ratio": float(np.exp(effect)),
                    "ci_low_ratio": float(np.exp(interval[0])),
                    "ci_high_ratio": float(np.exp(interval[1])),
                    "exact_two_sided_p": p_value,
                    "n_exact_assignments": assignments,
                    "multiplicity_family": (
                        f"{dataset.lower()}:{specification}:{endpoint}:"
                        "mask_variant_treatment_effects"
                    ),
                }
            )
    treatment_effects = pd.DataFrame(treatment_rows)
    if len(treatment_effects):
        treatment_effects = add_named_family_q(
            treatment_effects,
            "exact_two_sided_p",
            "multiplicity_family",
            "bh_q_within_named_sensitivity_family",
        )
        treatment_effects.to_csv(
            results / "mask_sensitivity_treatment_effects.tsv",
            sep="\t",
            index=False,
        )

    # Direct sensitivity estimand: does changing the mask alter the treatment
    # effect relative to the primary mask? Each animal first supplies a paired
    # variant-minus-primary value; treatment labels are then permuted exactly.
    index = [
        "dataset",
        "specification",
        "marker",
        "endpoint",
        "condition",
        "animal_id",
    ]
    wide = animal.pivot_table(
        index=index,
        columns="variant",
        values="log_enrichment",
    ).reset_index()
    delta_rows = []
    baseline_rows = []
    baseline_conditions = {"LPS": "Control", "MPTP": "Sham"}
    variants = [variant for variant in VARIANTS if variant != "primary"]
    for variant in variants:
        if variant not in wide or "primary" not in wide:
            continue
        working = wide.dropna(subset=[variant, "primary"]).copy()
        working["delta_variant_minus_primary"] = working[variant] - working["primary"]
        for keys, sub in working.groupby(
            ["dataset", "specification", "marker", "endpoint"],
        ):
            dataset, specification, marker, endpoint = keys
            baseline_values = sub.loc[
                sub.condition.eq(baseline_conditions[dataset]),
                "delta_variant_minus_primary",
            ].to_numpy(float)
            if len(baseline_values) >= 2:
                p_value, assignments = exact_sign_flip(baseline_values)
                interval = bootstrap_mean(
                    baseline_values,
                    stable_seed(
                        cfg["random_seed"],
                        "mask_sensitivity_baseline_delta",
                        *keys,
                        variant,
                    ),
                )
                mean_delta = float(baseline_values.mean())
                baseline_rows.append(
                    {
                        "dataset": dataset,
                        "specification": specification,
                        "marker": marker,
                        "endpoint": endpoint,
                        "variant": variant,
                        "condition": baseline_conditions[dataset],
                        "n_animals": len(baseline_values),
                        "mean_delta_log_enrichment_vs_primary": mean_delta,
                        "ratio_variant_to_primary": float(np.exp(mean_delta)),
                        "ci_low_ratio": float(np.exp(interval[0])),
                        "ci_high_ratio": float(np.exp(interval[1])),
                        "exact_sign_flip_p": p_value,
                        "n_exact_sign_assignments": assignments,
                        "multiplicity_family": (
                            f"{dataset.lower()}:{specification}:{endpoint}:baseline_mask_stability"
                        ),
                    }
                )
            for reference, treatment in contrasts[dataset]:
                effect, p_value, assignments, ref, trt = exact_two_group(
                    sub.delta_variant_minus_primary,
                    sub.condition,
                    reference,
                    treatment,
                )
                if not np.isfinite(effect):
                    continue
                interval = bootstrap_difference(
                    ref,
                    trt,
                    stable_seed(
                        cfg["random_seed"],
                        "mask_sensitivity_effect_delta",
                        *keys,
                        variant,
                        reference,
                        treatment,
                    ),
                )
                delta_rows.append(
                    {
                        "dataset": dataset,
                        "specification": specification,
                        "marker": marker,
                        "endpoint": endpoint,
                        "variant": variant,
                        "reference": reference,
                        "treatment": treatment,
                        "n_reference": len(ref),
                        "n_treatment": len(trt),
                        "change_in_treatment_effect_log_ratio": effect,
                        "ratio_of_treatment_effect_ratios": float(np.exp(effect)),
                        "ci_low_ratio_of_ratios": float(np.exp(interval[0])),
                        "ci_high_ratio_of_ratios": float(np.exp(interval[1])),
                        "exact_two_sided_p": p_value,
                        "n_exact_assignments": assignments,
                        "multiplicity_family": (
                            f"{dataset.lower()}:{specification}:{endpoint}:"
                            "mask_treatment_effect_stability"
                        ),
                    }
                )

    effect_deltas = pd.DataFrame(delta_rows)
    if len(effect_deltas):
        effect_deltas = add_named_family_q(
            effect_deltas,
            "exact_two_sided_p",
            "multiplicity_family",
            "bh_q_within_named_sensitivity_family",
        )
        effect_deltas.to_csv(
            results / "mask_sensitivity_treatment_effect_delta_vs_primary.tsv",
            sep="\t",
            index=False,
        )
    baseline_deltas = pd.DataFrame(baseline_rows)
    if len(baseline_deltas):
        baseline_deltas = add_named_family_q(
            baseline_deltas,
            "exact_sign_flip_p",
            "multiplicity_family",
            "bh_q_within_named_sensitivity_family",
        )
        baseline_deltas.to_csv(
            results / "mask_sensitivity_baseline_delta_vs_primary.tsv",
            sep="\t",
            index=False,
        )


def collect_outputs(cfg=None):
    cfg = cfg or load_config()
    selection_path = ROOT / "results" / "mask_sensitivity_selected_stacks.tsv"
    selected_ids = None
    expected_configuration_fingerprint = None
    selected_max_shifts = None
    selected_sampling_scheme = None
    if selection_path.exists():
        selection = pd.read_csv(selection_path, sep="\t")
        schemes = selection.sampling_scheme.dropna().astype(str).unique()
        if len(schemes) != 1:
            raise RuntimeError("mask-sensitivity selection has mixed sampling schemes")
        selected_sampling_scheme = schemes[0]
        max_shift_values = selection.requested_max_shifts.unique()
        if len(max_shift_values) != 1:
            raise RuntimeError("mask-sensitivity selection has mixed shift counts")
        selected_max_shifts = None if pd.isna(max_shift_values[0]) else int(max_shift_values[0])
        expected_configuration_fingerprint = mask_sensitivity_configuration_fingerprint(
            cfg,
            selected_max_shifts,
            selected_sampling_scheme,
        )
        selected_ids = {
            dataset.lower(): set(group.file_id) for dataset, group in selection.groupby("dataset")
        }
    rows = []
    for dataset in ["lps", "mptp"]:
        if selected_ids is not None and dataset not in selected_ids:
            continue
        for specification in ["uniform_standard", "uniform_sensitive"]:
            folder = (
                ROOT
                / "data"
                / "derived"
                / dataset
                / specification
                / "mask_sensitivity"
                / "summaries"
            )
            paths = sorted(folder.glob("*.json"))
            if selected_ids is not None:
                paths = [path for path in paths if path.stem in selected_ids.get(dataset, set())]
                observed_ids = {path.stem for path in paths}
                expected_ids = selected_ids.get(dataset, set())
                if observed_ids != expected_ids:
                    raise RuntimeError(
                        f"incomplete mask sensitivity for {dataset}/"
                        f"{specification}: missing="
                        f"{len(expected_ids - observed_ids)}"
                    )
            for path in paths:
                record = json.loads(path.read_text())
                source_paths = paths_for(
                    dataset.upper(),
                    specification,
                    record.get("file_id", path.stem),
                )
                upstream = validated_stack_summary(source_paths["stack_summary"])
                if (
                    record.get("mask_sensitivity_version") != MASK_SENSITIVITY_VERSION
                    or record.get("pipeline_version") != PIPELINE_VERSION
                    or record.get("torus_version") != TORUS_VERSION
                    or record.get("threshold_mask_version") != THRESHOLD_MASK_VERSION
                    or (
                        expected_configuration_fingerprint is not None
                        and record.get("mask_sensitivity_configuration_fingerprint")
                        != expected_configuration_fingerprint
                    )
                    or record.get("upstream_parameter_fingerprint")
                    != upstream.get("parameter_fingerprint")
                    or record.get("upstream_raw_boundary_sha256")
                    != upstream.get("raw_boundary_sha256")
                    or not source_paths["mask"].exists()
                    or not source_paths["shifts"].exists()
                    or not source_paths["threshold_masks"].exists()
                    or record.get("primary_microglia_mask_boundary_fingerprint")
                    != boundary_fingerprint(source_paths["mask"])
                    or record.get("primary_shift_boundary_fingerprint")
                    != boundary_fingerprint(source_paths["shifts"])
                    or record.get("threshold_masks_boundary_fingerprint")
                    != boundary_fingerprint(source_paths["threshold_masks"])
                    or (
                        selected_sampling_scheme is not None
                        and record.get("stack_sampling_scheme") != selected_sampling_scheme
                    )
                    or (
                        selected_ids is not None
                        and record.get("requested_max_shifts") != selected_max_shifts
                    )
                ):
                    raise RuntimeError(f"stale mask-sensitivity result: {path}")
                if selected_ids is not None and record["file_id"] not in selected_ids.get(
                    dataset, set()
                ):
                    continue
                for variant, values in record["variants"].items():
                    for endpoint in ["proximity_contact", "any_overlap", "complete", "partial"]:
                        if endpoint not in values:
                            raise RuntimeError(
                                f"missing {endpoint} in mask-sensitivity result: {path} / {variant}"
                            )
                        ep = values[endpoint]
                        rows.append(
                            {
                                "dataset": record["dataset"],
                                "specification": specification,
                                "file_id": record["file_id"],
                                "marker": record["marker"],
                                "condition": record["condition"],
                                "animal_id": record["animal_id"],
                                "region": record["region"],
                                "stack_num": record["stack_num"],
                                "variant": variant,
                                "endpoint": endpoint,
                                "n_synapses": record["n_synapses"],
                                "mask_volume_fraction": values["microglia_volume_fraction"],
                                "n_shifts": record["n_shifts"],
                                "stack_sampling_scheme": record.get(
                                    "stack_sampling_scheme",
                                    "unknown",
                                ),
                                **ep,
                            }
                        )
    out = pd.DataFrame(rows)
    if out.empty:
        return
    out["log_enrichment"] = np.log((out["observed"] + 0.5) / (out["null_mean"] + 0.5))
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    out.to_csv(results / "mask_sensitivity_stack_level.tsv", sep="\t", index=False)

    # Equal stack weight within region and equal region weight within animal.
    region = out.groupby(
        [
            "dataset",
            "specification",
            "marker",
            "condition",
            "animal_id",
            "variant",
            "endpoint",
            "region",
        ],
        as_index=False,
    ).agg(log_enrichment=("log_enrichment", "mean"), n_stacks=("file_id", "nunique"))
    animal = region.groupby(
        ["dataset", "specification", "marker", "condition", "animal_id", "variant", "endpoint"],
        as_index=False,
    ).agg(
        log_enrichment=("log_enrichment", "mean"),
        n_regions=("region", "nunique"),
        n_stacks=("n_stacks", "sum"),
    )
    animal["enrichment_ratio"] = np.exp(animal["log_enrichment"])
    animal.to_csv(results / "mask_sensitivity_animal_level.tsv", sep="\t", index=False)
    infer_mask_sensitivity(animal, cfg, results)
    infer_direct_marker_mask_sensitivity(region, cfg, results)

    # Directly paired detector-specification comparison at stack level.
    primary = out[out.variant.eq("primary")]
    paired = primary.pivot_table(
        index=[
            "dataset",
            "file_id",
            "marker",
            "condition",
            "animal_id",
            "region",
            "stack_num",
            "endpoint",
        ],
        columns="specification",
        values=["n_synapses", "log_enrichment"],
    ).reset_index()
    paired.columns = [
        "_".join(str(x) for x in col if x).strip("_") if isinstance(col, tuple) else col
        for col in paired.columns
    ]
    if {"log_enrichment_uniform_standard", "log_enrichment_uniform_sensitive"}.issubset(paired):
        paired["delta_log_enrichment_sensitive_minus_standard"] = (
            paired["log_enrichment_uniform_sensitive"] - paired["log_enrichment_uniform_standard"]
        )
        paired["bouton_count_ratio_sensitive_to_standard"] = paired[
            "n_synapses_uniform_sensitive"
        ] / paired["n_synapses_uniform_standard"].replace(0, np.nan)
    paired.to_csv(results / "specification_paired_stack_sensitivity.tsv", sep="\t", index=False)


def select_one_per_animal_marker_balanced_region(manifest):
    """Choose one stack per animal x marker while balancing represented regions."""
    chosen = []
    group_keys = ["dataset", "marker", "condition"]
    for group_values, group in manifest.groupby(group_keys, sort=True):
        usage = {region: 0 for region in sorted(group.region.unique())}
        for animal_id, animal in group.groupby("animal_id", sort=True):
            available = sorted(animal.region.unique())
            minimum = min(usage[region] for region in available)
            tied = [region for region in available if usage[region] == minimum]
            tied.sort(
                key=lambda region: stable_seed(
                    "mask_sensitivity_region_balance",
                    *group_values,
                    animal_id,
                    region,
                )
            )
            region = tied[0]
            usage[region] += 1
            picked = (
                animal[animal.region.eq(region)].sort_values(["stack_num", "manifest_row"]).iloc[0]
            )
            chosen.append(picked)
    if not chosen:
        return manifest.iloc[0:0].copy()
    return pd.DataFrame(chosen).sort_values("manifest_row").reset_index(drop=True)


def run_threshold_mask_jobs(rows, workers, force):
    print(
        f"Building/checking raw-Iba1 threshold variants for {len(rows)} stacks",
        flush=True,
    )
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    started = time.time()
    if workers == 1:
        iterator = (build_threshold_masks(row, force) for row in rows)
        for i, result in enumerate(iterator, 1):
            counts[result["status"]] += 1
            if i % 10 == 0 or i == len(rows):
                print(
                    f"[threshold {i}/{len(rows)}] {(time.time() - started) / 60:.1f}m {counts}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(build_threshold_masks, row, force) for row in rows]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                counts[result["status"]] += 1
                if i % 10 == 0 or i == len(rows):
                    print(
                        f"[threshold {i}/{len(rows)}] {(time.time() - started) / 60:.1f}m {counts}",
                        flush=True,
                    )
    if counts["failed"]:
        raise SystemExit(2)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--specification", choices=["uniform_standard", "uniform_sensitive", "ALL"], default="ALL"
    )
    parser.add_argument("--dataset", choices=["LPS", "MPTP", "ALL"], default="ALL")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--file-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sampling",
        choices=[
            "all_stacks",
            "one_per_animal_region",
            "one_per_animal_marker_balanced_region",
        ],
        default=None,
    )
    parser.add_argument(
        "--max-shifts",
        type=int,
        help=(
            "Frozen prefix of stored torus shifts; defaults to the release "
            "mask-sensitivity configuration"
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--threshold-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    if args.collect_only:
        collect_outputs(cfg)
        return
    if args.sampling is None:
        args.sampling = cfg["mask_sensitivity"]["sampling_scheme"]
    if args.max_shifts is None:
        args.max_shifts = int(cfg["mask_sensitivity"]["n_shifts"])
    manifest = pd.read_csv(MANIFEST_PATH, sep="\t")
    if args.dataset != "ALL":
        manifest = manifest[manifest.dataset.eq(args.dataset)]
    if args.file_id:
        manifest = manifest[manifest.file_id.eq(args.file_id)]
    if args.sampling == "one_per_animal_region":
        manifest = (
            manifest.sort_values("stack_num")
            .groupby(
                ["dataset", "marker", "condition", "animal_id", "region"],
                as_index=False,
                sort=True,
            )
            .head(1)
            .sort_values("manifest_row")
        )
    elif args.sampling == "one_per_animal_marker_balanced_region":
        manifest = select_one_per_animal_marker_balanced_region(manifest)
    if args.limit:
        manifest = manifest.head(args.limit)
    selection = manifest[
        [
            "dataset",
            "file_id",
            "marker",
            "condition",
            "animal_id",
            "region",
            "stack_num",
            "manifest_row",
        ]
    ].copy()
    selection["sampling_scheme"] = args.sampling
    selection["requested_max_shifts"] = args.max_shifts
    if not args.threshold_only:
        selection.to_csv(
            ROOT / "results" / "mask_sensitivity_selected_stacks.tsv",
            sep="\t",
            index=False,
        )
    specifications = (
        ["uniform_standard", "uniform_sensitive"]
        if args.specification == "ALL"
        else [args.specification]
    )
    rows = manifest.to_dict("records")
    run_threshold_mask_jobs(rows, args.workers, args.force)
    if args.threshold_only:
        return
    jobs = [(row, spec) for spec in specifications for row in rows]
    print(f"Queued {len(jobs)} mask-sensitivity jobs", flush=True)
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    started = time.time()
    if args.workers == 1:
        iterator = (
            process_one(
                row,
                spec,
                cfg,
                args.max_shifts,
                args.sampling,
                args.force,
            )
            for row, spec in jobs
        )
        for i, result in enumerate(iterator, 1):
            counts[result["status"]] += 1
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {(time.time() - started) / 60:.1f}m {counts}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_one,
                    row,
                    spec,
                    cfg,
                    args.max_shifts,
                    args.sampling,
                    args.force,
                )
                for row, spec in jobs
            ]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                counts[result["status"]] += 1
                if i % 20 == 0 or i == len(jobs):
                    print(
                        f"[{i}/{len(jobs)}] {(time.time() - started) / 60:.1f}m {counts}",
                        flush=True,
                    )
    collect_outputs(cfg)
    print(counts)
    if counts["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
