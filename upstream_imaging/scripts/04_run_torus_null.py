#!/usr/bin/env python3
"""Actual-label torus null with stable seeds and finite-shift p values."""

from __future__ import annotations

import argparse
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
from scipy.ndimage import binary_dilation

from common import MANIFEST_PATH, require_file
from common import (
    ROOT,
    PIPELINE_VERSION,
    MICROGLIA_MASK_VERSION,
    TORUS_VERSION,
    atomic_write_json,
    boundary_fingerprint,
    load_config,
    stable_seed,
    stack_detector_contract_is_current,
    torus_configuration_fingerprint,
)


MULTISCALE_RADII_UM = np.array([0.10, 0.20, 0.30, 0.50])


def paths_for(dataset: str, specification: str, file_id: str):
    base = ROOT / "data" / "derived" / dataset.lower()
    null = base / specification / "torus_null"
    return {
        "mg": base / "shared" / "microglia_masks" / f"{file_id}.npz",
        "labels": base / specification / "synapse_labels" / f"{file_id}.npz",
        "stack_summary": base / specification / "stack_summaries" / f"{file_id}.json",
        "iterations": null / "iterations" / f"{file_id}.npz",
        "summary": null / "summaries" / f"{file_id}.json",
        "failure": null / "failures" / f"{file_id}.json",
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


def count_overlap(labels, mask, totals, full_threshold, complete_eligible=None):
    n = len(totals)
    if n == 0:
        return 0, 0, 0, 0
    inside_ids = labels[mask]
    inside_ids = inside_ids[inside_ids > 0]
    inside = (
        np.bincount(inside_ids, minlength=n + 1)[1:] if len(inside_ids) else np.zeros(n, dtype=int)
    )
    frac = np.divide(inside, totals, out=np.zeros(n, dtype=float), where=totals > 0)
    if complete_eligible is None:
        complete_eligible = np.ones(n, dtype=bool)
    fully_overlapped = frac >= full_threshold
    complete = int(np.sum(fully_overlapped & complete_eligible))
    boundary_truncated = int(np.sum(fully_overlapped & ~complete_eligible))
    any_overlap = int(np.sum(inside > 0))
    partial = int(np.sum((inside > 0) & ~fully_overlapped))
    return any_overlap, complete, partial, boundary_truncated


def candidate_voxels(labels):
    """Return candidate voxel coordinates and label IDs in a fixed order."""
    z, y, x = np.nonzero(labels)
    ids = labels[z, y, x].astype(np.int32, copy=False)
    return z, y, x, ids


def count_overlap_sampled(
    label_ids, sampled_inside, totals, full_threshold, complete_eligible=None
):
    """Exact label-level counts from mask values sampled at candidate voxels."""
    n = len(totals)
    if n == 0:
        return 0, 0, 0, 0
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
    fully_overlapped = frac >= full_threshold
    complete = int(np.sum(fully_overlapped & complete_eligible))
    boundary_truncated = int(np.sum(fully_overlapped & ~complete_eligible))
    any_overlap = int(np.sum(inside > 0))
    partial = int(np.sum((inside > 0) & ~fully_overlapped))
    return any_overlap, complete, partial, boundary_truncated


def sample_torus(mask, z, y, x, shift_y=0, shift_x=0):
    """Sample ``roll(mask, (-dy, -dx))`` without constructing the roll."""
    return mask[
        z,
        (y + int(shift_y)) % mask.shape[1],
        (x + int(shift_x)) % mask.shape[2],
    ]


def periodic_xy_dilation(mask, radius_um, voxel_xy, voxel_z):
    """Dilate on the XY torus while retaining finite axial boundaries."""
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
    z_slice = slice(rz, -rz) if rz else slice(None)
    y_slice = slice(ry, -ry) if ry else slice(None)
    x_slice = slice(rx, -rx) if rx else slice(None)
    return dilated[z_slice, y_slice, x_slice]


def periodic_xy_distance(mask, max_radius_um, voxel_xy, voxel_z):
    """Distance to the mask on an XY torus with finite axial boundaries."""
    pad_xy = int(np.ceil(float(max_radius_um) / float(voxel_xy)))
    padded = np.pad(
        mask,
        ((0, 0), (pad_xy, pad_xy), (pad_xy, pad_xy)),
        mode="wrap",
    )
    distance = ndimage.distance_transform_edt(
        ~padded,
        sampling=(float(voxel_z), float(voxel_xy), float(voxel_xy)),
    )
    cropped = distance[
        :,
        pad_xy : pad_xy + mask.shape[1],
        pad_xy : pad_xy + mask.shape[2],
    ]
    return cropped.astype(np.float32, copy=False)


def proximity_edge_eligible(labels, radius_um, voxel_xy, voxel_z):
    """Flag labels whose voxel centres have a full radius inside the FOV."""
    n = int(labels.max())
    eligible = np.zeros(n, dtype=bool)
    if n == 0:
        return eligible
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


def physical_ball(radius_um, voxel_xy, voxel_z):
    """Discrete ellipsoid matching an EDT threshold in physical units."""
    rz = int(np.ceil(radius_um / voxel_z))
    rxy = int(np.ceil(radius_um / voxel_xy))
    zz, yy, xx = np.ogrid[-rz : rz + 1, -rxy : rxy + 1, -rxy : rxy + 1]
    return (zz * voxel_z) ** 2 + (yy * voxel_xy) ** 2 + (xx * voxel_xy) ** 2 <= radius_um**2 + 1e-12


def checkpoint_current(
    paths,
    n_shifts,
    force,
    configuration_fingerprint,
    specification,
    cfg,
):
    if (
        force
        or not paths["summary"].exists()
        or not paths["iterations"].exists()
        or not paths["stack_summary"].exists()
        or not paths["mg"].exists()
    ):
        return False
    try:
        previous = json.loads(paths["summary"].read_text())
        upstream = json.loads(paths["stack_summary"].read_text())
    except (OSError, json.JSONDecodeError):
        return False
    upstream_fingerprint = upstream.get("parameter_fingerprint")
    if not stack_detector_contract_is_current(
        upstream,
        specification,
        cfg,
    ):
        return False
    return (
        int(previous.get("n_shifts", -1)) == int(n_shifts)
        and "proximity_contact" in previous
        and "edge_corrected_multiscale_proximity" in previous
        and "boundary_truncated_overlap" in previous
        and previous.get("torus_version") == TORUS_VERSION
        and previous.get("pipeline_version") == PIPELINE_VERSION
        and previous.get("microglia_mask_version") == MICROGLIA_MASK_VERSION
        and previous.get("upstream_parameter_fingerprint") == upstream_fingerprint
        and previous.get("upstream_raw_boundary_sha256") == upstream.get("raw_boundary_sha256")
        and previous.get("torus_configuration_fingerprint") == configuration_fingerprint
        and previous.get("upstream_microglia_mask_boundary_fingerprint")
        == boundary_fingerprint(paths["mg"])
        and previous.get("iterations_boundary_fingerprint")
        == boundary_fingerprint(paths["iterations"])
    )


def process_one(row, specifications, cfg, n_shifts, force=False):
    dataset, file_id = row["dataset"], row["file_id"]
    if isinstance(specifications, str):
        specifications = [specifications]
    paths_by_spec = {spec: paths_for(dataset, spec, file_id) for spec in specifications}
    configuration_fingerprint = torus_configuration_fingerprint(
        cfg,
        n_shifts,
        MULTISCALE_RADII_UM,
    )
    needed = [
        spec
        for spec in specifications
        if not checkpoint_current(
            paths_by_spec[spec],
            n_shifts,
            force,
            configuration_fingerprint,
            spec,
            cfg,
        )
    ]
    if not needed:
        return {
            "dataset": dataset,
            "file_id": file_id,
            "status": "skipped",
            "specifications": specifications,
        }
    started = time.time()
    try:
        representative = paths_by_spec[needed[0]]
        mg_file = np.load(representative["mg"])
        stored_mask_version = (
            str(mg_file["mask_version"].item()) if "mask_version" in mg_file.files else ""
        )
        if stored_mask_version != MICROGLIA_MASK_VERSION:
            raise RuntimeError("stale microglia mask upstream of torus null")
        microglia_mask_boundary_fingerprint = boundary_fingerprint(representative["mg"])
        mg = mg_file["binary"].astype(bool, copy=False)
        full_threshold = float(cfg["torus_null"]["full_overlap_threshold"])
        voxel_xy = float(row["voxel_x_um"])
        voxel_z = float(row["voxel_z_um"])
        contact_distance = float(cfg["contact_distance_um"])
        observed_proximity_distance = ndimage.distance_transform_edt(
            ~mg,
            sampling=(voxel_z, voxel_xy, voxel_xy),
        ).astype(np.float32, copy=False)
        null_proximity_distance = periodic_xy_distance(
            mg,
            float(MULTISCALE_RADII_UM.max()),
            voxel_xy,
            voxel_z,
        )
        states = {}
        for specification in needed:
            paths = paths_by_spec[specification]
            upstream = json.loads(paths["stack_summary"].read_text())
            if (
                upstream.get("pipeline_version") != PIPELINE_VERSION
                or upstream.get("microglia_mask_version") != MICROGLIA_MASK_VERSION
                or not stack_detector_contract_is_current(
                    upstream,
                    specification,
                    cfg,
                )
                or str(upstream.get("raw_boundary_sha256", ""))
                != str(row.get("boundary_sha256", ""))
            ):
                raise RuntimeError(f"stale stack summary upstream of torus null: {specification}")
            labels = np.load(paths["labels"])["labels"].astype(
                np.int32,
                copy=False,
            )
            if mg.shape != labels.shape:
                raise ValueError(f"mask/label shape mismatch: {mg.shape} vs {labels.shape}")
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
            observed = count_overlap_sampled(
                candidate_ids,
                sample_torus(mg, candidate_z, candidate_y, candidate_x),
                totals,
                full_threshold,
                complete_eligible,
            )
            observed_candidate_distance = sample_torus(
                observed_proximity_distance,
                candidate_z,
                candidate_y,
                candidate_x,
            )
            # Use the same finite-field-eligible candidate population for the
            # observed primary endpoint and every periodic null shift.  Without
            # this small margin, a candidate on an XY acquisition face can see
            # wrapped microglia under the torus null but cannot see tissue
            # beyond the observed field.
            primary_proximity_eligible = proximity_edge_eligible(
                labels,
                contact_distance,
                voxel_xy,
                voxel_z,
            )
            observed_contact = count_any_eligible_sampled(
                candidate_ids,
                observed_candidate_distance <= contact_distance,
                primary_proximity_eligible,
            )
            # Use one common population with a full margin for the largest
            # radius.  Radius-specific sets would change abruptly at half the
            # 0.38-um axial step and confound spatial scale with selection.
            common_proximity_eligible = proximity_edge_eligible(
                labels,
                float(MULTISCALE_RADII_UM.max()),
                voxel_xy,
                voxel_z,
            )
            proximity_eligible = np.repeat(
                common_proximity_eligible[None, :],
                len(MULTISCALE_RADII_UM),
                axis=0,
            )
            observed_curve = np.array(
                [
                    count_any_eligible_sampled(
                        candidate_ids,
                        observed_candidate_distance <= float(radius),
                        proximity_eligible[radius_index],
                    )
                    for radius_index, radius in enumerate(MULTISCALE_RADII_UM)
                ],
                dtype=np.int32,
            )
            states[specification] = {
                "paths": paths,
                "labels": labels,
                "n": n,
                "totals": totals,
                "candidate_z": candidate_z,
                "candidate_y": candidate_y,
                "candidate_x": candidate_x,
                "candidate_ids": candidate_ids,
                "complete_eligible": complete_eligible,
                "primary_proximity_eligible": primary_proximity_eligible,
                "proximity_eligible": proximity_eligible,
                "observed_proximity_curve": observed_curve,
                "observed": observed,
                "observed_contact": observed_contact,
                "upstream": upstream,
                "null_any": np.zeros(n_shifts, dtype=np.int32),
                "null_complete": np.zeros(n_shifts, dtype=np.int32),
                "null_partial": np.zeros(n_shifts, dtype=np.int32),
                "null_boundary": np.zeros(n_shifts, dtype=np.int32),
                "null_contact": np.zeros(n_shifts, dtype=np.int32),
                "null_proximity_curve": np.zeros(
                    (n_shifts, len(MULTISCALE_RADII_UM)),
                    dtype=np.int32,
                ),
            }
        min_shift = int(np.ceil(float(cfg["torus_null"]["minimum_shift_um"]) / voxel_xy))
        _, height, width = mg.shape
        if height <= 2 * min_shift or width <= 2 * min_shift:
            raise ValueError("field too small for requested minimum torus shift")
        common_seed = stable_seed(
            cfg["random_seed"],
            dataset,
            file_id,
            "torus_common_shift_field",
        )
        rng = np.random.default_rng(common_seed)
        shifts_y = rng.integers(min_shift, height - min_shift, size=n_shifts, endpoint=False)
        shifts_x = rng.integers(min_shift, width - min_shift, size=n_shifts, endpoint=False)
        for i, (dy, dx) in enumerate(zip(shifts_y, shifts_x)):
            for state in states.values():
                sampled = sample_torus(
                    mg,
                    state["candidate_z"],
                    state["candidate_y"],
                    state["candidate_x"],
                    dy,
                    dx,
                )
                values = count_overlap_sampled(
                    state["candidate_ids"],
                    sampled,
                    state["totals"],
                    full_threshold,
                    state["complete_eligible"],
                )
                state["null_any"][i] = values[0]
                state["null_complete"][i] = values[1]
                state["null_partial"][i] = values[2]
                state["null_boundary"][i] = values[3]
                sampled_distance = sample_torus(
                    null_proximity_distance,
                    state["candidate_z"],
                    state["candidate_y"],
                    state["candidate_x"],
                    dy,
                    dx,
                )
                state["null_contact"][i] = count_any_eligible_sampled(
                    state["candidate_ids"],
                    sampled_distance <= contact_distance,
                    state["primary_proximity_eligible"],
                )
                for radius_index, radius in enumerate(MULTISCALE_RADII_UM):
                    state["null_proximity_curve"][i, radius_index] = count_any_eligible_sampled(
                        state["candidate_ids"],
                        sampled_distance <= float(radius),
                        state["proximity_eligible"][radius_index],
                    )

        def endpoint_summary(observed, null):
            mean = float(np.mean(null))
            return {
                "observed": int(observed),
                "null_mean": mean,
                "null_sd": float(np.std(null, ddof=1)) if len(null) > 1 else 0.0,
                "null_q025": float(np.quantile(null, 0.025)),
                "null_q975": float(np.quantile(null, 0.975)),
                "enrichment": float(observed / mean) if mean > 0 else None,
                "p_upper_finite": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
                "p_lower_finite": float((1 + np.sum(null <= observed)) / (len(null) + 1)),
            }

        for specification, state in states.items():
            paths = state["paths"]
            atomic_npz(
                paths["iterations"],
                shift_y=shifts_y.astype(np.int32),
                shift_x=shifts_x.astype(np.int32),
                n_any=state["null_any"],
                n_complete=state["null_complete"],
                n_partial=state["null_partial"],
                n_boundary_truncated=state["null_boundary"],
                n_proximity_contact=state["null_contact"],
                proximity_radii_um=MULTISCALE_RADII_UM.astype(np.float32),
                n_edge_corrected_proximity_curve=state["null_proximity_curve"],
            )
            observed = state["observed"]
            multiscale = {
                f"{float(radius):.2f}": {
                    "radius_um": float(radius),
                    "n_edge_eligible_candidates": int(
                        state["proximity_eligible"][radius_index].sum()
                    ),
                    **endpoint_summary(
                        state["observed_proximity_curve"][radius_index],
                        state["null_proximity_curve"][:, radius_index],
                    ),
                }
                for radius_index, radius in enumerate(MULTISCALE_RADII_UM)
            }
            summary = {
                "dataset": dataset,
                "specification": specification,
                "file_id": file_id,
                "marker": row["marker"],
                "condition": row["condition"],
                "animal_id": row["animal_id"],
                "region": row["region"],
                "stack_num": int(row["stack_num"]),
                "n_synapses": state["n"],
                "n_shifts": int(n_shifts),
                "n_complete_inclusion_eligible": int(state["complete_eligible"].sum()),
                "n_primary_proximity_eligible": int(state["primary_proximity_eligible"].sum()),
                "primary_proximity_acquisition_edge_corrected": True,
                "boundary_touching_candidates_excluded_from_complete_only": True,
                "torus_version": TORUS_VERSION,
                "torus_configuration_fingerprint": configuration_fingerprint,
                "pipeline_version": PIPELINE_VERSION,
                "microglia_mask_version": MICROGLIA_MASK_VERSION,
                "upstream_parameter_fingerprint": state["upstream"]["parameter_fingerprint"],
                "upstream_raw_boundary_sha256": state["upstream"].get("raw_boundary_sha256"),
                "upstream_microglia_mask_boundary_fingerprint": (
                    microglia_mask_boundary_fingerprint
                ),
                "iterations_boundary_fingerprint": boundary_fingerprint(paths["iterations"]),
                "seed": common_seed,
                "shift_field_shared_across_specifications": True,
                "torus_implementation": (
                    "exact candidate-voxel sampling of finite-field observed "
                    "and periodic-XY null distance transforms; identical "
                    "0.1-um edge-eligible candidate set for the primary "
                    "proximity endpoint"
                ),
                "minimum_shift_px": min_shift,
                "contact_distance_um": contact_distance,
                "proximity_contact": endpoint_summary(
                    state["observed_contact"],
                    state["null_contact"],
                ),
                "edge_corrected_multiscale_proximity": multiscale,
                "multiscale_proximity_role": (
                    "exploratory; primary endpoint remains the prespecified, "
                    "acquisition-edge-corrected "
                    f"{contact_distance:.2f}-um proximity; all curve radii "
                    "use the same candidates with a full 0.50-um FOV margin"
                ),
                "any_overlap": endpoint_summary(observed[0], state["null_any"]),
                "complete": endpoint_summary(
                    observed[1],
                    state["null_complete"],
                ),
                "partial": endpoint_summary(observed[2], state["null_partial"]),
                "boundary_truncated_overlap": endpoint_summary(
                    observed[3],
                    state["null_boundary"],
                ),
                "elapsed_s": time.time() - started,
            }
            atomic_write_json(paths["summary"], summary)
            if paths["failure"].exists():
                paths["failure"].unlink()
        return {
            "dataset": dataset,
            "file_id": file_id,
            "status": "ok",
            "specifications": needed,
            "elapsed_s": time.time() - started,
        }
    except Exception as exc:
        for specification in needed:
            paths = paths_by_spec[specification]
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
        return {"dataset": dataset, "file_id": file_id, "status": "failed", "error": repr(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--specification",
        choices=["uniform_standard", "uniform_sensitive", "ALL"],
        required=True,
    )
    parser.add_argument("--dataset", choices=["LPS", "MPTP", "ALL"], default="ALL")
    parser.add_argument("--marker", action="append")
    parser.add_argument("--condition", action="append")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--n-shifts", type=int)
    parser.add_argument("--file-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    n_shifts = args.n_shifts or int(cfg["torus_null"]["n_shifts"])
    manifest = pd.read_csv(require_file(MANIFEST_PATH), sep="\t")
    if args.dataset != "ALL":
        manifest = manifest[manifest.dataset.eq(args.dataset)]
    if args.marker:
        manifest = manifest[manifest.marker.isin(args.marker)]
    if args.condition:
        manifest = manifest[manifest.condition.isin(args.condition)]
    if args.file_id:
        manifest = manifest[manifest.file_id.eq(args.file_id)]
    if args.limit:
        manifest = manifest.head(args.limit)
    rows = manifest.to_dict("records")
    if not rows:
        raise SystemExit("No manifest rows selected")
    specifications = (
        ["uniform_standard", "uniform_sensitive"]
        if args.specification == "ALL"
        else [args.specification]
    )
    print(
        f"Queued {len(rows)} torus stack jobs for {','.join(specifications)}, "
        f"{n_shifts} shared shifts each",
        flush=True,
    )
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    started = time.time()
    if args.workers == 1:
        # Avoid platform semaphore requirements for deterministic serial runs
        # (including validation pilots and restricted compute environments).
        for i, row in enumerate(rows, 1):
            result = process_one(row, specifications, cfg, n_shifts, args.force)
            counts[result["status"]] += 1
            if i % 20 == 0 or i == len(rows):
                print(
                    f"[{i}/{len(rows)}] elapsed={(time.time() - started) / 60:.1f}m {counts}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_one,
                    row,
                    specifications,
                    cfg,
                    n_shifts,
                    args.force,
                )
                for row in rows
            ]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                counts[result["status"]] += 1
                if i % 20 == 0 or i == len(rows):
                    print(
                        f"[{i}/{len(rows)}] elapsed={(time.time() - started) / 60:.1f}m {counts}",
                        flush=True,
                    )
    print(counts)
    if counts["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
