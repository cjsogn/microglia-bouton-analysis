#!/usr/bin/env python3
"""Shared paths, configuration, hashing, and atomic-write helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(CODE_ROOT.parent / "code"))
from mg_paths import OUTPUT, require_file as require_file

ROOT = OUTPUT / "upstream_imaging"
CONFIG_PATH = CODE_ROOT / "config/reanalysis_config.json"
MANIFEST_PATH = Path(
    os.environ.get("MG_RAW_MANIFEST") or ROOT / "data/manifests/raw_stack_manifest.tsv"
)

# Release-contract versions are centralized so processing, aggregation, and
# downstream sensitivity tools cannot silently accept mutually stale outputs.
PIPELINE_VERSION = "physical_geometry_v2_11_boundary_category_summary"
MICROGLIA_MASK_VERSION = "physical_closing_process_fragments_no_hole_fill_v4"
TORUS_VERSION = "torus_v3_4_primary_edge_eligible_multiscale_distance"


@lru_cache(maxsize=1)
def _vendor_detector_config():
    """Load the frozen lightweight marker-parameter module once."""
    path = CODE_ROOT / "vendor" / "00_config.py"
    spec = importlib.util.spec_from_file_location(
        "release_detector_config",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load detector configuration: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_marker_parameters(
    dataset: str,
    marker: str,
    specification: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Return the exact intended detector dictionary for one stack class."""
    detector = _vendor_detector_config()
    params = dict(detector.get_marker_params(marker))
    if str(dataset).upper() == "MPTP" and marker == "TH":
        params.update(cfg["mptp_th_base"])
    spec = cfg["specifications"][specification]
    params["min_volume_um3"] = float(params["min_volume_um3"]) * spec["min_volume_factor"]
    params["max_volume_um3"] = float(params["max_volume_um3"]) * spec["max_volume_factor"]
    params["noise_sigma_k"] = float(params["noise_sigma_k"]) * spec["foreground_noise_k_factor"]
    params["global_intensity_factor"] = spec["global_intensity_factor"]
    if "rescue_min_volume_um3" in params:
        params["rescue_min_volume_um3"] = (
            float(params["rescue_min_volume_um3"]) * spec["min_volume_factor"]
        )
    if specification == "uniform_sensitive":
        params["raw_intensity_floor_policy"] = "uniform_standard_stack_reference_v1"
    return params


def expected_detector_parameter_fingerprint(
    dataset: str,
    marker: str,
    specification: str,
    cfg: dict[str, Any],
) -> str:
    return canonical_sha256(current_marker_parameters(dataset, marker, specification, cfg))


def _same_optional_finite_number(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(left_value) and math.isfinite(right_value) and left_value == right_value


def sensitive_intensity_reference_is_exact(record: dict[str, Any]) -> bool:
    """Verify the two numeric sensitive floors equal their frozen reference."""
    candidate_count = record.get("intensity_reference_candidate_count")
    try:
        valid_count = int(candidate_count) >= 0
    except (TypeError, ValueError):
        valid_count = False
    return bool(
        _same_optional_finite_number(
            record.get("object_noise_floor"),
            record.get("intensity_reference_object_noise_floor"),
        )
        and _same_optional_finite_number(
            record.get("rescue_base_floor"),
            record.get("intensity_reference_rescue_base_floor"),
        )
        and valid_count
    )


def stack_detector_contract_is_current(
    record: dict[str, Any],
    specification: str,
    cfg: dict[str, Any],
) -> bool:
    """Validate exact detector intent, including the sensitive reference."""
    parameters = record.get("parameters")
    expected = expected_detector_parameter_fingerprint(
        record.get("dataset", ""),
        record.get("marker", ""),
        specification,
        cfg,
    )
    current = (
        isinstance(parameters, dict)
        and record.get("parameter_fingerprint") == canonical_sha256(parameters)
        and record.get("parameter_fingerprint") == expected
    )
    if specification == "uniform_sensitive":
        standard = expected_detector_parameter_fingerprint(
            record.get("dataset", ""),
            record.get("marker", ""),
            "uniform_standard",
            cfg,
        )
        current = current and (
            record.get("raw_intensity_acceptance_policy")
            == "exact standard-stack marker-only floors"
            and record.get("intensity_reference_parameter_fingerprint") == standard
            and parameters.get("raw_intensity_floor_policy")
            == "uniform_standard_stack_reference_v1"
            and sensitive_intensity_reference_is_exact(record)
        )
    return bool(current)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def stable_seed(*parts: object, modulus: int = 2**32 - 1) -> int:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulus


def canonical_sha256(value: Any) -> str:
    """Hash JSON-like content independently of dict insertion order."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def torus_configuration_fingerprint(
    cfg: dict[str, Any],
    n_shifts: int,
    multiscale_radii_um=(0.10, 0.20, 0.30, 0.50),
) -> str:
    """Hash every configured choice that changes the released torus null."""
    return canonical_sha256(
        {
            "torus_version": TORUS_VERSION,
            "n_shifts": int(n_shifts),
            "random_seed": int(cfg["random_seed"]),
            "minimum_shift_um": float(cfg["torus_null"]["minimum_shift_um"]),
            "full_overlap_threshold": float(cfg["torus_null"]["full_overlap_threshold"]),
            "contact_distance_um": float(cfg["contact_distance_um"]),
            "multiscale_radii_um": [float(x) for x in multiscale_radii_um],
            "primary_acquisition_edge_correction": True,
            "multiscale_common_maximum_radius_margin": True,
        }
    )


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def boundary_fingerprint(path: str | Path, chunk_size: int = 1 << 20) -> str:
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode("ascii"))
    with path.open("rb") as handle:
        h.update(handle.read(chunk_size))
        if size > chunk_size:
            handle.seek(max(0, size - chunk_size))
            h.update(handle.read(chunk_size))
    return h.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def environment_summary() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
