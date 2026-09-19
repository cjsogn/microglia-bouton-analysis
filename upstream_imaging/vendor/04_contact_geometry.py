"""Operational 3D geometry between marker-positive candidates and Iba1 masks.

This module deliberately makes no claim about membrane crossing, lysosomal
localisation, phagocytosis, or biological clearance.  It is loaded only after
marker-channel candidate labels have been frozen.
"""

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import regionprops


_CONFIG_SPEC = importlib.util.spec_from_file_location(
    "geometry_config", str(Path(__file__).parent / "00_config.py")
)
config = importlib.util.module_from_spec(_CONFIG_SPEC)
_CONFIG_SPEC.loader.exec_module(config)

_SEG_SPEC = importlib.util.spec_from_file_location(
    "geometry_segmentation", str(Path(__file__).parent / "03_synapse_segmentation.py")
)
segmentation = importlib.util.module_from_spec(_SEG_SPEC)
_SEG_SPEC.loader.exec_module(segmentation)


def compute_distance_to_microglia(
    microglia_binary,
    voxel_xy=None,
    voxel_z=None,
):
    """Return signed Euclidean distance to the Iba1 mask in micrometres.

    Values are positive outside the mask and negative inside it.  Distances are
    between acquired voxel centres and respect anisotropic sampling.
    """
    voxel_xy = config.VOXEL_XY if voxel_xy is None else float(voxel_xy)
    voxel_z = config.VOXEL_Z if voxel_z is None else float(voxel_z)
    mask = np.asarray(microglia_binary, dtype=bool)
    spacing = (voxel_z, voxel_xy, voxel_xy)
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
    inside = ndimage.distance_transform_edt(mask, sampling=spacing)
    signed = outside
    signed[mask] = -inside[mask]
    return signed


def classify_contacts(
    candidate_labels,
    microglia_binary,
    distance_map,
    contact_distance_um=None,
    partial_threshold=None,
    full_threshold=None,
):
    """Classify marker-positive candidates using operational geometry only."""
    del partial_threshold  # Retained for compatibility with older call sites.
    contact_distance_um = (
        config.CONTACT_DISTANCE_UM if contact_distance_um is None else float(contact_distance_um)
    )
    full_threshold = (
        config.COMPLETE_INCLUSION_THRESHOLD if full_threshold is None else float(full_threshold)
    )
    labels = np.asarray(candidate_labels, dtype=np.int32)
    mask = np.asarray(microglia_binary, dtype=bool)
    ids = np.unique(labels)
    ids = ids[ids > 0]
    columns = [
        "label_id",
        "min_distance_to_microglia_um",
        "proximity_voxels",
        "overlap_voxels",
        "total_voxels",
        "overlap_fraction",
        "touches_z_boundary",
        "touches_xy_boundary",
        "complete_inclusion_eligible",
        "contact_category",
    ]
    if not len(ids):
        return pd.DataFrame(columns=columns)

    max_label = int(ids.max())
    total = np.bincount(labels.ravel(), minlength=max_label + 1)
    overlap = np.bincount(
        (labels * mask.astype(np.int32)).ravel(),
        minlength=max_label + 1,
    )
    proximity_mask = distance_map <= contact_distance_um
    proximity = np.bincount(
        (labels * proximity_mask.astype(np.int32)).ravel(),
        minlength=max_label + 1,
    )

    rows = []
    for prop in regionprops(labels, intensity_image=distance_map.astype(float)):
        label_id = int(prop.label)
        n_total = int(total[label_id])
        n_overlap = int(overlap[label_id])
        fraction = n_overlap / n_total if n_total else 0.0
        minimum_distance = float(prop.intensity_min)
        bbox = prop.bbox
        touches_z = bbox[0] == 0 or bbox[3] == labels.shape[0]
        touches_xy = (
            bbox[1] == 0 or bbox[4] == labels.shape[1] or bbox[2] == 0 or bbox[5] == labels.shape[2]
        )
        complete_eligible = not (touches_z or touches_xy)
        if fraction >= full_threshold and complete_eligible:
            category = "complete_inclusion"
        elif fraction >= full_threshold:
            category = "boundary_truncated_overlap"
        elif n_overlap > 0:
            category = "partial_overlap"
        elif minimum_distance <= contact_distance_um:
            category = "proximity_without_overlap"
        else:
            category = "no_contact"
        rows.append(
            {
                "label_id": label_id,
                "min_distance_to_microglia_um": minimum_distance,
                "proximity_voxels": int(proximity[label_id]),
                "overlap_voxels": n_overlap,
                "total_voxels": n_total,
                "overlap_fraction": fraction,
                "touches_z_boundary": bool(touches_z),
                "touches_xy_boundary": bool(touches_xy),
                "complete_inclusion_eligible": bool(complete_eligible),
                "contact_category": category,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def measure_synapses(
    candidate_labels,
    marker_raw,
    distance_map,
    marker,
    voxel_xy=None,
    voxel_z=None,
):
    """Measure every frozen marker-positive candidate in physical units."""
    voxel_xy = config.VOXEL_XY if voxel_xy is None else float(voxel_xy)
    voxel_z = config.VOXEL_Z if voxel_z is None else float(voxel_z)
    voxel_volume = voxel_xy * voxel_xy * voxel_z
    columns = [
        "label_id",
        "centroid_x_um",
        "centroid_y_um",
        "centroid_z_um",
        "volume_um3",
        "mean_intensity",
        "max_intensity",
        "integrated_intensity",
        "sphericity",
        "elongation",
        "bbox_min_x",
        "bbox_min_y",
        "bbox_min_z",
        "bbox_max_x",
        "bbox_max_y",
        "bbox_max_z",
    ]
    rows = []
    for prop in regionprops(candidate_labels, intensity_image=marker_raw):
        z, y, x = prop.centroid
        rows.append(
            {
                "label_id": int(prop.label),
                "centroid_x_um": float(x * voxel_xy),
                "centroid_y_um": float(y * voxel_xy),
                "centroid_z_um": float(z * voxel_z),
                "volume_um3": float(prop.area * voxel_volume),
                "mean_intensity": float(prop.intensity_mean),
                "max_intensity": float(prop.intensity_max),
                "integrated_intensity": float(prop.intensity_mean * prop.area),
                "sphericity": segmentation.compute_sphericity(
                    prop,
                    voxel_xy=voxel_xy,
                    voxel_z=voxel_z,
                ),
                "elongation": segmentation.compute_elongation(
                    prop,
                    voxel_xy=voxel_xy,
                    voxel_z=voxel_z,
                ),
                "bbox_min_x": int(prop.bbox[2]),
                "bbox_min_y": int(prop.bbox[1]),
                "bbox_min_z": int(prop.bbox[0]),
                "bbox_max_x": int(prop.bbox[5]),
                "bbox_max_y": int(prop.bbox[4]),
                "bbox_max_z": int(prop.bbox[3]),
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    frame["marker"] = marker
    return frame
