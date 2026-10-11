"""
Microglia segmentation from the iba1 channel.

Produces a binary mask of microglial cytoplasm (soma and processes) via 3D
Gaussian smoothing, local adaptive thresholding with a global Otsu floor,
morphological closing and volume-based fragment removal.  The primary mask
does not fill enclosed background cavities because Iba1 is not a nuclear
marker; capped hole filling is available only as a sensitivity parameter.
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import binary_closing, remove_small_objects
from skimage.filters import threshold_local

from pathlib import Path
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "config", str(Path(__file__).parent / "00_config.py")
)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)


def smooth_microglia(image, sigma_zyx=None):
    """
    3D Gaussian smoothing for iba1 channel before thresholding.

    Parameters
    ----------
    image : ndarray (Z, Y, X), preprocessed microglia channel
    sigma_zyx : tuple (sigma_z, sigma_y, sigma_x) in pixels

    Returns
    -------
    smoothed : ndarray (Z, Y, X)
    """
    if sigma_zyx is None:
        sigma_zyx = (
            config.MICROGLIA_SMOOTH_SIGMA_PX[2],  # Z
            config.MICROGLIA_SMOOTH_SIGMA_PX[1],  # Y
            config.MICROGLIA_SMOOTH_SIGMA_PX[0],  # X
        )
    return ndimage.gaussian_filter(image, sigma=sigma_zyx).astype(np.float32)


def adaptive_threshold_3d(image, block_size=None, factor=None):
    """
    Local adaptive thresholding combined with global Otsu threshold.

    A pixel is foreground if:
    1. Its intensity > local_mean * factor (local criterion), AND
    2. Its intensity > global_otsu_threshold * 0.5 (global floor to exclude
       uniform background regions where adaptive threshold is unreliable)

    Parameters
    ----------
    image : ndarray (Z, Y, X)
    block_size : int, must be odd, size of local region in pixels
    factor : float, threshold = local_mean * factor

    Returns
    -------
    binary : ndarray (Z, Y, X) bool
    """
    from skimage.filters import threshold_otsu

    if block_size is None:
        block_size = config.ADAPTIVE_BLOCK_SIZE_PX
    if factor is None:
        factor = config.ADAPTIVE_THRESHOLD_FACTOR

    # Ensure odd
    if block_size % 2 == 0:
        block_size += 1

    # Global Otsu threshold as floor (prevents classifying uniform background)
    try:
        global_thresh = threshold_otsu(image)
    except Exception:
        global_thresh = np.percentile(image, 95)
    global_floor = global_thresh * 0.5

    binary = np.zeros(image.shape, dtype=bool)

    for z in range(image.shape[0]):
        sl = image[z]
        local_mean = threshold_local(sl, block_size, method="mean", offset=0)
        # Both criteria: above local adaptive threshold AND above global floor
        binary[z] = (sl > (local_mean * factor)) & (sl > global_floor)

    return binary


def _physical_ball(radius_um, voxel_xy, voxel_z):
    rz = int(np.ceil(radius_um / voxel_z))
    rxy = int(np.ceil(radius_um / voxel_xy))
    zz, yy, xx = np.ogrid[-rz : rz + 1, -rxy : rxy + 1, -rxy : rxy + 1]
    return (zz * voxel_z) ** 2 + (yy * voxel_xy) ** 2 + (xx * voxel_xy) ** 2 <= radius_um**2 + 1e-12


def fill_holes_by_volume(binary, max_hole_volume_um3, voxel_xy=None, voxel_z=None):
    """Fill only enclosed background components below a physical volume cap.

    A non-positive cap returns the input unchanged.  ``None`` reproduces the
    unrestricted ``binary_fill_holes`` operation for sensitivity analyses.
    """
    if voxel_xy is None:
        voxel_xy = config.VOXEL_XY
    if voxel_z is None:
        voxel_z = config.VOXEL_Z
    if max_hole_volume_um3 is not None and max_hole_volume_um3 <= 0:
        return binary.copy()

    filled_all = ndimage.binary_fill_holes(binary)
    if max_hole_volume_um3 is None:
        return filled_all

    holes = filled_all & ~binary
    hole_labels, n_holes = ndimage.label(holes)
    if n_holes == 0:
        return binary.copy()
    voxel_volume = float(voxel_xy * voxel_xy * voxel_z)
    max_voxels = int(np.floor(max_hole_volume_um3 / voxel_volume + 1e-12))
    if max_voxels < 1:
        return binary.copy()
    sizes = np.bincount(hole_labels.ravel(), minlength=n_holes + 1)
    keep = sizes <= max_voxels
    keep[0] = False
    return binary | keep[hole_labels]


def morphological_refinement(
    binary,
    closing_radius_um=None,
    min_fragment_voxels=None,
    voxel_xy=None,
    voxel_z=None,
    max_hole_volume_um3=None,
):
    """
    Morphological post-processing of microglia binary mask.

    Steps:
    1. Binary closing to connect thin processes
    2. Remove small fragments (noise)
    3. Optionally fill only enclosed holes below a physical volume cap

    Parameters
    ----------
    binary : ndarray (Z, Y, X) bool
    closing_radius_um : float
    min_fragment_voxels : int
    max_hole_volume_um3 : float or None
        Physical cap for hole filling.  The configured primary value is 0
        (no filling).  Passing ``None`` explicitly is not interpreted as
        unrestricted filling here; use :func:`fill_holes_by_volume` directly for
        that sensitivity operation.

    Returns
    -------
    refined : ndarray (Z, Y, X) bool
    """
    if closing_radius_um is None:
        closing_radius_um = config.CLOSING_RADIUS_UM
    if min_fragment_voxels is None:
        min_fragment_voxels = config.MIN_MICROGLIA_VOXELS
    if voxel_xy is None:
        voxel_xy = config.VOXEL_XY
    if voxel_z is None:
        voxel_z = config.VOXEL_Z
    if max_hole_volume_um3 is None:
        max_hole_volume_um3 = config.MAX_MICROGLIA_HOLE_VOLUME_UM3

    # Closing connects thin processes into continuous branches
    if closing_radius_um > 0:
        selem = _physical_ball(closing_radius_um, voxel_xy, voxel_z)
        refined = binary_closing(binary, selem)
    else:
        refined = binary.copy()

    refined = remove_small_objects(refined, min_size=min_fragment_voxels)

    refined = fill_holes_by_volume(
        refined,
        max_hole_volume_um3,
        voxel_xy=voxel_xy,
        voxel_z=voxel_z,
    )

    return refined


def filter_by_volume(binary, min_volume_um3=None, voxel_xy=None, voxel_z=None):
    """
    Remove Iba1-positive fragments below the physical noise-floor volume.

    Parameters
    ----------
    binary : ndarray (Z, Y, X) bool
    min_volume_um3 : float
    voxel_xy, voxel_z : float

    Returns
    -------
    filtered : ndarray (Z, Y, X) bool
    n_objects : int, number of remaining objects
    labels : ndarray (Z, Y, X) int, labeled connected components
    """
    if min_volume_um3 is None:
        min_volume_um3 = config.MIN_MICROGLIA_VOLUME_UM3
    if voxel_xy is None:
        voxel_xy = config.VOXEL_XY
    if voxel_z is None:
        voxel_z = config.VOXEL_Z

    voxel_volume = voxel_xy * voxel_xy * voxel_z
    min_voxels = int(round(min_volume_um3 / voxel_volume))

    labels, n_objects = ndimage.label(binary)

    if min_voxels <= 0:
        return binary, n_objects, labels

    # Vectorized label lookup avoids rescanning the full 3-D field once for
    # every component (typically tens to hundreds of redundant scans).
    sizes = np.bincount(labels.ravel(), minlength=n_objects + 1)
    keep_ids = np.flatnonzero(sizes >= min_voxels)
    keep_ids = keep_ids[keep_ids > 0]
    remap = np.zeros(n_objects + 1, dtype=np.int32)
    remap[keep_ids] = np.arange(1, len(keep_ids) + 1, dtype=np.int32)
    new_labels = remap[labels]
    filtered = new_labels > 0
    return filtered, int(len(keep_ids)), new_labels


# Full microglia segmentation pipeline


def segment_microglia(microglia_preprocessed, voxel_xy=None, voxel_z=None, params=None):
    """
    Full microglia segmentation pipeline.

    Parameters
    ----------
    microglia_preprocessed : ndarray (Z, Y, X), preprocessed iba1 channel
    voxel_xy, voxel_z : float
    params : dict, optional parameter overrides

    Returns
    -------
    dict with:
        'binary': ndarray (Z, Y, X) bool, binary mask of all microglia
        'labels': ndarray (Z, Y, X) int, labeled individual components
        'n_objects': int
        'volume_fraction': float, fraction of total volume occupied by microglia
    """
    if voxel_xy is None:
        voxel_xy = config.VOXEL_XY
    if voxel_z is None:
        voxel_z = config.VOXEL_Z

    p = params or {}

    smoothed = smooth_microglia(microglia_preprocessed)

    binary = adaptive_threshold_3d(
        smoothed,
        block_size=p.get("block_size", config.ADAPTIVE_BLOCK_SIZE_PX),
        factor=p.get("factor", config.ADAPTIVE_THRESHOLD_FACTOR),
    )

    binary = morphological_refinement(
        binary,
        closing_radius_um=p.get("closing_radius_um", config.CLOSING_RADIUS_UM),
        min_fragment_voxels=p.get("min_fragment_voxels", config.MIN_MICROGLIA_VOXELS),
        voxel_xy=voxel_xy,
        voxel_z=voxel_z,
        max_hole_volume_um3=p.get(
            "max_hole_volume_um3",
            config.MAX_MICROGLIA_HOLE_VOLUME_UM3,
        ),
    )

    binary, n_objects, labels = filter_by_volume(
        binary,
        min_volume_um3=p.get("min_volume_um3", config.MIN_MICROGLIA_VOLUME_UM3),
        voxel_xy=voxel_xy,
        voxel_z=voxel_z,
    )

    volume_fraction = np.mean(binary)

    return {
        "binary": binary,
        "labels": labels,
        "n_objects": n_objects,
        "volume_fraction": volume_fraction,
    }
