"""Three-dimensional hysteresis segmentation and soma-seeded watershed."""

import gc

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.feature import peak_local_max
from skimage.filters import apply_hysteresis_threshold, threshold_otsu, threshold_triangle
from skimage.measure import label as relabel, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from . import parameters as config


def threshold_microglia_3d(stack):
    """Keep dim voxels connected to Otsu-positive signal in three dimensions."""
    if stack.max() == stack.min():
        return np.zeros(stack.shape, dtype=bool)
    noise_floor = stack.min() + (stack.max() - stack.min()) * 0.01
    vals = stack[stack > noise_floor]
    if len(vals) < 100:
        return np.zeros(stack.shape, dtype=bool)
    thresh_high = threshold_otsu(vals)
    thresh_low = threshold_triangle(vals)
    if thresh_low >= thresh_high:
        thresh_low = thresh_high * 0.5
    binary = apply_hysteresis_threshold(stack, thresh_low, thresh_high)
    binary = remove_small_objects(binary, min_size=30)
    # Connectivity 2 in three dimensions includes 18 neighbours.
    struct = ndimage.generate_binary_structure(3, 2)
    return ndimage.binary_closing(binary, structure=struct, iterations=1)


def detect_soma_seeds_3d(binary):
    """Find soma seeds in the smoothed physical distance transform."""
    distance = distance_transform_edt(binary, sampling=config.SPACING)
    distance_smooth = gaussian_filter(
        distance,
        sigma=(config.DT_SMOOTH_SIGMA_Z, config.DT_SMOOTH_SIGMA_XY, config.DT_SMOOTH_SIGMA_XY),
    )
    labeled = relabel(binary)
    min_dist_voxels = max(int(round(config.SOMA_RADIUS_UM / config.VOXEL_SIZE_Z)), 3)
    coords = peak_local_max(
        distance_smooth,
        min_distance=min_dist_voxels,
        threshold_abs=config.MIN_DT_PEAK_3D,
        labels=labeled,
    )
    del distance_smooth, labeled
    gc.collect()
    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (z, y, x) in enumerate(coords, start=1):
        markers[z, y, x] = i
    return markers, distance, len(coords)


def segment_stack(stack):
    """Apply the recorded volume and axial-boundary rules to a restored stack.

    Foreground is relabelled after filtering, as in the retained processing.
    Adjacent foreground can therefore form merged labels. The separate soma-core
    splitting stage is required before these masks are used for cell analysis.
    """
    shape = stack.shape
    binary = threshold_microglia_3d(stack)
    if not binary.any():
        return np.zeros(shape, dtype=np.int32)
    markers, distance, n_seeds = detect_soma_seeds_3d(binary)
    if n_seeds == 0:
        return np.zeros(shape, dtype=np.int32)
    labels = watershed(-distance, markers=markers, mask=binary)
    del binary, markers, distance
    gc.collect()
    minimum = config.MIN_CELL_VOL_UM3 / config.VOXEL_VOLUME
    maximum = config.MAX_CELL_VOL_UM3 / config.VOXEL_VOLUME
    mask = np.zeros_like(labels)
    for prop in regionprops(labels):
        if prop.area < minimum or prop.area > maximum:
            continue
        if prop.bbox[0] == 0 or prop.bbox[3] == shape[0]:
            continue
        mask[labels == prop.label] = prop.label
    del labels
    gc.collect()
    return relabel(mask > 0).astype(np.int32)
