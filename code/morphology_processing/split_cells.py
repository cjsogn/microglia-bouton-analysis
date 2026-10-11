"""Separate merged cell masks using connected soma cores and watershed."""

import numpy as np
from scipy.ndimage import distance_transform_edt, label as ndi_label
from skimage.segmentation import watershed

from . import parameters as config

CONN_3D = np.ones((3, 3, 3), dtype=bool)


def find_dt_seeds(cell_binary):
    """Find maxima within connected cores meeting the recorded radius and volume rules."""
    if cell_binary.sum() == 0:
        return []
    dist = distance_transform_edt(cell_binary, sampling=config.SPACING)
    soma_mask = dist >= config.DT_THRESH_UM
    if soma_mask.sum() == 0:
        return []
    cc, n = ndi_label(soma_mask, structure=CONN_3D)
    sizes = np.bincount(cc.ravel())
    min_vox = max(1, int(config.MIN_REGION_UM3 / config.VOXEL_VOLUME))
    seeds = []
    for k in range(1, n + 1):
        if sizes[k] < min_vox:
            continue
        coords_k = np.argwhere(cc == k)
        dt_vals = dist[coords_k[:, 0], coords_k[:, 1], coords_k[:, 2]]
        seeds.append(tuple(coords_k[int(np.argmax(dt_vals))]))
    return seeds


def split_cell(cell_binary, seeds):
    """Split a merged mask and absorb undersized parts into the largest remaining part."""
    if len(seeds) <= 1:
        return None
    dist = distance_transform_edt(cell_binary, sampling=config.SPACING)
    markers = np.zeros(cell_binary.shape, dtype=np.int32)
    for i, (z, y, x) in enumerate(seeds, start=1):
        markers[z, y, x] = i
    sub = watershed(-dist, markers=markers, mask=cell_binary)
    sub_volumes, sub_dts = {}, {}
    for label in range(1, sub.max() + 1):
        mask = sub == label
        if mask.sum() == 0:
            continue
        sub_volumes[label] = int(mask.sum())
        sub_dts[label] = float(distance_transform_edt(mask, sampling=config.SPACING).max())
    changed = True
    while changed and len(sub_volumes) > 1:
        changed = False
        labels = sorted(sub_volumes, key=lambda label: sub_volumes[label])
        for label in labels:
            volume_um3 = sub_volumes[label] * config.VOXEL_VOLUME
            if (
                volume_um3 < config.MIN_SUB_VOL_UM3 or sub_dts[label] < config.MIN_SUB_DT_UM
            ) and len(sub_volumes) > 1:
                other = max(
                    (key for key in sub_volumes if key != label),
                    key=lambda key: sub_volumes[key],
                )
                sub[sub == label] = other
                sub_volumes[other] += sub_volumes[label]
                sub_dts[other] = float(
                    distance_transform_edt(sub == other, sampling=config.SPACING).max()
                )
                del sub_volumes[label], sub_dts[label]
                changed = True
                break
    if len(sub_volumes) <= 1:
        return None
    new = np.zeros_like(sub)
    for new_id, old_id in enumerate(sorted(sub_volumes), start=1):
        new[sub == old_id] = new_id
    return new


def split_labels(labels):
    """Return final labels and splitting counts, preserving the recorded label assignment."""
    masks = labels.copy()
    if masks.max() == 0:
        return masks, 0, 0
    next_id = int(masks.max()) + 1
    n_split, n_new_total = 0, 0
    cell_ids = [int(cell) for cell in np.unique(masks) if cell > 0]
    for cell_id in cell_ids:
        cell = masks == cell_id
        if cell.sum() < 50:
            continue
        coords = np.argwhere(cell)
        z0, z1 = coords[:, 0].min(), coords[:, 0].max() + 1
        y0, y1 = coords[:, 1].min(), coords[:, 1].max() + 1
        x0, x1 = coords[:, 2].min(), coords[:, 2].max() + 1
        z0p, z1p = max(0, z0 - 1), min(cell.shape[0], z1 + 1)
        y0p, y1p = max(0, y0 - config.PAD), min(cell.shape[1], y1 + config.PAD)
        x0p, x1p = max(0, x0 - config.PAD), min(cell.shape[2], x1 + config.PAD)
        sub = cell[z0p:z1p, y0p:y1p, x0p:x1p].copy()
        seeds = find_dt_seeds(sub)
        if len(seeds) <= 1:
            continue
        new_sub = split_cell(sub, seeds)
        if new_sub is None or int(new_sub.max()) <= 1:
            continue
        n_new = int(new_sub.max())
        masks[cell] = 0
        for label in range(1, n_new + 1):
            sub_mask = new_sub == label
            if sub_mask.sum() == 0:
                continue
            full_mask = np.zeros_like(masks, dtype=bool)
            full_mask[z0p:z1p, y0p:y1p, x0p:x1p] = sub_mask
            masks[full_mask] = next_id
            next_id += 1
        n_split += 1
        n_new_total += n_new
    return masks, n_split, n_new_total
