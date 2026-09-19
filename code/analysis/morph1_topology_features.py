# -*- coding: utf-8 -*-
"""Extract terminal-path and auxiliary skeleton descriptors from retained 3D masks.

Terminal-path persistence is the difference in soma-rooted geodesic distance
between a tip and the first branch point reached toward the root. This operational
descriptor is not a full persistent-homology reconstruction of the arbor.
Distances use the recorded anisotropic voxel spacing in micrometres.

Additional descriptors are retained for reproducibility. They should not be
interpreted as validated biological states. Processing failures are reported,
and the downstream analysis validates coverage of the final eligible cell set."""

from __future__ import annotations


import os as _os
import sys as _sys
from pathlib import Path as _P

PKG = _P(
    _os.environ.get("MG_PACKAGE_ROOT")  # override to relocate
    or next(
        p
        for p in _P(__file__).resolve().parents  # else the dir holding
        if (p / "code").is_dir()
    )
)  # code/, at any depth
for _d in (_P(__file__).resolve().parent, PKG / "code", PKG / "code/analysis"):
    _sys.path.insert(0, str(_d))  # siblings, and across analysis subdirectories

from mg_paths import (
    MORPHOLOGY,
    output_dir,
)

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage.morphology import skeletonize

# Raw stacks, segmentation masks and the derived per-cell tables live outside the
# code tree because of their size. One constant points at them.
MORPH = MORPHOLOGY
OUT = output_dir("morphology")
MASK_DIR = MORPH / "output" / "masks_3d_split_v6"
VX, VZ = 0.1516, 0.5203
SPACING = np.array([VZ, VX, VX])
N_WORKERS = int(_os.environ.get("MG_WORKERS", "14"))
NEIGH = np.array(
    [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) != (0, 0, 0)
    ]
)


def skeleton_graph(skel):
    """Adjacency over skeleton voxels, 26-connected, with physical edge lengths."""
    pts = np.argwhere(skel)
    index = {tuple(p): i for i, p in enumerate(pts)}
    adj = defaultdict(list)
    for i, p in enumerate(pts):
        for d in NEIGH:
            q = tuple(p + d)
            j = index.get(q)
            if j is not None:
                adj[i].append((j, float(np.linalg.norm(d * SPACING))))
    return pts, adj


def geodesic(adj, n, root):
    """Dijkstra along the skeleton from the soma node."""
    import heapq

    dist = np.full(n, np.inf)
    dist[root] = 0.0
    pq = [(0.0, root)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return dist


def cell_features(mask, dt_um):
    """All descriptors for one binary cell mask. Returns None if the skeleton is too small."""
    skel = skeletonize(mask)
    if skel.sum() < 10:
        return None
    pts, adj = skeleton_graph(skel)
    n = len(pts)
    deg = np.array([len(adj[i]) for i in range(n)])
    # soma = skeleton node with the largest inscribed radius
    radii = np.array([dt_um[tuple(p)] for p in pts])
    root = int(np.argmax(radii))
    gd = geodesic(adj, n, root)
    if not np.isfinite(gd).any():
        return None
    ends = np.where(deg == 1)[0]
    branch = set(np.where(deg >= 3)[0].tolist())

    # ---- persistence: walk each terminal toward the soma until a branch point -----------
    bars, tort, taper = [], [], []
    for e in ends:
        if not np.isfinite(gd[e]):
            continue
        path, cur, prev = [e], e, -1
        while True:
            nxt = [v for v, _ in adj[cur] if v != prev and np.isfinite(gd[v]) and gd[v] < gd[cur]]
            if not nxt:
                break
            prev, cur = cur, min(nxt, key=lambda v: gd[v])
            path.append(cur)
            if cur in branch or cur == root:
                break
        if len(path) < 3:
            continue
        birth, death = gd[e], gd[cur]
        bars.append((birth, death))
        plen = birth - death  # path length along the branch
        straight = float(np.linalg.norm((pts[e] - pts[cur]) * SPACING))
        if straight > 1e-6:
            tort.append(plen / straight)
        r_prox, r_dist = radii[cur], radii[e]
        if plen > 1e-6:
            taper.append((r_prox - r_dist) / plen)

    if len(bars) < 2:
        return None
    lens = np.array([b - d for b, d in bars])
    lens = lens[lens > 0]
    if len(lens) < 2:
        return None
    p = lens / lens.sum()
    entropy = float(-(p * np.log(p + 1e-12)).sum())

    # ---- bifurcation angles ------------------------------------------------------------
    angles = []
    for b in branch:
        dirs = []
        for v, _ in adj[b]:
            step = (pts[v] - pts[b]) * SPACING
            nrm = np.linalg.norm(step)
            if nrm > 1e-6:
                dirs.append(step / nrm)
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                c = float(np.clip(np.dot(dirs[i], dirs[j]), -1, 1))
                angles.append(np.degrees(np.arccos(c)))

    # ---- Strahler order ------------------------------------------------------------------
    order = {}
    for i in np.argsort(-gd):
        i = int(i)
        if not np.isfinite(gd[i]):
            continue
        kids = [v for v, _ in adj[i] if np.isfinite(gd[v]) and gd[v] > gd[i]]
        if not kids:
            order[i] = 1
        else:
            ko = sorted((order.get(k, 1) for k in kids), reverse=True)
            order[i] = ko[0] + 1 if len(ko) > 1 and ko[0] == ko[1] else ko[0]

    feats = dict(
        tmd_n_bars=len(lens),
        tmd_total_persistence=float(lens.sum()),
        tmd_max_persistence=float(lens.max()),
        tmd_mean_persistence=float(lens.mean()),
        tmd_entropy=entropy,
        tmd_bars_gt5um=int((lens > 5).sum()),
        bifurcation_angle_mean=float(np.mean(angles)) if angles else np.nan,
        bifurcation_angle_sd=float(np.std(angles)) if len(angles) > 1 else np.nan,
        tortuosity_mean=float(np.mean(tort)) if tort else np.nan,
        tortuosity_max=float(np.max(tort)) if tort else np.nan,
        taper_mean=float(np.mean(taper)) if taper else np.nan,
        strahler_max=int(max(order.values())) if order else 1,
        path_length_max_um=float(np.nanmax(gd[np.isfinite(gd)])),
    )
    return feats, lens


def process_fov(stem):
    path = MASK_DIR / f"masks3d_{stem}.tif"
    masks = tifffile.imread(str(path))
    rows, bars = [], []
    for cid in np.unique(masks):
        if cid == 0:
            continue
        sl = ndimage.find_objects(masks == cid)[0]
        pad = tuple(slice(max(0, s.start - 2), s.stop + 2) for s in sl)
        crop = masks[pad] == cid
        dt = ndimage.distance_transform_edt(crop, sampling=SPACING)
        try:
            res = cell_features(crop, dt)
        except Exception as exc:
            raise RuntimeError(f"Topology failed for field {stem}, label {int(cid)}") from exc
        if res:
            f, lens = res
            f.update(fov=stem, label=int(cid))
            rows.append(f)
            bars.extend(dict(fov=stem, label=int(cid), persistence=float(v)) for v in lens)
    return rows, bars


def main():
    stems = [
        p.name.replace("masks3d_", "").replace(".tif", "")
        for p in sorted(MASK_DIR.glob("masks3d_*.tif"))
    ]
    if not stems:
        raise FileNotFoundError(f"No retained masks found in {MASK_DIR}")
    out, allbars = [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(process_fov, s): s for s in stems}
        for k, fut in enumerate(as_completed(futs), 1):
            r_, b_ = fut.result()
            out.extend(r_)
            allbars.extend(b_)
            if k % 50 == 0:
                print(f"  {k}/{len(stems)} fields, {len(out)} cells")
    df = pd.DataFrame(out).sort_values(["fov", "label"]).reset_index(drop=True)
    df.to_csv(OUT / "topology_features_v6.csv", index=False)
    pd.DataFrame(allbars).sort_values(["fov", "label", "persistence"]).to_csv(
        OUT / "topology_barcodes_v6.csv", index=False
    )
    print(f"\n{len(df)} cells, {df.fov.nunique()} fields")
    print(df.drop(columns=["fov", "label"]).describe().T[["mean", "std"]].to_string())
    print(f"\nwritten {OUT / 'topology_features_v6.csv'}")


if __name__ == "__main__":
    main()
