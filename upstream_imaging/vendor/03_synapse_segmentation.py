"""Outcome-blind segmentation of marker-positive bouton candidates.

The detector uses physically scaled multi-scale LoG responses, h-maxima seeds,
and marker-controlled watershed.  Candidate acceptance is based only on the
marker channel; Iba1 is never accepted as an input to this module.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import maximum_filter, label as ndi_label
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.morphology import reconstruction, remove_small_objects
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import threshold_otsu, threshold_triangle
import warnings

from pathlib import Path
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "config", str(Path(__file__).parent / "00_config.py")
)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)


def estimate_noise_sigma_mad(data):
    """
    Estimate noise standard deviation from *data* using the Median Absolute
    Deviation (MAD).  sigma = MAD * 1.4826 (consistency factor for Gaussian).

    Unlike percentile-based estimators, MAD is insensitive to a small fraction
    of signal outliers, making it ideal for images where most voxels are
    background noise but some contain real signal.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.size == 0:
        return 0.0
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    return float(mad * 1.4826)


def compute_rescue_base_floor(bg_unlabeled, is_axon_marker, marker_params):
    """Return the globally applicable floor used by the late rescue path.

    A supplied fixed floor takes precedence over every per-stack estimate so
    that the rescue path cannot reintroduce condition-dependent filtering.
    ``None`` is returned when an adaptive estimate lacks enough background.
    """
    fixed_floor = marker_params.get("fixed_noise_floor")
    reference_floor = marker_params.get("precomputed_rescue_base_floor")
    if fixed_floor is not None:
        base_floor = float(fixed_floor)
    elif reference_floor is not None:
        base_floor = float(reference_floor)
    else:
        bg_unlabeled = np.asarray(bg_unlabeled)
        bg_nonzero = bg_unlabeled[bg_unlabeled > 0]
        if len(bg_nonzero) <= 100:
            return None
        bg_med = float(np.median(bg_nonzero))
        bg_mad = estimate_noise_sigma_mad(bg_nonzero)
        base_floor = bg_med + marker_params.get("intensity_sigma_k", 3.0) * bg_mad
        if is_axon_marker:
            base_floor = max(
                base_floor,
                3.0 * float(np.percentile(bg_unlabeled, 99)),
            )
    return base_floor * float(marker_params.get("global_intensity_factor", 1.0))


def compute_log_response(image, sigma_range_um=None, n_scales=None, voxel_xy=None, voxel_z=None):
    """
    Compute scale-normalized Laplacian of Gaussian response across multiple scales.

    At each voxel, returns the maximum normalized LoG response across all scales.
    High values indicate likely bouton centers at the corresponding scale.

    Parameters
    ----------
    image : ndarray (Z, Y, X), preprocessed synapse channel
    sigma_range_um : tuple (min_sigma, max_sigma) in um
    n_scales : int, number of logarithmically spaced scales
    voxel_xy, voxel_z : float, voxel dimensions in um

    Returns
    -------
    log_response : ndarray (Z, Y, X), maximum normalized LoG response
    best_scale : ndarray (Z, Y, X), index of best-responding scale per voxel
    """
    if sigma_range_um is None:
        sigma_range_um = config.LOG_SIGMA_RANGE_UM
    if n_scales is None:
        n_scales = config.LOG_N_SCALES
    if voxel_xy is None:
        voxel_xy = config.VOXEL_XY
    if voxel_z is None:
        voxel_z = config.VOXEL_Z

    # Logarithmically spaced sigmas in physical units
    sigmas_um = np.logspace(np.log10(sigma_range_um[0]), np.log10(sigma_range_um[1]), n_scales)

    log_response = np.full(image.shape, -np.inf, dtype=np.float32)
    best_scale = np.zeros(image.shape, dtype=np.int8)

    for i, sigma_um in enumerate(sigmas_um):
        # Convert physical sigma to anisotropic voxel sigma
        sigma_xy_px = sigma_um / voxel_xy
        sigma_z_px = sigma_um / voxel_z
        sigma_zyx = (sigma_z_px, sigma_xy_px, sigma_xy_px)

        # Smooth at an isotropic *physical* scale, then evaluate the discrete
        # Laplacian in physical coordinates.  scipy.ndimage.laplace by itself
        # sums second differences in voxel-index units and therefore overweights
        # the axial term by (voxel_z / voxel_xy)^2 (about 18.5 here).
        # A zero-sum finite-difference kernel is used after smoothing because
        # derivative-of-Gaussian kernels are numerically biased when sigma_z is
        # below one acquired slice at the smallest scales.
        smoothed = ndimage.gaussian_filter(image, sigma=sigma_zyx)
        second_difference = np.array([1.0, -2.0, 1.0], dtype=np.float32)
        laplacian = (
            ndimage.correlate1d(smoothed, second_difference, axis=0, mode="reflect") / (voxel_z**2)
            + ndimage.correlate1d(smoothed, second_difference, axis=1, mode="reflect")
            / (voxel_xy**2)
            + ndimage.correlate1d(smoothed, second_difference, axis=2, mode="reflect")
            / (voxel_xy**2)
        )

        # Scale-normalized LoG (negative because we want bright blobs)
        # LoG gives negative response at blob centers for bright blobs
        norm_log = -(sigma_um**2) * laplacian

        # Keep maximum response across scales
        mask = norm_log > log_response
        log_response[mask] = norm_log[mask]
        best_scale[mask] = i

    # Clip negative values (noise floor)
    log_response = np.maximum(log_response, 0)

    return log_response, best_scale


def _physical_ellipsoid(radius_xy_um, radius_z_um, voxel_xy, voxel_z):
    """Boolean ellipsoid whose axes are specified in micrometres."""
    rxy = max(0, int(np.ceil(radius_xy_um / voxel_xy)))
    rz = max(0, int(np.ceil(radius_z_um / voxel_z)))
    zz, yy, xx = np.ogrid[-rz : rz + 1, -rxy : rxy + 1, -rxy : rxy + 1]
    z_term = (
        (zz * voxel_z / radius_z_um) ** 2 if radius_z_um > 0 else np.where(zz == 0, 0.0, np.inf)
    )
    y_term = (
        (yy * voxel_xy / radius_xy_um) ** 2 if radius_xy_um > 0 else np.where(yy == 0, 0.0, np.inf)
    )
    x_term = (
        (xx * voxel_xy / radius_xy_um) ** 2 if radius_xy_um > 0 else np.where(xx == 0, 0.0, np.inf)
    )
    return z_term + y_term + x_term <= 1.0 + 1e-12


def detect_seeds(
    log_response,
    h_fraction=None,
    min_distance_px=None,
    noise_sigma=None,
    min_distance_um=None,
    voxel_xy=None,
    voxel_z=None,
):
    """
    Detect bouton center seeds using h-maxima transform on LoG response.

    The h-maxima transform suppresses peaks with prominence < h, separating
    fused boutons. Each surviving peak = one bouton center.

    Parameters
    ----------
    log_response : ndarray (Z, Y, X), from compute_log_response
    h_fraction : float, h as fraction of median peak LoG response
    min_distance_px : int, minimum distance between seeds in XY pixels
    noise_sigma : float, optional
        MAD-based noise sigma of the LoG response.  When provided, seeds
        whose peak LoG value falls below ``2 * noise_sigma`` are discarded
        (they arise from noise-level LoG fluctuations, not real puncta).

    Returns
    -------
    seeds : ndarray (Z, Y, X) of int, labeled seed points (each seed = unique label)
    n_seeds : int
    seed_coords : ndarray (N, 3) of seed positions (z, y, x)
    """
    if h_fraction is None:
        h_fraction = config.H_MAXIMA_FRACTION
    if voxel_xy is None:
        voxel_xy = config.VOXEL_XY
    if voxel_z is None:
        voxel_z = config.VOXEL_Z
    if min_distance_um is None:
        if min_distance_px is None:
            min_distance_um = config.MIN_SEED_DISTANCE_UM
        else:
            min_distance_um = float(min_distance_px) * voxel_xy

    # Compute h adaptively from the LoG response distribution
    # Use the median of the top 1% of LoG values as reference
    nonzero = log_response[log_response > 0]
    if len(nonzero) == 0:
        return np.zeros(log_response.shape, dtype=np.int32), 0, np.empty((0, 3))

    p99 = np.percentile(nonzero, 99)
    peaks_above = nonzero[nonzero > p99 * 0.3]
    if len(peaks_above) == 0:
        peaks_above = nonzero
    median_peak = np.median(peaks_above)
    h = h_fraction * median_peak

    # h-maxima transform: suppress peaks with prominence < h
    # Implemented as: original - reconstruction(original - h, original)
    seed_image = log_response.astype(np.float64)
    shifted = seed_image - h
    shifted = np.clip(shifted, 0, None)

    # Morphological reconstruction (geodesic dilation of shifted under original)
    reconstructed = reconstruction(shifted, seed_image, method="dilation")

    # Peaks = voxels where original > reconstructed
    # (i.e., peaks that stood above the h-suppression)
    peak_map = seed_image - reconstructed
    peak_mask = peak_map > 0

    # Find local maxima with a physical-distance footprint.  At least the
    # immediately adjacent axial slice is included so the same axially blurred
    # punctum is not seeded independently on consecutive slices.
    footprint = _physical_ellipsoid(
        float(min_distance_um),
        max(float(min_distance_um), float(voxel_z)),
        float(voxel_xy),
        float(voxel_z),
    )
    max_filtered = maximum_filter(seed_image, footprint=footprint)
    local_max = (seed_image == max_filtered) & peak_mask & (seed_image > 0)

    # Noise quality filter: discard seeds whose peak LoG is below 2x the
    # MAD-estimated noise sigma.  These arise from random noise fluctuations
    # and do not correspond to real puncta.
    if noise_sigma is not None and noise_sigma > 0:
        local_max = local_max & (seed_image > 2.0 * noise_sigma)

    # Label connected seed regions and take centroids
    seeds_labeled, n_seeds = ndi_label(local_max)

    if n_seeds == 0:
        return seeds_labeled, 0, np.empty((0, 3))

    # Get seed coordinates (centroids of labeled regions)
    props = regionprops(seeds_labeled)
    seed_coords = np.array([p.centroid for p in props])

    return seeds_labeled, n_seeds, seed_coords


def compute_foreground_mask(
    log_response,
    synapse_preprocessed=None,
    seeds=None,
    noise_sigma_k=3.0,
    max_foreground_fraction=0.25,
    min_foreground_voxels=0,
    min_fg_threshold=None,
    fixed_fg_threshold=None,
):
    """
    Compute foreground mask using MAD-based threshold on the LoG response,
    hysteresis cleaning (seed-connected components only), and intensity-based
    recovery for large bright structures.

    Three stages:
      1. MAD-based threshold: foreground = LoG > k * sigma_noise_mad, where
         k = noise_sigma_k (per-marker).  Floored at median(positive LoG) to
         handle degenerate cases where MAD is very small.
      2. Hysteresis cleaning: label foreground connected components and discard
         any component that does not contain at least one seed.  This prevents
         watershed from growing into disconnected noise patches.
      3. Bright recovery (preserved): for large axon boutons whose LoG fades
         at the periphery.

    Parameters
    ----------
    log_response : ndarray (Z, Y, X)
    synapse_preprocessed : ndarray (Z, Y, X), optional
        Preprocessed synapse channel. Enables bright recovery for large boutons.
    seeds : ndarray (Z, Y, X) of int, optional
        Labeled seed map from detect_seeds().  When provided, enables
        hysteresis cleaning (only seed-connected foreground kept).
    noise_sigma_k : float
        Multiplier for MAD-based LoG noise threshold.  Higher = stricter.
    max_foreground_fraction : float, safety cap
    min_foreground_voxels : int
        Minimum number of voxels for a foreground connected component to
        survive hysteresis cleaning, even if it contains a seed.  Prevents
        tiny noise-level foreground fragments from spawning objects.
        Typically set to min_volume_um3 / voxel_volume.

    Returns
    -------
    foreground : ndarray (Z, Y, X) bool
    threshold : float
    method : str
    """
    # Get positive LoG values (negative values already clipped to 0)
    positive = log_response[log_response > 0]
    if len(positive) < 100:
        return np.zeros(log_response.shape, dtype=bool), 0.0, "empty"

    # Stage 1: threshold = max(MAD-based, Otsu)
    # MAD threshold works well for low-SNR punctate markers where Otsu fails
    # (unimodal LoG distribution).  Otsu works well for axon markers where
    # signal and noise are clearly bimodal.  Taking the maximum of both
    # ensures neither case regresses.
    sigma_mad = estimate_noise_sigma_mad(positive)
    mad_threshold = noise_sigma_k * sigma_mad

    try:
        otsu_threshold = float(threshold_otsu(positive))
    except Exception:
        otsu_threshold = 0.0

    # Floor: at least the median of positive LoG values (handles degenerate
    # cases where MAD ~ 0 because nearly all positive values are identical)
    median_positive = float(np.median(positive))
    mad_threshold = max(mad_threshold, median_positive)

    # Fixed threshold overrides all per-stack computation (cross-group uniformity)
    if fixed_fg_threshold is not None:
        threshold = float(fixed_fg_threshold)
        method = f"fixed({fixed_fg_threshold:.1f})"
    else:
        threshold = max(mad_threshold, otsu_threshold)
        if mad_threshold >= otsu_threshold:
            method = f"mad_k{noise_sigma_k:.1f}"
        else:
            method = f"otsu(>{mad_threshold:.2f}mad)"

        # Apply minimum absolute LoG threshold floor
        if min_fg_threshold is not None and threshold < min_fg_threshold:
            threshold = min_fg_threshold
            method = f"floor({min_fg_threshold:.1f})"

    foreground = log_response > threshold

    # Stage 2: Hysteresis cleaning (keep only seed-connected components)
    if seeds is not None and np.any(seeds > 0):
        fg_labels, n_fg_cc = ndi_label(foreground)
        if n_fg_cc > 0:
            # Build lookup arrays: for each foreground CC label, (a) does
            # it contain a seed? (b) is it large enough?  O(n_voxels).
            has_seed = np.zeros(n_fg_cc + 1, dtype=bool)
            seed_mask = seeds > 0
            fg_labels_at_seeds = fg_labels[seed_mask]
            fg_labels_at_seeds = fg_labels_at_seeds[fg_labels_at_seeds > 0]
            if len(fg_labels_at_seeds) > 0:
                has_seed[np.unique(fg_labels_at_seeds)] = True

            # Size filter: discard tiny CCs even if they contain a seed.
            # Tiny foreground islands from noise produce one-seed objects
            # that inflate counts (especially for axon markers).
            if min_foreground_voxels > 0:
                cc_sizes = np.bincount(fg_labels.ravel(), minlength=n_fg_cc + 1)
                big_enough = cc_sizes >= min_foreground_voxels
                big_enough[0] = True  # background label always "OK"
                has_seed = has_seed & big_enough

            # Discard CCs without seeds (or too small)
            keep_mask = has_seed[fg_labels]
            n_removed = int(np.sum(foreground) - np.sum(keep_mask))
            if n_removed > 0:
                foreground = keep_mask
                method += "+hysteresis"

    # Stage 3: Two-tier bright recovery for large boutons
    # Large boutons have strong LoG at edges but near-zero LoG at their
    # flat interior (LoG responds to curvature, not flat plateaus).
    # Tier 1: moderate brightness (>P25 of foreground) + moderate LoG.
    #         Recovers bouton periphery where LoG is weakening.
    # Tier 2: very bright (>P50 of foreground), purely intensity-based.
    #         Recovers flat interiors of large bright boutons where the
    #         LoG response is exactly 0 (clipped from negative).  No LoG
    #         requirement because the intensity threshold alone is already
    #         very selective (P50 of LoG-thresholded foreground).
    if synapse_preprocessed is not None:
        fg_intensities = synapse_preprocessed[foreground]
        if len(fg_intensities) > 100:
            tier1_int_thresh = np.percentile(fg_intensities, 25)
            tier2_int_thresh = np.percentile(fg_intensities, 75)
        else:
            bg_pos = synapse_preprocessed[synapse_preprocessed > 0]
            p99 = np.percentile(bg_pos, 99) if len(bg_pos) > 0 else 0
            tier1_int_thresh = p99
            tier2_int_thresh = p99 * 1.5

        low_log_thresh = max(threshold * 0.10, 0.5)

        # Tier 1: moderate bright + moderate LoG
        tier1 = (synapse_preprocessed > tier1_int_thresh) & (log_response > low_log_thresh)
        # Tier 2: very bright, no LoG requirement (captures flat interiors)
        tier2 = synapse_preprocessed > tier2_int_thresh

        bright_recovery = tier1 | tier2
        n_recovered = np.sum(bright_recovery & ~foreground)
        if n_recovered > 0:
            foreground = foreground | bright_recovery
            method += "+bright_recovery"

    # Safety cap: if foreground exceeds max fraction, raise threshold
    fg_fraction = np.mean(foreground)
    if fg_fraction > max_foreground_fraction:
        target_pctile = (1.0 - max_foreground_fraction) * 100
        threshold = np.percentile(positive, target_pctile)
        foreground = log_response > threshold
        method = f"capped_{max_foreground_fraction:.0%}"

    return foreground, float(threshold), method


def watershed_segmentation(log_response, seeds, foreground, synapse_preprocessed=None):
    """
    Marker-controlled 3D watershed with adaptive post-watershed trimming.

    Seeds grow into their local basins, constrained by the foreground mask.
    After watershed, each label is trimmed to keep voxels above 45%
    of the label's peak intensity (capped at the median to handle saturation).
    This prevents boutons from inflating into surrounding low-signal regions
    while preserving the full body of large bright boutons.

    Parameters
    ----------
    log_response : ndarray (Z, Y, X)
    seeds : ndarray (Z, Y, X) of int, labeled seed points
    foreground : ndarray (Z, Y, X) bool
    synapse_preprocessed : ndarray (Z, Y, X), optional
        Used for post-watershed intensity trimming

    Returns
    -------
    labels : ndarray (Z, Y, X) of int, each label = one bouton
    """
    # Watershed landscape: inverted LoG (grow from peaks into valleys)
    landscape = -log_response.astype(np.float64)

    # Constrain to foreground
    labels = watershed(landscape, markers=seeds, mask=foreground)

    # Post-watershed intensity trimming: shrink each label to its bright
    # core. Watershed territories extend to boundaries with neighbors,
    # producing masks larger than the actual fluorescent puncta. Trimming
    # to voxels above a fraction of the label's PEAK intensity captures
    # only the real bouton signal.
    #
    # The threshold is floored at the background noise level to prevent
    # noise objects (low peak) from inflating below the noise floor.
    # The object-median cap prevents saturated peaks from leaving only a tiny
    # core after the 45%-of-peak rule.
    if synapse_preprocessed is not None and labels.max() > 0:
        # Estimate background noise level for trimming floor
        bg_vals = synapse_preprocessed[labels == 0]
        if len(bg_vals) > 0:
            trim_floor = np.percentile(bg_vals, 95)
        else:
            trim_floor = 0

        props = regionprops(labels, intensity_image=synapse_preprocessed)
        trimmed = np.zeros_like(labels)

        for prop in props:
            # Restrict per-object work to the object's bounding box.  The
            # previous full-volume equality test was mathematically correct
            # but rescanned every voxel once per object (often hundreds of
            # full 3-D scans per stack).
            bbox = prop.bbox
            slc = tuple(slice(bbox[i], bbox[i + 3]) for i in range(3))
            labels_local = labels[slc]
            signal_local = synapse_preprocessed[slc]
            label_mask = labels_local == prop.label
            peak_intensity = prop.intensity_max
            # Adaptive threshold: 45% of peak, but capped at the object's
            # median intensity. For large
            # bright boutons with saturated peaks (peak >> median), the
            # cap prevents the threshold from being so high that only
            # the saturated core survives.
            object_vals = signal_local[label_mask]
            median_intensity = np.median(object_vals)
            local_thresh = max(min(peak_intensity * 0.45, median_intensity), trim_floor)
            keep = label_mask & (signal_local > local_thresh)
            if np.any(keep):
                trimmed_local = trimmed[slc]
                trimmed_local[keep] = prop.label

        labels = trimmed

    # Do not apply a fixed 3x3 opening here.  It contradicts the prespecified
    # physical minimum volumes (for example, VGLUT1 candidates may validly
    # occupy fewer than nine lateral pixels) and makes acceptance depend on an
    # undocumented pixel-scale rule.  Noise and size rejection occur below.

    return labels


def compute_elongation(prop, voxel_xy=None, voxel_z=None):
    """
    Compute elongation ratio from regionprops object.

    Uses the principal variances of voxel-centre coordinates in physical 3D
    space.  A within-voxel cuboid variance term prevents a one-slice object
    from being treated as having zero axial thickness.
    """
    if voxel_xy is None:
        voxel_xy = config.VOXEL_XY
    if voxel_z is None:
        voxel_z = config.VOXEL_Z
    coords = np.asarray(prop.coords, dtype=np.float64)
    if not len(coords):
        return 1.0
    spacing = np.array([voxel_z, voxel_xy, voxel_xy], dtype=np.float64)
    centres = (coords + 0.5) * spacing
    if len(centres) > 1:
        covariance = np.cov(centres, rowvar=False, bias=True)
    else:
        covariance = np.zeros((3, 3), dtype=np.float64)
    covariance += np.diag((spacing**2) / 12.0)
    eigvals = np.linalg.eigvalsh(covariance)
    eigvals = np.clip(eigvals, np.finfo(float).eps, None)
    return float(np.sqrt(eigvals[-1] / eigvals[0]))


def compute_sphericity(prop, voxel_xy=None, voxel_z=None):
    """
    Compute sphericity: ratio of sphere SA with same volume to actual SA.

    Surface area is estimated by counting exposed faces with their physical
    areas, so anisotropic axial sampling is respected.
    Returns value in [0, 1], where 1 = perfect sphere.
    """
    if voxel_xy is None:
        voxel_xy = config.VOXEL_XY
    if voxel_z is None:
        voxel_z = config.VOXEL_Z
    voxel_volume = voxel_xy * voxel_xy * voxel_z
    volume = prop.area * voxel_volume
    if volume <= 0:
        return 0.0

    # Equivalent sphere radius
    r_eq = (3 * volume / (4 * np.pi)) ** (1.0 / 3.0)
    sa_sphere = 4 * np.pi * r_eq**2

    image = np.pad(prop.image.astype(np.int8), 1)
    z_faces = np.abs(np.diff(image, axis=0)).sum()
    y_faces = np.abs(np.diff(image, axis=1)).sum()
    x_faces = np.abs(np.diff(image, axis=2)).sum()
    surface = z_faces * (voxel_xy**2) + (y_faces + x_faces) * (voxel_xy * voxel_z)
    return float(min(sa_sphere / surface, 1.0)) if surface > 0 else 0.0


def compute_tubeness_at_point(image, sigma_um, voxel_xy, voxel_z):
    """
    Compute Hessian-based tubeness measure for the image at a given scale.

    High tubeness = elongated/tubular structure (axon).
    Low tubeness = blob/spherical (bouton).

    Parameters
    ----------
    image : ndarray (Z, Y, X)
    sigma_um : float, scale in um (use bouton scale)
    voxel_xy, voxel_z : float

    Returns
    -------
    tubeness : ndarray (Z, Y, X), tubeness response
    """
    sigma_xy_px = sigma_um / voxel_xy
    sigma_z_px = sigma_um / voxel_z

    # Compute Hessian at this scale (2D per slice for speed)
    tubeness = np.zeros(image.shape, dtype=np.float32)

    for z in range(image.shape[0]):
        try:
            H = hessian_matrix(image[z], sigma=sigma_xy_px, order="rc")
            eigvals = hessian_matrix_eigvals(H)
            # Tubeness: |lambda_1| when lambda_1 >> lambda_2 (one strong curvature)
            # For bright tubes: lambda_1 < 0 (strong), lambda_2 ~ 0
            l1 = eigvals[0]  # most negative
            l2 = eigvals[1]  # least negative
            # Frangi-like tubeness for bright tubes on dark background
            tube_response = np.abs(l1) * np.exp(-0.5 * (l2 / (l1 + 1e-10)) ** 2)
            tube_response[l1 >= 0] = 0  # only dark lines (bright tubes)
            tubeness[z] = tube_response
        except Exception:
            pass

    return tubeness


def filter_objects(
    labels,
    synapse_raw,
    log_response,
    marker,
    voxel_xy=None,
    voxel_z=None,
    min_volume_um3=None,
    max_volume_um3=None,
    max_elongation=None,
    intensity_sigma_k=3.0,
    marker_params=None,
    microglia_mask=None,
):
    """
    Validate and filter segmented objects.

    Applies:
    - Optional morphological closing (vgat donut-filling)
    - Volume filter (min/max, marker-specific)
    - MAD-based intensity filter (unified for all markers)
    - Shape filter for axon markers (elongation threshold)
    Every candidate for a given marker/specification receives the same rules,
    irrespective of its eventual spatial relation to Iba1.

    Parameters
    ----------
    labels : ndarray (Z, Y, X) of int
    synapse_raw : ndarray (Z, Y, X), raw (non-normalized) synapse channel
    log_response : ndarray (Z, Y, X), LoG response
    marker : str, marker name
    voxel_xy, voxel_z : float
    min_volume_um3, max_volume_um3 : float
    max_elongation : float, only applied for axon markers
    intensity_sigma_k : float
        Multiplier for MAD-based background noise floor.
        noise_floor = median(bg) + intensity_sigma_k * MAD_sigma(bg)
    marker_params : dict, optional
        Full marker parameter dict (from config.get_marker_params).
        Used for marker-channel morphology and thresholds.
    microglia_mask : ndarray (Z, Y, X) bool, optional
        Accepted only so that it can be refused. An earlier version of this pipeline computed a
        per-label microglial overlap fraction here and let it influence candidate acceptance,
        which makes detection depend on the quantity the study goes on to measure. Passing
        anything other than None raises. The argument is kept rather than deleted so that the
        prohibition is visible at the call site and cannot be reintroduced silently; the guard is
        pinned by test_iba1_dependent_candidate_filter_fails_loudly.

    Returns
    -------
    filtered_labels : ndarray (Z, Y, X) of int, relabeled
    removed_counts : dict with removal reason counts
    """
    if voxel_xy is None:
        voxel_xy = config.VOXEL_XY
    if voxel_z is None:
        voxel_z = config.VOXEL_Z
    if min_volume_um3 is None:
        min_volume_um3 = config.MIN_BOUTON_VOLUME_UM3
    if max_volume_um3 is None:
        max_volume_um3 = config.MAX_BOUTON_VOLUME_UM3
    if max_elongation is None:
        max_elongation = config.MAX_ELONGATION_AXON_MARKERS
    if marker_params is None:
        marker_params = {}
    if microglia_mask is not None:
        raise ValueError(
            "Iba1-dependent candidate filtering is forbidden; pass microglia_mask=None"
        )

    voxel_volume = voxel_xy * voxel_xy * voxel_z
    is_axon_marker = marker.upper() in [m.upper() for m in config.AXON_MARKERS]

    # Morphological closing for donut-shaped puncta (vgat)
    if marker_params.get("morphological_closing", False):
        closing_radius_um = marker_params.get("closing_radius_um", 0.15)
        selem = _physical_ellipsoid(
            closing_radius_um,
            closing_radius_um,
            voxel_xy,
            voxel_z,
        )
        binary = labels > 0
        closed = ndimage.binary_closing(binary, structure=selem)
        new_voxels = closed & ~binary
        if np.any(new_voxels):
            dist, nearest_idx = ndimage.distance_transform_edt(~binary, return_indices=True)
            labels_closed = labels.copy()
            labels_closed[new_voxels] = labels[
                nearest_idx[0][new_voxels],
                nearest_idx[1][new_voxels],
                nearest_idx[2][new_voxels],
            ]
            labels = labels_closed

    props = regionprops(labels, intensity_image=synapse_raw)

    # Noise floor estimation.  ``precomputed_noise_floor`` is used only by an
    # explicit paired sensitivity analysis that freezes the standard
    # marker-only floor before applying more inclusive size/LoG criteria.
    # Unlike ``fixed_noise_floor`` (the cross-group MPTP-TH rule), it preserves
    # the ordinary maximum-intensity object statistic.
    precomputed_noise_floor = marker_params.get("precomputed_noise_floor")
    used_precomputed_noise_floor = precomputed_noise_floor is not None
    # If a fixed_noise_floor is provided (e.g. for cross-group uniformity),
    # use it directly and skip the per-stack MAD computation.
    if used_precomputed_noise_floor:
        noise_floor = float(precomputed_noise_floor)
    elif "fixed_noise_floor" in marker_params and marker_params["fixed_noise_floor"] is not None:
        noise_floor = float(marker_params["fixed_noise_floor"])
    else:
        # MAD-based: median(bg) + k * MAD_sigma(bg).  For axon markers, also
        # compute the original 3x P99 floor (which was very effective at
        # rejecting dim fragments) and take the maximum.  This prevents the
        # MAD floor from being too lenient when background has a long tail.
        bg_mask = labels == 0
        if np.any(bg_mask):
            bg_values = synapse_raw[bg_mask]
            # Use non-zero background to avoid MAD=0 when image is mostly zero
            # (common for sparse fluorescence markers like VNUT)
            bg_nonzero = bg_values[bg_values > 0]
            if len(bg_nonzero) > 100:
                bg_median = float(np.median(bg_nonzero))
                bg_sigma = estimate_noise_sigma_mad(bg_nonzero)
            else:
                bg_median = float(np.median(bg_values))
                bg_sigma = estimate_noise_sigma_mad(bg_values)
            mad_floor = bg_median + intensity_sigma_k * bg_sigma

            if is_axon_marker:
                p99_floor = 3.0 * float(np.percentile(bg_values, 99))
                noise_floor = max(mad_floor, p99_floor)
            else:
                noise_floor = mad_floor
        else:
            noise_floor = 0

        # Apply minimum noise floor if provided (ensures cross-group
        # comparability by preventing depleted-signal stacks from having
        # a noise floor far below the control detection limit).
        min_nf = marker_params.get("min_noise_floor")
        if min_nf is not None and noise_floor < float(min_nf):
            noise_floor = float(min_nf)

    # Apply any sensitivity multiplier globally, before Iba1 is revealed.
    # This is the unbiased counterpart of the original inside-microglia
    # relaxation.  The runner always passes microglia_mask=None.
    global_intensity_factor = float(marker_params.get("global_intensity_factor", 1.0))
    if not used_precomputed_noise_floor:
        noise_floor *= global_intensity_factor

    filtered_labels = np.zeros_like(labels)
    new_label = 0
    removed = {"too_small": 0, "too_large": 0, "too_dim": 0, "too_elongated": 0}

    for prop in props:
        volume_um3 = prop.area * voxel_volume

        # Volume filter
        if prop.area < int(
            marker_params.get(
                "min_support_voxels",
                config.MIN_CANDIDATE_SUPPORT_VOXELS,
            )
        ):
            removed["too_small"] += 1
            continue
        if volume_um3 < min_volume_um3:
            removed["too_small"] += 1
            continue
        if volume_um3 > max_volume_um3:
            removed["too_large"] += 1
            continue

        # Intensity filter.
        # When using a fixed_noise_floor (common threshold for cross-group
        # uniformity), compare against mean_intensity to ensure the average
        # signal exceeds the threshold. With per-stack adaptive noise floors,
        # use intensity_max (original behavior, more lenient).
        if "fixed_noise_floor" in marker_params and marker_params["fixed_noise_floor"] is not None:
            intensity_val = prop.mean_intensity
        else:
            intensity_val = prop.intensity_max
        if intensity_val <= noise_floor:
            removed["too_dim"] += 1
            continue

        # Shape filter for axon markers
        if is_axon_marker:
            elong = compute_elongation(
                prop,
                voxel_xy=voxel_xy,
                voxel_z=voxel_z,
            )
            if elong > max_elongation:
                removed["too_elongated"] += 1
                continue

        new_label += 1
        bbox = prop.bbox
        slc = tuple(slice(bbox[i], bbox[i + 3]) for i in range(3))
        source_local = labels[slc]
        target_local = filtered_labels[slc]
        target_local[source_local == prop.label] = new_label

    return filtered_labels, removed, noise_floor


def _resolved_marker_params(marker, params=None):
    mp = config.get_marker_params(marker)
    for key, value in (params or {}).items():
        mp[key] = value
    return mp


def prepare_shared_detection(
    synapse_preprocessed, marker, voxel_xy=None, voxel_z=None, params=None
):
    """Compute immutable LoG and seed primitives shared by specifications.

    Standard and globally sensitive definitions differ only after seed
    detection.  Sharing these marker-only primitives is therefore exact and
    avoids performing the ten-scale LoG and h-maxima reconstruction twice.
    """
    if voxel_xy is None:
        voxel_xy = config.VOXEL_XY
    if voxel_z is None:
        voxel_z = config.VOXEL_Z
    mp = _resolved_marker_params(marker, params)
    detection_parameters = {
        "sigma_range_um": tuple(
            mp.get(
                "sigma_range_um",
                config.LOG_SIGMA_RANGE_UM,
            )
        ),
        "n_scales": int(mp.get("n_scales", config.LOG_N_SCALES)),
        "h_maxima_fraction": float(
            mp.get(
                "h_maxima_fraction",
                config.H_MAXIMA_FRACTION,
            )
        ),
        "min_seed_distance_um": float(
            mp.get(
                "min_seed_distance_um",
                config.MIN_SEED_DISTANCE_UM,
            )
        ),
    }
    log_response, _ = compute_log_response(
        synapse_preprocessed,
        sigma_range_um=detection_parameters["sigma_range_um"],
        n_scales=detection_parameters["n_scales"],
        voxel_xy=voxel_xy,
        voxel_z=voxel_z,
    )
    positive_log = log_response[log_response > 0]
    noise_sigma_log = estimate_noise_sigma_mad(positive_log) if len(positive_log) else 0.0
    seeds, n_seeds, seed_coords = detect_seeds(
        log_response,
        h_fraction=detection_parameters["h_maxima_fraction"],
        min_distance_um=detection_parameters["min_seed_distance_um"],
        voxel_xy=voxel_xy,
        voxel_z=voxel_z,
        noise_sigma=noise_sigma_log,
    )
    return {
        "log_response": log_response,
        "noise_sigma_log": float(noise_sigma_log),
        "seeds": seeds,
        "n_seeds": int(n_seeds),
        "seed_coords": seed_coords,
        "detection_parameters": detection_parameters,
    }


def segment_synapses(
    synapse_preprocessed,
    synapse_raw,
    marker,
    microglia_mask=None,
    voxel_xy=None,
    voxel_z=None,
    params=None,
    shared_detection=None,
):
    """
    Full 5-stage synapse bouton segmentation pipeline.

    Parameters
    ----------
    synapse_preprocessed : ndarray (Z, Y, X), preprocessed synapse channel
    synapse_raw : ndarray (Z, Y, X), raw synapse channel (for intensity measurements)
    marker : str, marker name (affects filtering for axon markers)
    microglia_mask : ndarray (Z, Y, X) bool, optional
        Must be None; retained only to fail loudly if older calling code tries
        to make candidate detection depend on Iba1.
    voxel_xy, voxel_z : float
    params : dict, optional parameter overrides

    Returns
    -------
    dict with:
        'labels': ndarray (Z, Y, X) of int, each label = one bouton
        'n_boutons': int
        'log_response': LoG response image
        'foreground_mask': binary foreground
        'foreground_threshold': float
        'foreground_method': str
        'seed_coords': ndarray (N, 3)
        'n_seeds': int
        'removed_counts': dict
    """
    if voxel_xy is None:
        voxel_xy = config.VOXEL_XY
    if voxel_z is None:
        voxel_z = config.VOXEL_Z
    if microglia_mask is not None:
        raise ValueError("Iba1-dependent candidate segmentation is forbidden in this pipeline")

    # Resolve per-marker parameters
    # Start from marker-specific defaults, then overlay user overrides (params).
    # This preserves backwards compatibility with batch/stability analysis code
    # that passes custom params dicts.
    mp = _resolved_marker_params(marker, params)

    if shared_detection is None:
        shared_detection = prepare_shared_detection(
            synapse_preprocessed,
            marker,
            voxel_xy=voxel_xy,
            voxel_z=voxel_z,
            params=mp,
        )
    expected_detection_parameters = {
        "sigma_range_um": tuple(
            mp.get(
                "sigma_range_um",
                config.LOG_SIGMA_RANGE_UM,
            )
        ),
        "n_scales": int(mp.get("n_scales", config.LOG_N_SCALES)),
        "h_maxima_fraction": float(
            mp.get(
                "h_maxima_fraction",
                config.H_MAXIMA_FRACTION,
            )
        ),
        "min_seed_distance_um": float(
            mp.get(
                "min_seed_distance_um",
                config.MIN_SEED_DISTANCE_UM,
            )
        ),
    }
    if shared_detection.get("detection_parameters") != expected_detection_parameters:
        raise ValueError("shared marker detection parameters do not match specification")
    log_response = shared_detection["log_response"]
    noise_sigma_log = float(shared_detection["noise_sigma_log"])
    # Seed rescue mutates this map downstream, so each specification gets an
    # independent copy while the immutable LoG response remains shared.
    seeds = shared_detection["seeds"].copy()
    n_seeds = int(shared_detection["n_seeds"])
    seed_coords = shared_detection["seed_coords"].copy()

    if n_seeds == 0:
        empty = np.zeros(synapse_preprocessed.shape, dtype=np.int32)
        return {
            "labels": empty,
            "n_boutons": 0,
            "log_response": log_response,
            "foreground_mask": np.zeros(synapse_preprocessed.shape, dtype=bool),
            "foreground_threshold": 0.0,
            "foreground_method": "none",
            "seed_coords": seed_coords,
            "n_seeds": 0,
            "removed_counts": {},
            "noise_floor": 0.0,
        }

    # Stage 3: Foreground mask (MAD threshold + hysteresis + bright recovery)
    voxel_volume = voxel_xy * voxel_xy * voxel_z
    min_fg_vox = max(
        1, int(round(mp.get("min_volume_um3", config.MIN_BOUTON_VOLUME_UM3) / voxel_volume))
    )

    foreground, fg_threshold, fg_method = compute_foreground_mask(
        log_response,
        synapse_preprocessed=synapse_preprocessed,
        seeds=seeds,
        noise_sigma_k=mp.get("noise_sigma_k", 3.0),
        min_foreground_voxels=min_fg_vox,
        min_fg_threshold=mp.get("min_fg_threshold", None),
        fixed_fg_threshold=mp.get("fixed_fg_threshold", None),
    )

    # Stage 3.5: Intensity-based seed rescue for large bright boutons
    # Bright recovery (Stage 3) may add foreground regions that had no LoG
    # seed (their LoG interior response was ~0).  Without a seed, the
    # watershed cannot label them.  For each foreground connected component
    # above a generous minimum size that lacks a seed, place one at the raw
    # intensity peak.  The minimum is 5x min_fg_vox to avoid rescuing tiny
    # bright noise CCs (which would create thousands of spurious objects).
    min_rescue_vox = 5 * min_fg_vox
    if np.any(foreground):
        fg_cc_labels, n_fg_cc = ndi_label(foreground)
        if n_fg_cc > 0:
            # Which foreground CCs already have a seed?
            seed_mask_rescue = seeds > 0
            has_seed_rescue = np.zeros(n_fg_cc + 1, dtype=bool)
            fg_at_seeds = fg_cc_labels[seed_mask_rescue]
            fg_at_seeds = fg_at_seeds[fg_at_seeds > 0]
            if len(fg_at_seeds) > 0:
                has_seed_rescue[np.unique(fg_at_seeds)] = True

            # Size filter: only rescue CCs substantially larger than min volume
            cc_sizes_rescue = np.bincount(fg_cc_labels.ravel(), minlength=n_fg_cc + 1)

            max_seed_label = int(seeds.max())
            n_rescued = 0
            cc_slices = ndimage.find_objects(fg_cc_labels)
            for cc_i in range(1, n_fg_cc + 1):
                if has_seed_rescue[cc_i]:
                    continue
                if cc_sizes_rescue[cc_i] < min_rescue_vox:
                    continue
                # Place seed at raw intensity peak within this CC
                slc = cc_slices[cc_i - 1]
                if slc is None:
                    continue
                cc_local = fg_cc_labels[slc] == cc_i
                raw_local = synapse_raw[slc]
                values = np.where(cc_local, raw_local, -np.inf)
                local_peak = np.unravel_index(np.argmax(values), values.shape)
                peak_idx = tuple(local_peak[d] + slc[d].start for d in range(3))
                max_seed_label += 1
                seeds[peak_idx] = max_seed_label
                n_rescued += 1

            if n_rescued > 0:
                n_seeds += n_rescued
                fg_method += f"+int_seeds({n_rescued})"

    # Stage 4: Marker-controlled watershed + intensity trimming
    labels = watershed_segmentation(
        log_response,
        seeds,
        foreground,
        synapse_preprocessed=synapse_preprocessed,
    )

    # Stage 5: Object validation (MAD intensity + marker-specific volumes)
    filtered_labels, removed_counts, intensity_noise_floor = filter_objects(
        labels,
        synapse_raw,
        log_response,
        marker,
        voxel_xy=voxel_xy,
        voxel_z=voxel_z,
        min_volume_um3=mp.get("min_volume_um3", config.MIN_BOUTON_VOLUME_UM3),
        max_volume_um3=mp.get("max_volume_um3", config.MAX_BOUTON_VOLUME_UM3),
        max_elongation=mp.get("max_elongation", config.MAX_ELONGATION_AXON_MARKERS),
        intensity_sigma_k=mp.get("intensity_sigma_k", 3.0),
        marker_params=mp,
        microglia_mask=microglia_mask,
    )

    # Stage 6: Watershed-based rescue for bright uncaptured objects
    # The LoG pipeline may miss large/flat-topped boutons whose LoG response
    # is too weak for detection.  Find bright voxels (1.5x noise floor) not
    # in any label, run a mini-watershed to split connected varicosities
    # (e.g., VNUT beaded axon strings), and add objects that pass volume
    # bounds.  The watershed naturally splits at narrow/dim connections.
    is_axon_marker = marker.upper() in [m.upper() for m in config.AXON_MARKERS]
    bg_unlabeled = synapse_raw[filtered_labels == 0]
    # The rescue path must use the same location- and condition-independent
    # intensity rule as the primary object filter.  In particular, a supplied
    # fixed_noise_floor (used for MPTP TH) must not be silently replaced here
    # by a per-stack adaptive estimate.
    base_floor = compute_rescue_base_floor(bg_unlabeled, is_axon_marker, mp)
    if base_floor is not None:
        # Rescue threshold: factor * noise_floor (marker-specific).
        # Must be strict enough to avoid rescuing medium-bright noise.
        rescue_factor = mp.get("rescue_intensity_factor", 2.5)
        rescue_floor = rescue_factor * base_floor

        uncaptured_bright = (synapse_raw > rescue_floor) & (filtered_labels == 0)
        n_uncaptured = np.sum(uncaptured_bright)
        if n_uncaptured > 10:
            # Find local intensity maxima within uncaptured bright regions
            rescue_footprint = _physical_ellipsoid(
                config.MIN_SEED_DISTANCE_UM,
                max(config.MIN_SEED_DISTANCE_UM, voxel_z),
                voxel_xy,
                voxel_z,
            )
            local_max_r = (
                maximum_filter(synapse_raw, footprint=rescue_footprint) == synapse_raw
            ) & uncaptured_bright
            rescue_seeds, n_rescue_seeds = ndi_label(local_max_r)

            if n_rescue_seeds > 0:
                # Watershed on inverted intensity to split varicosities
                landscape_r = -synapse_raw.astype(np.float64)
                rescue_labels = watershed(landscape_r, markers=rescue_seeds, mask=uncaptured_bright)

                # Filter rescued objects by volume (marker-specific minimum)
                min_vol_r = mp.get("rescue_min_volume_um3", 0.30)
                max_vol_r = mp.get("max_volume_um3", config.MAX_BOUTON_VOLUME_UM3)
                min_vox_r = max(1, int(round(min_vol_r / voxel_volume)))
                max_vox_r = int(round(max_vol_r / voxel_volume))

                rescue_sizes = np.bincount(
                    rescue_labels.ravel(), minlength=int(rescue_labels.max()) + 1
                )

                max_label = int(filtered_labels.max())
                n_rescued_final = 0
                rescue_slices = ndimage.find_objects(rescue_labels)
                for rc in range(1, int(rescue_labels.max()) + 1):
                    if rescue_sizes[rc] < min_vox_r or rescue_sizes[rc] > max_vox_r:
                        continue
                    slc = rescue_slices[rc - 1]
                    if slc is None:
                        continue
                    if is_axon_marker:
                        rescue_prop = regionprops((rescue_labels[slc] == rc).astype(np.uint8))[0]
                        if compute_elongation(
                            rescue_prop,
                            voxel_xy=voxel_xy,
                            voxel_z=voxel_z,
                        ) > mp.get(
                            "max_elongation",
                            config.MAX_ELONGATION_AXON_MARKERS,
                        ):
                            removed_counts["too_elongated"] += 1
                            continue
                    max_label += 1
                    rescue_local = rescue_labels[slc]
                    filtered_local = filtered_labels[slc]
                    filtered_local[rescue_local == rc] = max_label
                    n_rescued_final += 1

                if n_rescued_final > 0:
                    removed_counts["rescued_bright"] = n_rescued_final
                    fg_method += f"+rescue({n_rescued_final})"

    n_boutons = int(filtered_labels.max())

    return {
        "labels": filtered_labels,
        "n_boutons": int(n_boutons),
        "log_response": log_response,
        "foreground_mask": foreground,
        "foreground_threshold": fg_threshold,
        "foreground_method": fg_method,
        "seed_coords": seed_coords,
        "n_seeds": n_seeds,
        "removed_counts": removed_counts,
        "noise_floor": float(intensity_noise_floor),
        "rescue_base_floor": (float(base_floor) if base_floor is not None else None),
    }
