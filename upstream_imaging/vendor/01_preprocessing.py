"""
Image preprocessing for the microglia-synapse analysis.

Loads LSM stacks and extracts the marker (Ch0) and Iba1 (Ch1) channels.
The outcome-blind release does not estimate spectral correction from Iba1-
selected voxels.  The marker channel is processed with white top-hat
background subtraction, per-slice P99.5 normalization,
Richardson-Lucy XY deconvolution, and light 3D Gaussian denoising. The
Iba1 channel is normalized and denoised only. Filenames are also parsed
for experimental metadata.
"""

import numpy as np
from pathlib import Path
import tifffile
from scipy import ndimage
from skimage.morphology import disk, white_tophat
import warnings
import re

try:
    import cv2
except ImportError:  # pragma: no cover - retained for portable reruns
    cv2 = None

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "config", str(Path(__file__).parent / "00_config.py")
)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

TOPHAT_BACKEND = "opencv_exact_flat_disk" if cv2 is not None else "skimage_flat_disk"


# LSM file loading


def load_lsm(filepath):
    """
    Load an LSM file and return raw data + voxel size.

    Returns
    -------
    dict with keys:
        'data': numpy array, shape (Z, C, Y, X), dtype float32
        'voxel_xy': float, um per pixel XY
        'voxel_z': float, um per pixel Z
        'shape_original': original shape before reordering
        'axes': axis order string from tifffile
    """
    filepath = Path(filepath)
    with tifffile.TiffFile(filepath) as tif:
        if not tif.is_lsm:
            raise ValueError(f"{filepath.name} is not an LSM file")

        series = tif.series[0]
        data = series.asarray()
        axes = series.axes

        # Read voxel size from LSM metadata (fallback to config defaults)
        voxel_xy = config.VOXEL_XY
        voxel_z = config.VOXEL_Z

        if hasattr(tif, "lsm_metadata") and tif.lsm_metadata:
            meta = tif.lsm_metadata
            if "VoxelSizeX" in meta:
                voxel_xy = meta["VoxelSizeX"] * 1e6  # m -> um
                voxel_z = meta["VoxelSizeZ"] * 1e6

        shape_original = data.shape

        # Reorder to (Z, C, Y, X)
        if axes == "ZCYX":
            pass
        elif axes == "CZYX":
            data = np.transpose(data, (1, 0, 2, 3))
        elif axes == "ZYX":
            data = data[:, np.newaxis, :, :]
        elif axes == "TZCYX":
            # Time-series: take first timepoint
            data = data[0]
        else:
            warnings.warn(f"Unexpected axes '{axes}' in {filepath.name}, assuming ZCYX")

        data = data.astype(np.float32)

    return {
        "data": data,
        "voxel_xy": voxel_xy,
        "voxel_z": voxel_z,
        "shape_original": shape_original,
        "axes": axes,
    }


def extract_channels(data):
    """
    Extract synapse (Ch0) and microglia (Ch1) channels.

    Parameters
    ----------
    data : ndarray, shape (Z, C, Y, X)

    Returns
    -------
    synapse_raw : ndarray (Z, Y, X), float32
    microglia_raw : ndarray (Z, Y, X), float32
    """
    n_channels = data.shape[1]
    if n_channels < 2:
        raise ValueError(f"Expected >= 2 channels, got {n_channels}")

    synapse_raw = data[:, config.CHANNEL_SYNAPSE, :, :]
    microglia_raw = data[:, config.CHANNEL_MICROGLIA, :, :]
    return synapse_raw, microglia_raw


# Preprocessing steps


def white_tophat_background_subtraction(image_3d, radius_px=None):
    """
    White top-hat transform for background subtraction.

    Applied per Z-slice using a 2D disk structuring element.
    Removes diffuse/uneven background while preserving punctate structures.

    Parameters
    ----------
    image_3d : ndarray (Z, Y, X)
    radius_px : int, structuring element radius in pixels

    Returns
    -------
    background_subtracted : ndarray (Z, Y, X)
    """
    if radius_px is None:
        radius_px = config.TOPHAT_RADIUS_PX

    selem = disk(radius_px)
    result = np.zeros_like(image_3d)

    for z in range(image_3d.shape[0]):
        if cv2 is not None:
            # This flat-footprint opening with BORDER_REFLECT is exactly
            # equivalent to skimage.morphology.white_tophat for this disk,
            # while avoiding the generic morphology path's severe slowdown
            # on 1512x1512 float32 optical sections.
            opened = cv2.morphologyEx(
                image_3d[z],
                cv2.MORPH_OPEN,
                selem.astype(np.uint8),
                borderType=cv2.BORDER_REFLECT,
            )
            result[z] = image_3d[z] - opened
        else:
            result[z] = white_tophat(image_3d[z], selem)

    return result


def normalize_per_slice(image_3d, percentile=None):
    """
    Per-slice intensity normalization.

    For each Z-slice, compute the given percentile of foreground signal.
    Scale each slice so that this percentile maps to the stack's median
    percentile value. Preserves relative intensity within each slice while
    equalizing across depth.

    Parameters
    ----------
    image_3d : ndarray (Z, Y, X), already background-subtracted
    percentile : float, default 99.5

    Returns
    -------
    normalized : ndarray (Z, Y, X)
    """
    if percentile is None:
        percentile = config.NORMALIZATION_PERCENTILE

    nz = image_3d.shape[0]
    slice_pvals = np.zeros(nz)

    for z in range(nz):
        sl = image_3d[z]
        # Use only nonzero pixels (foreground after tophat)
        fg = sl[sl > 0]
        if len(fg) > 0:
            slice_pvals[z] = np.percentile(fg, percentile)
        else:
            slice_pvals[z] = 0.0

    # Reference value: median of non-zero slice percentiles
    valid = slice_pvals[slice_pvals > 0]
    if len(valid) == 0:
        return image_3d.copy()
    reference = np.median(valid)

    normalized = np.zeros_like(image_3d)
    for z in range(nz):
        if slice_pvals[z] > 0:
            normalized[z] = image_3d[z] * (reference / slice_pvals[z])
        else:
            normalized[z] = image_3d[z]

    return normalized


def deconvolve_xy(image_3d, psf_sigma_um=None, n_iterations=6, voxel_xy=None):
    """
    Per-slice 2D Richardson-Lucy deconvolution in XY.

    Sharpens puncta boundaries and makes intensity profiles more peaked,
    improving LoG blob detection and producing tighter object masks.
    Applied per-slice to avoid Z-boundary artifacts from thin stacks.

    Uses a 2D Gaussian PSF derived from the confocal optical parameters
    (Plan-Apo 63x/1.40 Oil, lambda ~488 nm for green channel).

    Parameters
    ----------
    image_3d : ndarray (Z, Y, X), background-subtracted, normalized
    psf_sigma_um : float, PSF sigma in um (default: theoretical for NA=1.40)
    n_iterations : int, number of Richardson-Lucy iterations
    voxel_xy : float, um per pixel

    Returns
    -------
    deconvolved : ndarray (Z, Y, X), float32
    """
    from skimage.restoration import richardson_lucy

    if voxel_xy is None:
        voxel_xy = config.VOXEL_XY
    if psf_sigma_um is None:
        # Theoretical confocal PSF: FWHM_xy = 0.51 * lambda / NA
        # For green (488nm), NA=1.40: FWHM = 0.178 um, sigma = 0.076 um
        psf_sigma_um = 0.076

    psf_sigma_px = psf_sigma_um / voxel_xy  # ~0.86 pixels

    # Build 2D Gaussian PSF kernel (must be normalized to sum=1)
    kernel_radius = int(np.ceil(4 * psf_sigma_px))
    size = 2 * kernel_radius + 1
    y, x = np.mgrid[-kernel_radius : kernel_radius + 1, -kernel_radius : kernel_radius + 1]
    psf_2d = np.exp(-(x**2 + y**2) / (2 * psf_sigma_px**2))
    psf_2d /= psf_2d.sum()
    psf_2d = psf_2d.astype(np.float64)

    deconvolved = np.zeros_like(image_3d, dtype=np.float32)

    for z in range(image_3d.shape[0]):
        sl = image_3d[z].astype(np.float64)
        # Richardson-Lucy requires strictly positive input
        sl = np.clip(sl, 1e-6, None)
        deconv = richardson_lucy(sl, psf_2d, num_iter=n_iterations, clip=False)
        deconvolved[z] = deconv.astype(np.float32)

    return deconvolved


def light_gaussian_denoise(image_3d, sigma_zyx=None):
    """
    Light 3D Gaussian filter with anisotropic sigma.

    Very gentle smoothing to reduce shot noise without blurring boutons.

    Parameters
    ----------
    image_3d : ndarray (Z, Y, X)
    sigma_zyx : tuple of (sigma_z, sigma_y, sigma_x) in pixels
        scipy expects (Z, Y, X) order

    Returns
    -------
    denoised : ndarray (Z, Y, X)
    """
    if sigma_zyx is None:
        # Config stores as (X, Y, Z) physical -> convert to (Z, Y, X) pixels
        sigma_zyx = (
            config.GAUSSIAN_SIGMA_PX[2],  # Z
            config.GAUSSIAN_SIGMA_PX[1],  # Y
            config.GAUSSIAN_SIGMA_PX[0],  # X
        )

    return ndimage.gaussian_filter(image_3d, sigma=sigma_zyx).astype(np.float32)


# Spectral bleed-through correction


def estimate_bleedthrough_coefficient(synapse_raw, microglia_raw, erosion_voxels=3):
    """
    Estimate spectral bleed-through from iba1 (Ch1) into synapse (Ch0).

    Uses voxels deep inside microglia (eroded mask) where synapse signal
    should be minimal. The slope of the linear regression Ch0 = a * Ch1 + b
    gives the bleed-through coefficient a.

    Parameters
    ----------
    synapse_raw : ndarray (Z, Y, X)
    microglia_raw : ndarray (Z, Y, X)
    erosion_voxels : int, erosion iterations for interior estimation

    Returns
    -------
    coefficient : float, bleed-through coefficient (slope)
    r_squared : float, R² of the fit
    """
    from scipy.stats import linregress
    from skimage.filters import threshold_otsu

    # Quick microglia mask via Otsu on raw iba1
    try:
        thresh = threshold_otsu(microglia_raw[microglia_raw > 0])
    except Exception:
        thresh = np.percentile(microglia_raw, 95)
    mg_mask = microglia_raw > thresh

    # Erode to get deep interior
    eroded = ndimage.binary_erosion(mg_mask, iterations=erosion_voxels)
    interior_voxels = np.sum(eroded)

    if interior_voxels < 1000:
        return 0.0, 0.0

    ch0_interior = synapse_raw[eroded].ravel()
    ch1_interior = microglia_raw[eroded].ravel()

    # Subsample for speed if very large
    if len(ch0_interior) > 500000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(ch0_interior), 500000, replace=False)
        ch0_interior = ch0_interior[idx]
        ch1_interior = ch1_interior[idx]

    result = linregress(ch1_interior, ch0_interior)
    coefficient = max(result.slope, 0.0)  # bleed-through can only be positive
    r_squared = result.rvalue**2

    return coefficient, r_squared


def correct_bleedthrough(synapse_raw, microglia_raw, coefficient=None, min_r_squared=0.05):
    """
    Apply linear spectral unmixing to remove iba1 bleed-through from Ch0.

    corrected = synapse_raw - coefficient * microglia_raw

    Only applies correction if the estimated bleed-through explains at least
    min_r_squared of the variance.

    Parameters
    ----------
    synapse_raw : ndarray (Z, Y, X)
    microglia_raw : ndarray (Z, Y, X)
    coefficient : float, if None, estimates from data
    min_r_squared : float, minimum R² to apply correction

    Returns
    -------
    corrected : ndarray (Z, Y, X), float32
    coefficient_used : float, the coefficient that was applied (0 if skipped)
    r_squared : float
    """
    if coefficient is None:
        coefficient, r_squared = estimate_bleedthrough_coefficient(synapse_raw, microglia_raw)
    else:
        r_squared = 1.0  # user-provided, always apply

    if coefficient <= 0 or r_squared < min_r_squared:
        return synapse_raw.copy(), 0.0, r_squared

    corrected = synapse_raw.astype(np.float32) - coefficient * microglia_raw.astype(np.float32)
    corrected = np.maximum(corrected, 0)  # clip negative values

    return corrected, coefficient, r_squared


# Full preprocessing pipeline


def preprocess_synapse_channel(synapse_raw):
    """
    Full preprocessing for synapse channel.

    Steps:
    1. White top-hat background subtraction (2D per slice)
    2. Per-slice intensity normalization (P99.5)
    3. Richardson-Lucy XY deconvolution
    4. Light 3D Gaussian denoising

    Parameters
    ----------
    synapse_raw : ndarray (Z, Y, X), float32

    Returns
    -------
    dict with:
        'preprocessed': final preprocessed image
        'tophat': after background subtraction (before normalization)
    """
    tophat = white_tophat_background_subtraction(synapse_raw)

    normalized = normalize_per_slice(tophat)

    # Deconvolution sharpens puncta boundaries and makes intensity profiles
    # more peaked, improving LoG response for both small and large boutons.
    deconvolved = deconvolve_xy(normalized)

    # Light Gaussian denoising suppresses Richardson-Lucy noise amplification.
    denoised = light_gaussian_denoise(deconvolved)

    return {
        "preprocessed": denoised,
        "tophat": tophat,
    }


def preprocess_microglia_channel(microglia_raw):
    """
    Preprocessing for microglia channel.

    Microglia are large structures, so we skip top-hat (which removes large
    structures) and use only normalization + smoothing.

    Steps:
    1. Per-slice intensity normalization (P99.5) on raw
    2. Light 3D Gaussian denoising

    Parameters
    ----------
    microglia_raw : ndarray (Z, Y, X), float32

    Returns
    -------
    dict with:
        'preprocessed': final preprocessed image
    """
    normalized = normalize_per_slice(microglia_raw)

    denoised = light_gaussian_denoise(normalized)

    return {
        "preprocessed": denoised,
    }


def preprocess_file(filepath):
    """
    Load and preprocess a single LSM file.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    dict with:
        'synapse_preprocessed': preprocessed synapse channel
        'synapse_raw': raw synapse channel
        'synapse_tophat': top-hat result (for QC)
        'microglia_preprocessed': preprocessed microglia channel
        'microglia_raw': raw microglia channel
        'voxel_xy': float
        'voxel_z': float
        'shape': (Z, Y, X)
        'filepath': str
    """
    loaded = load_lsm(filepath)
    data = loaded["data"]

    synapse_raw, microglia_raw = extract_channels(data)

    # Outcome-blind preprocessing: do not estimate marker correction from
    # Iba1-defined voxels.  The original data-derived correction assumes that
    # signal inside Iba1 cannot be true marker cargo, which is the endpoint
    # being tested.  An external single-stain coefficient can be introduced
    # later as a separately calibrated sensitivity.
    if config.APPLY_DATA_DERIVED_BLEEDTHROUGH_CORRECTION:
        synapse_corrected, bt_coefficient, bt_r2 = correct_bleedthrough(synapse_raw, microglia_raw)
    else:
        synapse_corrected = synapse_raw.copy()
        bt_coefficient, bt_r2 = 0.0, 0.0

    syn_result = preprocess_synapse_channel(synapse_corrected)
    mg_result = preprocess_microglia_channel(microglia_raw)

    return {
        "synapse_preprocessed": syn_result["preprocessed"],
        "synapse_raw": synapse_raw,
        "synapse_corrected": synapse_corrected,
        "synapse_tophat": syn_result["tophat"],
        "microglia_preprocessed": mg_result["preprocessed"],
        "microglia_raw": microglia_raw,
        "voxel_xy": loaded["voxel_xy"],
        "voxel_z": loaded["voxel_z"],
        "shape": synapse_raw.shape,
        "filepath": str(filepath),
        "bleedthrough_coefficient": bt_coefficient,
        "bleedthrough_r2": bt_r2,
    }


# Filename parsing


def parse_filename(filepath):
    """
    Parse LSM filename to extract experimental metadata.

    Filename format: {marker} {condition}{animal_number} {region} z{stack_number}.lsm
    Examples:
        vglut1 kiv1 ca1 z1.lsm
        5ht liv2 cortex 2 z2.lsm
        gad65 2 livl3 SLM z1.lsm

    Returns
    -------
    dict with keys: marker, condition, condition_code, animal_id, animal_num,
                    region, stack_num, filename, file_id
    """
    filepath = Path(filepath)
    filename_lower = filepath.stem.lower()

    # Marker: use parent directory name (reliable)
    marker = filepath.parent.name

    # Treatment: kiv, liv, livl (livl must be checked before liv)
    treatment_match = re.search(r"(kiv|livl|liv)(\d+)", filename_lower)
    if treatment_match:
        condition_code = treatment_match.group(1)
        animal_num = int(treatment_match.group(2))
        condition = config.TREATMENT_MAP.get(condition_code, condition_code)
        animal_id = f"{condition_code}{animal_num}"
    else:
        condition_code = "unknown"
        condition = "unknown"
        animal_num = 0
        animal_id = "unknown"

    # Z-stack number
    z_match = re.search(r"z(\d+)", filename_lower)
    stack_num = int(z_match.group(1)) if z_match else 1

    # Region: search for known region patterns
    region = "unknown"
    # Check most specific patterns first
    region_patterns = [
        ("cortex 2", "cortex_L2"),
        ("cortex2", "cortex_L2"),
        ("cortex 4", "cortex_L4"),
        ("cortex4", "cortex_L4"),
        ("tempcort", "temporal_cortex"),
        ("slm", "SLM"),
        ("ca1", "CA1"),
        ("dg", "DG"),
        ("stria", "striatum"),
        ("amygdala", "amygdala"),
        ("thal", "thalamus"),
    ]
    for pattern, normalized in region_patterns:
        if pattern in filename_lower:
            region = normalized
            break

    # Create unique file identifier
    file_id = f"{marker}_{animal_id}_{region}_z{stack_num}"

    return {
        "marker": marker,
        "condition": condition,
        "condition_code": condition_code,
        "animal_id": animal_id,
        "animal_num": animal_num,
        "region": region,
        "stack_num": stack_num,
        "filename": filepath.name,
        "filepath": str(filepath),
        "file_id": file_id,
    }


def should_exclude(filepath):
    """Check if a file should be excluded based on filename patterns."""
    fp = Path(filepath)
    # Skip macOS resource forks
    if fp.name.startswith("._"):
        return True
    name_lower = fp.stem.lower()
    for pattern in config.EXCLUDE_PATTERNS:
        if pattern in name_lower:
            return True
    return False


def get_all_lsm_files(data_root=None):
    """
    Discover all LSM files with parsed metadata.

    Returns
    -------
    list of dicts with file info
    """
    if data_root is None:
        data_root = config.DATA_ROOT

    data_root = Path(data_root)
    all_files = []

    for marker_dir in sorted(data_root.iterdir()):
        if not marker_dir.is_dir():
            continue
        for lsm_file in sorted(marker_dir.glob("*.lsm")):
            if should_exclude(lsm_file):
                continue
            info = parse_filename(lsm_file)
            all_files.append(info)

    return all_files
