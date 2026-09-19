"""
Configuration for the microglia-synapse interaction analysis pipeline:
paths, voxel dimensions, channel maps, and marker-specific parameters.
"""

import os
from pathlib import Path

# Path configuration
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from mg_paths import OUTPUT, EXTERNAL

DATA_ROOT = EXTERNAL
PROJECT_ROOT = OUTPUT / "upstream_imaging"
OUTPUT_DIR = PROJECT_ROOT / "data" / "derived" / "vendor_default"

# Subdirectories
MASKS_DIR = OUTPUT_DIR / "masks"
MICROGLIA_MASKS_DIR = MASKS_DIR / "microglia"
SYNAPSE_MASKS_DIR = MASKS_DIR / "synapses"
OBJECT_DATA_DIR = OUTPUT_DIR / "object_data"
CONTACT_DATA_DIR = OUTPUT_DIR / "contact_data"
INTERNALIZED_DATA_DIR = OUTPUT_DIR / "internalized_data"
SUMMARY_DIR = OUTPUT_DIR / "summary"
QC_DIR = OUTPUT_DIR / "qc"

# Directories are created only by the scripts that write to them.  Importing a
# configuration module must not mutate the release tree.

# Voxel dimensions (from LSM metadata)
VOXEL_XY = 0.0884  # um per pixel in X and Y
VOXEL_Z = 0.3800  # um per pixel in Z
VOXEL_VOLUME_UM3 = VOXEL_XY * VOXEL_XY * VOXEL_Z  # ~0.00297 um^3

# Channel configuration
CHANNEL_SYNAPSE = 0  # Ch0 (green) = synaptic marker
CHANNEL_MICROGLIA = 1  # Ch1 (red) = iba1

# Dataset structure
MARKERS = ["vglut1", "vglut2", "vgat", "gad65", "5HT", "ChaT", "TH", "VNUT"]

# Markers that label axons + varicosities (need elongation/tubeness filtering)
AXON_MARKERS = ["5HT", "TH", "ChaT"]

# Markers that produce clearly punctate labeling
PUNCTATE_MARKERS = ["vglut1", "vglut2", "vgat", "gad65", "VNUT"]

# Treatment condition mapping from filename patterns
TREATMENT_MAP = {
    "kiv": "Control",
    "liv": "LPS_72h",
    "livl": "LPS_60d",
}

# Condition ordering for plots
CONDITION_ORDER = ["Control", "LPS_72h", "LPS_60d"]
CONDITION_CODES = ["kiv", "liv", "livl"]

# Brain regions
REGIONS = ["CA1", "DG", "SLM", "cortex_L2", "cortex_L4", "striatum", "amygdala"]
REGION_NORMALIZE = {
    "ca1": "CA1",
    "dg": "DG",
    "slm": "SLM",
    "cortex 2": "cortex_L2",
    "cortex2": "cortex_L2",
    "cortex 4": "cortex_L4",
    "cortex4": "cortex_L4",
    "stria": "striatum",
    "amygdala": "amygdala",
    "tempcort": "temporal_cortex",
    "thal": "thalamus",
}

# VNUT has different regions
VNUT_REGIONS = ["CA1", "DG", "SLM", "cortex_L2", "cortex_L4"]

# Preprocessing parameters
# Background subtraction: White top-hat transform
TOPHAT_RADIUS_UM = 2.0  # um, structuring element radius for background estimation
TOPHAT_RADIUS_PX = int(round(TOPHAT_RADIUS_UM / VOXEL_XY))  # ~23 pixels

# Per-slice intensity normalization
NORMALIZATION_PERCENTILE = 99.5  # percentile for reference intensity

# The original pipeline estimated marker-channel bleed-through from voxels
# selected using Iba1.  That makes marker preprocessing depend on the outcome
# channel.  The overlap-blind reanalysis keeps this False.  A non-zero
# coefficient may only be supplied from an external single-stain calibration.
APPLY_DATA_DERIVED_BLEEDTHROUGH_CORRECTION = False

# Light denoising: 3D Gaussian (anisotropic)
GAUSSIAN_SIGMA_UM = (0.05, 0.05, 0.10)  # (XY, XY, Z) in um
GAUSSIAN_SIGMA_PX = (
    GAUSSIAN_SIGMA_UM[0] / VOXEL_XY,  # ~0.57 px
    GAUSSIAN_SIGMA_UM[1] / VOXEL_XY,  # ~0.57 px
    GAUSSIAN_SIGMA_UM[2] / VOXEL_Z,  # ~0.26 px
)

# Microglia segmentation parameters
# Smoothing before thresholding
MICROGLIA_SMOOTH_SIGMA_UM = (0.1, 0.1, 0.2)  # (X, Y, Z) in um
MICROGLIA_SMOOTH_SIGMA_PX = (
    MICROGLIA_SMOOTH_SIGMA_UM[0] / VOXEL_XY,  # ~1.13 px
    MICROGLIA_SMOOTH_SIGMA_UM[1] / VOXEL_XY,
    MICROGLIA_SMOOTH_SIGMA_UM[2] / VOXEL_Z,  # ~0.53 px
)

# Local adaptive thresholding
ADAPTIVE_BLOCK_SIZE_UM = 5.0  # um, size of local region
ADAPTIVE_BLOCK_SIZE_PX = int(round(ADAPTIVE_BLOCK_SIZE_UM / VOXEL_XY))  # ~57 px
# Ensure odd
if ADAPTIVE_BLOCK_SIZE_PX % 2 == 0:
    ADAPTIVE_BLOCK_SIZE_PX += 1
ADAPTIVE_THRESHOLD_FACTOR = 0.8  # foreground if intensity > local_mean * factor

# Morphological refinement
CLOSING_RADIUS_UM = 0.3  # um
CLOSING_RADIUS_PX = max(1, int(round(CLOSING_RADIUS_UM / VOXEL_XY)))  # ~3 px
# Iba1 is a cytoplasmic/process marker.  Unrestricted 3-D hole filling can
# incorrectly convert genuine nuclear and inter-process voids into segmented
# Iba1 volume.  The primary mask therefore does not fill enclosed cavities;
# physically capped filling is retained as a prespecified sensitivity check.
MAX_MICROGLIA_HOLE_VOLUME_UM3 = 0.0

# Size filtering
MIN_MICROGLIA_VOXELS = 50  # minimum Iba1-positive fragment size (~0.15 um^3)
# The geometry endpoint concerns all resolved Iba1-positive processes, not only
# connected components large enough to resemble a complete cell.  A historical
# 5-um3 second-stage cutoff removed visible disconnected process segments from
# thin stacks.  The final physical cutoff therefore matches the prespecified
# 50-voxel noise floor above; 5 um3 is retained as a sensitivity mask.
MIN_MICROGLIA_VOLUME_UM3 = 0.15

# Synapse segmentation parameters
# Multi-scale LoG blob detection
BOUTON_DIAMETER_RANGE_UM = (0.3, 3.0)  # min, max expected diameter in um
LOG_N_SCALES = 10  # number of logarithmically spaced LoG scales
# Sigma range: diameter / (2 * sqrt(2)) for LoG blob detection
LOG_SIGMA_RANGE_UM = (
    BOUTON_DIAMETER_RANGE_UM[0] / (2 * 1.414),  # ~0.106 um
    BOUTON_DIAMETER_RANGE_UM[1] / (2 * 1.414),  # ~1.061 um
)

# h-maxima for seed detection
H_MAXIMA_FRACTION = 0.15  # h as fraction of median peak LoG response

# Minimum distance between bouton seeds
MIN_SEED_DISTANCE_UM = 0.3  # um
MIN_SEED_DISTANCE_PX = max(2, int(round(MIN_SEED_DISTANCE_UM / VOXEL_XY)))  # ~3.4 px

# Object validation: volume filtering
MIN_BOUTON_VOLUME_UM3 = 0.01  # um^3
MAX_BOUTON_VOLUME_UM3 = 8.0  # um^3
MIN_BOUTON_VOXELS = max(1, int(round(MIN_BOUTON_VOLUME_UM3 / VOXEL_VOLUME_UM3)))
MAX_BOUTON_VOXELS = int(round(MAX_BOUTON_VOLUME_UM3 / VOXEL_VOLUME_UM3))
MIN_CANDIDATE_SUPPORT_VOXELS = 2

# Marker-specific segmentation parameters (noise-adaptive)
# Per-marker volume bounds, noise threshold multipliers, and morphological opts.
# noise_sigma_k: multiplier for MAD-based LoG foreground threshold (higher = stricter)
# intensity_sigma_k: multiplier for MAD-based raw intensity filter
# h_maxima_fraction: fraction of median peak LoG used as h in h-maxima seed detection
MARKER_PARAMS = {
    # Punctate markers
    "vglut1": {
        "min_volume_um3": 0.005,
        "max_volume_um3": 5.0,
        "noise_sigma_k": 3.0,
        "intensity_sigma_k": 3.0,
        "h_maxima_fraction": 0.05,
    },
    "vglut2": {
        "min_volume_um3": 0.12,
        "max_volume_um3": 5.0,
        "noise_sigma_k": 5.0,
        "intensity_sigma_k": 5.0,
        "h_maxima_fraction": 0.20,
        "rescue_intensity_factor": 1.0,
        "rescue_min_volume_um3": 0.20,
    },
    "vgat": {
        "min_volume_um3": 0.04,
        "max_volume_um3": 5.0,
        "noise_sigma_k": 5.0,
        "intensity_sigma_k": 5.0,
        "h_maxima_fraction": 0.15,
        "morphological_closing": True,
        "closing_radius_um": 0.15,
    },
    "gad65": {
        "min_volume_um3": 0.03,
        "max_volume_um3": 5.0,
        "noise_sigma_k": 4.5,
        "intensity_sigma_k": 4.5,
        "h_maxima_fraction": 0.15,
    },
    "VNUT": {
        "min_volume_um3": 0.10,
        "max_volume_um3": 3.0,
        "noise_sigma_k": 5.0,
        "intensity_sigma_k": 6.0,
        "h_maxima_fraction": 0.15,
        "rescue_intensity_factor": 1.2,
        "rescue_min_volume_um3": 0.20,
    },
    # Axon markers
    "5HT": {
        "min_volume_um3": 0.03,
        "max_volume_um3": 8.0,
        "noise_sigma_k": 3.0,
        "intensity_sigma_k": 3.0,
        "max_elongation": 4.5,
    },
    "ChaT": {
        "min_volume_um3": 0.05,
        "max_volume_um3": 8.0,
        "noise_sigma_k": 5.0,
        "intensity_sigma_k": 5.0,
        "max_elongation": 4.5,
    },
    "TH": {
        "min_volume_um3": 0.08,
        "max_volume_um3": 8.0,
        "noise_sigma_k": 4.5,
        "intensity_sigma_k": 4.5,
        "max_elongation": 4.5,
    },
}

# Object validation: elongation filtering (for axon markers)
MAX_ELONGATION_AXON_MARKERS = 4.5  # objects more elongated than this are axon segments

# Global defaults (used when a marker is not in MARKER_PARAMS)
_MARKER_PARAM_DEFAULTS = {
    "min_volume_um3": MIN_BOUTON_VOLUME_UM3,
    "max_volume_um3": MAX_BOUTON_VOLUME_UM3,
    "min_support_voxels": MIN_CANDIDATE_SUPPORT_VOXELS,
    "noise_sigma_k": 3.0,
    "intensity_sigma_k": 3.0,
    "h_maxima_fraction": H_MAXIMA_FRACTION,
    "max_elongation": MAX_ELONGATION_AXON_MARKERS,
    "morphological_closing": False,
    "closing_radius_um": 0.0,
    # Location-independent intensity multiplier.  Both released
    # specifications retain 1.0; sensitivity is confined to size/LoG criteria.
    "global_intensity_factor": 1.0,
    # Post-pipeline rescue for bright objects missed by LoG.
    "rescue_intensity_factor": 2.5,  # rescue threshold = factor * noise_floor
    "rescue_min_volume_um3": 0.30,  # minimum volume for rescued objects
}


def get_marker_params(marker):
    """Return merged parameter dict for *marker* (marker-specific over defaults)."""
    out = dict(_MARKER_PARAM_DEFAULTS)
    out.update(MARKER_PARAMS.get(marker, {}))
    return out


# Operational Iba1 relation parameters.  The complete-inclusion endpoint means
# only that every segmented candidate voxel lies within segmented Iba1 volume.
CONTACT_DISTANCE_UM = 0.1
COMPLETE_INCLUSION_THRESHOLD = 1.0

# Batch processing
N_CORES = 6  # number of parallel jobs (24 GB RAM, ~2.2 GB per worker)
BATCH_BACKEND = "loky"  # joblib backend

# Filename patterns to exclude
EXCLUDE_PATTERNS = ["test", "ekstra", "zest", "extra", "pfar"]

# Statistical analysis
ALPHA = 0.05
MULTIPLE_TESTING_METHOD = "fdr_bh"  # Benjamini-Hochberg FDR
MIN_SAMPLES_PER_GROUP = 3

# Stability / parameter optimization
N_STABILITY_FILES = 20  # number of files for parameter stability analysis
STABILITY_PARAM_GRID = {
    "h_maxima_fraction": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "min_bouton_volume_um3": [0.005, 0.01, 0.02],
    "max_bouton_volume_um3": [3.0, 5.0, 8.0],
}

# QC thresholds (automated sanity checks)
QC_OUTLIER_SD = 3.0  # flag files >3 SD from mean density
QC_MICROGLIA_FRACTION_RANGE = (0.01, 0.15)  # expected 1-15% of neuropil
