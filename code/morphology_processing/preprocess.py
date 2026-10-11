"""Background subtraction and deconvolution of morphology Iba1 stacks.

The downstream three-dimensional segmentation uses the deconvolved volume,
without slice-wise contrast equalization.
"""

import numpy as np
import tifffile
from skimage.morphology import disk, white_tophat
from skimage.restoration import richardson_lucy

from . import parameters as config


def generate_theoretical_psf():
    """Return the normalized Gaussian PSF used in the recorded processing."""
    r_lateral = 0.61 * config.EMISSION_WAVELENGTH / config.OBJECTIVE_NA
    r_axial = 2 * config.IMMERSION_RI * config.EMISSION_WAVELENGTH / config.OBJECTIVE_NA**2
    sigma_xy = r_lateral / (2.355 * config.VOXEL_SIZE_XY)
    sigma_z = r_axial / (2.355 * config.VOXEL_SIZE_Z)
    sz, sxy = config.DECONV_PSF_SIZE_Z, config.DECONV_PSF_SIZE_XY
    z = np.arange(sz) - sz // 2
    y = np.arange(sxy) - sxy // 2
    x = np.arange(sxy) - sxy // 2
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    psf = np.exp(-0.5 * ((xx / sigma_xy) ** 2 + (yy / sigma_xy) ** 2 + (zz / sigma_z) ** 2))
    psf /= psf.sum()
    return psf


def load_lsm(path):
    """Load the recorded Iba1 channel as a ZYX float32 volume.

    Four-dimensional inputs follow the acquisition's ZCYX layout, with Iba1
    at channel zero. Inputs with other axis layouts must be prepared explicitly.
    """
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        data = series.asarray()
        axes = series.axes
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    elif data.ndim == 4:
        if axes != "ZCYX":
            raise ValueError(f"Expected ZCYX acquisition layout, received {axes}: {path}")
        data = data[:, 0, :, :]
    if data.ndim != 3:
        raise ValueError(f"Expected a three-dimensional Iba1 stack: {path}")
    if not np.isfinite(data).all():
        raise ValueError(f"Nonfinite image values: {path}")
    return data.astype(np.float32)


def preprocess_stack(stack, psf):
    """Return the deconvolved volume used for cell segmentation."""
    stack_min, stack_max = stack.min(), stack.max()
    if stack_max <= stack_min:
        return np.zeros_like(stack)
    normalized = (stack - stack_min) / (stack_max - stack_min)
    selem = disk(config.ROLLING_BALL_RADIUS)
    bg_sub = np.zeros_like(normalized)
    for z in range(normalized.shape[0]):
        bg_sub[z] = white_tophat(normalized[z], selem)
    bg_sub = np.clip(bg_sub, 0, None)
    bg_sub += 1e-6
    return richardson_lucy(bg_sub, psf, num_iter=config.DECONV_ITERATIONS, clip=True)
