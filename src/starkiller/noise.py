"""Estimate the noise in an image that also contains stars and nebulosity.

This is PixInsight's MRS noise evaluation, after Starck and Murtagh's
"multiresolution support": a pixel belongs to the noise if its wavelet
coefficients are insignificant at every scale. On the M13 lights it agrees
with PixInsight to about 0.02%.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from starkiller.wavelets import B3_NOISE_PER_LAYER, b3_layers

# A coefficient is significant above this many noise deviations.
SIGNIFICANCE = 3.0
MRS_TOLERANCE = 1e-4
# The noise pixels are those within three sigma at every scale, so their
# spread understates the noise by this factor.
MRS_BIAS = 0.974
# Pixels this close to black or white are clipped and carry no noise.
RANGE = (0.00002, 0.99998)
# Use the largest number of layers that leaves this fraction of noise pixels.
MIN_NOISE_FRACTION = 0.01


@dataclass(frozen=True)
class Noise:
    """Standard deviation of the noise and the fraction of pixels it came from."""

    sigma: float
    fraction: float


def _k_sigma(values: npt.NDArray[np.float32], tolerance: float = 0.01, passes: int = 10) -> float:
    """Standard deviation with values beyond three deviations clipped out."""
    sigma = float(values.std(dtype=np.float64, ddof=1))
    for _ in range(passes):
        clipped = float(values[np.abs(values) < SIGNIFICANCE * sigma].std(dtype=np.float64, ddof=1))
        if abs(sigma - clipped) / sigma < tolerance:
            return clipped
        sigma = clipped
    return sigma


def noise_mrs(image: npt.NDArray[np.floating], layer_count: int = 4) -> Noise:
    """Noise of a 2-D image from its multiresolution support."""
    layers, residual = b3_layers(image, layer_count)
    in_range = (image > RANGE[0]) & (image < RANGE[1])
    detail = image - residual

    sigma = _k_sigma(layers[0][in_range]) / B3_NOISE_PER_LAYER[0]
    while True:
        noise = in_range.copy()
        for layer, per_layer in zip(layers, B3_NOISE_PER_LAYER, strict=False):
            noise &= np.abs(layer) < SIGNIFICANCE * sigma * per_layer
        count = int(noise.sum())
        if count < 2:
            return Noise(sigma, 0.0)
        previous, sigma = sigma, float(detail[noise].std(dtype=np.float64, ddof=1))
        if abs(previous - sigma) / previous < MRS_TOLERANCE:
            return Noise(sigma / MRS_BIAS, count / image.size)


def evaluate_noise(image: npt.NDArray[np.floating]) -> Noise:
    """Noise of a 2-D image, using as many wavelet layers as the image allows.

    Crowded images leave few pure-noise pixels at four layers, so the count
    drops until enough remain.
    """
    for layer_count in (4, 3, 2):
        noise = noise_mrs(image, layer_count)
        if noise.fraction >= MIN_NOISE_FRACTION:
            return noise
    layers, _ = b3_layers(image, 1)
    return Noise(_k_sigma(layers[0]) / B3_NOISE_PER_LAYER[0], 1.0)
