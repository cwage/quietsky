"""Sharpen a deep-sky master: drop its finest noise, then deconvolve the star profile.

Stars in a stacked image are blurred by seeing, focus and the resampling of
registration into a profile much wider than the pixel grid. Richardson-Lucy
deconvolution with that profile, measured from the stars themselves,
tightens stars and the small features of nebulae. Deconvolution also
amplifies noise, so the finest wavelet layers, which hold no detail at this
scale, are thresholded first, and the sky, where there is nothing to
sharpen, keeps its original values.
"""

import numpy as np
import numpy.typing as npt

from quietsky.estimators import mad
from quietsky.stars import Stars, detect_stars, windows
from quietsky.wavelets import B3_NOISE_PER_LAYER, b3_layers

Float32 = npt.NDArray[np.float32]
Float64 = npt.NDArray[np.float64]

MAD_TO_SIGMA = 1.4826
# Half-width of the window a star's profile is measured in.
PSF_HALF = 7
# Stars with a peak above this are flat-topped and measure too wide.
PSF_SATURATION = 0.9
# The measuring window is this many times wider than the current profile
# estimate: wide enough to see the profile, narrow enough to shut out
# neighbours and noise.
PSF_WINDOW = 2.0
PSF_PASSES = 8
# Wavelet layers smoothed away to get the local sky the support mask is
# measured against: about 64 pixels.
SUPPORT_LAYERS = 6


def gaussian_blur(image: Float32, sigma: float) -> Float32:
    """Blur a 2-D image with a Gaussian of the given standard deviation, in pixels."""
    fy = np.fft.fftfreq(image.shape[0])[:, None]
    fx = np.fft.rfftfreq(image.shape[1])[None, :]
    kernel = np.exp(-2 * (np.pi * sigma) ** 2 * (fy**2 + fx**2))
    blurred: Float32 = np.fft.irfft2(np.fft.rfft2(image) * kernel, s=image.shape).astype(np.float32)
    return blurred


def richardson_lucy(image: Float32, sigma: float, iterations: int) -> Float32:
    """Deconvolve a Gaussian blur of the given sigma by Richardson-Lucy iteration.

    Each step compares the blurred estimate with the image and corrects the
    estimate by the ratio, so it stays positive and brightness is kept.
    """
    observed = np.clip(image, 1e-6, None).astype(np.float64)
    fy = np.fft.fftfreq(image.shape[0])[:, None]
    fx = np.fft.rfftfreq(image.shape[1])[None, :]
    transfer = np.exp(-2 * (np.pi * sigma) ** 2 * (fy**2 + fx**2))

    def blur(values: Float64) -> Float64:
        result: Float64 = np.fft.irfft2(np.fft.rfft2(values) * transfer, s=image.shape)
        return result

    estimate = observed.copy()
    for _ in range(iterations):
        estimate *= blur(observed / np.maximum(blur(estimate), 1e-9))
    result: Float32 = estimate.astype(np.float32)
    return result


def measure_psf(image: Float32, stars: Stars | None = None, count: int = 200) -> float:
    """Standard deviation, in pixels, of the Gaussian that fits the stars.

    The brightest unsaturated stars with no neighbour inside their window
    are measured by their second moments under a Gaussian window twice as
    wide as the profile, which keeps noise and faint neighbours out. A
    Gaussian seen through a Gaussian window is a narrower Gaussian, so the
    true width follows from the measured one; the window is then set from
    that width and the measurement repeated until it settles. The median
    over the stars is returned.
    """
    luminance = image if image.ndim == 2 else image.mean(axis=-1)
    if stars is None:
        stars = detect_stars(luminance)
    height, width = luminance.shape
    rows = np.rint(stars.y).astype(np.intp)
    columns = np.rint(stars.x).astype(np.intp)
    inside = (
        (rows >= PSF_HALF)
        & (rows < height - PSF_HALF)
        & (columns >= PSF_HALF)
        & (columns < width - PSF_HALF)
    )
    candidates = np.nonzero(inside)[0][: 3 * count]
    separation = np.hypot(
        stars.x[candidates, None] - stars.x[None, :], stars.y[candidates, None] - stars.y[None, :]
    )
    separation[np.arange(len(candidates)), candidates] = np.inf
    isolated = separation.min(axis=1) > 2 * PSF_HALF
    unsaturated = luminance[rows[candidates], columns[candidates]] < PSF_SATURATION
    chosen = candidates[isolated & unsaturated][:count]
    if len(chosen) == 0:
        raise ValueError("no isolated unsaturated stars to measure")

    light = windows(luminance, rows[chosen], columns[chosen], PSF_HALF).astype(np.float64)
    # The sky under each star: the median of the window's edge.
    edge = np.concatenate(
        [light[:, 0, :], light[:, -1, :], light[:, 1:-1, 0], light[:, 1:-1, -1]], axis=1
    )
    light = np.clip(light - np.median(edge, axis=1)[:, None, None], 0, None)
    offsets = np.arange(-PSF_HALF, PSF_HALF + 1, dtype=np.float64)
    dy = offsets[None, :, None] - (stars.y[chosen] - rows[chosen])[:, None, None]
    dx = offsets[None, None, :] - (stars.x[chosen] - columns[chosen])[:, None, None]
    distance2 = dy**2 + dx**2

    sigma = 2.0
    for _ in range(PSF_PASSES):
        window_variance = (PSF_WINDOW * sigma) ** 2
        weighted = light * np.exp(-distance2 / (2 * window_variance))
        total = weighted.sum(axis=(1, 2))
        # Half the mean squared distance: the variance of the windowed profile.
        seen = (weighted * distance2).sum(axis=(1, 2)) / np.maximum(total, 1e-30) / 2
        seen = np.clip(seen, 1e-6, 0.99 * window_variance)
        true = seen * window_variance / (window_variance - seen)
        sigma = float(np.sqrt(np.median(true[total > 0])))
    return sigma


def denoise(image: Float32, noise: float, threshold: float = 3.0, layers: int = 2) -> Float32:
    """Shrink the finest wavelet layers towards zero where they are within the noise.

    Values of the finest layer within threshold noise deviations are set to
    zero and the rest moved towards zero by that much (a soft threshold).
    Each coarser layer uses half the threshold of the one before, as it
    holds more detail and less noise. noise is the standard deviation of
    the pixel noise; each layer's share of it is known for the B3 spline.
    """
    detail, residual = b3_layers(image, layers)
    result = residual.astype(np.float64)
    for index, layer in enumerate(detail):
        limit = threshold / 2**index * noise * B3_NOISE_PER_LAYER[index]
        result += np.sign(layer) * np.maximum(np.abs(layer) - limit, 0)
    output: Float32 = result.astype(np.float32)
    return output


def _local_minimum(image: Float32, radius: int) -> Float32:
    """The smallest value within radius pixels of each pixel."""
    height, width = image.shape
    padded = np.pad(image, radius, mode="edge")
    result = image.copy()
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result = np.minimum(result, padded[dy : dy + height, dx : dx + width])
    return result


def deconvolve(
    image: Float32, sigma: float, iterations: int, noise: float, support: float = 3.0
) -> Float32:
    """Deconvolve the star profile where there is signal; leave the sky as it is.

    The sky level is taken off and a pedestal of ten noise deviations put
    under the image so that Richardson-Lucy sees positive values; the
    pedestal is flat, which the iteration leaves alone. The result replaces
    the image where the signal, smoothed by the profile, stands more than
    support deviations of its own noise above the local sky, fading in over
    the next support deviations, so deconvolution never amplifies the noise
    of empty sky. Black registration borders are steps the profile cannot
    explain, and nothing within its reach of one is touched either. Dark
    rings round stars are held at the darkest original value nearby.
    """
    sky = float(np.median(image))
    # Reflected margins keep the Fourier wrap-around, and the bend the local
    # sky takes at a reflected edge, outside the image.
    margin = 2 * 2**SUPPORT_LAYERS
    signal = np.pad(image - sky, margin, mode="reflect").astype(np.float32)
    pedestal = max(10 * noise, 1e-6)
    data = np.clip(signal + pedestal, 1e-3 * pedestal, None).astype(np.float32)
    estimate = richardson_lucy(data, sigma, iterations) - pedestal
    # Richardson-Lucy digs a dark ring round a star whose wings are wider
    # than the profile. Nothing is let fall below the darkest original pixel
    # within the profile's reach, which is where such a ring would sit.
    estimate = np.maximum(estimate, _local_minimum(signal, int(np.ceil(2 * sigma))))

    # Signal at the scale of the profile, above the local sky: a nebula's own
    # broad glow does not open the mask, only what stands out from it.
    smooth = gaussian_blur(signal, sigma)
    _, local_sky = b3_layers(smooth, SUPPORT_LAYERS)
    inside = (slice(margin, -margin), slice(margin, -margin))
    contrast = (smooth - local_sky)[inside]
    signal, estimate = signal[inside], estimate[inside]
    smooth_noise = MAD_TO_SIGMA * mad(contrast, float(np.median(contrast)))
    weight = np.clip((contrast - support * smooth_noise) / (support * smooth_noise), 0, 1)
    weight = weight * weight * (3 - 2 * weight)
    border = gaussian_blur((image <= 0).astype(np.float32), 2 * sigma)
    weight[border > 1e-3] = 0
    output: Float32 = (signal + weight * (estimate - signal) + sky).astype(np.float32)
    return output
