"""Denoise a linear master before it is stretched.

The integration leaves noise set by the sky and the exposure, and the
stretch lifts it with the faint signal. Nothing can take it out of the
data; this hides it. Each wavelet layer of the image is shrunk towards
zero where its coefficients are within the noise, under a mask that
spares bright signal, and with the colour treated harder than the
brightness, since the eye forgives smoothed colour and not smoothed
detail.

The noise in a registered stack is correlated between neighbouring
pixels: debayering and the resampling of registration spread each pixel
into its neighbours. So the share of the noise in each layer is not the
white-noise table's, and each layer's noise is measured on its own, from
the pixels that are insignificant at every scale.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quietsky.deconvolve import gaussian_blur, measure_psf
from quietsky.noise import MRS_MAX_ITERATIONS, MRS_TOLERANCE, RANGE, SIGNIFICANCE, k_sigma
from quietsky.stars import Stars, detect_stars
from quietsky.stretch import TARGET_BACKGROUND, autostretch
from quietsky.wavelets import b3_layers

Float32 = npt.NDArray[np.float32]
Float64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]

# Standard deviation of a unit Gaussian truncated at SIGNIFICANCE deviations:
# the spread of a layer's coefficients over the pixels within the noise
# understates the noise by this factor.
TRUNCATED_STD = 0.9866
# Thresholds in noise deviations, finest layer first. The brightness keeps
# its coarser layers nearly whole, where faint nebula lives; the colour is
# smoothed harder and deeper, as colour noise is coarse and the eye forgives
# smooth colour. Stopping the colour short, at four layers, left the red
# halos of bright stars as crisp rings; six layers leave them a soft glow.
LUMINANCE_THRESHOLDS = (3.0, 2.0, 1.0)
CHROMA_THRESHOLDS = (4.0, 4.0, 4.0, 4.0, 4.0, 4.0)
# Wavelet layers smoothed away before the luminance is stretched into the
# mask: enough to drop the noise, not enough to lose the stars.
MASK_SMOOTHING = 2
# Stretched luminance at which the mask reaches zero: bright nebula and star
# cores are left as they are, well before the stretch reaches white.
MASK_PROTECTED = 0.75
# Each detected star is protected under a Gaussian this many times wider
# than the star profile, so its wings are kept as well as its core.
STAR_PROTECTION = 1.5
# The profile width assumed when there are no stars to measure one from.
DEFAULT_PSF = 2.0

Region = tuple[int, int, int, int]


def unclipped(image: Float32) -> Bool:
    """The pixels that are not clipped to black or white, which carry no noise."""
    result: Bool = (image > RANGE[0]) & (image < RANGE[1])
    return result


def layer_noise(
    image: Float32, count: int, sky: Region | None = None, valid: Bool | None = None
) -> list[float]:
    """Noise standard deviation of each of count wavelet layers, finest first.

    With a sky region (x, y, width, height) the noise is the clipped spread
    of each layer over it. Otherwise it is measured as noise_mrs measures
    the pixel noise: over the pixels whose coefficients are within the noise
    at every scale, with the noise of each layer re-estimated from those
    pixels until it settles, among the valid pixels (default: those not
    clipped to black or white). Each estimate is corrected for the truncation
    the selection makes.
    """
    layers, _ = b3_layers(image, count)
    if sky is not None:
        x, y, width, height = sky
        window = (slice(y, y + height), slice(x, x + width))
        return [k_sigma(layer[window]) / TRUNCATED_STD for layer in layers]
    in_range = unclipped(image) if valid is None else valid
    sigmas = [k_sigma(layer[in_range]) / TRUNCATED_STD for layer in layers]
    for _ in range(MRS_MAX_ITERATIONS):
        support = in_range.copy()
        for layer, sigma in zip(layers, sigmas, strict=True):
            support &= np.abs(layer) < SIGNIFICANCE * sigma
        if support.sum() < 2:
            break
        estimates = [
            float(layer[support].std(dtype=np.float64, ddof=1)) / TRUNCATED_STD for layer in layers
        ]
        settled = all(
            abs(new - old) / old < MRS_TOLERANCE for new, old in zip(estimates, sigmas, strict=True)
        )
        sigmas = estimates
        if settled:
            break
    return sigmas


def shrink_layers(image: Float32, sigmas: list[float], thresholds: list[float]) -> Float32:
    """Shrink each wavelet layer towards zero where it is within the noise.

    A coefficient within threshold deviations of the layer's noise is set to
    zero; a larger one is shrunk by the square of that limit over itself
    (the non-negative garrote). Unlike a soft threshold, which takes the
    whole limit off every coefficient and so dims every star and nebula by
    a little, this leaves strong coefficients nearly alone, so the residual
    holds grain and not the shapes of the signal.
    """
    detail, residual = b3_layers(image, len(sigmas))
    result = residual.astype(np.float64)
    for layer, sigma, threshold in zip(detail, sigmas, thresholds, strict=True):
        limit = threshold * sigma
        if limit <= 0:
            result += layer
            continue
        magnitude = np.abs(layer.astype(np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            kept = np.where(magnitude > limit, layer - limit**2 / layer, 0.0)
        result += kept
    output: Float32 = result.astype(np.float32)
    return output


def star_footprint(luminance: Float32, stars: Stars | None = None) -> Float32:
    """How much of each pixel belongs to a star: 1 at the centres, falling off with the profile.

    The stars are detected if not given, and their profile measured from
    the isolated bright ones; each then marks a Gaussian STAR_PROTECTION
    times as wide as the profile. Faint stars are the ones a denoise would
    otherwise eat, as their coefficients are only a few times the noise.
    """
    if stars is None:
        stars = detect_stars(luminance)
    if len(stars) == 0:
        return np.zeros(luminance.shape, np.float32)
    try:
        sigma = measure_psf(luminance, stars)
    except ValueError:
        sigma = DEFAULT_PSF
    width = STAR_PROTECTION * sigma
    centres = np.zeros(luminance.shape, np.float32)
    rows = np.clip(np.rint(stars.y).astype(np.intp), 0, luminance.shape[0] - 1)
    columns = np.clip(np.rint(stars.x).astype(np.intp), 0, luminance.shape[1] - 1)
    centres[rows, columns] = 1
    # A blurred unit impulse peaks at 1 / (2 pi width^2); scaled back to 1.
    footprint: Float32 = np.clip(
        gaussian_blur(centres, width) * (2 * np.pi * width**2), 0, 1
    ).astype(np.float32)
    return footprint


def protection_mask(
    luminance: Float32, smoothing: int = MASK_SMOOTHING, stars: Stars | None = None
) -> Float32:
    """How much to denoise each pixel: 1 on the sky, falling to 0 on signal.

    The luminance is smoothed, stretched as the preview would be, and
    inverted, so faint sky and faint nebula are smoothed in full, bright
    nebula little and star cores not at all; then every detected star,
    however faint, is taken out under its footprint.
    """
    _, smooth = b3_layers(luminance, smoothing)
    stretched = autostretch(smooth)
    mask: Float32 = np.clip(
        1 - (stretched - TARGET_BACKGROUND) / (MASK_PROTECTED - TARGET_BACKGROUND), 0, 1
    ).astype(np.float32)
    mask *= 1 - star_footprint(luminance, stars)
    return mask


@dataclass(frozen=True)
class Component:
    """The noise of one luminance or chrominance component, by layer, before and after."""

    name: str
    before: list[float]
    after: list[float]


def denoise_image(
    image: Float32,
    thresholds: tuple[float, ...] = LUMINANCE_THRESHOLDS,
    chroma_thresholds: tuple[float, ...] = CHROMA_THRESHOLDS,
    mask: Float32 | None = None,
    sky: Region | None = None,
    amount: float = 1.0,
) -> tuple[Float32, Float32, list[Component]]:
    """Denoise a linear (H, W) or (H, W, C) image.

    The luminance (the mean of the channels, or the image itself when mono)
    is shrunk under the given thresholds, finest layer first, and the
    chrominance (each channel less the luminance) under chroma_thresholds.
    Each result is blended with the original by the mask: 1 means the
    denoised value, 0 the original, and the default mask is
    protection_mask. The luminance is blended by amount as well, so some
    of the fine grain can be kept and the sky stays sky rather than
    plastic; the colour is always taken in full, as the eye forgives
    smooth colour. The noise of each layer is measured from the image,
    over the pixels the luminance leaves unclipped, or from the sky region
    (x, y, width, height) when given.

    Returns the denoised image, the mask used and the noise of each
    component by layer before and after.
    """
    colour = image.ndim == 3
    luminance = image.mean(axis=-1).astype(np.float32) if colour else image
    if mask is None:
        mask = protection_mask(luminance)
    valid = unclipped(luminance)
    components = []

    def shrink(name: str, component: Float32, limits: tuple[float, ...]) -> Float32:
        if not limits or all(limit <= 0 for limit in limits):
            return component
        before = layer_noise(component, len(limits), sky, valid)
        result = shrink_layers(component, before, list(limits))
        components.append(Component(name, before, layer_noise(result, len(limits), sky, valid)))
        return result

    cleaned = shrink("L", luminance, thresholds)
    cleaned = (luminance + amount * mask * (cleaned - luminance)).astype(np.float32)
    if not colour:
        return cleaned, mask, components
    names = "RGB" if image.shape[-1] == 3 else [str(c) for c in range(image.shape[-1])]
    chroma = [(image[..., index] - luminance).astype(np.float32) for index in range(len(names))]
    shrunk = [
        shrink(name, component, chroma_thresholds)
        for name, component in zip(names, chroma, strict=True)
    ]
    # Shrinking each colour on its own leaves a little brightness in their
    # sum; taking it out keeps the colour free of brightness, so the
    # luminance is exactly the one denoised above.
    level = np.mean(shrunk, axis=0)
    channels = [
        cleaned + original + mask * (result - level - original)
        for original, result in zip(chroma, shrunk, strict=True)
    ]
    output: Float32 = np.stack(channels, axis=-1).astype(np.float32)
    return output, mask, components
