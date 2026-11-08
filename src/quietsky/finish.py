"""Make the display version of a linear master.

A stacked master is linear and nearly black. The finish takes the sky out,
balances the colour on the stars, stretches each channel with the same
arcsinh so that the sky lands at a chosen grey, sets the sky neutral and
adds saturation. That is the finish the Heart was given by hand; its
settings are the defaults here, and the knobs beyond them (a tone curve
for the nebula's range, local contrast at coarse scales, saturation that
spares the sky, a denoise before the stretch, star reduction) are off or
at those values until asked for.

The colour is balanced on the light of whole stars, not on their peak
pixels. The Heart script balanced the brightest pixels, which on the
Seestar made the star cores white and everything else red: its red
channel is a third wider than the green and blue, so the peaks hold less
of the red light than the wings and the faint stars do. Each detected
star's flux is measured in an aperture per channel, and the median star
is made white.

Everything that sets the stretch (sky, noise, colour gains, black point,
stretch factor) is measured on the master as it came, so a denoise or
any other change to the pixels is shown under the same stretch.

Even balanced, a star whose channels differ in width is not white in
every pixel: the broad channel is weaker at the peak and stronger in the
wings, and the stretch, being concave, favours the wings. Matching the
widths (the sharper channels blurred to the widest) before the stretch
makes every pixel of a star the colour of its light; so does sharpening
the master to one width beforehand.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from quietsky.deconvolve import gaussian_blur, local_minimum, measure_psf
from quietsky.denoise import DEFAULT_PSF, denoise_image, star_footprint
from quietsky.stars import Stars, detect_stars, windows
from quietsky.stretch import midtones
from quietsky.wavelets import b3_layers

Float32 = npt.NDArray[np.float32]
Float64 = npt.NDArray[np.float64]

MAD_TO_SIGMA = 1.4826
# Where the sky median lands in the output, and the saturation, as the
# Heart was finished by hand.
SKY_TARGET = 0.13
SATURATION = 1.3
# The black point sits this many noise deviations below the sky.
BLACK_POINT = -2.5
# The stretch's top: this percentile of the luminance maps to white.
TOP_PERCENTILE = 99.99
# Colour balance: the stars' flux is summed in a square aperture this many
# PSF widths to each side of the centre (at least and at most the pixels
# given), with the local sky taken from a ring this wide outside it.
APERTURE_SIGMAS = 4.0
APERTURE_HALF = (4, 10)
APERTURE_RING = 3
# A star counts if no pixel of its aperture reaches this level, and each
# channel's flux is this many deviations above the aperture's noise.
STAR_SATURATED = 0.9
STAR_SNR = 10.0
# With fewer stars than this, the brightest pixels at this percentile and
# above, not saturated, are made to average white instead, as the Heart
# script did.
MIN_STARS = 5
STAR_PERCENTILE = 99.7
# The nebula's range is set by this percentile of the luminance.
RANGE_PERCENTILE = 99.0
# Saturation weighted by signal reaches its full value this many noise
# deviations above the sky.
CHROMA_SIGNAL = 3.0
# Local contrast works on these wavelet layers (16, 32 and 64 px).
LOCAL_CONTRAST_LAYERS = (4, 5, 6)
# Star reduction pulls each star towards the minimum within this radius.
STAR_REDUCTION_RADIUS = 2
# Channels within this many pixels of width of the widest are not blurred.
MATCH_TOLERANCE = 0.05
STRETCH_SEARCH = (1.0, 1e9)
# Factors tried, on a log scale, to find where the range peaks.
RANGE_GRID = 120
BISECTION_STEPS = 100


@dataclass(frozen=True)
class Tone:
    """What the master measured as, and the stretch chosen from it."""

    sky: Float64
    noise: float
    gains: Float64
    black: float
    top: float
    stretch: float
    balance: float
    # The stars' width per channel, in pixels, when the widths were matched.
    widths: Float64 | None = None


@dataclass(frozen=True)
class Settings:
    """The knobs of a finish; the defaults give the Heart as it was finished by hand."""

    sky_target: float = SKY_TARGET
    # The black point, in noise deviations below the sky: nearer than
    # BLACK_POINT clips the dark side of the sky's noise, which deepens the
    # blacks and, as the stretch then has less far to lift the sky, spares
    # the stars and the nebula's cores the compression.
    black_point: float = BLACK_POINT
    # The white point: this percentile of the luminance maps to white. The
    # default puts only the star cores there and squeezes the nebula into
    # the bottom of the scale; a lower one, say 99.9, lets the nebula and
    # the stars use the whole range and clips the brightest cores white.
    top_percentile: float = TOP_PERCENTILE
    range_target: float | None = None
    saturation: float = SATURATION
    # How much of the saturation the sky gets: 1 the same as the nebula, 0 none.
    sky_chroma: float = 1.0
    local_contrast: float = 1.0
    star_reduction: float = 0.0
    denoise: str | None = None
    denoise_amount: float = 1.0
    # Colour gains to use instead of the ones measured on the stars.
    gains: tuple[float, float, float] | None = None
    # Blur the sharper channels to the width of the widest before the stretch.
    match_psf: bool = False
    # The number of frames that reached each pixel, for the denoise.
    coverage: npt.NDArray[np.integer] | None = None


def aperture_half(sigma: float) -> int:
    """Half the side of the aperture, in pixels, for stars of this width."""
    return int(np.clip(np.ceil(APERTURE_SIGMAS * sigma), *APERTURE_HALF))


def star_fluxes(rgb: Float64, stars: Stars, half: int) -> tuple[Float64, npt.NDArray[np.bool_]]:
    """Each star's flux per channel, less the local sky, and whether its aperture is unsaturated.

    The aperture is a square half pixels to each side of the star's centre
    pixel; the local sky is the median of a ring APERTURE_RING pixels wide
    around it. Stars too close to the edge for the ring are left out.
    """
    outer = half + APERTURE_RING
    rows = np.rint(stars.y).astype(np.intp)
    columns = np.rint(stars.x).astype(np.intp)
    height, width = rgb.shape[:2]
    inside = (
        (rows >= outer) & (rows < height - outer) & (columns >= outer) & (columns < width - outer)
    )
    rows, columns = rows[inside], columns[inside]
    offsets = np.arange(-outer, outer + 1)
    box = (np.abs(offsets[:, None]) <= half) & (np.abs(offsets[None, :]) <= half)
    ring = ~box
    fluxes = np.empty((len(rows), 3))
    unsaturated = np.ones(len(rows), bool)
    for channel in range(3):
        light = windows(rgb[..., channel], rows, columns, outer)
        local = np.median(light[:, ring], axis=1)
        fluxes[:, channel] = (light[:, box] - local[:, None]).sum(axis=1)
        unsaturated &= light[:, box].max(axis=1) < STAR_SATURATED
    return fluxes, unsaturated


def star_colour(rgb: Float64, stars: Stars) -> Float64 | None:
    """The colour of the median star, as each channel's share of its light, or None with too few.

    rgb is the master with the sky still in, so saturation can be seen.
    The aperture is sized from the stars' width in the luminance, which
    holds all of a star in every channel even when one channel is broader
    than the others. Stars whose flux in any channel is within STAR_SNR
    deviations of the aperture's noise are left out, as a ratio of noisy
    fluxes leans.
    """
    try:
        sigma = measure_psf(rgb.mean(axis=2).astype(np.float32), stars)
    except ValueError:
        sigma = DEFAULT_PSF
    half = aperture_half(sigma)
    fluxes, unsaturated = star_fluxes(rgb, stars, half)
    noise = np.array(
        [
            MAD_TO_SIGMA * float(np.median(np.abs(rgb[..., c] - np.median(rgb[..., c]))))
            for c in range(3)
        ]
    )
    clear = (fluxes > STAR_SNR * noise * (2 * half + 1)).all(axis=1) & unsaturated
    if clear.sum() < MIN_STARS:
        return None
    fluxes = fluxes[clear]
    colour: Float64 = np.median(fluxes / fluxes.mean(axis=1, keepdims=True), axis=0)
    return colour


def bright_pixel_colour(rgb: Float64, sky: Float64) -> Float64:
    """The colour of the brightest unsaturated pixels: the Heart script's star colour."""
    rgb = rgb - sky
    lum = rgb.mean(axis=2)
    bright = (lum > np.percentile(lum, STAR_PERCENTILE)) & (
        rgb.max(axis=2) < STAR_SATURATED - sky.max()
    )
    colour: Float64 = np.median(rgb[bright] / rgb[bright].mean(axis=1, keepdims=True), axis=0)
    return colour


def measure(
    rgb: Float64, stars: Stars | None = None, valid: npt.NDArray[np.bool_] | None = None
) -> tuple[Float64, float, Float64]:
    """The sky per channel, the noise of the luminance, and the star colour gains.

    The sky is the median, as the frame is mostly sky and faint nebula.
    The gains make the median star's light white, each star's flux being
    summed over an aperture per channel; with too few stars to measure,
    the brightest unsaturated pixels are made to average white instead.
    valid marks the pixels the sky and noise are taken over.
    """
    if valid is None:
        valid = np.ones(rgb.shape[:2], np.bool_)
    sky = np.median(rgb[valid], axis=0)
    lum = (rgb - sky).mean(axis=2)
    noise = MAD_TO_SIGMA * float(np.median(np.abs(lum[valid] - np.median(lum[valid]))))
    if stars is None:
        stars = detect_stars(lum.astype(np.float32))
    colour = star_colour(rgb, stars)
    if colour is None:
        colour = bright_pixel_colour(rgb, sky)
    gains: Float64 = 1 / colour
    return sky, noise, gains


def match_psf(image: Float32, stars: Stars | None = None) -> tuple[Float32, Float64]:
    """The image with each channel blurred to the width of the widest, and the widths found.

    A Gaussian blurred by a Gaussian is a Gaussian whose variance is the
    sum of the two, so each channel is blurred by the width that takes it
    to the widest. Channels whose width cannot be measured are left alone.
    """
    if stars is None:
        stars = detect_stars(image.mean(axis=2))
    widths = np.full(3, np.nan)
    for channel in range(3):
        try:
            widths[channel] = measure_psf(image[..., channel], stars)
        except ValueError:
            pass
    if np.isnan(widths).all():
        return image, widths
    widest = float(np.nanmax(widths))
    matched = image.copy()
    for channel in range(3):
        if np.isnan(widths[channel]) or widths[channel] > widest - MATCH_TOLERANCE:
            continue
        extra = float(np.sqrt(widest**2 - widths[channel] ** 2))
        matched[..., channel] = gaussian_blur(image[..., channel], extra)
    return matched, widths


def _arcsinh(values: Float64, factor: float, top: float) -> Float64:
    result: Float64 = np.arcsinh(factor * values) / np.arcsinh(factor * top)
    return result


def _bisect(target: float, low: float, high: float, value: Callable[[float], float]) -> float:
    """The argument in (low, high), searched on a log scale, where the rising value meets target."""
    for _ in range(BISECTION_STEPS):
        mid = float(np.sqrt(low * high))
        low, high = (mid, high) if value(mid) < target else (low, mid)
    return float(np.sqrt(low * high))


def _balance_pinning(value: float, target: float) -> float:
    """The midtones balance that maps value to target: the transfer function, roles swapped."""
    balance = (target - 1) * value / ((2 * target - 1) * value - target)
    return float(np.clip(balance, 1e-6, 1 - 1e-6))


def choose_stretch(
    lum: Float64,
    black: float,
    sky_target: float,
    range_target: float | None,
    top_percentile: float = TOP_PERCENTILE,
) -> tuple[float, float, float]:
    """The arcsinh factor and midtones balance that put the sky and the nebula where asked.

    Returns the top, the stretch factor and the balance. With no range the
    factor alone is chosen, so that the sky median lands at sky_target,
    and the balance is 0.5 (no curve), as the Heart was finished. With a
    range, the curve is chosen so that the sky lands at sky_target and the
    luminance's RANGE_PERCENTILE lands range_target above it: for each
    factor the balance that pins the sky follows in closed form. The range
    that leaves rises with the factor and then falls, as the arcsinh first
    spreads the faint end and then squeezes everything towards white; the
    factor is searched on the falling side, which continues from the
    plain arcsinh, and a range beyond the peak gets the peak.
    """
    top = float(np.percentile(lum, top_percentile)) - black
    level = float(np.median(lum)) - black
    low, high = STRETCH_SEARCH

    def sky_at(factor: float) -> float:
        return float(_arcsinh(np.array(level), factor, top))

    if range_target is None:
        return top, _bisect(sky_target, low, high, sky_at), 0.5

    bright = float(np.percentile(lum, RANGE_PERCENTILE)) - black

    def balance_at(factor: float) -> float:
        return _balance_pinning(sky_at(factor), sky_target)

    def range_at(factor: float) -> float:
        lifted = float(_arcsinh(np.array(bright), factor, top))
        return float(midtones(np.array(lifted, np.float32), balance_at(factor))) - sky_target

    grid = np.logspace(np.log10(low), np.log10(high), RANGE_GRID)
    ranges = [range_at(float(factor)) for factor in grid]
    peak = int(np.argmax(ranges))
    if range_target >= ranges[peak]:
        factor = float(grid[peak])
    else:
        factor = _bisect(-range_target, float(grid[peak]), high, lambda a: -range_at(a))
    return top, factor, balance_at(factor)


def _stretch(values: Float64, tone: Tone) -> Float64:
    out = _arcsinh(np.clip(values - tone.black, 0, None), tone.stretch, tone.top)
    if tone.balance != 0.5:
        out = midtones(out.astype(np.float32), tone.balance).astype(np.float64)
    return out


def finish(image: Float32, settings: Settings | None = None) -> tuple[Float64, Tone]:
    """The display version of a linear (H, W, 3) master, in [0, 1], and how it was made."""
    settings = settings or Settings()
    rgb = image.astype(np.float64)
    # A master is zero where no light reached it. Those pixels are filled
    # with the sky, so that nothing measured or stretched sees an edge
    # there, and come out black.
    covered = rgb.max(axis=2) > 0
    fill = np.median(rgb[covered], axis=0)
    if not covered.all():
        rgb = np.where(covered[..., None], rgb, fill)
        image = rgb.astype(np.float32)
    sky, noise, gains = measure(rgb, valid=covered)
    if settings.gains is not None:
        gains = np.array(settings.gains, np.float64)
    black = settings.black_point * noise
    lum = ((rgb - sky) * gains).mean(axis=2)
    top, stretch, balance = choose_stretch(
        lum[covered], black, settings.sky_target, settings.range_target, settings.top_percentile
    )
    tone = Tone(sky, noise, gains, black, top, stretch, balance)

    if settings.match_psf:
        image, widths = match_psf(image)
        tone = replace(tone, widths=widths)
        rgb = image.astype(np.float64)
    if settings.denoise:
        # Denoised with the unreached pixels at zero, which the noise
        # estimates leave out, rather than the fill, which they would take
        # for noiseless; then filled again.
        denoised, _, _ = denoise_image(
            np.where(covered[..., None], image, 0).astype(np.float32),
            method=settings.denoise,
            amount=settings.denoise_amount,
            coverage=settings.coverage,
        )
        rgb = np.where(covered[..., None], denoised.astype(np.float64), fill)
    rgb = (rgb - sky) * gains
    chan = np.stack([_stretch(rgb[..., c], tone) for c in range(3)], axis=-1)
    # Neutral sky: each channel's median lands on the same level.
    chan -= np.median(chan[covered], axis=0) - np.median(chan[covered])

    if settings.local_contrast != 1.0:
        layers, _ = b3_layers(chan.mean(axis=2).astype(np.float32), max(LOCAL_CONTRAST_LAYERS) + 1)
        coarse: Float32 = np.sum([layers[index] for index in LOCAL_CONTRAST_LAYERS], axis=0)
        chan += ((settings.local_contrast - 1) * coarse)[..., None]

    if settings.star_reduction > 0:
        linear = lum.astype(np.float32)
        footprint = star_footprint(linear, detect_stars(linear))
        floor = np.stack(
            [
                local_minimum(chan[..., c].astype(np.float32), STAR_REDUCTION_RADIUS)
                for c in range(3)
            ],
            axis=-1,
        )
        chan -= settings.star_reduction * footprint[..., None] * (chan - floor)

    grey = chan.mean(axis=2, keepdims=True)
    strength = settings.saturation - 1
    if settings.sky_chroma != 1.0:
        # Chroma weighted by signal: anything CHROMA_SIGNAL noise deviations
        # above the sky, in the linear master, gets it all; the sky itself
        # gets sky_chroma of it.
        signal = np.clip((lum - np.median(lum)) / (CHROMA_SIGNAL * noise), 0, 1)
        weight = settings.sky_chroma + (1 - settings.sky_chroma) * signal
        out: Float64 = np.clip(grey + (chan - grey) * (1 + strength * weight[..., None]), 0, 1)
    else:
        out = np.clip(grey + (chan - grey) * (1 + strength), 0, 1)
    out[~covered] = 0
    return out, tone


def to_8bit(out: Float64) -> npt.NDArray[np.uint8]:
    """Round to 8-bit levels, as the PNG is written."""
    return (out * 255 + 0.5).astype(np.uint8)
