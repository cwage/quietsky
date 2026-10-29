"""Make the display version of a linear master.

A stacked master is linear and nearly black. The finish takes the sky out,
balances the colour on the stars, stretches each channel with the same
arcsinh so that the sky lands at a chosen grey, sets the sky neutral and
adds saturation. That is the finish the Heart was given by hand; its
settings are the defaults here, and the knobs beyond them (a tone curve
for the nebula's range, local contrast at coarse scales, saturation that
spares the sky, a denoise before the stretch, star reduction) are off or
at those values until asked for.

Everything that sets the stretch (sky, noise, colour gains, black point,
stretch factor) is measured on the master as it came, so a denoise or
any other change to the pixels is shown under the same stretch.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quietsky.deconvolve import local_minimum
from quietsky.denoise import denoise_image, star_footprint
from quietsky.stars import detect_stars
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
# Colour balance: the brightest pixels at this percentile and above, not
# saturated, are made to average white.
STAR_PERCENTILE = 99.7
STAR_SATURATED = 0.9
# The nebula's range is set by this percentile of the luminance.
RANGE_PERCENTILE = 99.0
# Saturation weighted by signal reaches its full value this many noise
# deviations above the sky.
CHROMA_SIGNAL = 3.0
# Local contrast works on these wavelet layers (16, 32 and 64 px).
LOCAL_CONTRAST_LAYERS = (4, 5, 6)
# Star reduction pulls each star towards the minimum within this radius.
STAR_REDUCTION_RADIUS = 2
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


@dataclass(frozen=True)
class Settings:
    """The knobs of a finish; the defaults give the Heart as it was finished by hand."""

    sky_target: float = SKY_TARGET
    range_target: float | None = None
    saturation: float = SATURATION
    # How much of the saturation the sky gets: 1 the same as the nebula, 0 none.
    sky_chroma: float = 1.0
    local_contrast: float = 1.0
    star_reduction: float = 0.0
    denoise: str | None = None
    denoise_amount: float = 1.0


def measure(rgb: Float64) -> tuple[Float64, float, Float64]:
    """The sky per channel, the noise of the luminance, and the star colour gains.

    The sky is the median, as the frame is mostly sky and faint nebula.
    The gains make the brightest unsaturated pixels, which are stars,
    average white.
    """
    sky = np.median(rgb, axis=(0, 1))
    rgb = rgb - sky
    lum = rgb.mean(axis=2)
    noise = MAD_TO_SIGMA * float(np.median(np.abs(lum - np.median(lum))))
    bright = (lum > np.percentile(lum, STAR_PERCENTILE)) & (
        rgb.max(axis=2) < STAR_SATURATED - sky.max()
    )
    star_colour = np.median(rgb[bright] / rgb[bright].mean(axis=1, keepdims=True), axis=0)
    gains: Float64 = 1 / star_colour
    return sky, noise, gains


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
    lum: Float64, black: float, sky_target: float, range_target: float | None
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
    top = float(np.percentile(lum, TOP_PERCENTILE)) - black
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
    sky, noise, gains = measure(rgb)
    black = BLACK_POINT * noise
    lum = ((rgb - sky) * gains).mean(axis=2)
    top, stretch, balance = choose_stretch(lum, black, settings.sky_target, settings.range_target)
    tone = Tone(sky, noise, gains, black, top, stretch, balance)

    if settings.denoise:
        denoised, _, _ = denoise_image(
            image, method=settings.denoise, amount=settings.denoise_amount
        )
        rgb = denoised.astype(np.float64)
    rgb = (rgb - sky) * gains
    chan = np.stack([_stretch(rgb[..., c], tone) for c in range(3)], axis=-1)
    # Neutral sky: each channel's median lands on the same level.
    chan -= np.median(chan, axis=(0, 1)) - np.median(chan)

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
    return out, tone


def to_8bit(out: Float64) -> npt.NDArray[np.uint8]:
    """Round to 8-bit levels, as the PNG is written."""
    return (out * 255 + 0.5).astype(np.uint8)
