"""Remove the sensor's and the optics' fixed signal from a frame.

A raw frame is the light that reached the sensor, dimmed towards the corners
and under dust, plus a constant offset (the bias) and thermal signal that
grows with exposure and temperature (the dark current). Calibration
subtracts the master bias and a scaled master dark, then divides by the
master flat.
"""

import numpy as np
import numpy.typing as npt

from quietsky.estimators import mad
from quietsky.noise import k_sigma_noise

Float32 = npt.NDArray[np.float32]

# Dark pixels this many deviations above the dark's median carry enough
# thermal signal to judge the scaling by.
OPTIMIZATION_THRESHOLD = 3.0
MAD_TO_SIGMA = 1.4826
# Side of the central square the noise is measured on, and how finely and how
# far the scaling factor is searched.
OPTIMIZATION_WINDOW = 1024
OPTIMIZATION_TOLERANCE = 0.0005
MAX_DARK_SCALE = 10.0
GOLDEN = 0.61803399
FIT_HALF_WIDTH = 0.3
FIT_POINTS = 13


def calibrate(
    frame: Float32,
    bias: Float32 | None = None,
    dark: Float32 | None = None,
    dark_scale: float = 1.0,
    flat: Float32 | None = None,
) -> Float32:
    """Subtract the master bias and dark_scale times the master dark, and divide by the flat.

    Each master is optional. The master dark still contains the bias, as a
    stack of raw dark frames does. With a bias, the bias is taken out of the
    dark before scaling. Without one the dark is subtracted whole, bias and
    all, which is only right for a dark_scale of one.

    The flat evens out vignetting and dust shadows. It must have a mean of
    one over the whole frame (see unit_flat), so that the division leaves
    the overall level of the frame alone.
    """
    result: Float32 = frame
    if bias is not None:
        result = result - bias
    if dark is not None:
        thermal = dark if bias is None else dark - bias
        result = result - np.float32(dark_scale) * thermal
    if flat is not None:
        result = result / flat
    return result


def unit_flat(flat: Float32) -> Float32:
    """Scale a master flat to a mean of one.

    One mean is taken over all pixels, also for a colour mosaic, so dividing
    by the result imprints the colour of the flat's light source on the frame.
    """
    scaled: Float32 = flat / np.float32(flat.mean(dtype=np.float64))
    return scaled


def _bin2(image: Float32) -> Float32:
    """Average 2x2 blocks, which turns a colour mosaic into a mono image."""
    height, width = image.shape[0] // 2 * 2, image.shape[1] // 2 * 2
    blocks = image[:height, :width].reshape(height // 2, 2, width // 2, 2)
    return blocks.mean(axis=(1, 3), dtype=np.float32)


def _central_window(image: Float32) -> Float32:
    top = max((image.shape[0] - OPTIMIZATION_WINDOW) // 2, 0)
    left = max((image.shape[1] - OPTIMIZATION_WINDOW) // 2, 0)
    return image[top : top + OPTIMIZATION_WINDOW, left : left + OPTIMIZATION_WINDOW]


def optimization_dark(dark: Float32, mosaic: bool) -> tuple[Float32, float]:
    """The part of a bias-subtracted master dark that scaling is judged by.

    Returns the dark with everything at or below the threshold set to zero,
    and the threshold. A mosaic is binned 2x2 first.
    """
    if mosaic:
        dark = _bin2(dark)
    location = float(np.median(dark))
    threshold = location + OPTIMIZATION_THRESHOLD * MAD_TO_SIGMA * mad(dark, location)
    return np.where(dark > threshold, dark, np.float32(0)), threshold


def optimize_dark(frame: Float32, bias: Float32, dark: Float32, mosaic: bool) -> float:
    """The dark scaling factor that leaves the least noise in the frame.

    Dark current depends on temperature as well as exposure, so the right
    factor is rarely just the ratio of exposure times. This searches for the
    factor that minimises the noise where the dark has signal.

    The method is PixInsight's, and the threshold agrees with the one it
    logs. The noise barely changes near its minimum, and in small steps as
    pixels cross the clipping limit, so the minimum is only defined to within
    several percent: on the M13 lights we come out about 6% above the factors
    PixInsight chose, 11% at most. When the frame's exposure is much shorter than the
    dark's the noise hardly depends on the factor at all and the result is
    not meaningful.
    """
    target = frame - bias
    if mosaic:
        target = _bin2(target)
    target = _central_window(target)
    reference = _central_window(optimization_dark(dark - bias, mosaic)[0])

    def noise(scale: float) -> float:
        return k_sigma_noise(target - np.float32(scale) * reference)

    # Golden section search for the neighbourhood of the minimum.
    low, high = 0.0, MAX_DARK_SCALE
    a = high - GOLDEN * (high - low)
    b = low + GOLDEN * (high - low)
    noise_a, noise_b = noise(a), noise(b)
    while high - low > OPTIMIZATION_TOLERANCE:
        if noise_a < noise_b:
            high, b, noise_b = b, a, noise_a
            a = high - GOLDEN * (high - low)
            noise_a = noise(a)
        else:
            low, a, noise_a = a, b, noise_b
            b = low + GOLDEN * (high - low)
            noise_b = noise(b)
    found = a if noise_a < noise_b else b

    # The search can settle in any of the small steps. A parabola through the
    # noise either side smooths over them.
    scales = np.linspace(max(found - FIT_HALF_WIDTH, 0.0), found + FIT_HALF_WIDTH, FIT_POINTS)
    curvature, slope, _ = np.polyfit(scales, [noise(float(scale)) for scale in scales], 2)
    if curvature <= 0:
        return found
    return float(np.clip(-slope / (2 * curvature), scales[0], scales[-1]))
