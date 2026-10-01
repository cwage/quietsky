"""Robust estimates of where pixel values sit and how widely they spread.

Integration uses these to bring frames onto a common level and scale before
combining them. They follow PixInsight's definitions, so the numbers agree
with the estimates it records in a master frame's history.
"""

import numpy as np
import numpy.typing as npt

# Tuning constant of the biweight: values further than this many reference
# deviations from the centre get no weight.
BIWEIGHT_K = 9
# Iterative k-sigma: clip at this many deviations, stop when the scale
# changes by less than the tolerance, and correct the clipped estimate so it
# matches the standard deviation of a normal distribution.
IKSS_CLIP = 4.0
IKSS_TOLERANCE = 1e-6
IKSS_NORMALIZATION = 0.991


def mad(values: npt.NDArray[np.floating], center: float) -> float:
    """Median absolute deviation from center, not scaled to a sigma."""
    return float(np.median(np.abs(values - center)))


def biweight_midvariance(values: npt.NDArray[np.floating], center: float, sigma: float) -> float:
    """Variance estimate that smoothly discounts values far from center.

    sigma is a reference estimate of dispersion, normally the MAD.
    """
    reach = BIWEIGHT_K * sigma
    if 1.0 + reach == 1.0:
        return 0.0
    deviation = values - center
    y2 = (deviation / reach) ** 2
    inside = y2 < 1
    deviation, y2 = deviation[inside], y2[inside]
    numerator = float((deviation**2 * (1 - y2) ** 4).sum())
    denominator = float(((1 - y2) * (1 - 5 * y2)).sum()) ** 2
    if 1.0 + denominator == 1.0:
        return 0.0
    return values.size * numerator / denominator


def ikss(values: npt.ArrayLike) -> tuple[float, float]:
    """Location and scale by iterative k-sigma clipping.

    Each pass takes the median and the biweight midvariance scale of the
    values still in play, then drops those more than four scales from the
    median. Stars and other outliers go in the first passes, leaving the
    level and noise of the background.
    """
    ordered = np.sort(np.asarray(values, dtype=np.float64), axis=None)
    low, high = 0, ordered.size
    previous = 1.0
    while high - low >= 1:
        kept = ordered[low:high]
        location = float(0.5 * (kept[(kept.size - 1) // 2] + kept[kept.size // 2]))
        scale = float(np.sqrt(biweight_midvariance(kept, location, mad(kept, location))))
        if 1.0 + scale == 1.0:
            return location, 0.0
        if (previous - scale) / scale < IKSS_TOLERANCE:
            return location, IKSS_NORMALIZATION * scale
        previous = scale
        low = int(np.searchsorted(ordered, location - IKSS_CLIP * scale, "left"))
        high = int(np.searchsorted(ordered, location + IKSS_CLIP * scale, "right"))
    return 0.0, 0.0
