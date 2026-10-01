"""Repair defective pixels in calibrated frames.

Some pixels stay wrong after calibration, usually ones whose dark current is
too high or too erratic for the master dark to remove. They are replaced by
the average of their neighbours.

PixInsight's 2018 run used CosmeticCorrection's automatic detection. Its
replacement value is reproduced here exactly, but the rule by which it
picked pixels could not be recovered (see the README), so detection here
works from the master dark instead, which is PixInsight's other mode.
"""

import numpy as np
import numpy.typing as npt

from starkiller.estimators import mad

Float32 = npt.NDArray[np.float32]
Bool = npt.NDArray[np.bool_]

MAD_TO_SIGMA = 1.4826


def hot_pixels(dark: Float32, sigma: float) -> Bool:
    """Pixels of a bias-subtracted master dark that stand out from the rest.

    A pixel is hot if it lies more than sigma robust deviations above the
    median of the dark.
    """
    location = float(np.median(dark))
    hot: Bool = dark > location + sigma * MAD_TO_SIGMA * mad(dark, location)
    return hot


def replace_from_neighbours(image: Float32, defective: Bool, mosaic: bool) -> Float32:
    """Replace the marked pixels by the mean of their eight neighbours.

    In a colour mosaic the neighbours are the nearest pixels of the same
    colour, two pixels away. Neighbours beyond the edge of the image are left
    out of the mean; defective neighbours are not.
    """
    step = 2 if mosaic else 1
    padded = np.pad(image.astype(np.float64), step, constant_values=np.nan)
    rows, columns = np.nonzero(defective)
    neighbours = np.stack(
        [
            padded[rows + step + dy, columns + step + dx]
            for dy in (-step, 0, step)
            for dx in (-step, 0, step)
            if (dy, dx) != (0, 0)
        ]
    )
    result = image.copy()
    result[rows, columns] = np.nanmean(neighbours, axis=0)
    return result
