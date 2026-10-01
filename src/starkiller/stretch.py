"""Non-linear stretches for looking at linear data.

A stacked astronomical image is linear: nearly everything sits just above
black. The automatic stretch here is the same idea as PixInsight's
ScreenTransferFunction and is meant for previews, not for final output.
"""

import numpy as np
import numpy.typing as npt

Float32 = npt.NDArray[np.float32]

# Shadows clip this many deviations below the median; the median lands here.
SHADOWS_CLIP = -2.8
TARGET_BACKGROUND = 0.25
MAD_TO_SIGMA = 1.4826


def midtones(x: Float32, balance: float) -> Float32:
    """Midtones transfer function: maps 0 to 0, 1 to 1 and balance to 0.5."""
    return ((balance - 1) * x / ((2 * balance - 1) * x - balance)).astype(np.float32)


def _autostretch_channel(x: Float32) -> Float32:
    median = float(np.median(x))
    sigma = MAD_TO_SIGMA * float(np.median(np.abs(x - median)))
    shadows = min(max(median + SHADOWS_CLIP * sigma, 0.0), 1.0)
    if median <= shadows or shadows >= 1.0:
        return np.clip(x, 0, 1)
    # The balance that sends the rescaled median to the target is the
    # transfer function itself, with the roles of the two swapped.
    m, b = (median - shadows) / (1 - shadows), TARGET_BACKGROUND
    balance = (b - 1) * m / ((2 * b - 1) * m - b)
    return midtones(np.clip((x - shadows) / (1 - shadows), 0, 1), balance)


def autostretch(data: Float32) -> Float32:
    """Stretch linear data in [0, 1] so the sky background becomes dark grey.

    Colour channels are stretched independently, which also neutralises the
    background colour.
    """
    if data.ndim == 2:
        return _autostretch_channel(data)
    return np.stack([_autostretch_channel(data[..., c]) for c in range(data.shape[-1])], axis=-1)
