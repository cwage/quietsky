"""The à trous ("with holes") wavelet transform.

It splits an image into layers of detail at doubling scales, plus a smooth
residual, all the same size as the image. The layers and the residual add
back up to the image exactly.
"""

import numpy as np
import numpy.typing as npt

Float32 = npt.NDArray[np.float32]

# Standard deviation of each B3-spline layer when the image is white noise of
# unit standard deviation.
B3_NOISE_PER_LAYER = (0.8907, 0.2007, 0.0856, 0.0413, 0.0205, 0.0103)


def _b3_smooth(image: Float32, step: int) -> Float32:
    """Blur with the B3 spline kernel [1, 4, 6, 4, 1] / 16, taps step apart."""
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (2 * step, 2 * step)
        padded = np.pad(image, padding, mode="reflect")
        size = image.shape[axis]

        def tap(offset: int, axis: int = axis, size: int = size) -> tuple[slice, ...]:
            window = slice(offset, offset + size)
            return (window, slice(None)) if axis == 0 else (slice(None), window)

        image = (
            padded[tap(0)]
            + padded[tap(4 * step)]
            + 4 * (padded[tap(step)] + padded[tap(3 * step)])
            + 6 * padded[tap(2 * step)]
        ) / np.float32(16)
    return image


def b3_layers(image: npt.NDArray[np.floating], count: int) -> tuple[list[Float32], Float32]:
    """Split a 2-D image into count detail layers, finest first, and a residual."""
    smooth = image.astype(np.float32)
    layers = []
    for index in range(count):
        smoother = _b3_smooth(smooth, 2**index)
        layers.append(smooth - smoother)
        smooth = smoother
    return layers, smooth
