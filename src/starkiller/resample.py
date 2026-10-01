"""Redraw a frame in the geometry of another.

Each pixel of the output takes its value from the place the transformation
sends it to in the input frame, which generally falls between pixels.
Lanczos-3 interpolation keeps stars sharp, at the price of faint dark rings
around them; clamping holds those back, as PixInsight's does.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt

Float32 = npt.NDArray[np.float32]
Float64 = npt.NDArray[np.float64]

LANCZOS_ORDER = 3
# Offsets of the six input pixels used along each axis, from the pixel at or
# before the sampling position, and whether the kernel is negative there.
_TAPS = tuple(range(1 - LANCZOS_ORDER, LANCZOS_ORDER + 1))
_NEGATIVE = {-1, 2}
CLAMPING_THRESHOLD = 0.3
CHUNK_ROWS = 64


def _lanczos(distance: Float64) -> Float64:
    """The Lanczos-3 kernel."""
    scaled = np.pi * distance
    with np.errstate(divide="ignore", invalid="ignore"):
        value = LANCZOS_ORDER * np.sin(scaled) * np.sin(scaled / LANCZOS_ORDER) / scaled**2
    return np.where(distance == 0, 1.0, value)


def interpolate(
    image: npt.NDArray[np.floating], x: Float64, y: Float64, clamping: float | None
) -> Float64:
    """Values of an (H, W) or (H, W, C) image at fractional positions (x, y).

    Positions outside the image give zero. With clamping, the negative lobes
    of the kernel are scaled back wherever they contribute more than that
    fraction of what the positive lobes do, which is where ringing shows.
    """
    height, width = image.shape[:2]
    column = np.floor(x).astype(np.intp)
    row = np.floor(y).astype(np.intp)
    weights_x = [_lanczos(x - column - tap) for tap in _TAPS]
    weights_y = [_lanczos(y - row - tap) for tap in _TAPS]

    channels = image.shape[2:]
    lobes = np.zeros((2, len(x), *channels))  # positive and negative contributions
    lobe_weights = np.zeros((2, len(x)))
    for tap_y, weight_y in zip(_TAPS, weights_y, strict=True):
        rows = np.clip(row + tap_y, 0, height - 1)
        for tap_x, weight_x in zip(_TAPS, weights_x, strict=True):
            columns = np.clip(column + tap_x, 0, width - 1)
            negative = int((tap_x in _NEGATIVE) != (tap_y in _NEGATIVE))
            weight = np.abs(weight_x * weight_y)
            lobes[negative] += image[rows, columns] * weight.reshape(-1, *([1] * len(channels)))
            lobe_weights[negative] += weight

    positive, negative_sum = lobes
    weight_positive = lobe_weights[0].reshape(-1, *([1] * len(channels)))
    weight_negative = np.broadcast_to(
        lobe_weights[1].reshape(-1, *([1] * len(channels))), negative_sum.shape
    )
    if clamping is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = negative_sum / positive
        # No clamping up to the threshold, then progressively less of the
        # negative lobes, down to none once they would outweigh the positive.
        excess = np.clip((ratio - clamping) / (1 - clamping), 0, 1)
        keep = np.where(positive > 0, 1 - excess**2, 1.0)
        negative_sum = negative_sum * keep
        weight_negative = weight_negative * keep
    result: Float64 = (positive - negative_sum) / (weight_positive - weight_negative)
    outside = (x < 0) | (x > width - 1) | (y < 0) | (y > height - 1)
    result[outside] = 0
    return result


def resample(
    image: npt.NDArray[np.floating],
    matrix: Float64,
    clamping: float | None = CLAMPING_THRESHOLD,
    workers: int | None = None,
) -> Float32:
    """Redraw image so that output pixel (x, y) shows the input at matrix @ (x, y, 1).

    matrix is the projective transformation from the output (reference)
    geometry to the input frame, as solve_transformation gives it.
    """
    height, width = image.shape[:2]
    result = np.empty(image.shape, np.float32)
    columns = np.arange(width, dtype=np.float64)

    def rows_from(top: int) -> None:
        bottom = min(top + CHUNK_ROWS, height)
        x, y = np.meshgrid(columns, np.arange(top, bottom, dtype=np.float64))
        u, v, w = matrix @ np.stack([x.ravel(), y.ravel(), np.ones(x.size)])
        values = interpolate(image, u / w, v / w, clamping)
        result[top:bottom] = values.reshape(bottom - top, width, *image.shape[2:])

    with ThreadPoolExecutor(workers or os.process_cpu_count()) as pool:
        list(pool.map(rows_from, range(0, height, CHUNK_ROWS)))
    return result
