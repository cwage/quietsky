"""Model the sky background of a linear image and take it out.

Light pollution and leftover vignetting lay a smooth gradient over a stacked
image. It is measured in small boxes spread over the frame, the boxes that
sit on stars or on the target are discarded, and a smooth surface through
the rest is subtracted.

This follows PixInsight's DynamicBackgroundExtraction with automatically
placed samples. Given PixInsight's own samples, the surface here reproduces
its background model to about 0.2% of the model's range. The choice of
which samples to keep is ours.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Float32 = npt.NDArray[np.float32]
Float64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]

# The surface is evaluated on a grid this many pixels apart and interpolated
# in between; it varies far more slowly than that.
MODEL_STEP = 16
# Our regularisation of the spline for PixInsight's smoothing parameter,
# found by fitting its model.
SMOOTHING_SCALE = 0.05
# Samples are compared with a quadratic fit through all of them. One this
# many robust deviations above it sits on an object; the limit below is
# looser because nothing real is darker than the sky.
REJECT_ABOVE = 3.0
REJECT_BELOW = 6.0
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class Samples:
    """Background samples: positions in pixels, a value per channel, and which were kept."""

    x: Float64
    y: Float64
    values: Float64  # (N, C)
    kept: Bool


def _quadratic_terms(x: Float64, y: Float64) -> Float64:
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)


def _objects(x: Float64, y: Float64, values: Float64) -> Bool:
    """Samples that stand out from a smooth background in any channel."""
    terms = _quadratic_terms(x, y)
    rejected = np.zeros(len(x), np.bool_)
    for _ in range(10):
        previous = rejected.copy()
        for channel in range(values.shape[1]):
            keep = ~rejected
            coefficients = np.linalg.lstsq(terms[keep], values[keep, channel], rcond=None)[0]
            residual = values[:, channel] - terms @ coefficients
            sigma = MAD_TO_SIGMA * float(np.median(np.abs(residual[keep])))
            rejected |= (residual > REJECT_ABOVE * sigma) | (residual < -REJECT_BELOW * sigma)
        if (rejected == previous).all():
            break
    return rejected


def sample_background(
    image: npt.NDArray[np.floating], samples_per_row: int = 16, radius: int = 8
) -> Samples:
    """Measure the background of an (H, W, C) image on a grid of small boxes.

    Each sample is the median of a box of side 2 * radius + 1, which ignores
    the stars inside it. Samples that still stand above their neighbours,
    because they sit on a nebula, a galaxy or a cluster, are marked as not
    kept.
    """
    height, width = image.shape[:2]
    columns = np.linspace(radius, width - 1 - radius, samples_per_row + 1).round().astype(int)
    rows_count = max(int(round(samples_per_row * height / width)), 2) + 1
    rows = np.linspace(radius, height - 1 - radius, rows_count).round().astype(int)
    x, y = (grid.ravel() for grid in np.meshgrid(columns, rows))
    values = np.array(
        [
            np.median(
                image[row - radius : row + radius + 1, column - radius : column + radius + 1],
                axis=(0, 1),
            )
            for column, row in zip(x, y, strict=True)
        ],
        dtype=np.float64,
    ).reshape(len(x), -1)
    fx, fy = x / width, y / height
    return Samples(x.astype(np.float64), y.astype(np.float64), values, ~_objects(fx, fy, values))


def _kernel(x: Float64, y: Float64, to_x: Float64, to_y: Float64) -> Float64:
    """Thin-plate spline kernel r^2 ln r^2 between two sets of points."""
    distance2 = (x[:, None] - to_x[None, :]) ** 2 + (y[:, None] - to_y[None, :]) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(distance2 > 0, distance2 * np.log(distance2), 0.0)


def fit_surface(
    x: Float64, y: Float64, values: Float64, shape: tuple[int, int], smoothing: float = 0.25
) -> Float32:
    """A smooth surface through sample values, as an image of the given shape.

    x and y are sample positions in pixels and values one channel's sample
    values. The surface is a thin-plate spline; with smoothing above zero it
    passes near the samples rather than through them, so that noise in the
    samples does not ripple the model. 0.25 is PixInsight's default.
    """
    height, width = shape
    sx, sy = x / width, y / height
    count = len(x)
    plane = np.stack([np.ones(count), sx, sy], axis=1)
    system = np.zeros((count + 3, count + 3))
    system[:count, :count] = _kernel(sx, sy, sx, sy) + SMOOTHING_SCALE * smoothing * np.eye(count)
    system[:count, count:] = plane
    system[count:, :count] = plane.T
    solution = np.linalg.solve(system, np.concatenate([values, np.zeros(3)]))

    # Evaluate coarsely, then interpolate linearly up to the full image.
    coarse_x = np.linspace(0, width - 1, max(width // MODEL_STEP, 2))
    coarse_y = np.linspace(0, height - 1, max(height // MODEL_STEP, 2))
    grid_x, grid_y = (grid.ravel() for grid in np.meshgrid(coarse_x / width, coarse_y / height))
    coarse = (
        _kernel(grid_x, grid_y, sx, sy) @ solution[:count]
        + solution[count]
        + solution[count + 1] * grid_x
        + solution[count + 2] * grid_y
    ).reshape(len(coarse_y), len(coarse_x))

    def weights(size: int, points: Float64) -> tuple[npt.NDArray[np.intp], Float64]:
        position = np.arange(size) / (size - 1) * (len(points) - 1)
        index = np.minimum(position.astype(np.intp), len(points) - 2)
        return index, position - index

    column, across = weights(width, coarse_x)
    row, down = weights(height, coarse_y)
    wide = coarse[:, column] * (1 - across) + coarse[:, column + 1] * across
    full = wide[row] * (1 - down[:, None]) + wide[row + 1] * down[:, None]
    result: Float32 = full.astype(np.float32)
    return result


def extract_background(
    image: npt.NDArray[np.floating],
    samples_per_row: int = 16,
    radius: int = 8,
    smoothing: float = 0.25,
) -> tuple[Float32, Float32, Samples]:
    """Subtract the sky background from a linear (H, W) or (H, W, C) image.

    Returns the corrected image, the background model and the samples the
    model was built from. The model's median is added back, so the image
    keeps its overall level and only the unevenness goes.
    """
    channels = image[..., None] if image.ndim == 2 else image
    samples = sample_background(channels, samples_per_row, radius)
    kept = samples.kept
    model = np.stack(
        [
            fit_surface(
                samples.x[kept],
                samples.y[kept],
                samples.values[kept, channel],
                channels.shape[:2],
                smoothing,
            )
            for channel in range(channels.shape[-1])
        ],
        axis=-1,
    )
    corrected: Float32 = (channels - model + np.median(model, axis=(0, 1))).astype(np.float32)
    if image.ndim == 2:
        return corrected[..., 0], model[..., 0], samples
    return corrected, model, samples
