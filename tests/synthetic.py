"""Synthetic star fields with known star positions."""

import numpy as np
import numpy.typing as npt

PSF_SIGMA = 1.1  # pixels; a FWHM of about 2.6
NOISE = 0.0005


def star_field(
    shape: tuple[int, int] = (400, 600), count: int = 80, seed: int = 1, noise_seed: int = 0
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A noisy sky with a gradient and Gaussian stars; returns image, x and y.

    Stars are kept apart so each can be told from its neighbours.
    """
    rng = np.random.default_rng(seed)
    height, width = shape
    x = np.empty(0)
    y = np.empty(0)
    while len(x) < count:
        cx, cy = rng.uniform(20, width - 20), rng.uniform(20, height - 20)
        if len(x) == 0 or np.hypot(x - cx, y - cy).min() > 12:
            x, y = np.append(x, cx), np.append(y, cy)
    flux = 10 ** rng.uniform(-1.0, 0.3, count)
    return render(shape, x, y, flux, noise_seed), x, y


def render(
    shape: tuple[int, int],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    flux: npt.NDArray[np.float64],
    noise_seed: int = 0,
) -> npt.NDArray[np.float32]:
    """Draw Gaussian stars of the given fluxes on a noisy sky with a gradient."""
    rows, columns = np.mgrid[: shape[0], : shape[1]]
    image = 0.01 + 0.00002 * columns
    for cx, cy, f in zip(x, y, flux, strict=True):
        distance2 = (rows - cy) ** 2 + (columns - cx) ** 2
        image += f * np.exp(-distance2 / (2 * PSF_SIGMA**2)) / (2 * np.pi * PSF_SIGMA**2)
    image += np.random.default_rng(noise_seed).normal(0, NOISE, shape)
    result: npt.NDArray[np.float32] = image.astype(np.float32)
    return result
