"""Find stars in a frame and measure where they are.

Registration needs star positions good to a fraction of a pixel. Stars are
picked out as small bright structures: the image is band-pass filtered with
wavelets, which removes the sky background and nebulosity on one side and
pixel noise on the other, and the peaks that stand clear of the remaining
noise are taken as stars.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from starkiller.estimators import mad
from starkiller.wavelets import b3_layers

Float32 = npt.NDArray[np.float32]
Float64 = npt.NDArray[np.float64]

MAD_TO_SIGMA = 1.4826
# Half-width of the window a star's position and flux are measured in.
WINDOW = 3
# Stars closer to the edge than this are not measured.
MARGIN = 8
# A real star spreads over several pixels. If the brightest pixel holds more
# than this share of the light in the 3x3 around it, it is a hot pixel.
MAX_PEAK_SHARE = 0.8
# Detections closer together than this are taken to be the same star.
MIN_SEPARATION = 4.0
# Registration needs a few hundred stars at most; keep the brightest.
MAX_STARS = 4000


@dataclass(frozen=True)
class Stars:
    """Star positions in pixels (x along columns, y along rows) and fluxes, brightest first."""

    x: Float64
    y: Float64
    flux: Float64

    def __len__(self) -> int:
        return len(self.x)


def _windows(
    image: Float32, rows: npt.NDArray[np.intp], columns: npt.NDArray[np.intp], half: int
) -> Float32:
    """The square of pixels around each (row, column), as an (N, size, size) array."""
    offsets = np.arange(-half, half + 1)
    windows: Float32 = image[
        rows[:, None, None] + offsets[:, None], columns[:, None, None] + offsets[None, :]
    ]
    return windows


def detect_stars(image: npt.NDArray[np.floating], threshold: float = 5.0) -> Stars:
    """Detect stars in a mono (H, W) or colour (H, W, C) image.

    threshold is how many noise deviations a star's peak must rise above
    the background of the filtered image.
    """
    luminance = image if image.ndim == 2 else image.mean(axis=-1)
    layers, _ = b3_layers(luminance, 3)
    # Structures of about one to four pixels: sharper than nebulosity,
    # smoother than noise.
    detail = layers[0] + layers[1] + layers[2]
    filtered = layers[1] + layers[2]
    noise = MAD_TO_SIGMA * mad(filtered, float(np.median(filtered)))

    # Peaks: above the threshold and not lower than any of their neighbours.
    inner = filtered[MARGIN:-MARGIN, MARGIN:-MARGIN]
    peak = inner > threshold * noise
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy, dx) != (0, 0):
                neighbour = filtered[
                    MARGIN + dy : filtered.shape[0] - MARGIN + dy,
                    MARGIN + dx : filtered.shape[1] - MARGIN + dx,
                ]
                peak &= inner >= neighbour
    rows, columns = np.nonzero(peak)
    rows, columns = rows + MARGIN, columns + MARGIN

    # Hot pixels are single-pixel spikes before filtering.
    core = np.clip(_windows(detail, rows, columns, 1), 0, None)
    share = core[:, 1, 1] / core.sum(axis=(1, 2))
    rows, columns = rows[share <= MAX_PEAK_SHARE], columns[share <= MAX_PEAK_SHARE]

    # Position: centre of light of the filtered image around the peak.
    light = np.clip(_windows(filtered, rows, columns, WINDOW), 0, None).astype(np.float64)
    flux = light.sum(axis=(1, 2))
    offsets = np.arange(-WINDOW, WINDOW + 1)
    x = columns + (light.sum(axis=1) * offsets).sum(axis=1) / flux
    y = rows + (light.sum(axis=2) * offsets).sum(axis=1) / flux

    order = np.argsort(-flux)[:MAX_STARS]
    x, y, flux = x[order], y[order], flux[order]

    # A saturated star has a flat top with several peaks; keep the brightest.
    separation = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    brighter_nearby = np.triu(separation < MIN_SEPARATION, k=1).any(axis=0)
    keep = ~brighter_nearby
    return Stars(x[keep], y[keep], flux[keep])
