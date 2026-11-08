"""The colour of the stars in finished PNGs, as blobs, for judging a finish.

    python tools/star_colour.py finished.png [other.png ...]

Each star found in the PNG is summed over a square aperture less the
median of a ring around it, which is how the eye takes a star in: not its
peak pixel, and not the cores alone. Faint stars (peak under FAINT_PEAK
over a background near the sky's level) and bright ones are reported
apart, as each channel's share of the light and as red and blue over
green. A white star is 1.0 in every channel; the Seestar's own renders
of the Bubble come out at a red to green ratio of about 1.35 for faint
stars, and the finish before issue #69 at 1.7.
"""

import sys

import numpy as np
from PIL import Image

from quietsky.stars import detect_stars, windows

HALF = 4
OUTER = 8
# Faint stars peak below this, bright ones from it up to this many levels short of clipping.
FAINT_PEAK = 120
CLIPPED = 250
# A faint star counts if its peak stands this far over the sky and the ring around it is
# within this much of the frame's sky level, so that nebula under it does not colour it.
OVER_SKY = 15
NEAR_SKY = 6


def star_colours(png: np.ndarray) -> dict[str, tuple[np.ndarray, int]]:
    """For faint and bright stars, the median colour (each channel's share) and the count."""
    luminance = png.mean(axis=2).astype(np.float32)
    stars = detect_stars(luminance)
    rows = np.rint(stars.y).astype(np.intp)
    columns = np.rint(stars.x).astype(np.intp)
    height, width = luminance.shape
    inside = (
        (rows >= OUTER) & (rows < height - OUTER) & (columns >= OUTER) & (columns < width - OUTER)
    )
    rows, columns = rows[inside], columns[inside]
    offsets = np.arange(-OUTER, OUTER + 1)
    box = (np.abs(offsets[:, None]) <= HALF) & (np.abs(offsets[None, :]) <= HALF)
    flux = np.empty((len(rows), 3))
    ring = np.empty((len(rows), 3))
    for channel in range(3):
        light = windows(png[..., channel], rows, columns, OUTER)
        ring[:, channel] = np.median(light[:, ~box], axis=1)
        flux[:, channel] = (light[:, box] - ring[:, channel, None]).sum(axis=1)
    peak = windows(luminance, rows, columns, OUTER)[:, OUTER, OUTER]
    sky = float(np.median(luminance))
    faint = (
        (peak < FAINT_PEAK) & (peak > sky + OVER_SKY) & (np.abs(ring.mean(axis=1) - sky) < NEAR_SKY)
    )
    bright = (peak >= FAINT_PEAK) & (peak < CLIPPED)
    result = {}
    for label, chosen in (("faint", faint), ("bright", bright)):
        light = flux[chosen]
        colour = np.median(light / light.mean(axis=1, keepdims=True), axis=0)
        result[label] = (colour, int(chosen.sum()))
    return result


def main(paths: list[str]) -> None:
    for path in paths:
        png = np.asarray(Image.open(path).convert("RGB")).astype(np.float64)
        print(path)
        for label, (colour, count) in star_colours(png).items():
            print(
                f"  {label:6s} stars {count:5d}  R,G,B share {np.round(colour, 3)}"
                f"  R/G {colour[0] / colour[1]:.3f}  B/G {colour[2] / colour[1]:.3f}"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
