"""Render a master through the scope's own tone curve, taken from its stacked FITS and JPG.

    python tools/scope_curve.py SCOPE.fit SCOPE.jpg MASTER.fits -o OUT.png \\
        [--master-crop X,Y,W,H] [--compare DIR]

The scope stacks a night and renders it; its FITS and JPG share a pixel
grid, so the JPG value against the linear FITS value, per channel, is its
tone curve (the median JPG value in each bin of FITS values, held to
rise). The curve is applied to the master, scaled into the scope's units
first: the scope's FITS is registered onto the master, and the slope of
the scope's values over the master's on nebula pixels, both less their
sky, gives the scale. So the master, which may be many nights deep, comes
out rendered exactly as the scope renders one night.

With --compare, the scope's JPG is registered onto the master's frame
too, and both are written cropped to the master's area for a side-by-side
of the same pixels. The master is taken as it is: a denoised or sharpened
master renders through the same curve.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from quietsky.deconvolve import local_minimum
from quietsky.denoise import star_footprint
from quietsky.frame import Frame, load, normalized, save_fits
from quietsky.register import solve_transformation
from quietsky.resample import resample
from quietsky.stars import detect_stars

Float32 = npt.NDArray[np.float32]
Float64 = npt.NDArray[np.float64]

MAD_TO_SIGMA = 1.4826
# Bins of FITS value the curve is read at: quantiles, so the sky, where
# most pixels are, is resolved finely and the stars coarsely.
CURVE_BINS = 400
# Nebula pixels for the scale: this many noise deviations above the sky,
# and below this many, with stars left out.
SCALE_SIGNAL = (3.0, 60.0)
BORDER = 16
# The scope renders its star cores brighter than the nebula's scale would,
# and its saturated stars white. Between these two levels of the master
# the scale blends from the nebula's to the one that takes the master's
# white to the scope's.
STAR_BLEND = (0.1, 0.9)


def tone_curve(fits: Float64, jpg: Float64) -> list[tuple[Float64, Float64]]:
    """Per channel, the FITS values and the JPG values (0 to 1) they map to."""
    curves = []
    for channel in range(3):
        x, y = fits[..., channel].ravel(), jpg[..., channel].ravel()
        # Quantile bins resolve the sky, where most pixels are; linear bins
        # above the 99.9th percentile resolve the stars up to white.
        bright = np.quantile(x, 0.999)
        edges = np.unique(
            np.concatenate(
                [
                    np.quantile(x, np.linspace(0, 0.999, CURVE_BINS)),
                    np.linspace(bright, x.max(), 40),
                ]
            )
        )
        which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
        centres = np.array([x[which == b].mean() for b in range(len(edges) - 1)])
        medians = np.array([np.median(y[which == b]) for b in range(len(edges) - 1)])
        keep = ~np.isnan(centres)
        centres, medians = centres[keep], np.maximum.accumulate(medians[keep])
        curves.append((centres, medians))
    return curves


def apply_curve(values: Float64, curves: list[tuple[Float64, Float64]]) -> Float64:
    out = np.empty_like(values)
    for channel, (x, y) in enumerate(curves):
        out[..., channel] = np.interp(values[..., channel], x, y, left=y[0], right=y[-1])
    return out


def box(image: npt.NDArray[Any], region: tuple[int, int, int, int]) -> npt.NDArray[Any]:
    x, y, w, h = region
    return image[y : y + h, x : x + w]


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scope_fits", type=Path)
    parser.add_argument("scope_jpg", type=Path)
    parser.add_argument("master", type=Path, help="a linear master, background removed")
    parser.add_argument("-o", "--output", type=Path, required=True, help="PNG to write")
    parser.add_argument(
        "--master-crop",
        type=lambda t: tuple(int(v) for v in t.split(",")),
        help="X,Y,W,H of the master's frame the master is a crop of (the frame the scope "
        "is registered onto)",
    )
    parser.add_argument(
        "--frame", type=Path, help="the full master frame the crop was cut from, to register onto"
    )
    parser.add_argument("--compare", type=Path, help="directory for the same-crop comparison")
    args = parser.parse_args(argv)

    scope = normalized(load(args.scope_fits).data).astype(np.float64)
    jpg = np.asarray(Image.open(args.scope_jpg).convert("RGB")).astype(np.float64) / 255
    if jpg.shape != scope.shape:
        raise SystemExit(f"the JPG is {jpg.shape[:2]} and the FITS {scope.shape[:2]}")
    curves = tone_curve(scope, jpg)
    scope_sky = np.median(scope, axis=(0, 1))
    print("scope sky", scope_sky.round(5))
    for channel, (x, y) in enumerate(curves):
        at = [
            np.interp(v, x, y) * 255
            for v in (scope_sky[channel] * np.array([0.98, 1.0, 1.05, 1.2, 2.0, 5.0]))
        ]
        print(f"  curve {'RGB'[channel]}: sky*0.98,1,1.05,1.2,2,5 -> {np.round(at, 0)}")

    master = normalized(load(args.master).data).astype(np.float64)
    frame_path = args.frame or args.master
    frame = normalized(load(frame_path).data)
    reference = detect_stars(frame)
    crop = args.master_crop or (0, 0, master.shape[1], master.shape[0])
    transformation = solve_transformation(
        reference, detect_stars(normalized(load(args.scope_fits).data))
    )
    print(f"scope registered: {transformation.pairs} stars, rms {transformation.rms:.2f} px")
    aligned = box(resample(scope.astype(np.float32), transformation.matrix), crop).astype(
        np.float64
    )
    covered = (
        local_minimum(
            ((aligned.min(axis=2) > 0) & (master.min(axis=2) > 0)).astype(np.float32), BORDER
        )
        > 0
    )
    master_sky = np.median(master[covered], axis=0)
    aligned_sky = np.median(aligned[covered], axis=0)
    lum = (master - master_sky).mean(axis=2).astype(np.float32)
    noise = MAD_TO_SIGMA * float(np.median(np.abs(lum[covered] - np.median(lum[covered]))))
    foot = star_footprint(lum, detect_stars(lum))
    nebula = (
        covered & (foot == 0) & (lum > SCALE_SIGNAL[0] * noise) & (lum < SCALE_SIGNAL[1] * noise)
    )
    scales = []
    for channel in range(3):
        x = (master[..., channel] - master_sky[channel])[nebula]
        y = (aligned[..., channel] - aligned_sky[channel])[nebula]
        scales.append(float((x * y).sum() / (x * x).sum()))
    print(
        "master sky",
        master_sky.round(5),
        "scale to the scope's units R,G,B",
        np.round(scales, 3),
        f"on {int(nebula.sum())} nebula pixels",
    )

    nebula_scale = np.array(scales)
    white_scale = (1 - scope_sky) / (1 - master_sky)
    low, high = STAR_BLEND
    blend = np.clip((master.max(axis=2) - low) / (high - low), 0, 1)[..., None]
    scale = nebula_scale * (1 - blend) + white_scale * blend
    mapped = scope_sky + (master - master_sky) * scale
    rendered = apply_curve(mapped, curves)
    rendered[master.max(axis=2) <= 0] = 0
    Image.fromarray((np.clip(rendered, 0, 1) * 255 + 0.5).astype(np.uint8)).save(args.output)
    print(f"wrote {args.output}")

    if args.compare:
        args.compare.mkdir(parents=True, exist_ok=True)
        jpg_aligned = box(resample(jpg.astype(np.float32), transformation.matrix), crop)
        Image.fromarray((np.clip(jpg_aligned, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
            args.compare / "scope-jpg-registered.png"
        )
        save_fits(args.compare / "scope-registered.fits", Frame(aligned.astype(np.float32)))
        print(f"wrote {args.compare / 'scope-jpg-registered.png'}")


if __name__ == "__main__":
    main(sys.argv[1:])
