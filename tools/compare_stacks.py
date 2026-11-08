"""Compare other stacks of a field with a master: depth, star width, contrast, linearity.

    python tools/compare_stacks.py MASTER.fits scope=Stacked_696.fit night1=light.fits \\
        --crop 224,160,764,1720 --region 230,900,160,160 -o out/compare

Each stack is registered onto the master, resampled and cropped to the
master's area, so the same pixels are compared. Its fluxes are scaled to
the master's by the same stars, so noise comes out in the master's units.
The master itself is the first row. What is reported:

- sky noise per channel, at the pixel scale, with the coarse background
  taken out, on empty pixels every stack covers;
- the stars' width per channel, and how many stars stand five noise
  deviations clear, which says how deep the stack goes;
- the region's contrast: its mean over the sky just outside it, in noise
  deviations of the luminance;
- linearity against the master: the slope of this stack's values over the
  master's on faint nebula, bright nebula and star pixels. A plain mean of
  the same subs gives one slope throughout; a stack that has been denoised
  or sharpened compresses the faint nebula and lifts the star cores.

Written for the Seestar's own stacked FITS (issue #69's comparison), whose
stacks turned out to be processed: half the noise at 1 px and the same at
4 px, stars narrower than the subs, star cores two to four times brighter.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from quietsky.deconvolve import local_minimum, measure_psf
from quietsky.denoise import star_footprint
from quietsky.finish import aperture_half, star_fluxes
from quietsky.frame import Frame, load, normalized, save_fits
from quietsky.register import RegistrationError, solve_transformation
from quietsky.resample import resample
from quietsky.stars import Stars, detect_stars
from quietsky.wavelets import b3_layers

Float32 = npt.NDArray[np.float32]
Float64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]

MAD_TO_SIGMA = 1.4826
# Black borders are kept this far away from any measured pixel.
BORDER = 16
# Stars used to scale a stack to the master: the brightest unsaturated ones.
SCALE_STARS = 300
# The ring of sky round the region, in pixels.
REGION_RING = 60
# Signal classes for the linearity slopes, in noise deviations of the master.
FAINT = (3.0, 15.0)
BRIGHT = (15.0, 60.0)
STAR = (60.0, 2000.0)


def box(image: npt.NDArray[Any], region: tuple[int, int, int, int]) -> npt.NDArray[Any]:
    x, y, w, h = region
    return image[y : y + h, x : x + w]


def flat(channel: npt.NDArray[np.floating]) -> Float32:
    """The channel less its coarse (64 px) background."""
    _, coarse = b3_layers(channel.astype(np.float32), 6)
    result: Float32 = channel.astype(np.float32) - coarse
    return result


def registered(data: Float32, reference: Stars) -> tuple[Float32, str]:
    """The stack resampled onto the reference's frame, and a note on the fit."""
    transformation = solve_transformation(reference, detect_stars(data))
    return resample(data, transformation.matrix), (
        f"{transformation.pairs} stars matched, rms {transformation.rms:.2f} px"
    )


def slope(x: npt.NDArray[np.floating[Any]], y: npt.NDArray[np.floating[Any]], mask: Bool) -> float:
    """The least-squares slope of y over x through the origin, on the masked pixels."""
    xs, ys = x[mask], y[mask]
    if len(xs) < 10:
        return float("nan")
    return float((xs * ys).sum() / (xs * xs).sum())


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("master", type=Path, help="the master light the others are compared with")
    parser.add_argument("stacks", nargs="+", help="name=FITS of each other stack")
    parser.add_argument("--crop", type=_four_ints, help="X,Y,W,H of the master to compare over")
    parser.add_argument(
        "--region", type=_four_ints, help="X,Y,W,H in the crop whose contrast is reported"
    )
    parser.add_argument("-o", "--output", type=Path, help="directory for the registered crops")
    args = parser.parse_args(argv)

    master_frame = load(args.master)
    master_full = normalized(master_frame.data)
    reference_stars = detect_stars(master_full)
    crop = args.crop or (0, 0, master_full.shape[1], master_full.shape[0])
    master = box(master_full, crop).astype(np.float64)
    sky = np.median(master, axis=(0, 1))
    master -= sky
    lum = master.mean(axis=2).astype(np.float32)
    stars = detect_stars(lum)
    sigma = measure_psf(lum, stars)
    half = aperture_half(sigma)
    master_flux, master_ok = star_fluxes(master + sky, stars, half)
    noise = MAD_TO_SIGMA * float(np.median(np.abs(flat(lum))))
    footprint = star_footprint(lum, stars)
    print(f"master {args.master.name}: {len(stars)} stars, sigma {sigma:.2f} px")

    rows = []
    versions: list[tuple[str, Float64, str | None, Frame]] = [
        ("master", (master + sky).astype(np.float64), None, master_frame)
    ]
    for item in args.stacks:
        name, path = item.split("=", 1)
        frame = load(Path(path))
        loaded = normalized(frame.data)
        note: str | None
        try:
            aligned, note = registered(loaded, reference_stars)
        except RegistrationError as error:
            print(f"{name}: not registered, {error}")
            continue
        cropped = box(aligned, crop).astype(np.float64)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            save_fits(
                args.output / f"{name}.fits",
                Frame(np.ascontiguousarray(cropped.astype(np.float32)), frame.header),
            )
        versions.append((name, cropped, note, frame))

    for name, raw, note, frame in versions:
        subs = frame.header.get("NCOMBINE", frame.header.get("STACKCNT", "?"))
        covered = local_minimum((raw.min(axis=2) > 0).astype(np.float32), BORDER) > 0
        data = (raw - np.median(raw[covered], axis=0)).astype(np.float64)
        flux, ok = star_fluxes(raw, stars, half)
        use = ok & master_ok & (flux.sum(axis=1) > 0) & (master_flux.sum(axis=1) > 0)
        brightest = np.argsort(-master_flux[use].sum(axis=1))[:SCALE_STARS]
        scale = float(
            np.median(master_flux[use][brightest].sum(axis=1) / flux[use][brightest].sum(axis=1))
        )
        data *= scale
        this_lum = data.mean(axis=2).astype(np.float32)
        own_stars = detect_stars(this_lum)
        level = flat(this_lum)
        empty = (
            covered
            & (star_footprint(this_lum, own_stars) == 0)
            & (np.abs(level) < 3 * MAD_TO_SIGMA * np.median(np.abs(level[covered])))
        )
        noises = [
            MAD_TO_SIGMA * float(np.median(np.abs(flat(data[..., c])[empty]))) for c in range(3)
        ]
        widths = []
        for c in range(3):
            try:
                widths.append(measure_psf(data[..., c].astype(np.float32), own_stars))
            except ValueError:
                widths.append(float("nan"))
        contrast = ""
        if args.region:
            x, y, w, h = args.region
            around = box(
                this_lum,
                (x - REGION_RING, y - REGION_RING, w + 2 * REGION_RING, h + 2 * REGION_RING),
            ).copy()
            around[REGION_RING : REGION_RING + h, REGION_RING : REGION_RING + w] = np.nan
            lum_noise = MAD_TO_SIGMA * float(np.median(np.abs(level[empty])))
            inside = float(np.mean(box(this_lum, args.region)))
            contrast = f"{(inside - float(np.nanmedian(around))) / lum_noise:.1f}"
            if not box(covered, args.region).all():
                contrast += " (not covered)"
        signal = lum / noise
        classes = {
            "faint nebula": covered & (footprint == 0) & (signal > FAINT[0]) & (signal < FAINT[1]),
            "bright nebula": covered
            & (footprint == 0)
            & (signal > BRIGHT[0])
            & (signal < BRIGHT[1]),
            "star pixels": covered & (footprint > 0.3) & (signal > STAR[0]) & (signal < STAR[1]),
        }
        slopes = {
            label: [slope(master[..., c], data[..., c], mask) for c in range(3)]
            for label, mask in classes.items()
        }
        rows.append(
            f"| {name} | {subs} | {scale:.3f} | "
            + ", ".join(f"{n * 1e6:.1f}" for n in noises)
            + " | "
            + ", ".join(f"{w:.2f}" for w in widths)
            + f" | {len(own_stars)} | {contrast} | {100 * covered.mean():.0f}% |"
        )
        if note:
            print(f"{name}: {note}")
        for label, values in slopes.items():
            print(f"  {name} over master, {label}: " + ", ".join(f"{v:.2f}" for v in values))

    print()
    print(
        "| stack | subs | scale | noise R,G,B (1e-6, master units) | star sigma R,G,B (px) "
        "| stars at 5 sigma | region contrast | covered |"
    )
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))


def _four_ints(text: str) -> tuple[int, int, int, int]:
    parts = tuple(int(value) for value in text.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected X,Y,W,H")
    return parts[0], parts[1], parts[2], parts[3]


if __name__ == "__main__":
    main(sys.argv[1:])
