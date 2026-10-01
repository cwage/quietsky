"""Command line interface."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from starkiller.background import extract_background
from starkiller.calibrate import calibrate, optimize_dark, unit_flat
from starkiller.debayer import BAYER_PATTERNS, debayer_vng
from starkiller.frame import FITS_SUFFIXES, RAW_SUFFIXES, Frame, load, normalized, save_fits
from starkiller.integrate import (
    flux_normalization,
    frame_estimates,
    integrate,
    integrate_lights,
)
from starkiller.pipeline import Session, frame_noise, noise_keywords, preprocess
from starkiller.register import RegistrationError, solve_transformation
from starkiller.resample import resample
from starkiller.stars import detect_stars
from starkiller.stretch import autostretch


def _info(args: argparse.Namespace) -> None:
    for path in args.files:
        frame = load(path)
        data = frame.data
        height, width = data.shape[:2]
        print(f"{path}: {width}x{height} {data.dtype}", end="")
        print(f" min {data.min():g} median {np.median(data):g} max {data.max():g}")
        for key, value in frame.header.items():
            print(f"  {key:8} {value}")


def _stack(args: argparse.Namespace) -> None:
    frames = [load(path) for path in args.files]
    stack = np.stack([normalized(frame.data) for frame in frames])
    if args.light:
        channels = stack[..., None] if stack.ndim == 3 else stack
        results = integrate_lights(
            channels, frame_noise(frames, channels), args.sigma_low, args.sigma_high
        )
        image = np.stack([result.image for result in results], axis=-1)
        image = image[..., 0] if stack.ndim == 3 else image
    else:
        normalization = flux_normalization(frame_estimates(stack)[0]) if args.flat else None
        results = [integrate(stack, args.sigma_low, args.sigma_high, normalization)]
        image = results[0].image
    total = stack.size
    low = sum(result.frame_rejected_low.sum() for result in results)
    high = sum(result.frame_rejected_high.sum() for result in results)
    print(f"integrated {len(args.files)} frames")
    print(f"rejected low {low} ({low / total:.3%}), high {high} ({high / total:.3%})")
    save_fits(args.output, Frame(image, {"NCOMBINE": len(args.files)}))
    print(f"wrote {args.output}")


def _calibrate(args: argparse.Namespace) -> None:
    bias = normalized(load(args.bias).data) if args.bias else None
    dark = normalized(load(args.dark).data) if args.dark else None
    flat = unit_flat(normalized(load(args.flat).data)) if args.flat else None
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        frame = load(path)
        data = normalized(frame.data)
        scale = args.dark_scale
        if scale is None:
            # Without a bias the dark cannot be scaled and is subtracted whole.
            scale = 1.0
            if bias is not None and dark is not None:
                scale = optimize_dark(data, bias, dark, mosaic="BAYERPAT" in frame.header)
        calibrated = calibrate(data, bias, dark, scale, flat)
        output = args.output / f"{path.stem}_c.fits"
        save_fits(output, Frame(calibrated, frame.header | {"DARKSCAL": round(scale, 4)}))
        print(f"{output}: dark scale {scale:.3f}")


def _debayer(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        frame = load(path)
        pattern = args.pattern or frame.header.get("BAYERPAT")
        if not isinstance(pattern, str):
            raise SystemExit(f"{path}: no BAYERPAT in the header; give --pattern")
        header = {key: value for key, value in frame.header.items() if key != "BAYERPAT"}
        rgb = debayer_vng(normalized(frame.data), pattern)
        header |= noise_keywords(rgb)
        output = args.output / f"{path.stem}_d.fits"
        save_fits(output, Frame(rgb, header))
        print(f"{output}: {pattern}")


def _register(args: argparse.Namespace) -> None:
    reference = detect_stars(normalized(load(args.reference).data))
    print(f"{args.reference}: reference, {len(reference)} stars")
    args.output.mkdir(parents=True, exist_ok=True)
    failed = 0
    for path in args.files:
        frame = load(path)
        data = normalized(frame.data)
        try:
            transformation = solve_transformation(reference, detect_stars(data))
        except RegistrationError as error:
            print(f"{path}: skipped, {error}")
            failed += 1
            continue
        output = args.output / f"{path.stem}_r.fits"
        save_fits(output, Frame(resample(data, transformation.matrix), frame.header))
        print(f"{output}: {transformation.pairs} stars matched, rms {transformation.rms:.2f} px")
    if failed:
        raise SystemExit(f"{failed} frame(s) could not be registered")


def _background(args: argparse.Namespace) -> None:
    frame = load(args.file)
    corrected, model, samples = extract_background(
        normalized(frame.data), args.samples_per_row, smoothing=args.smoothing
    )
    save_fits(args.output, Frame(corrected, frame.header))
    print(f"{len(samples.x)} samples, {int(samples.kept.sum())} kept")
    print(f"background varied by {float(np.ptp(model)):.3g} across the frame")
    print(f"wrote {args.output}")
    if args.model:
        save_fits(args.model, Frame(model))
        print(f"wrote {args.model}")


def _frames_in(directory: Path | None) -> list[Path]:
    if directory is None:
        return []
    suffixes = RAW_SUFFIXES | FITS_SUFFIXES | {".xisf"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in suffixes)


def _preprocess(args: argparse.Namespace) -> None:
    session = Session(
        _frames_in(args.bias), _frames_in(args.dark), _frames_in(args.flat), _frames_in(args.light)
    )
    if not session.light:
        raise SystemExit(f"no light frames in {args.light}")
    preprocess(session, args.output)


def _preview(args: argparse.Namespace) -> None:
    stretched = autostretch(normalized(load(args.file).data))
    Image.fromarray((stretched * 255 + 0.5).astype(np.uint8)).save(args.output)
    print(f"wrote {args.output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="starkiller", description=__doc__)
    commands = parser.add_subparsers(required=True)

    info = commands.add_parser("info", help="show size, statistics and header of frames")
    info.add_argument("files", nargs="+", type=Path)
    info.set_defaults(run=_info)

    stack = commands.add_parser(
        "stack", help="average bias, dark, flat or registered light frames with outlier rejection"
    )
    stack.add_argument("files", nargs="+", type=Path)
    stack.add_argument("-o", "--output", type=Path, required=True, help="FITS file to write")
    stack.add_argument("--sigma-low", type=float, default=4.0)
    stack.add_argument("--sigma-high", type=float, default=3.0)
    kind = stack.add_mutually_exclusive_group()
    kind.add_argument(
        "--flat",
        action="store_true",
        help="match the brightness of the frames before combining them, as flats need",
    )
    kind.add_argument(
        "--light",
        action="store_true",
        help="match level and scale, weight frames by noise and ignore black or saturated "
        "pixels, as registered lights need",
    )
    stack.set_defaults(run=_stack)

    calibration = commands.add_parser(
        "calibrate",
        help="subtract a master bias and a scaled master dark from frames, and divide by a "
        "flat; each master is optional",
    )
    calibration.add_argument("files", nargs="+", type=Path)
    calibration.add_argument("--bias", type=Path, help="master bias")
    calibration.add_argument("--dark", type=Path, help="master dark")
    calibration.add_argument("--flat", type=Path, help="master flat to divide by (for lights)")
    calibration.add_argument(
        "--dark-scale",
        type=float,
        help="factor to scale the dark by (default: the factor that minimises noise)",
    )
    calibration.add_argument(
        "-o", "--output", type=Path, required=True, help="directory for the calibrated frames"
    )
    calibration.set_defaults(run=_calibrate)

    debayer = commands.add_parser("debayer", help="turn colour mosaics into RGB images (VNG)")
    debayer.add_argument("files", nargs="+", type=Path)
    debayer.add_argument(
        "--pattern", choices=BAYER_PATTERNS, help="Bayer pattern (default: BAYERPAT in the header)"
    )
    debayer.add_argument(
        "-o", "--output", type=Path, required=True, help="directory for the RGB frames"
    )
    debayer.set_defaults(run=_debayer)

    register = commands.add_parser(
        "register", help="align frames to a reference frame by their stars"
    )
    register.add_argument("files", nargs="+", type=Path)
    register.add_argument("--reference", type=Path, required=True, help="frame to align to")
    register.add_argument(
        "-o", "--output", type=Path, required=True, help="directory for the aligned frames"
    )
    register.set_defaults(run=_register)

    whole = commands.add_parser(
        "preprocess",
        help="run the whole pipeline on a session: masters, calibration, debayer, "
        "registration and integration",
    )
    for name in ("bias", "dark", "flat"):
        whole.add_argument(f"--{name}", type=Path, help=f"directory of {name} frames, if any")
    whole.add_argument("--light", type=Path, required=True, help="directory of light frames")
    whole.add_argument(
        "-o", "--output", type=Path, required=True, help="directory for masters and aligned lights"
    )
    whole.set_defaults(run=_preprocess)

    background = commands.add_parser(
        "background", help="subtract the sky background gradient from a linear image"
    )
    background.add_argument("file", type=Path)
    background.add_argument("-o", "--output", type=Path, required=True, help="FITS file to write")
    background.add_argument("--model", type=Path, help="also write the background model here")
    background.add_argument(
        "--samples-per-row", type=int, default=16, help="background samples across the frame"
    )
    background.add_argument(
        "--smoothing", type=float, default=0.25, help="how loosely the model follows the samples"
    )
    background.set_defaults(run=_background)

    preview = commands.add_parser("preview", help="write an auto-stretched PNG of a frame")
    preview.add_argument("file", type=Path)
    preview.add_argument("-o", "--output", type=Path, required=True, help="PNG file to write")
    preview.set_defaults(run=_preview)

    args = parser.parse_args(argv)
    args.run(args)
