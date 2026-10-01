"""Command line interface."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from starkiller.calibrate import calibrate, optimize_dark, unit_flat
from starkiller.frame import Frame, load, normalized, save_fits
from starkiller.integrate import flux_gains, integrate
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
    stack = np.stack([normalized(load(path).data) for path in args.files])
    gains = flux_gains(stack) if args.flat else None
    result = integrate(stack, args.sigma_low, args.sigma_high, gains)
    total = stack[0].size * len(args.files)
    low, high = result.frame_rejected_low.sum(), result.frame_rejected_high.sum()
    print(f"integrated {len(args.files)} frames")
    print(f"rejected low {low} ({low / total:.3%}), high {high} ({high / total:.3%})")
    save_fits(args.output, Frame(result.image, {"NCOMBINE": len(args.files)}))
    print(f"wrote {args.output}")


def _calibrate(args: argparse.Namespace) -> None:
    bias = normalized(load(args.bias).data)
    dark = normalized(load(args.dark).data)
    flat = unit_flat(normalized(load(args.flat).data)) if args.flat else None
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        frame = load(path)
        data = normalized(frame.data)
        scale = args.dark_scale
        if scale is None:
            scale = optimize_dark(data, bias, dark, mosaic="BAYERPAT" in frame.header)
        calibrated = calibrate(data, bias, dark, scale, flat)
        output = args.output / f"{path.stem}_c.fits"
        save_fits(output, Frame(calibrated, frame.header | {"DARKSCAL": round(scale, 4)}))
        print(f"{output}: dark scale {scale:.3f}")


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
        "stack", help="average bias, dark or flat frames with outlier rejection"
    )
    stack.add_argument("files", nargs="+", type=Path)
    stack.add_argument("-o", "--output", type=Path, required=True, help="FITS file to write")
    stack.add_argument("--sigma-low", type=float, default=4.0)
    stack.add_argument("--sigma-high", type=float, default=3.0)
    stack.add_argument(
        "--flat",
        action="store_true",
        help="match the brightness of the frames before combining them, as flats need",
    )
    stack.set_defaults(run=_stack)

    calibration = commands.add_parser(
        "calibrate",
        help="subtract a master bias and a scaled master dark from frames, and divide by a flat",
    )
    calibration.add_argument("files", nargs="+", type=Path)
    calibration.add_argument("--bias", type=Path, required=True, help="master bias")
    calibration.add_argument("--dark", type=Path, required=True, help="master dark")
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

    preview = commands.add_parser("preview", help="write an auto-stretched PNG of a frame")
    preview.add_argument("file", type=Path)
    preview.add_argument("-o", "--output", type=Path, required=True, help="PNG file to write")
    preview.set_defaults(run=_preview)

    args = parser.parse_args(argv)
    args.run(args)
