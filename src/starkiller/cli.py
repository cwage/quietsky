"""Command line interface."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from starkiller.frame import Frame, load, normalized, save_fits
from starkiller.integrate import integrate
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
    result = integrate(stack, args.sigma_low, args.sigma_high)
    total = stack[0].size * len(args.files)
    low, high = result.frame_rejected_low.sum(), result.frame_rejected_high.sum()
    print(f"integrated {len(args.files)} frames")
    print(f"rejected low {low} ({low / total:.3%}), high {high} ({high / total:.3%})")
    save_fits(args.output, Frame(result.image, {"NCOMBINE": len(args.files)}))
    print(f"wrote {args.output}")


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
        "stack", help="average frames with outlier rejection (bias and dark frames)"
    )
    stack.add_argument("files", nargs="+", type=Path)
    stack.add_argument("-o", "--output", type=Path, required=True, help="FITS file to write")
    stack.add_argument("--sigma-low", type=float, default=4.0)
    stack.add_argument("--sigma-high", type=float, default=3.0)
    stack.set_defaults(run=_stack)

    preview = commands.add_parser("preview", help="write an auto-stretched PNG of a frame")
    preview.add_argument("file", type=Path)
    preview.add_argument("-o", "--output", type=Path, required=True, help="PNG file to write")
    preview.set_defaults(run=_preview)

    args = parser.parse_args(argv)
    args.run(args)
