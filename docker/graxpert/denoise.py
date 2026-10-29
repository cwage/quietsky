#!/usr/local/bin/python
"""graxpert-denoise IN OUT [--strength S] [--ai-version V] [--batch-size N]

Run GraXpert's AI denoise on a linear FITS master and write the result to
OUT. GraXpert writes next to its input and our data is mounted read-only,
so the input is copied to a scratch directory first and the result moved
out. GraXpert rescales an image whose values pass 1 onto [0, 1] by its
minimum and maximum, so the result is fitted against the input and put
back on the input's scale. Fails if GraXpert wrote nothing.

The service has no network. GraXpert still tries to list the models on its
server when it starts and logs the failure with retries; that is expected,
and the model in the image is used.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("input", type=Path, help="a linear master, FITS")
    parser.add_argument("output", type=Path, help="FITS to write")
    parser.add_argument("--strength", type=float, default=0.5, help="0 to 1 (default 0.5)")
    parser.add_argument(
        "--ai-version", help="denoise model version (default: the one in the image)"
    )
    parser.add_argument("--batch-size", type=int, help="tiles denoised in parallel, 1 to 32")
    args = parser.parse_args()
    if not args.input.exists():
        sys.exit(f"{args.input}: no such file")
    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / ("input" + args.input.suffix.lower())
        shutil.copyfile(args.input, source)
        command = [
            "graxpert",
            str(source),
            "-cli",
            "-cmd",
            "denoising",
            "-gpu",
            "false",
            "-strength",
            str(args.strength),
            "-output",
            "result",
        ]
        if args.ai_version:
            command += ["-ai_version", args.ai_version]
        if args.batch_size:
            command += ["-batch_size", str(args.batch_size)]
        print("running:", " ".join(command), flush=True)
        status = subprocess.call(command)
        result = Path(scratch) / ("result" + (".xisf" if source.suffix == ".xisf" else ".fits"))
        if status != 0 or not result.exists():
            wrote = "a" if result.exists() else "no"
            sys.exit(f"GraXpert exited with {status} and wrote {wrote} result")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        restore_scale(args.input, result, args.output)
    print(f"wrote {args.output}")


def restore_scale(original: Path, result: Path, output: Path) -> None:
    """Write result to output on the original's flux scale.

    A line is fitted through the result against the original over a sample
    of pixels, with outliers (the pixels the denoise changed most) clipped,
    and undone. An unchanged scale fits a slope of 1 and is left alone.
    """
    with fits.open(original) as hdus:
        before = np.asarray(hdus[0].data, dtype=np.float64).ravel()
    with fits.open(result) as hdus:
        after = np.asarray(hdus[0].data, dtype=np.float64)
        header = hdus[0].header
    sample = np.random.default_rng(0).choice(before.size, min(before.size, 200_000), replace=False)
    x, y = before[sample], after.ravel()[sample]
    keep = np.ones(len(x), bool)
    slope, offset = 1.0, 0.0
    for _ in range(3):
        slope, offset = np.polyfit(x[keep], y[keep], 1)
        error = y - (slope * x + offset)
        keep = np.abs(error) < 3 * np.std(error[keep])
    if abs(slope - 1) > 1e-3 or abs(offset) > 1e-3 * abs(float(np.median(x))):
        print(f"GraXpert rescaled the image (slope {slope:.4f}, offset {offset:.3g}); undoing that")
        after = (after - offset) / slope
    fits.writeto(output, after.astype(np.float32), header, overwrite=True)


if __name__ == "__main__":
    main()
