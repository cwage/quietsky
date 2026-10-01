"""Preprocess the full M13 session as PixInsight loaded it and compare master lights.

Runs the whole pipeline on the raw frames in $QUIETSKY_DATA, with the raw
values truncated the way PixInsight 1.8.5 read them, and reports how the
master light differs from PixInsight's. Takes roughly twenty minutes.

    docker compose run --rm quietsky python tools/compare_m13_end_to_end.py out/m13_legacy
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

from quietsky.frame import Frame, load, load_raw
from quietsky.pipeline import Session, preprocess
from quietsky.xisf import read_xisf

SESSION = Path(os.environ["QUIETSKY_DATA"]) / "2018-09-12" / "m13"
# Every light covers this part of the frame; nearer the edge some are black.
INNER = (slice(150, -150), slice(150, -150))


def as_pixinsight_loaded(path: Path) -> Frame:
    """Load a raw frame with the two low bits dropped, as dcraw did for PixInsight."""
    frame = load_raw(path)
    return Frame(((frame.data >> 2) / 65535).astype(np.float32), frame.header)


def main(output: Path) -> None:
    def frames(directory: str) -> list[Path]:
        return sorted((SESSION / directory).glob("*.arw"))

    session = Session(frames("offset"), frames("dark"), frames("flat"), frames("light"))
    start = time.time()
    master = preprocess(
        session,
        output,
        load_frame=as_pixinsight_loaded,
        report=lambda message: print(f"[{time.time() - start:6.0f}s] {message}", flush=True),
    )
    ours = load(master).data
    theirs = read_xisf(SESSION / "output" / "master" / "light-FILTER_m13-BINNING_1.xisf")[0].data
    print(f"finished in {time.time() - start:.0f}s")
    for channel, name in enumerate("RGB"):
        a = ours[INNER][..., channel].astype(np.float64)
        b = np.asarray(theirs[INNER][..., channel], dtype=np.float64)
        ratio = float(np.median(a / b))
        relative = (a / ratio - b) / b
        noise = float(np.std(b[b < np.percentile(b, 50)]))
        print(
            f"{name}: correlation {np.corrcoef(a.ravel(), b.ravel())[0, 1]:.6f}, "
            f"level ours/PixInsight {ratio:.4f}, "
            f"rms difference after matching level {np.sqrt((relative**2).mean()):.4%} "
            f"of the pixel value, {np.sqrt(((a / ratio - b) ** 2).mean()) / noise:.2f} "
            f"of the background noise"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
