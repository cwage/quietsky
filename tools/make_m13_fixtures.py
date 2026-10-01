"""Cut the M13 test fixtures out of the full data set.

Reads the 2018-09-12 M13 session (Sony raw frames plus the output of a
PixInsight BatchPreprocessing run) from $STARKILLER_DATA and writes a small
crop of every frame, and of PixInsight's master frames, to tests/data/m13.

    docker compose run --rm starkiller python tools/make_m13_fixtures.py
"""

import json
import os
from pathlib import Path

import numpy as np

from starkiller.frame import Frame, load_raw, save_fits
from starkiller.xisf import read_xisf

SESSION = Path(os.environ["STARKILLER_DATA"]) / "2018-09-12" / "m13"
OUTPUT = Path(__file__).parent.parent / "tests" / "data" / "m13"

# A window that keeps M13 in view in all the light frames. The origin is
# even in both axes so the crop has the same Bayer pattern as the full frame.
TOP, LEFT, SIZE = 1252, 2016, 256
WINDOW = (slice(TOP, TOP + SIZE), slice(LEFT, LEFT + SIZE))

# Raw frame directories, as named in the session, and our name for each kind.
KINDS = {"offset": "bias", "dark": "dark", "flat": "flat", "light": "light"}
MASTERS = {
    "bias": "bias-BINNING_1.xisf",
    "dark": "dark-BINNING_1-EXPTIME_30.xisf",
    "flat": "flat-FILTER_m13-BINNING_1.xisf",
    "light": "light-FILTER_m13-BINNING_1.xisf",
}


def crop_frames() -> None:
    for directory, kind in KINDS.items():
        paths = sorted((SESSION / directory).glob("*.arw"))
        (OUTPUT / kind).mkdir(parents=True, exist_ok=True)
        for path in paths:
            frame = load_raw(path)
            header = frame.header | {
                "IMAGETYP": kind.upper(),
                "XORGSUBF": LEFT,
                "YORGSUBF": TOP,
                "ORIGFILE": path.name,
            }
            save_fits(OUTPUT / kind / f"{path.stem}.fits", Frame(frame.data[WINDOW], header))
        print(f"{kind}: {len(paths)} frames")


def crop_masters() -> None:
    (OUTPUT / "reference").mkdir(parents=True, exist_ok=True)
    for kind, name in MASTERS.items():
        integration, rejection_low, rejection_high = read_xisf(SESSION / "output" / "master" / name)
        np.savez_compressed(
            OUTPUT / "reference" / f"master_{kind}.npz",
            integration=integration.data[WINDOW],
            rejection_low=rejection_low.data[WINDOW],
            rejection_high=rejection_high.data[WINDOW],
        )
        # The processing history records the settings PixInsight used and how
        # many pixels it rejected in each full frame.
        history = [text for key, text in integration.keywords if key == "HISTORY"]
        (OUTPUT / "reference" / f"master_{kind}.json").write_text(json.dumps(history, indent=1))
        print(f"master {kind}")


if __name__ == "__main__":
    crop_frames()
    crop_masters()
