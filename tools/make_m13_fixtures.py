"""Cut the M13 test fixtures out of the full data set.

Reads the 2018-09-12 M13 session (Sony raw frames plus the output of a
PixInsight BatchPreprocessing run) from $STARKILLER_DATA and writes a small
crop of every frame, and of PixInsight's master frames, to tests/data/m13.

    docker compose run --rm starkiller python tools/make_m13_fixtures.py
"""

import json
import os
import re
from pathlib import Path

import numpy as np

from starkiller.estimators import ikss
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
# Lights whose intermediate stages are kept as crops: the first (a stray 10 s
# exposure at ISO 1600), one from the middle of the session and the last.
SAMPLE_LIGHTS = ("2018-09-12-20_22_50", "2018-09-12-20_38_15", "2018-09-12-20_56_55")
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


def integration_inputs(log: str, kind: str) -> list[str]:
    """Names of the files PixInsight integrated into a master, in its order.

    The order is not always that of the file names, and it is the order the
    per-frame entries of the master's history follow.
    """
    section = log.split(f"* Begin integration of {kind} frames")[1]
    section = section.split("Pixel rejection counts:")[0]
    first = re.search(r"Opening files:\s*(?:\[.*?\]\s*)*(/\S+)", section)
    assert first is not None
    rest = re.findall(r"\] \[\d+\] (/\S+)", section)
    return [Path(path).name for path in [first.group(1), *rest]]


def debayer_noise(log: str) -> dict[str, list[list[float]]]:
    """Noise PixInsight measured in each debayered light.

    Maps the debayered file name to a (sigma, fraction of pixels) pair per
    colour channel.
    """
    section = log.split("* Begin demosaicing of light frames")[1].split("* End demosaicing")[0]
    channel = r"s\d = (\S+), n\d = (\S+) \(MRS\)\n.*?"
    blocks = re.findall(channel * 3 + r"Writing output file: \S*/(\S+)", section, re.DOTALL)
    return {
        block[6]: [[float(block[i]), float(block[i + 1])] for i in (0, 2, 4)] for block in blocks
    }


def dark_scales(log: str, kind: str) -> dict[str, float]:
    """The dark scaling factor PixInsight chose for each calibrated frame."""
    section = log.split(f"* Begin calibration of {kind} frames")[1]
    section = section.split(f"* End calibration of {kind} frames")[0]
    pairs = re.findall(r"Writing output file: \S*/(\S+)_c\.xisf\n.*?\n.*?k0 = (\S+)", section)
    return {name: float(scale) for name, scale in pairs}


def crop_calibrated_flats() -> None:
    directory = SESSION / "output" / "calibrated" / "flat"
    crops = {
        path.name.removesuffix("_c.xisf"): read_xisf(path)[0].data[WINDOW]
        for path in sorted(directory.glob("*_c.xisf"))
    }
    np.savez_compressed(OUTPUT / "reference" / "calibrated_flat.npz", **crops)  # type: ignore[arg-type]
    # Integration matches the flats' brightness using the level of each whole
    # frame, which the crops alone cannot give.
    locations = {
        path.name.removesuffix("_c.xisf"): ikss(read_xisf(path)[0].data)[0]
        for path in sorted(directory.glob("*_c.xisf"))
    }
    (OUTPUT / "reference" / "calibrated_flat_locations.json").write_text(
        json.dumps(locations, indent=1)
    )
    print(f"calibrated flats: {len(crops)} frames")


def crop_light_stages() -> None:
    """Crops of the sample lights after each stage of PixInsight's pipeline."""
    directory = SESSION / "output" / "calibrated" / "light"
    calibrated = {
        name: read_xisf(directory / f"{name}_c.xisf")[0].data[WINDOW] for name in SAMPLE_LIGHTS
    }
    np.savez_compressed(OUTPUT / "reference" / "calibrated_light.npz", **calibrated)  # type: ignore[arg-type]
    print(f"light stages: {len(SAMPLE_LIGHTS)} frames")


def crop_masters() -> None:
    (OUTPUT / "reference").mkdir(parents=True, exist_ok=True)
    log = next((SESSION / "output" / "logs").glob("*.log")).read_text()
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
        record = {"inputs": integration_inputs(log, kind), "history": history}
        (OUTPUT / "reference" / f"master_{kind}.json").write_text(json.dumps(record, indent=1))
        print(f"master {kind}")
    # Dividing by the flat scales it by the mean of the whole master flat.
    flat = read_xisf(SESSION / "output" / "master" / MASTERS["flat"])[0].data
    (OUTPUT / "reference" / "master_flat_mean.json").write_text(
        json.dumps(float(flat.mean(dtype=np.float64)))
    )
    scales = {kind: dark_scales(log, kind) for kind in ("flat", "light")}
    (OUTPUT / "reference" / "dark_scales.json").write_text(json.dumps(scales, indent=1))
    print(f"dark scales: {len(scales['flat'])} flats, {len(scales['light'])} lights")
    noise = debayer_noise(log)
    (OUTPUT / "reference" / "debayer_noise.json").write_text(json.dumps(noise, indent=1))
    print(f"debayer noise: {len(noise)} frames")


if __name__ == "__main__":
    crop_frames()
    crop_masters()
    crop_calibrated_flats()
    crop_light_stages()
