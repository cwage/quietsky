"""Cut the M13 test fixtures out of the full data set.

Reads the 2018-09-12 M13 session (Sony raw frames plus the output of a
PixInsight BatchPreprocessing run) from $QUIETSKY_DATA and writes a small
crop of every frame, and of PixInsight's master frames, to tests/data/m13.

    docker compose run --rm quietsky python tools/make_m13_fixtures.py
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from quietsky.estimators import ikss
from quietsky.frame import Frame, load_raw, save_fits
from quietsky.xisf import read_xisf

SESSION = Path(os.environ["QUIETSKY_DATA"]) / "2018-09-12" / "m13"
OUTPUT = Path(__file__).parent.parent / "tests" / "data" / "m13"

# A window that keeps M13 in view in all the light frames. The origin is
# even in both axes so the crop has the same Bayer pattern as the full frame.
TOP, LEFT, SIZE = 1252, 2016, 256
WINDOW = (slice(TOP, TOP + SIZE), slice(LEFT, LEFT + SIZE))

# Raw frame directories, as named in the session, and our name for each kind.
KINDS = {"offset": "bias", "dark": "dark", "flat": "flat", "light": "light"}
# A smaller window inside the first, on the cluster core, for the stage that
# needs every light: the 43 registered frames, in colour.
CORE_TOP, CORE_LEFT, CORE_SIZE = TOP + 80, LEFT + 80, 96
CORE = (slice(CORE_TOP, CORE_TOP + CORE_SIZE), slice(CORE_LEFT, CORE_LEFT + CORE_SIZE))

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
    stages = {
        "calibrated_light": (directory, "_c"),
        "cosmetized_light": (directory / "cosmetized", "_c_cc"),
        "debayered_light": (directory / "debayered", "_c_cc_d"),
    }
    for stage, (folder, suffix) in stages.items():
        crops = {
            name: read_xisf(folder / f"{name}{suffix}.xisf")[0].data[WINDOW]
            for name in SAMPLE_LIGHTS
        }
        np.savez_compressed(OUTPUT / "reference" / f"{stage}.npz", **crops)  # type: ignore[arg-type]
    print(f"light stages: {len(SAMPLE_LIGHTS)} frames")


def registration_matrices(log: str) -> dict[str, list[list[float]]]:
    """The transformation PixInsight printed for each light, by raw frame name.

    It maps reference-frame pixel coordinates to the light's. PixInsight
    prints six decimals, which loses the perspective terms in the last row.
    """
    section = log.split("* Begin registration of light frames")[1]
    section = section.split("* End registration of light frames")[0]
    text = "\n".join(line.split("] ", 1)[-1] for line in section.splitlines())
    row = r"\s+".join([r"([+-]?\d+\.\d+)"] * 3)
    blocks = re.findall(
        rf"Transformation matrix:\n\s*{row}\n\s*{row}\n\s*{row}"
        r".*?Writing output file: \S*/(\S+)_c_cc_d_r\.xisf",
        text,
        re.DOTALL,
    )
    return {
        block[9]: [[float(value) for value in block[i : i + 3]] for i in (0, 3, 6)]
        for block in blocks
    }


def registered_lights(log: str) -> None:
    """Crops of all the registered lights, and what integration needs to know of each.

    Integration normalises a frame by the level and scale of the whole frame
    and weights it by the noise PixInsight measured after demosaicing, none
    of which a crop can give.
    """
    directory = SESSION / "output" / "registered" / "m13"
    names = integration_inputs(log, "light")
    master = read_xisf(SESSION / "output" / "master" / MASTERS["light"])[0]
    history = [text for key, text in master.keywords if key == "HISTORY"]
    noise = [
        [float(value) for value in match.group(1).split()]
        for line in history
        if (match := re.search(r"noiseEstimates_\d+: (.+)", line))
    ]

    def estimates(name: str) -> list[tuple[float, float]]:
        data = read_xisf(directory / name)[0].data
        return [ikss(np.asarray(data[..., channel])) for channel in range(3)]

    with ThreadPoolExecutor(8) as pool:
        per_frame = list(pool.map(estimates, names))
    record = {
        name: {
            "location": [location for location, _ in frame],
            "scale": [scale for _, scale in frame],
            "noise": frame_noise,
        }
        for name, frame, frame_noise in zip(names, per_frame, noise, strict=True)
    }
    (OUTPUT / "reference" / "registered_light_estimates.json").write_text(
        json.dumps(record, indent=1)
    )
    crops = {name: read_xisf(directory / name)[0].data[CORE] for name in names}
    np.savez_compressed(OUTPUT / "reference" / "registered_light.npz", **crops)  # type: ignore[arg-type]
    print(f"registered lights: {len(names)} frames")


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
    matrices = registration_matrices(log)
    (OUTPUT / "reference" / "registration.json").write_text(json.dumps(matrices, indent=1))
    print(f"registration: {len(matrices)} matrices")
    noise = debayer_noise(log)
    (OUTPUT / "reference" / "debayer_noise.json").write_text(json.dumps(noise, indent=1))
    print(f"debayer noise: {len(noise)} frames")


if __name__ == "__main__":
    crop_frames()
    crop_masters()
    crop_calibrated_flats()
    crop_light_stages()
    registered_lights(next((SESSION / "output" / "logs").glob("*.log")).read_text())
