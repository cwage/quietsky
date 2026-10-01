"""The M13 fixture set, made by tools/make_m13_fixtures.py.

tests/data/m13 holds a 256x256 crop of every raw frame of a 2018 session
(bias, dark, flat and light) and the same crop of the master frames that
PixInsight 1.8.5 produced from the full frames.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from starkiller.frame import load_fits

DATA = Path(__file__).parent / "data" / "m13"
# The full session, for the tests marked "nas".
SESSION = Path(os.environ.get("STARKILLER_DATA", "/data")) / "2018-09-12" / "m13"


def as_pixinsight_loaded(data: npt.NDArray[np.uint16]) -> npt.NDArray[np.float32]:
    """Raw sensor values as PixInsight 1.8.5 loaded them.

    Its dcraw-based loader dropped the two low bits of Sony's 14-bit samples,
    then scaled 16-bit integers onto [0, 1].
    """
    return ((data >> 2) / 65535).astype(np.float32)


def frames(kind: str) -> npt.NDArray[np.uint16]:
    """The cropped raw frames of one kind, as an (N, H, W) stack in time order."""
    return np.stack([load_fits(path).data for path in sorted((DATA / kind).glob("*.fits"))])


def master(kind: str) -> dict[str, npt.NDArray[Any]]:
    """Crops of PixInsight's master: integration, rejection_low, rejection_high."""
    with np.load(DATA / "reference" / f"master_{kind}.npz") as arrays:
        return dict(arrays)


def rejected_per_frame(history: list[str], side: str) -> list[int]:
    """Per-frame rejection counts ("Low" or "High") from a master's history."""
    return [
        int(match.group(1))
        for line in history
        if (match := re.search(rf"rejected{side}_\d+: (\d+)", line))
    ]


def estimates_per_frame(history: list[str], name: str) -> list[list[float]]:
    """A per-frame, per-channel quantity such as "scaleEstimates" from a history."""
    return [
        [float(value) for value in match.group(1).split()]
        for line in history
        if (match := re.search(rf"ImageIntegration\.{name}_\d+: (.+)", line))
    ]


def _master_record(kind: str) -> dict[str, list[str]]:
    record: dict[str, list[str]] = json.loads(
        (DATA / "reference" / f"master_{kind}.json").read_text()
    )
    return record


def master_history(kind: str) -> list[str]:
    """PixInsight's processing history for a master, covering the full frames."""
    return _master_record(kind)["history"]


def master_inputs(kind: str) -> list[str]:
    """Names of the files PixInsight integrated into a master, in its order.

    Per-frame entries in the history follow this order, which is not always
    the order of the file names.
    """
    return _master_record(kind)["inputs"]
