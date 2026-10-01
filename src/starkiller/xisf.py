"""Read XISF, PixInsight's image file format.

Covers monolithic files with attached pixel blocks, uncompressed or
zlib-compressed, which is what PixInsight's preprocessing scripts write.
"""

import struct
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

SIGNATURE = b"XISF0100"
NAMESPACE = "{http://www.pixinsight.com/xisf}"
SAMPLE_FORMATS = {
    "UInt8": "u1",
    "UInt16": "u2",
    "UInt32": "u4",
    "Float32": "f4",
    "Float64": "f8",
}


@dataclass(frozen=True)
class XisfImage:
    """One image of an XISF file: (H, W), or (H, W, C) for colour."""

    id: str
    data: npt.NDArray[Any]
    # FITS keywords as (name, value) pairs; HISTORY and COMMENT keep their
    # text in the comment field, so those appear as ("HISTORY", text).
    keywords: list[tuple[str, str]]


def _unshuffle(block: bytes, item_size: int) -> bytes:
    """Undo XISF byte shuffling: all first bytes, then all second bytes, ..."""
    items = len(block) // item_size
    planes = np.frombuffer(block, np.uint8, items * item_size).reshape(item_size, items)
    return planes.T.tobytes() + block[items * item_size :]


def _read_block(
    path: Path, element: ET.Element, dtype: np.dtype[Any], shape: tuple[int, ...]
) -> npt.NDArray[Any]:
    location = element.get("location", "").split(":")
    if location[0] != "attachment":
        raise NotImplementedError(f"XISF block location {location[0]!r}")
    position, size = int(location[1]), int(location[2])

    compression = element.get("compression")
    if compression is None:
        return np.memmap(path, dtype, mode="r", offset=position, shape=shape)

    codec, *sizes = compression.split(":")
    name, _, shuffled = codec.partition("+")
    if name != "zlib":
        raise NotImplementedError(f"XISF compression codec {name!r}")
    with open(path, "rb") as file:
        file.seek(position)
        block = zlib.decompress(file.read(size))
    if shuffled:
        block = _unshuffle(block, int(sizes[1]))
    return np.frombuffer(block, dtype).reshape(shape)


def read_xisf(path: Path) -> list[XisfImage]:
    """Read every image in an XISF file.

    Uncompressed pixel data is memory-mapped rather than read, so slicing a
    region out of a large file is cheap.
    """
    with open(path, "rb") as file:
        if file.read(8) != SIGNATURE:
            raise ValueError(f"not an XISF file: {path}")
        (length,) = struct.unpack("<I", file.read(4))
        file.read(4)
        root = ET.fromstring(file.read(length))

    images = []
    for element in root.iter(f"{NAMESPACE}Image"):
        width, height, channels = (int(n) for n in element.get("geometry", "").split(":"))
        order = ">" if element.get("byteOrder") == "big" else "<"
        dtype = np.dtype(order + SAMPLE_FORMATS[element.get("sampleFormat", "")])
        if element.get("pixelStorage", "Planar") == "Planar":
            data = _read_block(path, element, dtype, (channels, height, width))
            data = data[0] if channels == 1 else np.moveaxis(data, 0, -1)
        else:
            data = _read_block(path, element, dtype, (height, width, channels))
            data = data[..., 0] if channels == 1 else data
        keywords = [
            (name, k.get("comment", "") if name in ("HISTORY", "COMMENT") else k.get("value", ""))
            for k in element.iter(f"{NAMESPACE}FITSKeyword")
            if (name := k.get("name", ""))
        ]
        images.append(XisfImage(element.get("id", ""), data, keywords))
    return images
