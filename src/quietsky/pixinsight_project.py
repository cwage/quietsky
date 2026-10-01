"""Read images stored in a PixInsight project.

A project (.xosm) keeps the pixels of each open image, and of the earlier
states in its processing history, as files in a .data directory beside it.
Those give the exact input and output of a process that was run, which the
reference tests use.
"""

import struct
import zlib
from pathlib import Path

import numpy as np
import numpy.typing as npt

MAGIC = b"egamiwar"
HEADER_SIZE = 128
BLOCK_HEADER_SIZE = 32


def read_project_image(path: Path) -> npt.NDArray[np.float32]:
    """Read a 32-bit floating point image from a project's data directory, as (H, W, C)."""
    data = path.read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"not a PixInsight project image: {path}")
    width, height, channels = struct.unpack("<3I", data[28:40])
    position = HEADER_SIZE
    planes = []
    # Each channel is a list of blocks, zlib-compressed unless that would not
    # make them smaller, with the four bytes of each sample stored apart.
    for _ in range(channels):
        (count,) = struct.unpack("<I", data[position : position + 4])
        position += 4
        plane = bytearray()
        for _ in range(count):
            stored, size = struct.unpack("<QQ", data[position : position + 16])
            position += BLOCK_HEADER_SIZE
            block = data[position : position + stored]
            position += stored
            plane += block if stored == size else zlib.decompress(block)
        shuffled = np.frombuffer(bytes(plane), np.uint8).reshape(4, -1)
        planes.append(shuffled.T.copy().view("<f4").reshape(height, width))
    return np.stack(planes, axis=-1)
