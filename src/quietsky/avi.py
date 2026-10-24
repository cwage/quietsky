"""Read uncompressed 8-bit video frames from AVI files.

Planetary and lunar cameras, the Seestar among them, record the sensor's
undemosaiced data as AVI files with one uncompressed frame per chunk. Files
over 1 GB follow the OpenDML extension: the first RIFF "AVI " segment is
followed by RIFF "AVIX" segments, each with a "movi" list of more frames.
The frames are found by walking the chunks and read through a memory map,
so a clip of any length costs no memory until a frame is asked for.
"""

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from quietsky.frame import HeaderValue

# Chunk ids of video frames: uncompressed ("db") or compressed ("dc") data of stream 0.
_FRAME_CHUNKS = {b"00db", b"00dc"}


@dataclass(frozen=True)
class Video:
    """The frames of an AVI file, each an (H, W) uint8 array.

    header holds what the capture program wrote beside the file (the
    Seestar's .avi.txt), such as the Bayer pattern under BAYERPAT.
    """

    path: Path
    width: int
    height: int
    microseconds_per_frame: int
    offsets: list[int]
    top_down: bool
    header: dict[str, HeaderValue] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.offsets)

    def frame(self, index: int) -> npt.NDArray[np.uint8]:
        data = np.memmap(
            self.path,
            np.uint8,
            "r",
            offset=self.offsets[index],
            shape=(self.height, self.width),
        )
        return np.array(data if self.top_down else data[::-1])


def _chunks(data: bytes | np.memmap, start: int, end: int) -> list[tuple[bytes, int, int]]:
    """The (id, offset of the body, size of the body) of the chunks between two offsets."""
    chunks = []
    offset = start
    while offset + 8 <= end:
        chunk_id = bytes(data[offset : offset + 4])
        (size,) = struct.unpack("<I", bytes(data[offset + 4 : offset + 8]))
        chunks.append((chunk_id, offset + 8, size))
        offset += 8 + size + (size & 1)
    return chunks


def _riff_segments(data: bytes | np.memmap) -> list[tuple[int, int]]:
    """The (offset of the body, size of the body) of each top-level RIFF segment.

    The Seestar records the size of its "AVIX" segments 8 bytes too large, so
    the next segment is looked for a few bytes either side of where its size
    says it should start.
    """
    segments = []
    offset = 0
    while offset + 12 <= len(data):
        found = next(
            (
                start
                for start in sorted(range(offset - 16, offset + 17), key=lambda o: abs(o - offset))
                if 0 <= start and bytes(data[start : start + 4]) == b"RIFF"
            ),
            None,
        )
        if found is None:
            break
        (size,) = struct.unpack("<I", bytes(data[found + 4 : found + 8]))
        segments.append((found + 8, size))
        offset = found + 8 + size + (size & 1)
    return segments


def _sidecar_header(path: Path) -> dict[str, HeaderValue]:
    """Keywords from a Seestar "name.avi.txt" file, if there is one."""
    sidecar = path.with_name(path.name + ".txt")
    if not sidecar.exists():
        return {}
    values: dict[str, str] = {}
    for line in sidecar.read_text(errors="replace").splitlines():
        key, equals, value = line.partition("=")
        if equals:
            values[key.strip()] = value.strip()
    header: dict[str, HeaderValue] = {}
    if "Bayer" in values:
        # The Seestar names the first row only (GR); the second is the other pair.
        first = values["Bayer"].upper()
        second = {"GR": "BG", "RG": "GB", "BG": "GR", "GB": "RG"}.get(first, "")
        if second:
            header["BAYERPAT"] = first + second
    if "DATE-OBS" in values:
        header["DATE-OBS"] = values["DATE-OBS"]
    if "OBJECT" in values:
        header["OBJECT"] = values["OBJECT"]
    for key in ("SITELAT", "SITELONG"):
        try:
            header[key] = float(values[key])
        except (KeyError, ValueError):
            pass
    return header


def read_video(path: Path) -> Video:
    """Index the frames of an uncompressed 8-bit AVI file."""
    data = np.memmap(path, np.uint8, "r")
    width = height = microseconds = 0
    top_down = False
    offsets: list[int] = []
    for riff_body, riff_size in _riff_segments(data):
        riff_end = min(riff_body + riff_size, len(data))
        for chunk_id, body, size in _chunks(data, riff_body + 4, riff_end):
            if chunk_id != b"LIST":
                continue
            kind = bytes(data[body : body + 4])
            if kind == b"hdrl":
                for header_id, header_body, header_size in _chunks(data, body + 4, body + size):
                    if header_id == b"avih":
                        fields = struct.unpack("<10I", bytes(data[header_body : header_body + 40]))
                        microseconds, width, height = fields[0], fields[8], fields[9]
                    if header_id == b"LIST":
                        for stream_id, stream_body, _ in _chunks(
                            data, header_body + 4, header_body + header_size
                        ):
                            if stream_id == b"strf":
                                info = struct.unpack(
                                    "<IiiHH", bytes(data[stream_body : stream_body + 16])
                                )
                                if info[4] != 8:
                                    raise ValueError(
                                        f"{path}: {info[4]}-bit frames, only 8-bit are supported"
                                    )
                                # A negative height in the bitmap header means rows run
                                # from the top down.
                                top_down = info[2] < 0
            elif kind == b"movi":
                for frame_id, frame_body, frame_size in _chunks(
                    data, body + 4, min(body + size, riff_end)
                ):
                    complete = frame_body + frame_size <= len(data)
                    if frame_id in _FRAME_CHUNKS and frame_size == width * height and complete:
                        offsets.append(frame_body)
    if not offsets:
        raise ValueError(f"no uncompressed frames found in {path}")
    return Video(path, width, height, microseconds, offsets, top_down, _sidecar_header(path))
