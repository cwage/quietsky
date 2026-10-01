"""Load and save image frames: camera raw files, FITS and XISF."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import rawpy
from astropy.io import fits

from quietsky.xisf import read_xisf

HeaderValue = str | int | float | bool

# FITS keywords that describe how the pixels are stored, not the image.
STRUCTURAL_KEYWORDS = {"SIMPLE", "BITPIX", "NAXIS", "EXTEND", "BSCALE", "BZERO", "ROWORDER"}
RAW_SUFFIXES = {".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf", ".raf", ".rw2"}
FITS_SUFFIXES = {".fit", ".fits", ".fts"}


@dataclass(frozen=True)
class Frame:
    """Pixel data with FITS-style header keywords.

    data is (H, W) for a mono image or an undemosaiced colour mosaic, and
    (H, W, C) for a colour image. Integer data holds the values as stored.
    """

    data: npt.NDArray[Any]
    header: dict[str, HeaderValue] = field(default_factory=dict)


def load_raw(path: Path) -> Frame:
    """Load the undemosaiced sensor data of a camera raw file."""
    with rawpy.imread(str(path)) as raw:
        data = raw.raw_image_visible.copy()
        if raw.raw_pattern is None:
            raise ValueError(f"not a Bayer-mosaic raw file: {path}")
        colours = raw.color_desc.decode()
        header: dict[str, HeaderValue] = {
            "BAYERPAT": "".join(colours[i] for i in raw.raw_pattern.ravel()),
            "BLKLEVEL": int(raw.black_level_per_channel[0]),
            "WHTLEVEL": int(raw.white_level),
            "EXPTIME": float(raw.other.shutter_speed),
            "ISOSPEED": int(raw.other.iso_speed),
            "FOCALLEN": float(raw.other.focal_length),
            # Camera clock time: raw files don't record a time zone.
            "DATE-LOC": raw.other.timestamp.isoformat(),
        }
    return Frame(data, header)


def load_fits(path: Path) -> Frame:
    """Load the first image in a FITS file."""
    with fits.open(path) as hdus:
        hdu = next(h for h in hdus if h.data is not None)
        data = np.asarray(hdu.data)
        header = {
            key: value
            for key, value in hdu.header.items()
            if key
            and key not in ("COMMENT", "HISTORY")
            and key.rstrip("0123456789") not in STRUCTURAL_KEYWORDS
        }
    if data.ndim == 3:
        data = np.moveaxis(data, 0, -1)
    return Frame(data, header)


def save_fits(path: Path, frame: Frame) -> None:
    data = np.moveaxis(frame.data, -1, 0) if frame.data.ndim == 3 else frame.data
    hdu = fits.PrimaryHDU(data)
    hdu.header.update(frame.header)
    # FITS puts the origin at the bottom left; we keep row 0 at the top.
    hdu.header["ROWORDER"] = "TOP-DOWN"
    hdu.writeto(path, overwrite=True)


def load_header(path: Path) -> dict[str, HeaderValue]:
    """The header of a frame, without reading the pixels of a FITS file."""
    if path.suffix.lower() not in FITS_SUFFIXES:
        return load(path).header
    with fits.open(path) as hdus:
        hdu = next(h for h in hdus if h.header["NAXIS"] > 0)
        return {
            key: value
            for key, value in hdu.header.items()
            if key
            and key not in ("COMMENT", "HISTORY")
            and key.rstrip("0123456789") not in STRUCTURAL_KEYWORDS
        }


class DiskStack:
    """One channel of many floating point FITS frames, read from disk on demand.

    It stands in for an (N, H, W) array in the two ways integration uses a
    stack: stack[i] is a whole frame, and stack[:, rows, :] a band of rows
    across every frame. Only what is asked for is read, so the number of
    frames is not limited by memory.
    """

    def __init__(self, paths: Sequence[Path], channel: int = 0) -> None:
        self._paths = list(paths)
        self._offsets = []
        for path in self._paths:
            with fits.open(path) as hdus:
                hdu = next(h for h in hdus if h.header["NAXIS"] > 0)
                header = hdu.header
                if header["BITPIX"] not in (-32, -64):
                    raise ValueError(f"not a floating point FITS image: {path}")
                self._dtype = np.dtype(">f4" if header["BITPIX"] == -32 else ">f8")
                self._height, self._width = header["NAXIS2"], header["NAXIS1"]
                channels = header["NAXIS3"] if header["NAXIS"] == 3 else 1
                if channel >= channels:
                    raise ValueError(f"{path} has no channel {channel}")
                plane = channel * self._height * self._width * self._dtype.itemsize
                self._offsets.append(hdu.fileinfo()["datLoc"] + plane)
        self.shape = (len(self._paths), self._height, self._width)

    def __len__(self) -> int:
        return len(self._paths)

    def _rows(self, index: int, rows: slice) -> npt.NDArray[np.float32]:
        top, bottom, _ = rows.indices(self._height)
        with open(self._paths[index], "rb") as file:
            file.seek(self._offsets[index] + top * self._width * self._dtype.itemsize)
            data = np.fromfile(file, self._dtype, (bottom - top) * self._width)
        return data.reshape(bottom - top, self._width).astype(np.float32)

    def __getitem__(self, key: int | tuple[slice, slice, slice]) -> npt.NDArray[np.float32]:
        if isinstance(key, tuple):
            return np.stack([self._rows(index, key[1]) for index in range(len(self))])
        return self._rows(key, slice(None))


def load(path: Path) -> Frame:
    """Load a frame, choosing the format from the file suffix."""
    suffix = path.suffix.lower()
    if suffix in RAW_SUFFIXES:
        return load_raw(path)
    if suffix in FITS_SUFFIXES:
        return load_fits(path)
    if suffix == ".xisf":
        image = read_xisf(path)[0]
        return Frame(np.asarray(image.data), dict(image.keywords))
    raise ValueError(f"unsupported file type: {path}")


def normalized(data: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """Convert pixel data to float32, mapping integer types onto [0, 1]."""
    if np.issubdtype(data.dtype, np.integer):
        return (data / np.iinfo(data.dtype).max).astype(np.float32)
    return data.astype(np.float32, copy=False)
