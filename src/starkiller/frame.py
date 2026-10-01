"""Load and save image frames: camera raw files, FITS and XISF."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import rawpy
from astropy.io import fits

from starkiller.xisf import read_xisf

HeaderValue = str | int | float | bool

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
            if key and key not in ("COMMENT", "HISTORY")
        }
    if data.ndim == 3:
        data = np.moveaxis(data, 0, -1)
    return Frame(data, header)


def save_fits(path: Path, frame: Frame) -> None:
    data = np.moveaxis(frame.data, -1, 0) if frame.data.ndim == 3 else frame.data
    hdu = fits.PrimaryHDU(data)
    for key, value in frame.header.items():
        if key not in hdu.header:
            hdu.header[key] = value
    # FITS puts the origin at the bottom left; we keep row 0 at the top.
    hdu.header["ROWORDER"] = "TOP-DOWN"
    hdu.writeto(path, overwrite=True)


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
