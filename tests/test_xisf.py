import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from quietsky.xisf import read_xisf

HEADER_SIZE = 4096


def write_xisf(path: Path, images: str, blocks: bytes) -> None:
    """Write a monolithic XISF file whose pixel blocks start at HEADER_SIZE."""
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<xisf version="1.0" xmlns="http://www.pixinsight.com/xisf">{images}</xisf>'
    ).encode()
    header = b"XISF0100" + struct.pack("<I", len(xml)) + bytes(4) + xml
    path.write_bytes(header.ljust(HEADER_SIZE, b"\0") + blocks)


def test_reads_every_image_with_its_keywords(tmp_path: Path) -> None:
    mono = np.arange(6, dtype="<f4").reshape(2, 3) / 10
    colour = np.arange(24, dtype="<u2").reshape(3, 2, 4)  # stored as planes
    images = (
        f'<Image id="integration" geometry="3:2:1" sampleFormat="Float32" colorSpace="Gray"'
        f' location="attachment:{HEADER_SIZE}:{mono.nbytes}">'
        '<FITSKeyword name="IMAGETYP" value="\'Master Bias\'" comment="Type of image"/>'
        '<FITSKeyword name="HISTORY" value="" comment="ImageIntegration.numberOfImages: 20"/>'
        "</Image>"
        f'<Image id="extra" geometry="4:2:3" sampleFormat="UInt16" colorSpace="RGB"'
        f' location="attachment:{HEADER_SIZE + mono.nbytes}:{colour.nbytes}"/>'
    )
    path = tmp_path / "master.xisf"
    write_xisf(path, images, mono.tobytes() + colour.tobytes())

    first, second = read_xisf(path)

    assert first.id == "integration"
    np.testing.assert_array_equal(first.data, mono)
    assert first.keywords == [
        ("IMAGETYP", "'Master Bias'"),
        ("HISTORY", "ImageIntegration.numberOfImages: 20"),
    ]
    assert second.id == "extra"
    assert second.data.shape == (2, 4, 3)
    np.testing.assert_array_equal(second.data, np.moveaxis(colour, 0, -1))


@pytest.mark.parametrize("shuffled", [False, True])
def test_reads_zlib_compressed_pixels(tmp_path: Path, shuffled: bool) -> None:
    data = np.arange(12, dtype="<u2").reshape(3, 4) * 1000
    raw = data.tobytes()
    if shuffled:
        raw = np.frombuffer(raw, np.uint8).reshape(-1, 2).T.tobytes()
        compression = f"zlib+sh:{data.nbytes}:2"
    else:
        compression = f"zlib:{data.nbytes}"
    block = zlib.compress(raw)
    image = (
        f'<Image id="image" geometry="4:3:1" sampleFormat="UInt16" colorSpace="Gray"'
        f' compression="{compression}" location="attachment:{HEADER_SIZE}:{len(block)}"/>'
    )
    path = tmp_path / "compressed.xisf"
    write_xisf(path, image, block)

    np.testing.assert_array_equal(read_xisf(path)[0].data, data)


def test_refuses_files_that_are_not_xisf(tmp_path: Path) -> None:
    path = tmp_path / "plain.xisf"
    path.write_bytes(b"SIMPLE  =                    T")

    with pytest.raises(ValueError, match="not an XISF file"):
        read_xisf(path)
