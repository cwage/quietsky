from pathlib import Path

import numpy as np
import pytest

from quietsky.frame import DiskStack, Frame, load, load_header, normalized, save_fits


def test_fits_round_trip_keeps_integer_mosaic_and_header(tmp_path: Path) -> None:
    data = np.arange(12, dtype=np.uint16).reshape(3, 4) * 5000
    path = tmp_path / "frame.fits"

    save_fits(path, Frame(data, {"BAYERPAT": "RGGB", "EXPTIME": 30.0}))
    loaded = load(path)

    assert loaded.data.dtype == np.uint16
    np.testing.assert_array_equal(loaded.data, data)
    assert loaded.header["BAYERPAT"] == "RGGB"
    assert loaded.header["EXPTIME"] == 30.0


def test_header_of_an_integer_frame_can_be_reused_for_float_data(tmp_path: Path) -> None:
    integer = tmp_path / "integer.fits"
    save_fits(integer, Frame(np.array([[1000, 2000]], dtype=np.uint16), {"EXPTIME": 30.0}))
    header = load(integer).header
    data = np.array([[0.25, 0.5]], dtype=np.float32)
    path = tmp_path / "float.fits"

    save_fits(path, Frame(data, header))

    np.testing.assert_array_equal(load(path).data, data)
    assert load(path).header == {"EXPTIME": 30.0}


def test_fits_round_trip_keeps_colour_channels_last(tmp_path: Path) -> None:
    data = np.random.default_rng(1).random((3, 4, 3), dtype=np.float32)
    path = tmp_path / "colour.fits"

    save_fits(path, Frame(data))

    np.testing.assert_array_equal(load(path).data, data)


def test_normalized_maps_integers_onto_unit_range() -> None:
    result = normalized(np.array([0, 65535], dtype=np.uint16))

    assert result.dtype == np.float32
    assert result.tolist() == [0.0, 1.0]


def test_normalized_leaves_float_values_alone() -> None:
    data = np.array([0.25, 1.5], dtype=np.float32)

    np.testing.assert_array_equal(normalized(data), data)


def test_load_refuses_unknown_file_types(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported file type"):
        load(tmp_path / "notes.txt")


def test_disk_stack_reads_frames_and_bands_of_rows_from_files(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    frames = [rng.random((6, 8, 3), dtype=np.float32) for _ in range(4)]
    paths = [tmp_path / f"frame{index}.fits" for index in range(4)]
    for path, frame in zip(paths, frames, strict=True):
        save_fits(path, Frame(frame, {"EXPTIME": 30.0}))

    stack = DiskStack(paths, channel=1)

    assert stack.shape == (4, 6, 8)
    assert len(stack) == 4
    np.testing.assert_array_equal(stack[2], frames[2][..., 1])
    expected = np.stack([frame[2:5, :, 1] for frame in frames])
    np.testing.assert_array_equal(stack[:, 2:5, :], expected)


def test_disk_stack_reads_mono_frames(tmp_path: Path) -> None:
    frame = np.random.default_rng(1).random((5, 7), dtype=np.float32)
    save_fits(tmp_path / "mono.fits", Frame(frame))

    np.testing.assert_array_equal(DiskStack([tmp_path / "mono.fits"])[0], frame)


def test_disk_stack_refuses_integer_frames(tmp_path: Path) -> None:
    save_fits(tmp_path / "raw.fits", Frame(np.zeros((4, 4), np.uint16)))

    with pytest.raises(ValueError, match="not a floating point FITS image"):
        DiskStack([tmp_path / "raw.fits"])


def test_load_header_gives_the_keywords_without_the_pixels(tmp_path: Path) -> None:
    save_fits(tmp_path / "frame.fits", Frame(np.zeros((4, 4), np.float32), {"NOISE00": 1e-5}))

    assert load_header(tmp_path / "frame.fits") == {"NOISE00": 1e-5}


def test_disk_stack_places_frames_of_different_sizes_on_one_canvas(tmp_path: Path) -> None:
    big = np.full((10, 12), 1.0, np.float32)
    small = np.full((8, 6), 2.0, np.float32)
    save_fits(tmp_path / "a.fits", Frame(big, {"PLACEX": 0, "PLACEY": 0}))
    save_fits(tmp_path / "b.fits", Frame(small, {"PLACEX": 5, "PLACEY": 4}))
    save_fits(tmp_path / "c.fits", Frame(small, {"PLACEX": -2, "PLACEY": 6}))

    stack = DiskStack([tmp_path / "a.fits", tmp_path / "b.fits", tmp_path / "c.fits"])

    # The canvas is the union: from x -2 to 11 and y 0 to 13.
    assert stack.origin == (-2, 0)
    assert stack.shape == (3, 14, 14)
    whole = stack[1]
    assert whole.shape == (14, 14)
    assert (whole[4:12, 7:13] == 2.0).all()
    assert whole.sum() == small.sum()
    assert (stack[2][6:14, 0:6] == 2.0).all()
    band = stack[:, 3:7, :]
    assert band.shape == (3, 4, 14)
    assert (band[0, :, 2:14] == 1.0).all() and (band[0, :, :2] == 0.0).all()
    assert (band[1, 0] == 0.0).all() and (band[1, 1:, 7:13] == 2.0).all()
    # Frames without a placement sit at (0, 0), and a stack of them is as before.
    save_fits(tmp_path / "plain.fits", Frame(big))
    plain = DiskStack([tmp_path / "plain.fits", tmp_path / "a.fits"])
    assert plain.origin == (0, 0) and plain.shape == (2, 10, 12)
