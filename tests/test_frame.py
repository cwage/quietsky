from pathlib import Path

import numpy as np
import pytest

from starkiller.frame import Frame, load, normalized, save_fits


def test_fits_round_trip_keeps_integer_mosaic_and_header(tmp_path: Path) -> None:
    data = np.arange(12, dtype=np.uint16).reshape(3, 4) * 5000
    path = tmp_path / "frame.fits"

    save_fits(path, Frame(data, {"BAYERPAT": "RGGB", "EXPTIME": 30.0}))
    loaded = load(path)

    assert loaded.data.dtype == np.uint16
    np.testing.assert_array_equal(loaded.data, data)
    assert loaded.header["BAYERPAT"] == "RGGB"
    assert loaded.header["EXPTIME"] == 30.0


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
