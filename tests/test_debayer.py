import numpy as np
import pytest

import m13
from quietsky.debayer import debayer_vng
from quietsky.xisf import read_xisf

PATTERNS = ["RGGB", "BGGR", "GRBG", "GBRG"]


def mosaic_of(rgb: np.ndarray, pattern: str) -> np.ndarray:
    """Sample an RGB image through a Bayer filter."""
    mosaic = np.empty(rgb.shape[:2], np.float32)
    for index, letter in enumerate(pattern):
        row, column = divmod(index, 2)
        mosaic[row::2, column::2] = rgb[row::2, column::2, "RGB".index(letter)]
    return mosaic


@pytest.mark.parametrize("pattern", PATTERNS)
def test_a_uniform_colour_is_recovered_exactly(pattern: str) -> None:
    rgb = np.broadcast_to(np.array([0.2, 0.5, 0.8], np.float32), (12, 16, 3))

    result = debayer_vng(mosaic_of(rgb, pattern), pattern)

    np.testing.assert_allclose(result, rgb, rtol=1e-6)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_a_smooth_gradient_is_recovered_closely(pattern: str) -> None:
    rows, columns = np.mgrid[:32, :32]
    ramp = (0.2 + 0.01 * rows + 0.005 * columns).astype(np.float32)
    rgb = np.stack([ramp, 0.5 * ramp, 0.25 * ramp], axis=-1)

    result = debayer_vng(mosaic_of(rgb, pattern), pattern)

    np.testing.assert_allclose(result[2:-2, 2:-2], rgb[2:-2, 2:-2], rtol=0.02)


def test_the_sensor_sample_is_kept_in_its_own_channel() -> None:
    mosaic = np.random.default_rng(1).random((16, 16), dtype=np.float32)

    result = debayer_vng(mosaic, "RGGB")

    inner = (slice(2, -2), slice(2, -2))
    np.testing.assert_array_equal(result[..., 0][inner][0::2, 0::2], mosaic[inner][0::2, 0::2])
    np.testing.assert_array_equal(result[..., 1][inner][0::2, 1::2], mosaic[inner][0::2, 1::2])
    np.testing.assert_array_equal(result[..., 2][inner][1::2, 1::2], mosaic[inner][1::2, 1::2])


def test_the_border_repeats_the_nearest_inner_pixel() -> None:
    result = debayer_vng(np.random.default_rng(1).random((12, 12), dtype=np.float32))

    np.testing.assert_array_equal(result[0], result[2])
    np.testing.assert_array_equal(result[:, -1], result[:, -3])


def test_rejects_patterns_that_are_not_bayer() -> None:
    with pytest.raises(ValueError, match="not a Bayer pattern"):
        debayer_vng(np.zeros((8, 8), np.float32), "RGBG")


def test_debayered_lights_match_pixinsight() -> None:
    reference = m13.debayered_lights()
    for name, mosaic in m13.cosmetized_lights().items():
        result = debayer_vng(mosaic, "RGGB")

        # The crop's own edge lacks the neighbours the full frame had.
        np.testing.assert_array_equal(result[4:-4, 4:-4], reference[name][4:-4, 4:-4])


@pytest.mark.nas
def test_full_frame_debayered_light_matches_pixinsight() -> None:
    directory = m13.SESSION / "output" / "calibrated" / "light"
    name = "2018-09-12-20_45_50"
    mosaic = read_xisf(directory / "cosmetized" / f"{name}_c_cc.xisf")[0].data
    reference = read_xisf(directory / "debayered" / f"{name}_c_cc_d.xisf")[0].data

    result = debayer_vng(mosaic, "RGGB")

    # Identical, borders included, except where a gradient sits on the
    # threshold and rounding decides: 30 pixels when this was written.
    assert (result != reference).any(axis=-1).sum() <= 40
