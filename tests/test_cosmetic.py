import numpy as np
import pytest

import m13
from starkiller.cosmetic import hot_pixels, replace_from_neighbours
from starkiller.xisf import read_xisf


def test_hot_pixels_are_the_outliers_of_the_dark() -> None:
    dark = np.random.default_rng(1).normal(0, 1e-5, (64, 64)).astype(np.float32)
    dark[10, 20] = 5e-4
    dark[40, 7] = 2e-4

    hot = hot_pixels(dark, sigma=6)

    assert np.argwhere(hot).tolist() == [[10, 20], [40, 7]]


def test_mosaic_pixels_are_replaced_from_neighbours_of_the_same_colour() -> None:
    image = np.zeros((6, 6), np.float32)
    image[0::2, 0::2] = 0.4  # one colour of the mosaic
    image[2, 2] = 9.0
    defective = np.zeros((6, 6), np.bool_)
    defective[2, 2] = True

    result = replace_from_neighbours(image, defective, mosaic=True)

    assert result[2, 2] == np.float32(0.4)
    assert (result == image)[~defective].all()


def test_mono_pixels_are_replaced_from_adjacent_neighbours() -> None:
    image = np.arange(9, dtype=np.float32).reshape(3, 3)
    defective = np.zeros((3, 3), np.bool_)
    defective[1, 1] = True

    result = replace_from_neighbours(image, defective, mosaic=False)

    assert result[1, 1] == 4.0  # mean of 0..8 without the centre


def test_pixels_at_the_edge_use_the_neighbours_that_exist() -> None:
    image = np.array([[9.0, 1.0], [2.0, 3.0]], np.float32)
    defective = np.array([[True, False], [False, False]])

    result = replace_from_neighbours(image, defective, mosaic=False)

    assert result[0, 0] == 2.0


def lowered_by_pixinsight(calibrated: np.ndarray, corrected: np.ndarray) -> np.ndarray:
    """The hot pixels PixInsight replaced, away from the image edge."""
    lowered = corrected < calibrated
    lowered[:2] = lowered[-2:] = lowered[:, :2] = lowered[:, -2:] = False
    return lowered


def test_replacement_matches_pixinsight_at_the_pixels_it_corrected() -> None:
    corrected = m13.cosmetized_lights()
    total = 0
    for name, calibrated in m13.calibrated_lights().items():
        lowered = lowered_by_pixinsight(calibrated, corrected[name])
        total += int(lowered.sum())

        result = replace_from_neighbours(calibrated, lowered, mosaic=True)

        np.testing.assert_allclose(result[lowered], corrected[name][lowered], rtol=2e-7)
    assert total >= 20  # the crops hold enough corrected pixels to mean something


@pytest.mark.nas
def test_full_frame_replacement_matches_pixinsight() -> None:
    directory = m13.SESSION / "output" / "calibrated" / "light"
    name = "2018-09-12-20_24_07"
    calibrated = np.asarray(read_xisf(directory / f"{name}_c.xisf")[0].data)
    corrected = np.asarray(read_xisf(directory / "cosmetized" / f"{name}_c_cc.xisf")[0].data)
    lowered = lowered_by_pixinsight(calibrated, corrected)

    result = replace_from_neighbours(calibrated, lowered, mosaic=True)

    assert lowered.sum() == 39
    np.testing.assert_allclose(result[lowered], corrected[lowered], rtol=2e-7)
