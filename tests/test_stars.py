import numpy as np
import pytest

import m13
import synthetic
from starkiller.stars import detect_stars
from starkiller.xisf import read_xisf


def nearest(x: np.ndarray, y: np.ndarray, to_x: np.ndarray, to_y: np.ndarray) -> np.ndarray:
    """Distance from each (to_x, to_y) to the closest (x, y)."""
    distance: np.ndarray = np.hypot(x[:, None] - to_x[None, :], y[:, None] - to_y[None, :]).min(0)
    return distance


def test_finds_every_star_and_nothing_else() -> None:
    image, x, y = synthetic.star_field()

    stars = detect_stars(image)

    assert len(stars) == len(x)
    assert nearest(stars.x, stars.y, x, y).max() < 1.0


def test_positions_are_good_to_a_tenth_of_a_pixel() -> None:
    image, x, y = synthetic.star_field()

    stars = detect_stars(image)

    error = nearest(stars.x, stars.y, x, y)
    assert np.median(error) < 0.05
    assert np.percentile(error, 90) < 0.1


def test_stars_come_brightest_first() -> None:
    stars = detect_stars(synthetic.star_field()[0])

    assert (np.diff(stars.flux) <= 0).all()


def test_a_hot_pixel_is_not_a_star() -> None:
    image, x, _ = synthetic.star_field()
    image[100, 100] += 0.2

    stars = detect_stars(image)

    assert len(stars) == len(x)


def test_a_saturated_star_is_found_once() -> None:
    x, y = np.array([100.3, 60.0]), np.array([80.6, 40.0])
    image = synthetic.render((160, 200), x, y, np.array([40.0, 1.0]))
    image = np.minimum(image, 0.5)  # the bright star now has a flat top

    stars = detect_stars(image)

    assert len(stars) == 2
    assert nearest(stars.x, stars.y, x, y).max() < 0.5


def test_colour_images_are_searched_in_their_combined_light() -> None:
    image, x, y = synthetic.star_field()
    colour = np.stack([image, 0.5 * image, 0.25 * image], axis=-1)

    stars = detect_stars(colour)

    assert len(stars) == len(x)
    assert np.median(nearest(stars.x, stars.y, x, y)) < 0.05


@pytest.mark.nas
def test_reference_light_has_stars_in_numbers_like_pixinsight() -> None:
    directory = m13.SESSION / "output" / "calibrated" / "light" / "debayered"
    light = read_xisf(directory / "2018-09-12-20_22_50_c_cc_d.xisf")[0].data

    # PixInsight found 445 in this frame with its own detector and settings.
    assert 300 < len(detect_stars(light, threshold=15)) < 700
