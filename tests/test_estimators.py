import numpy as np
import pytest

import m13
from starkiller.estimators import biweight_midvariance, ikss, mad
from starkiller.frame import load_raw
from starkiller.xisf import read_xisf


def background(sigma: float = 0.002, size: int = 200_000) -> np.ndarray:
    return np.random.default_rng(1).normal(0.1, sigma, size)


def test_mad_is_the_median_distance_from_the_centre() -> None:
    assert mad(np.array([1.0, 2.0, 4.0, 8.0, 100.0]), center=4.0) == 3.0


def test_biweight_midvariance_of_normal_data_is_its_variance() -> None:
    values = background()

    variance = biweight_midvariance(values, 0.1, mad(values, 0.1))

    assert variance == pytest.approx(0.002**2, rel=0.02)


def test_ikss_recovers_level_and_noise_of_normal_data() -> None:
    location, scale = ikss(background())

    assert location == pytest.approx(0.1, abs=1e-5)
    assert scale == pytest.approx(0.002, rel=0.02)


def test_ikss_ignores_stars() -> None:
    values = background()
    values[:10_000] += np.random.default_rng(2).uniform(0.05, 0.8, 10_000)

    location, scale = ikss(values)

    assert location == pytest.approx(0.1, abs=1e-4)
    assert scale == pytest.approx(0.002, rel=0.05)
    assert values.std() > 20 * scale


def test_ikss_of_constant_data_has_no_scale() -> None:
    assert ikss(np.full(100, 0.25)) == (0.25, 0.0)


def test_ikss_of_nothing_is_zero() -> None:
    assert ikss(np.array([])) == (0.0, 0.0)


@pytest.mark.nas
def test_ikss_matches_pixinsight_on_calibrated_flats() -> None:
    history = m13.master_history("flat")
    locations = m13.estimates_per_frame(history, "locationEstimates")
    scales = m13.estimates_per_frame(history, "scaleEstimates")
    directory = m13.SESSION / "output" / "calibrated" / "flat"

    for index in (0, 3, 17):
        flat = directory / m13.master_inputs("flat")[index]
        location, scale = ikss(read_xisf(flat)[0].data)

        # PixInsight records five significant digits.
        assert location == pytest.approx(locations[index][0], rel=5e-5)
        assert scale == pytest.approx(scales[index][0], rel=5e-5)


@pytest.mark.nas
def test_ikss_matches_pixinsight_on_a_registered_light() -> None:
    history = m13.master_history("light")
    locations = m13.estimates_per_frame(history, "locationEstimates")
    scales = m13.estimates_per_frame(history, "scaleEstimates")
    index = 5  # a frame with black borders left by registration
    light = m13.SESSION / "output" / "registered" / "m13" / m13.master_inputs("light")[index]
    data = read_xisf(light)[0].data

    for channel in range(3):
        location, scale = ikss(data[..., channel])

        assert location == pytest.approx(locations[index][channel], rel=5e-5)
        assert scale == pytest.approx(scales[index][channel], rel=5e-5)


@pytest.mark.nas
def test_mad_matches_pixinsight_on_a_dark() -> None:
    scales = m13.estimates_per_frame(m13.master_history("dark"), "scaleEstimates")
    dark = sorted((m13.SESSION / "dark").glob("*.arw"))[0]
    data = m13.as_pixinsight_loaded(load_raw(dark).data)

    assert mad(data, float(np.median(data))) == pytest.approx(scales[0][0], rel=5e-5)
