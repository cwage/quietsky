import numpy as np
import pytest

import m13
from starkiller.calibrate import calibrate, optimization_dark, optimize_dark, unit_flat
from starkiller.frame import load_raw
from starkiller.xisf import read_xisf

# PixInsight logs the dark scaling factor to three decimals, so its output
# can differ from ours by that much of the dark, plus float32 rounding.
SCALE_PRECISION = 0.0005
ROUNDING = 8e-9


def synthetic_session(scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A frame whose thermal signal is scale times that of the master dark."""
    rng = np.random.default_rng(1)
    shape = (256, 256)
    bias = np.full(shape, 0.002, np.float32)
    thermal = np.zeros(shape, np.float32)
    hot = rng.random(shape) < 0.05
    thermal[hot] = rng.uniform(0.0005, 0.005, int(hot.sum()))
    dark = bias + thermal
    sky = rng.normal(0.01, 0.0001, shape).astype(np.float32)
    return bias + sky + np.float32(scale) * thermal, bias, dark


def test_calibrate_subtracts_bias_and_scaled_dark() -> None:
    bias = np.full((2, 2), 0.002, np.float32)
    dark = bias + np.array([[0, 0.001], [0, 0]], np.float32)
    frame = np.full((2, 2), 0.01, np.float32)

    result = calibrate(frame, bias, dark, dark_scale=2.0)

    np.testing.assert_allclose(result, [[0.008, 0.006], [0.008, 0.008]], rtol=1e-6)


def test_calibrate_without_a_bias_subtracts_the_whole_dark() -> None:
    dark = np.array([[0.002, 0.003]], np.float32)  # bias and thermal signal together
    frame = np.array([[0.012, 0.013]], np.float32)

    np.testing.assert_allclose(calibrate(frame, dark=dark), [[0.01, 0.01]], rtol=1e-6)


def test_calibrate_without_a_dark_subtracts_only_the_bias() -> None:
    bias = np.full((1, 2), 0.002, np.float32)
    frame = np.array([[0.012, 0.013]], np.float32)

    np.testing.assert_allclose(calibrate(frame, bias), [[0.01, 0.011]], rtol=1e-6)


def test_calibrate_with_no_masters_returns_the_frame() -> None:
    frame = np.array([[0.012, 0.013]], np.float32)

    np.testing.assert_array_equal(calibrate(frame), frame)


def test_calibrate_with_a_flat_evens_out_vignetting() -> None:
    rows, columns = np.mgrid[:64, :64]
    vignette = (1 - ((rows - 32) ** 2 + (columns - 32) ** 2) / 4000).astype(np.float32)
    bias = np.full((64, 64), 0.002, np.float32)
    frame = bias + np.float32(0.01) * vignette  # an even sky seen through the optics
    flat = np.float32(0.3) * vignette

    result = calibrate(frame, bias, bias, dark_scale=1.0, flat=unit_flat(flat))

    np.testing.assert_allclose(result, 0.01 * vignette.mean(), rtol=1e-5)


def test_unit_flat_has_a_mean_of_one() -> None:
    flat = np.random.default_rng(1).uniform(0.2, 0.4, (32, 32)).astype(np.float32)

    assert unit_flat(flat).mean() == pytest.approx(1.0, rel=1e-6)


def test_optimization_dark_keeps_only_pixels_with_thermal_signal() -> None:
    _, bias, dark = synthetic_session(1.0)

    kept, threshold = optimization_dark(dark - bias, mosaic=False)

    assert threshold == pytest.approx(0.0, abs=1e-9)
    assert (kept > 0).mean() == pytest.approx(0.05, abs=0.01)


@pytest.mark.parametrize("scale", [0.5, 1.3, 2.0])
@pytest.mark.parametrize("mosaic", [False, True])
def test_optimize_dark_finds_the_scale_of_the_thermal_signal(scale: float, mosaic: bool) -> None:
    frame, bias, dark = synthetic_session(scale)

    assert optimize_dark(frame, bias, dark, mosaic) == pytest.approx(scale, abs=0.02)


def test_calibrated_flats_match_pixinsight() -> None:
    bias = m13.master("bias")["integration"]
    dark = m13.master("dark")["integration"]
    scales = m13.dark_scales("flat")
    flats = m13.as_pixinsight_loaded(m13.frames("flat"))
    reference = m13.calibrated_flats()

    for name, flat in zip(m13.frame_names("flat"), flats, strict=True):
        result = calibrate(flat, bias, dark, scales[name])

        allowed = SCALE_PRECISION * np.abs(dark - bias) + ROUNDING
        assert (np.abs(result - reference[name]) <= allowed).all()


def test_calibrated_lights_match_pixinsight() -> None:
    bias = m13.master("bias")["integration"]
    dark = m13.master("dark")["integration"]
    flat = (m13.master("flat")["integration"] / m13.master_flat_mean()).astype(np.float32)
    scales = m13.dark_scales("light")
    lights = dict(zip(m13.frame_names("light"), m13.frames("light"), strict=True))

    for name, reference in m13.calibrated_lights().items():
        light = m13.as_pixinsight_loaded(lights[name])

        result = calibrate(light, bias, dark, scales[name], flat)

        allowed = SCALE_PRECISION * np.abs(dark - bias) / flat + 1e-6 * reference
        assert (np.abs(result - reference) <= allowed).all()


@pytest.mark.nas
def test_full_frame_calibrated_light_matches_pixinsight() -> None:
    output = m13.SESSION / "output"
    bias = np.asarray(read_xisf(output / "master" / "bias-BINNING_1.xisf")[0].data)
    dark = np.asarray(read_xisf(output / "master" / "dark-BINNING_1-EXPTIME_30.xisf")[0].data)
    flat = unit_flat(
        np.asarray(read_xisf(output / "master" / "flat-FILTER_m13-BINNING_1.xisf")[0].data)
    )
    name = "2018-09-12-20_45_50"
    light = m13.as_pixinsight_loaded(load_raw(m13.SESSION / "light" / f"{name}.arw").data)
    reference = read_xisf(output / "calibrated" / "light" / f"{name}_c.xisf")[0].data

    result = calibrate(light, bias, dark, m13.dark_scales("light")[name], flat)

    allowed = SCALE_PRECISION * np.abs(dark - bias) / flat + 1e-6 * reference
    assert (np.abs(result - reference) <= allowed).all()


@pytest.mark.nas
def test_full_frame_calibrated_flat_matches_pixinsight() -> None:
    output = m13.SESSION / "output"
    bias = np.asarray(read_xisf(output / "master" / "bias-BINNING_1.xisf")[0].data)
    dark = np.asarray(read_xisf(output / "master" / "dark-BINNING_1-EXPTIME_30.xisf")[0].data)
    name = "2018-09-12-21_15_41"
    flat = m13.as_pixinsight_loaded(load_raw(m13.SESSION / "flat" / f"{name}.arw").data)
    reference = read_xisf(output / "calibrated" / "flat" / f"{name}_c.xisf")[0].data

    result = calibrate(flat, bias, dark, m13.dark_scales("flat")[name])

    allowed = SCALE_PRECISION * np.abs(dark - bias) + ROUNDING
    assert (np.abs(result - reference) <= allowed).all()


@pytest.mark.nas
def test_optimization_threshold_matches_pixinsight() -> None:
    output = m13.SESSION / "output"
    bias = np.asarray(read_xisf(output / "master" / "bias-BINNING_1.xisf")[0].data)
    dark = np.asarray(read_xisf(output / "master" / "dark-BINNING_1-EXPTIME_30.xisf")[0].data)

    kept, threshold = optimization_dark(dark - bias, mosaic=True)

    # "Td0 = 0.00000819 (54389 px = 1.795%)" in PixInsight's log.
    assert f"{threshold:.8f}" == "0.00000819"
    assert int((kept > 0).sum()) == 54389


@pytest.mark.nas
def test_optimized_dark_scale_is_close_to_pixinsight_on_lights() -> None:
    output = m13.SESSION / "output"
    bias = np.asarray(read_xisf(output / "master" / "bias-BINNING_1.xisf")[0].data)
    dark = np.asarray(read_xisf(output / "master" / "dark-BINNING_1-EXPTIME_30.xisf")[0].data)
    scales = m13.dark_scales("light")

    for name in ("2018-09-12-20_24_07", "2018-09-12-20_45_50", "2018-09-12-20_56_55"):
        light = m13.as_pixinsight_loaded(load_raw(m13.SESSION / "light" / f"{name}.arw").data)

        scale = optimize_dark(light, bias, dark, mosaic=True)

        # The minimum is only defined to within several percent.
        assert scale == pytest.approx(scales[name], rel=0.12)
