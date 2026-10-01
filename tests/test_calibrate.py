import numpy as np
import pytest

import m13
from starkiller.calibrate import calibrate, optimization_dark, optimize_dark
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
