"""Compare our output with PixInsight's on the M13 session."""

import numpy as np
import pytest

import m13
from starkiller.frame import load_raw
from starkiller.integrate import flux_gains, integrate
from starkiller.xisf import read_xisf


@pytest.mark.parametrize("kind", ["bias", "dark"])
def test_master_matches_pixinsight(kind: str) -> None:
    # Bias and dark integration works one pixel stack at a time, so a crop of
    # PixInsight's full-frame master is what integrating the crops must give.
    stack = m13.as_pixinsight_loaded(m13.frames(kind))
    reference = m13.master(kind)

    result = integrate(stack)

    count = len(stack)
    np.testing.assert_array_equal(result.image, reference["integration"])
    np.testing.assert_array_equal(
        np.rint(result.rejection_low * count), np.rint(reference["rejection_low"] * count)
    )
    np.testing.assert_array_equal(
        np.rint(result.rejection_high * count), np.rint(reference["rejection_high"] * count)
    )


@pytest.mark.nas
@pytest.mark.parametrize(
    ("kind", "directory", "master"),
    [
        ("bias", "offset", "bias-BINNING_1.xisf"),
        ("dark", "dark", "dark-BINNING_1-EXPTIME_30.xisf"),
    ],
)
def test_full_frame_master_matches_pixinsight(kind: str, directory: str, master: str) -> None:
    paths = sorted((m13.SESSION / directory).glob("*.arw"))
    stack = np.stack([m13.as_pixinsight_loaded(load_raw(path).data) for path in paths])
    reference = read_xisf(m13.SESSION / "output" / "master" / master)[0]

    result = integrate(stack)

    np.testing.assert_array_equal(result.image, reference.data)
    history = m13.master_history(kind)
    assert result.frame_rejected_low.tolist() == m13.rejected_per_frame(history, "Low")
    assert result.frame_rejected_high.tolist() == m13.rejected_per_frame(history, "High")


def flat_gains() -> np.ndarray:
    """Brightness-matching factors for the flats, from the full-frame levels."""
    locations = m13.calibrated_flat_locations()
    ordered = np.array(
        [locations[name.removesuffix("_c.xisf")] for name in m13.master_inputs("flat")]
    )
    gains: np.ndarray = ordered[0] / ordered
    return gains


def test_master_flat_matches_pixinsight() -> None:
    calibrated = m13.calibrated_flats()
    stack = np.stack(
        [calibrated[name.removesuffix("_c.xisf")] for name in m13.master_inputs("flat")]
    )
    reference = m13.master("flat")

    result = integrate(stack, gains=flat_gains())

    count = len(stack)
    np.testing.assert_array_equal(result.image, reference["integration"])
    np.testing.assert_array_equal(
        np.rint(result.rejection_low * count), np.rint(reference["rejection_low"] * count)
    )
    np.testing.assert_array_equal(
        np.rint(result.rejection_high * count), np.rint(reference["rejection_high"] * count)
    )


@pytest.mark.nas
def test_full_frame_master_flat_matches_pixinsight() -> None:
    directory = m13.SESSION / "output" / "calibrated" / "flat"
    stack = np.stack([read_xisf(directory / name)[0].data for name in m13.master_inputs("flat")])
    reference = read_xisf(m13.SESSION / "output" / "master" / "flat-FILTER_m13-BINNING_1.xisf")[0]

    gains = flux_gains(stack)
    result = integrate(stack, gains=gains)

    np.testing.assert_allclose(gains, flat_gains(), rtol=1e-12)
    # Bit-identical except at a handful of pixels where a value sits so close
    # to the rejection limit that PixInsight decides the other way: 8 of the
    # 12 million when this was written.
    assert (result.image != reference.data).sum() <= 10
    history = m13.master_history("flat")
    low = np.array(m13.rejected_per_frame(history, "Low"))
    high = np.array(m13.rejected_per_frame(history, "High"))
    assert np.abs(result.frame_rejected_low - low).max() <= 2
    assert np.abs(result.frame_rejected_high - high).max() <= 2
