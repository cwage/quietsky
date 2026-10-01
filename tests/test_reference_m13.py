"""Compare our output with PixInsight's on the M13 session."""

import numpy as np
import pytest

import m13
from starkiller.frame import load_raw
from starkiller.integrate import integrate
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
