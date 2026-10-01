"""Compare our output with PixInsight's on the M13 session."""

from pathlib import Path

import numpy as np
import pytest

import m13
from quietsky.frame import DiskStack, Frame, load_raw, save_fits
from quietsky.integrate import (
    Integration,
    Normalization,
    flux_normalization,
    frame_estimates,
    integrate,
    level_and_scale_normalization,
    noise_weights,
)
from quietsky.xisf import read_xisf


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


def flat_normalization() -> Normalization:
    """Brightness matching for the flats, from the full-frame levels."""
    locations = m13.calibrated_flat_locations()
    names = [name.removesuffix("_c.xisf") for name in m13.master_inputs("flat")]
    return flux_normalization(np.array([locations[name] for name in names]))


def test_master_flat_matches_pixinsight() -> None:
    calibrated = m13.calibrated_flats()
    stack = np.stack(
        [calibrated[name.removesuffix("_c.xisf")] for name in m13.master_inputs("flat")]
    )
    reference = m13.master("flat")

    result = integrate(stack, normalization=flat_normalization())

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

    normalization = flux_normalization(frame_estimates(stack)[0])
    result = integrate(stack, normalization=normalization)

    np.testing.assert_allclose(normalization.multiply, flat_normalization().multiply, rtol=1e-12)
    # Bit-identical except at a handful of pixels where a value sits so close
    # to the rejection limit that PixInsight decides the other way: 8 of the
    # 12 million when this was written.
    assert (result.image != reference.data).sum() <= 10
    history = m13.master_history("flat")
    low = np.array(m13.rejected_per_frame(history, "Low"))
    high = np.array(m13.rejected_per_frame(history, "High"))
    assert np.abs(result.frame_rejected_low - low).max() <= 2
    assert np.abs(result.frame_rejected_high - high).max() <= 2


def integrate_light_channel(stack: np.ndarray | DiskStack, channel: int) -> Integration:
    """Integrate one colour channel of registered lights the way PixInsight's run did."""
    normalization = level_and_scale_normalization(
        m13.registered_light_estimates("location")[:, channel],
        m13.registered_light_estimates("scale")[:, channel],
    )
    weights = noise_weights(m13.registered_light_estimates("noise")[:, channel], normalization)
    return integrate(stack, normalization=normalization, weights=weights, valid_range=(0.0, 0.98))


def test_master_light_matches_pixinsight() -> None:
    lights = m13.registered_lights()
    reference = m13.master("light")
    count = len(lights)

    for channel in range(3):
        result = integrate_light_channel(lights[..., channel], channel)

        # Equal to the last bit almost everywhere; the weighted sum can round
        # differently by one unit in the last place.
        expected = reference["integration"][m13.CORE][..., channel]
        np.testing.assert_allclose(result.image, expected, rtol=2e-7)
        assert (result.image == expected).mean() > 0.999
        for ours, theirs in (
            (result.rejection_low, reference["rejection_low"]),
            (result.rejection_high, reference["rejection_high"]),
        ):
            np.testing.assert_array_equal(
                np.rint(ours * count), np.rint(theirs[m13.CORE][..., channel] * count)
            )


def test_light_weights_match_pixinsight() -> None:
    recorded = np.array(m13.estimates_per_frame(m13.master_history("light"), "imageWeights"))

    for channel in range(3):
        normalization = level_and_scale_normalization(
            m13.registered_light_estimates("location")[:, channel],
            m13.registered_light_estimates("scale")[:, channel],
        )
        weights = noise_weights(m13.registered_light_estimates("noise")[:, channel], normalization)

        # PixInsight records six digits.
        np.testing.assert_allclose(weights, recorded[:, channel], rtol=5e-6)


@pytest.mark.nas
@pytest.mark.parametrize("top", [0, 1300])
def test_bands_of_the_full_master_light_match_pixinsight(top: int) -> None:
    # The whole frame takes minutes; two bands of rows across the full width
    # cover the black borders left by registration and the cluster.
    rows = slice(top, top + 32)
    directory = m13.SESSION / "output" / "registered" / "m13"
    frames = [read_xisf(directory / name)[0].data for name in m13.master_inputs("light")]
    master = m13.SESSION / "output" / "master" / "light-FILTER_m13-BINNING_1.xisf"
    reference = read_xisf(master)[0].data

    for channel in range(3):
        stack = np.stack([np.asarray(frame[rows, :, channel]) for frame in frames])

        result = integrate_light_channel(stack, channel)

        expected = reference[rows, :, channel]
        np.testing.assert_allclose(result.image, expected, rtol=2e-7, atol=1e-12)
        assert (result.image == expected).mean() > 0.999


def test_integrating_from_disk_gives_the_same_as_from_memory(tmp_path: Path) -> None:
    lights = m13.registered_lights()
    paths = [tmp_path / f"light{index:02d}.fits" for index in range(len(lights))]
    for path, light in zip(paths, lights, strict=True):
        save_fits(path, Frame(light))

    for channel in range(3):
        from_disk = integrate_light_channel(DiskStack(paths, channel), channel)
        from_memory = integrate_light_channel(lights[..., channel], channel)

        np.testing.assert_array_equal(from_disk.image, from_memory.image)
        np.testing.assert_array_equal(from_disk.rejection_high, from_memory.rejection_high)
        assert from_disk.frame_rejected_low.tolist() == from_memory.frame_rejected_low.tolist()
