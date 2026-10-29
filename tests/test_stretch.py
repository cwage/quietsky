import numpy as np
import pytest

from quietsky.stretch import TARGET_BACKGROUND, autostretch, midtones


def linear_sky(median: float, seed: int = 1) -> np.ndarray:
    """A faint, noisy background with a few bright stars, as a stack looks."""
    sky = np.random.default_rng(seed).normal(median, median / 50, (64, 64)).astype(np.float32)
    sky[::16, ::16] = 0.8
    return sky


def test_midtones_fixes_the_ends_and_sends_the_balance_to_half() -> None:
    x = np.array([0.0, 0.2, 1.0], dtype=np.float32)

    assert midtones(x, 0.2).tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_autostretch_brings_the_background_to_the_target() -> None:
    stretched = autostretch(linear_sky(0.001))

    assert np.median(stretched) == pytest.approx(TARGET_BACKGROUND, abs=0.01)
    assert stretched.min() >= 0
    assert stretched.max() <= 1
    assert stretched[0, 0] > 0.95  # stars stay bright


def test_autostretch_treats_colour_channels_separately() -> None:
    colour = np.stack([linear_sky(0.001), linear_sky(0.004, seed=2)], axis=-1)

    stretched = autostretch(colour)

    assert stretched.shape == colour.shape
    assert np.median(stretched[..., 0]) == pytest.approx(TARGET_BACKGROUND, abs=0.01)
    assert np.median(stretched[..., 1]) == pytest.approx(TARGET_BACKGROUND, abs=0.01)


def test_autostretch_can_take_its_parameters_from_a_reference() -> None:
    rng = np.random.default_rng(0)
    noisy = (0.02 + rng.normal(0, 0.001, (50, 60))).astype(np.float32)
    quiet = (0.02 + rng.normal(0, 0.000001, noisy.shape)).astype(np.float32)
    quiet[10, 10] = 0.021
    # On its own, a nearly flat image is stretched to its own vanishing noise
    # and a small step becomes bright; under the noisy reference's stretch it
    # stays small, as it would in the noisy image.
    alone = autostretch(quiet)
    referenced = autostretch(quiet, reference=noisy)
    assert alone[10, 10] - alone[0, 0] > 0.5
    assert referenced[10, 10] - referenced[0, 0] < 0.2
    assert np.allclose(autostretch(noisy, reference=noisy), autostretch(noisy))
