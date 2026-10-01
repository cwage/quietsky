import numpy as np
import pytest

import m13
from starkiller.noise import evaluate_noise, noise_mrs
from starkiller.wavelets import B3_NOISE_PER_LAYER, b3_layers
from starkiller.xisf import read_xisf

SIGMA = 0.002


def noisy(shape: tuple[int, int] = (256, 256), seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.1, SIGMA, shape).astype(np.float32)


def test_layers_and_residual_add_back_up_to_the_image() -> None:
    image = noisy()

    layers, residual = b3_layers(image, 4)

    assert len(layers) == 4
    np.testing.assert_allclose(sum(layers) + residual, image, atol=1e-6)


def test_white_noise_spreads_over_the_layers_as_tabulated() -> None:
    layers, _ = b3_layers(np.random.default_rng(1).normal(0, 1, (512, 512)), 4)

    spread = [float(layer.std()) for layer in layers]

    assert spread == pytest.approx(B3_NOISE_PER_LAYER[:4], rel=0.03)


def test_noise_of_a_flat_noisy_field_is_its_sigma() -> None:
    noise = noise_mrs(noisy())

    assert noise.sigma == pytest.approx(SIGMA, rel=0.02)
    assert noise.fraction > 0.9


def test_noise_is_not_inflated_by_stars_or_gradients() -> None:
    image = noisy()
    rows, columns = np.mgrid[:256, :256]
    image += (columns / 256 * 0.05).astype(np.float32)  # a sky gradient
    for row, column in np.random.default_rng(2).integers(8, 248, (40, 2)):
        distance2 = (rows - row) ** 2 + (columns - column) ** 2
        image += (0.5 * np.exp(-distance2 / 8)).astype(np.float32)  # a star

    noise = evaluate_noise(image)

    assert noise.sigma == pytest.approx(SIGMA, rel=0.03)
    assert image.std() > 5 * noise.sigma


@pytest.mark.nas
def test_noise_matches_pixinsight_on_debayered_lights() -> None:
    reference = m13.debayer_noise()
    directory = m13.SESSION / "output" / "calibrated" / "light" / "debayered"

    for name in sorted(reference)[::20]:
        data = read_xisf(directory / name)[0].data

        for channel, (sigma, fraction) in enumerate(reference[name]):
            noise = evaluate_noise(data[..., channel])

            # PixInsight prints four digits; ours can differ in the last one.
            assert noise.sigma == pytest.approx(sigma, rel=3e-4)
            assert noise.fraction == pytest.approx(fraction, abs=2e-4)
