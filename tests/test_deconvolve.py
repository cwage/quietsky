from pathlib import Path

import numpy as np
import pytest

import synthetic
from quietsky.cli import main
from quietsky.deconvolve import (
    deconvolve,
    denoise,
    gaussian_blur,
    measure_psf,
    richardson_lucy,
)
from quietsky.frame import Frame, load, save_fits
from quietsky.noise import evaluate_noise


def blurred_field(extra_sigma: float) -> tuple[np.ndarray, float]:
    """A synthetic star field blurred further; returns it and its total star sigma."""
    image, _, _ = synthetic.star_field()
    if extra_sigma > 0:
        # Blurred with reflected margins, so the Fourier blur does not wrap
        # one edge of the field round to the other.
        padded = np.pad(image, 32, mode="reflect")
        image = gaussian_blur(padded, extra_sigma)[32:-32, 32:-32]
        # The blur smooths the field's noise too; fresh noise keeps it a
        # noisy image, as the noise estimators expect.
        image += np.random.default_rng(5).normal(0, synthetic.NOISE, image.shape).astype(np.float32)
    return image, float(np.hypot(synthetic.PSF_SIGMA, extra_sigma))


def test_measure_psf_recovers_the_width_of_synthetic_stars() -> None:
    for extra in (0.0, 1.5, 2.5):
        image, sigma = blurred_field(extra)
        assert measure_psf(image) == pytest.approx(sigma, rel=0.08)


def test_measure_psf_ignores_saturated_and_crowded_stars() -> None:
    image, sigma = blurred_field(1.5)
    # A flat-topped star and a close pair would both measure too wide.
    image[100:105, 100:105] = 1.0
    image[200, 300] += 0.5
    image[200, 304] += 0.5
    assert measure_psf(image) == pytest.approx(sigma, rel=0.08)


def test_denoise_lowers_the_noise_and_keeps_the_sky_level() -> None:
    rng = np.random.default_rng(3)
    rows = np.arange(300, dtype=np.float32)[:, None]
    sky = 0.02 + 0.00002 * rows + np.zeros((300, 400), np.float32)
    image = (sky + rng.normal(0, 0.001, sky.shape)).astype(np.float32)

    cleaned = denoise(image, evaluate_noise(image).sigma)

    assert evaluate_noise(cleaned).sigma < 0.5 * evaluate_noise(image).sigma
    assert np.abs(cleaned - sky).mean() < np.abs(image - sky).mean()
    assert cleaned.mean() == pytest.approx(image.mean(), rel=1e-3)


def test_deconvolve_tightens_stars_keeps_their_light_and_leaves_the_sky() -> None:
    image, sigma = blurred_field(2.0)
    noise = evaluate_noise(image).sigma

    sharpened = deconvolve(image, sigma, iterations=20, noise=noise)

    assert measure_psf(sharpened) < 0.7 * sigma
    # The light of a star, within a window, stays the same.
    _, x, y = synthetic.star_field()
    row, column = int(round(y[0])), int(round(x[0]))
    window = (slice(row - 10, row + 11), slice(column - 10, column + 11))
    sky = np.median(image)
    assert (sharpened[window] - sky).sum() == pytest.approx((image[window] - sky).sum(), rel=0.02)
    # Empty sky is left alone, but for the odd noise peak that passes for signal.
    distance = np.hypot(
        np.arange(image.shape[1])[None, :] - x[:, None, None],
        np.arange(image.shape[0])[None, :, None] - y[:, None, None],
    ).min(axis=0)
    empty = distance > 15
    change = sharpened[empty] - image[empty]
    assert np.mean(change != 0) < 0.03
    assert np.abs(change).max() < 5 * noise


def test_deconvolve_does_not_dig_dark_rings_round_stars() -> None:
    # A star with wider wings than the profile deconvolved with it.
    image, sigma = blurred_field(2.0)
    rows, columns = np.mgrid[: image.shape[0], : image.shape[1]]
    wide = 0.3 * np.exp(-((rows - 200) ** 2 + (columns - 300) ** 2) / (2 * (2 * sigma) ** 2))
    image = (image + wide).astype(np.float32)
    noise = evaluate_noise(image).sigma

    sharpened = deconvolve(image, sigma, iterations=20, noise=noise)

    ring = image[180:221, 280:321]
    assert sharpened[180:221, 280:321].min() >= ring.min() - 1e-6


def test_richardson_lucy_is_the_lunar_one() -> None:
    image, sigma = blurred_field(1.0)
    assert richardson_lucy(image, sigma, 5).shape == image.shape


def test_sharpen_command_writes_a_sharper_frame(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image, sigma = blurred_field(2.0)
    colour = np.stack([image, image * 0.8, image * 0.6], axis=-1).astype(np.float32)
    source = tmp_path / "light.fits"
    output = tmp_path / "sharp.fits"
    save_fits(source, Frame(colour, {"NCOMBINE": 7}))

    png = tmp_path / "sharp.png"
    main(["sharpen", str(source), "-o", str(output), "--iterations", "10", "--png", str(png)])

    result = load(output)
    assert result.data.shape == colour.shape
    assert result.header["NCOMBINE"] == 7
    assert measure_psf(result.data[..., 0]) < measure_psf(colour[..., 0])
    assert png.exists()
    assert "R: stars sigma" in capsys.readouterr().out


def test_sharpen_command_gives_every_colour_the_same_star_profile(tmp_path: Path) -> None:
    sharp, _ = blurred_field(0.0)
    soft, _ = blurred_field(2.0)
    colour = np.stack([soft, sharp, sharp], axis=-1).astype(np.float32)
    source = tmp_path / "light.fits"
    output = tmp_path / "sharp.fits"
    save_fits(source, Frame(colour))

    main(["sharpen", str(source), "-o", str(output), "--iterations", "10"])

    widths = [measure_psf(load(output).data[..., c]) for c in range(3)]
    assert max(widths) == pytest.approx(min(widths), rel=0.05)
    assert widths[0] < measure_psf(soft)
