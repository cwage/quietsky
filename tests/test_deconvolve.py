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
    star_profile,
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


MOFFAT_ALPHA, MOFFAT_BETA = 2.0, 2.5


def moffat(radius: np.ndarray) -> np.ndarray:
    """A Moffat profile, as seeing gives: a Gaussian-like core with wings that fall as a power."""
    result: np.ndarray = (1 + (radius / MOFFAT_ALPHA) ** 2) ** -MOFFAT_BETA
    return result


def moffat_field() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The synthetic field's stars drawn with a Moffat profile instead of a Gaussian."""
    _, x, y = synthetic.star_field()
    rng = np.random.default_rng(1)
    flux = 10 ** rng.uniform(-1.0, 0.3, len(x))
    rows, columns = np.mgrid[:400, :600]
    image = 0.01 + 0.00002 * columns
    total = np.pi * MOFFAT_ALPHA**2 / (MOFFAT_BETA - 1)
    for cx, cy, f in zip(x, y, flux, strict=True):
        image = image + f * moffat(np.hypot(rows - cy, columns - cx)) / total
    image += np.random.default_rng(0).normal(0, synthetic.NOISE, image.shape)
    return image.astype(np.float32), x, y


def test_star_profile_measures_the_profile_as_it_is() -> None:
    image, _, _ = moffat_field()
    kernel = star_profile(image)
    half = kernel.shape[0] // 2
    offsets = np.arange(-half, half + 1)
    radius = np.hypot(offsets[:, None], offsets[None, :])
    expected = moffat(radius)
    expected[radius > half] = 0
    expected /= expected.sum()
    assert kernel.sum() == pytest.approx(1.0)
    # The peak is a little soft, averaged over the innermost half-pixel bin;
    # the wings, which a Gaussian fit would miss, are right to a tenth.
    assert np.abs(kernel - expected).max() < 0.08 * expected.max()
    wings = (radius >= 2) & (radius <= 7)
    assert kernel[wings] == pytest.approx(expected[wings], rel=0.1)
    gaussian = np.exp(-(radius**2) / (2 * measure_psf(image) ** 2))
    gaussian /= gaussian.sum()
    assert kernel[radius == 5].mean() > 3 * gaussian[radius == 5].mean()


def test_stars_keep_their_wings_under_a_tightened_core() -> None:
    """Richardson-Lucy empties the near wings into the core and leaves the far halo: a ring.

    Within a star's reach nothing falls below the original, so the core
    brightens over wings that stay, whichever kernel is used.
    """
    image, x, y = moffat_field()
    noise = evaluate_noise(image).sigma
    sigma = measure_psf(image)
    rows, columns = np.mgrid[: image.shape[0], : image.shape[1]]
    for kernel in (None, star_profile(image)):
        sharpened = deconvolve(image, sigma, iterations=20, noise=noise, kernel=kernel)
        assert measure_psf(sharpened) < 0.8 * sigma
        for cx, cy in list(zip(x, y, strict=True))[:10]:
            radius = np.hypot(rows - cy, columns - cx)
            # The near wings, where the ring would be, lose nothing on average.
            wings = (radius >= 1.5 * sigma) & (radius <= 5 * sigma)
            assert (sharpened - image)[wings].mean() > -noise
            assert sharpened[radius < 1].max() > 1.2 * image[radius < 1].max()


def test_richardson_lucy_takes_a_kernel_as_well_as_a_sigma() -> None:
    image, sigma = blurred_field(1.0)
    half = 7
    offsets = np.arange(-half, half + 1)
    kernel = np.exp(-(offsets[:, None] ** 2 + offsets[None, :] ** 2) / (2 * sigma**2))
    kernel /= kernel.sum()
    by_sigma = richardson_lucy(image, sigma, 5)
    by_kernel = richardson_lucy(image, kernel, 5)
    assert np.abs(by_kernel - by_sigma).max() < 1e-3 * image.max()


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


def test_deconvolve_tightens_stars_loses_no_light_and_leaves_the_sky() -> None:
    image, sigma = blurred_field(2.0)
    noise = evaluate_noise(image).sigma

    sharpened = deconvolve(image, sigma, iterations=20, noise=noise)

    assert measure_psf(sharpened) < 0.85 * sigma
    # The light of a star, within a window, is kept: its wings stay under
    # the tightened core, so it gains a little rather than losing any.
    _, x, y = synthetic.star_field()
    row, column = int(round(y[0])), int(round(x[0]))
    window = (slice(row - 10, row + 11), slice(column - 10, column + 11))
    sky = np.median(image)
    before, after = (image[window] - sky).sum(), (sharpened[window] - sky).sum()
    assert before <= after < 1.5 * before
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
    out = capsys.readouterr().out
    assert "R: stars sigma" in out and "of a star's light lies beyond" in out
    main(["sharpen", str(source), "-o", str(output), "--iterations", "10", "--kernel", "gaussian"])
    assert "light lies beyond" not in capsys.readouterr().out


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
