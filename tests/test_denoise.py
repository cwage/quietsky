from pathlib import Path

import numpy as np
import pytest

from quietsky.cli import main
from quietsky.deconvolve import gaussian_blur, measure_psf
from quietsky.denoise import (
    denoise_image,
    layer_noise,
    non_local_means,
    protection_mask,
    shrink_layers,
)
from quietsky.frame import Frame, load, save_fits
from quietsky.noise import evaluate_noise
from quietsky.stars import Stars
from quietsky.wavelets import B3_NOISE_PER_LAYER

SIGMA = 0.001
STAR_SIGMA = 1.5
SKY = 0.02
# A dozen stars kept apart, the first of them bright.
STAR_X = np.array([300.0, 40, 90, 160, 230, 350, 60, 210, 330, 120, 280, 370])
STAR_Y = np.array([100.0, 30, 250, 60, 270, 40, 150, 180, 220, 20, 30, 280])
STAR_FLUX = np.array([7.0, 3, 2.5, 2, 1.5, 1.2, 1, 0.8, 0.6, 0.5, 0.4, 0.3])
STARS = Stars(STAR_X, STAR_Y, STAR_FLUX)
NEBULA = (slice(120, 181), slice(90, 151))


def white_noise(shape: tuple[int, int] = (300, 400), seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, SIGMA, shape).astype(np.float32)


def correlated_noise(shape: tuple[int, int] = (300, 400), seed: int = 0) -> np.ndarray:
    """Noise spread into its neighbours, as debayering and resampling spread it."""
    return gaussian_blur(white_noise(shape, seed), 0.7)


def field() -> tuple[np.ndarray, np.ndarray]:
    """A sky with a gradient, a faint nebula and stars; returns it and its truth."""
    rows, columns = np.mgrid[:300, :400].astype(np.float32)
    truth = SKY + 0.00002 * rows
    truth += 0.004 * np.exp(-((rows - 150) ** 2 + (columns - 120) ** 2) / (2 * 30.0**2))
    for x, y, flux in zip(STAR_X, STAR_Y, STAR_FLUX, strict=True):
        distance2 = (rows - y) ** 2 + (columns - x) ** 2
        truth += flux * np.exp(-distance2 / (2 * STAR_SIGMA**2)) / (2 * np.pi * STAR_SIGMA**2)
    truth = truth.astype(np.float32)
    return (truth + correlated_noise()).astype(np.float32), truth


def binned(image: np.ndarray, size: int = 8) -> np.ndarray:
    height, width = (image.shape[0] // size) * size, (image.shape[1] // size) * size
    return image[:height, :width].reshape(height // size, size, width // size, size).mean((1, 3))


def test_layer_noise_of_white_noise_follows_the_table() -> None:
    sigmas = layer_noise((0.5 + white_noise()).astype(np.float32), 3)
    for sigma, share in zip(sigmas, B3_NOISE_PER_LAYER, strict=False):
        assert sigma == pytest.approx(SIGMA * share, rel=0.03)


def test_layer_noise_of_correlated_noise_leaves_the_table() -> None:
    image = (0.5 + correlated_noise()).astype(np.float32)
    pixel = float(image.std())
    sigmas = layer_noise(image, 3)
    # The finest layer holds less of the noise than white noise would put there,
    # and the coarser layers more.
    assert sigmas[0] < 0.8 * pixel * B3_NOISE_PER_LAYER[0]
    assert sigmas[1] > 1.2 * pixel * B3_NOISE_PER_LAYER[1]
    assert sigmas[2] > 1.2 * pixel * B3_NOISE_PER_LAYER[2]


def test_layer_noise_from_a_sky_region_ignores_the_rest() -> None:
    image = (0.5 + white_noise()).astype(np.float32)
    image[:, 200:] += 0.3  # a bright, noiseless-looking step and plateau
    sigmas = layer_noise(image, 2, sky=(10, 10, 150, 250))
    for sigma, share in zip(sigmas, B3_NOISE_PER_LAYER, strict=False):
        assert sigma == pytest.approx(SIGMA * share, rel=0.05)


def test_layer_noise_of_a_flat_image_is_zero() -> None:
    assert layer_noise(np.full((100, 100), 0.5, np.float32), 2) == [0.0, 0.0]


def test_shrink_layers_leaves_strong_coefficients_nearly_alone() -> None:
    image = (0.5 + white_noise()).astype(np.float32)
    image[150, 200] += 0.2
    sigmas = layer_noise(image, 2)
    shrunk = shrink_layers(image, sigmas, [3.0, 3.0])
    assert shrunk[150, 200] == pytest.approx(image[150, 200], abs=0.5 * SIGMA)
    away = (slice(0, 100), slice(0, 400))
    assert shrunk[away].std() < 0.5 * image[away].std()


def test_denoise_lowers_fine_noise_and_keeps_the_signal() -> None:
    image, truth = field()
    cleaned, mask, components = denoise_image(image)

    assert evaluate_noise(cleaned).sigma < 0.5 * evaluate_noise(image).sigma
    # The coarse noise, which the binned image shows, is mostly left.
    sky = (slice(200, 300), slice(0, 400))
    coarse_before = binned(image[sky] - truth[sky]).std()
    coarse_after = binned(cleaned[sky] - truth[sky]).std()
    assert 0.6 * coarse_before < coarse_after <= coarse_before
    # Closer to the truth everywhere, and the nebula is not flattened.
    assert np.abs(cleaned - truth).mean() < 0.6 * np.abs(image - truth).mean()
    level = SKY + 0.00002 * 150
    assert (cleaned[NEBULA] - level).sum() == pytest.approx((truth[NEBULA] - level).sum(), rel=0.05)
    # The stars keep their light and their width.
    star = (slice(90, 111), slice(290, 311))
    level = SKY + 0.00002 * 100
    assert (cleaned[star] - level).sum() == pytest.approx((truth[star] - level).sum(), rel=0.02)
    assert measure_psf(cleaned, STARS) == pytest.approx(measure_psf(image, STARS), rel=0.05)
    assert mask[100, 300] < 0.1 < 0.9 < mask[250, 20]
    assert [c.name for c in components] == ["L"]
    assert all(a < b for a, b in zip(components[0].after, components[0].before, strict=True))


def test_mask_keeps_the_original_where_it_is_zero() -> None:
    image, _ = field()
    mask = np.ones(image.shape, np.float32)
    mask[:, 200:] = 0
    cleaned, _, _ = denoise_image(image, mask=mask)
    assert np.array_equal(cleaned[:, 200:], image[:, 200:])
    assert not np.array_equal(cleaned[:, :200], image[:, :200])


def test_protection_mask_spares_bright_signal() -> None:
    image, _ = field()
    mask = protection_mask(image)
    assert mask.min() >= 0 and mask.max() <= 1
    assert mask[100, 300] < mask[150, 120] < mask[250, 20]


def test_colour_noise_is_smoothed_harder_than_luminance() -> None:
    luminance, _ = field()
    channels = [luminance + correlated_noise(seed=seed) for seed in (1, 2, 3)]
    image = np.stack(channels, axis=-1).astype(np.float32)
    cleaned, _, components = denoise_image(image)

    assert cleaned.shape == image.shape
    assert [c.name for c in components] == ["L", "R", "G", "B"]
    ratios = {c.name: [a / b for a, b in zip(c.after, c.before, strict=True)] for c in components}
    # Beyond the finest layer, where both are shrunk to nearly nothing, the
    # colour's noise is cut by more than the brightness's at every scale.
    for name in "RGB":
        assert all(c < lum for c, lum in zip(ratios[name][1:3], ratios["L"][1:3], strict=True))
    # The colour balance of the whole frame is untouched.
    assert cleaned.mean(axis=(0, 1)) == pytest.approx(image.mean(axis=(0, 1)), rel=1e-3)


def test_denoise_command_writes_the_result_mask_and_residual(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image, _ = field()
    colour = np.stack([image, image * 0.8, image * 0.6], axis=-1).astype(np.float32)
    source = tmp_path / "light_bg.fits"
    save_fits(source, Frame(colour, {"NCOMBINE": 7}))
    output, mask, residual, png = (tmp_path / n for n in ("dn.fits", "m.fits", "r.fits", "dn.png"))

    main(
        [
            "denoise",
            str(source),
            "-o",
            str(output),
            "--threshold",
            "3,2",
            "--chroma-threshold",
            "4,4,4",
            "--mask-output",
            str(mask),
            "--residual",
            str(residual),
            "--png",
            str(png),
        ]
    )

    result = load(output)
    assert result.data.shape == colour.shape
    assert result.header["NCOMBINE"] == 7
    assert evaluate_noise(result.data[..., 0]).sigma < evaluate_noise(colour[..., 0]).sigma
    assert load(mask).data.shape == image.shape
    assert np.allclose(load(residual).data, colour - result.data, atol=1e-7)
    assert png.exists()
    out = capsys.readouterr().out
    assert "L:" in out and "R:" in out and "wrote" in out


def test_denoise_command_takes_a_mask_and_a_sky_region(tmp_path: Path) -> None:
    image, _ = field()
    source, output, mask_file = (tmp_path / n for n in ("in.fits", "out.fits", "mask.fits"))
    save_fits(source, Frame(image))
    mask = np.zeros(image.shape, np.float32)
    mask[:, :200] = 1
    save_fits(mask_file, Frame(mask))

    main(
        ["denoise", str(source), "-o", str(output), "--mask", str(mask_file), "--sky", "0,0,200,80"]
    )

    result = load(output).data
    assert np.array_equal(result[:, 200:], image[:, 200:])
    assert evaluate_noise(result[:, :200]).sigma < 0.5 * evaluate_noise(image[:, :200]).sigma


def test_amount_scales_the_luminance_change_and_not_the_colour() -> None:
    luminance, _ = field()
    channels = [luminance + correlated_noise(seed=seed) for seed in (1, 2, 3)]
    image = np.stack(channels, axis=-1).astype(np.float32)
    full, _, _ = denoise_image(image)
    half, _, _ = denoise_image(image, amount=0.5)
    change_full = full.mean(axis=-1) - image.mean(axis=-1)
    change_half = half.mean(axis=-1) - image.mean(axis=-1)
    assert np.allclose(change_half, 0.5 * change_full, atol=1e-7)
    colour_full = (full[..., 0] - full[..., 1]) - (image[..., 0] - image[..., 1])
    colour_half = (half[..., 0] - half[..., 1]) - (image[..., 0] - image[..., 1])
    assert np.allclose(colour_half, colour_full, atol=1e-7)


def test_faint_stars_are_protected_whole() -> None:
    image, _ = field()
    cleaned, mask, _ = denoise_image(image)
    # The faintest star peaks at a few times the noise; the mask takes it out
    # anyway, so the denoise leaves it exactly as it was.
    x, y = int(STAR_X[-1]), int(STAR_Y[-1])
    assert mask[y, x] < 0.05
    assert cleaned[y, x] == pytest.approx(image[y, x], abs=0.05 * SIGMA)
    assert mask[y, x + 12] > 0.8


def test_non_local_means_lowers_noise_and_keeps_an_edge_sharp() -> None:
    image = (0.5 + white_noise()).astype(np.float32)
    image[:, 200:] += 0.02
    cleaned = non_local_means(image, SIGMA, 1.2 * SIGMA)
    assert cleaned[:, :190].std() < 0.5 * image[:, :190].std()
    assert cleaned[:, 210:].std() < 0.5 * image[:, 210:].std()
    # The step stays a step: the columns either side of it keep their levels.
    assert cleaned[:, 199].mean() == pytest.approx(0.5, abs=0.2 * SIGMA)
    assert cleaned[:, 200].mean() == pytest.approx(0.52, abs=0.2 * SIGMA)


def test_nlm_method_lowers_noise_and_keeps_the_signal() -> None:
    image, truth = field()
    cleaned, _, components = denoise_image(image, method="nlm")
    assert evaluate_noise(cleaned).sigma < 0.5 * evaluate_noise(image).sigma
    assert np.abs(cleaned - truth).mean() < 0.7 * np.abs(image - truth).mean()
    level = SKY + 0.00002 * 150
    assert (cleaned[NEBULA] - level).sum() == pytest.approx((truth[NEBULA] - level).sum(), rel=0.05)
    star = (slice(90, 111), slice(290, 311))
    level = SKY + 0.00002 * 100
    assert (cleaned[star] - level).sum() == pytest.approx((truth[star] - level).sum(), rel=0.02)
    assert measure_psf(cleaned, STARS) == pytest.approx(measure_psf(image, STARS), rel=0.05)
    assert all(a < b for a, b in zip(components[0].after, components[0].before, strict=True))


def test_unknown_method_is_refused() -> None:
    image, _ = field()
    with pytest.raises(ValueError, match="method"):
        denoise_image(image, method="magic")


def test_denoise_command_takes_the_nlm_method(tmp_path: Path) -> None:
    image, _ = field()
    colour = np.stack([image, image * 0.8, image * 0.6], axis=-1).astype(np.float32)
    source, output = tmp_path / "in.fits", tmp_path / "out.fits"
    save_fits(source, Frame(colour))
    main(["denoise", str(source), "-o", str(output), "--method", "nlm", "--strength", "1.5"])
    result = load(output).data
    assert result.shape == colour.shape
    assert evaluate_noise(result[..., 0]).sigma < 0.5 * evaluate_noise(colour[..., 0]).sigma
