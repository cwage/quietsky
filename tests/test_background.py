from pathlib import Path

import numpy as np
import pytest

import m13
import synthetic
from starkiller.background import extract_background, fit_surface, sample_background
from starkiller.pixinsight_project import read_project_image

M27 = Path(__file__).parent / "data" / "m27" / "dbe.npz"
M27_PROJECT = m13.SESSION.parent.parent / "2018-09-14" / "m27" / "m27.data"
SHAPE = (400, 600)


def gradient(shape: tuple[int, int] = SHAPE) -> np.ndarray:
    """A sky that brightens towards one corner, as light pollution does."""
    rows, columns = np.mgrid[: shape[0], : shape[1]]
    u, v = columns / shape[1], rows / shape[0]
    brightness: np.ndarray = 0.004 * u + 0.002 * v * v + 0.001 * u * v
    return brightness


def field_with_gradient() -> np.ndarray:
    image, _, _ = synthetic.star_field(SHAPE, 80)
    return image + gradient().astype(np.float32)


def unevenness(image: np.ndarray) -> float:
    """Spread of the background level over the frame, measured in coarse blocks."""
    blocks = image[:400, :600].reshape(8, 50, 12, 50)
    return float(np.ptp(np.median(blocks, axis=(1, 3))))


def test_removes_a_gradient_and_keeps_the_level() -> None:
    image = field_with_gradient()

    corrected, model, _ = extract_background(image)

    assert unevenness(image) > 0.004
    assert unevenness(corrected) < 2 * synthetic.NOISE / 10
    assert np.median(corrected) == pytest.approx(np.median(image), abs=synthetic.NOISE)
    # The model is the gradient, up to a constant.
    difference = model - gradient() - 0.01 - 0.00002 * np.mgrid[:400, :600][1]
    assert np.ptp(difference[20:-20, 20:-20]) < synthetic.NOISE


def test_stars_are_left_alone() -> None:
    plain, x, y = synthetic.star_field(SHAPE, 80)
    image = plain + gradient().astype(np.float32)

    corrected, _, _ = extract_background(image)

    def height_above_surroundings(picture: np.ndarray) -> np.ndarray:
        """Each star's peak pixel less the median of the box around it."""
        return np.array(
            [
                picture[row, column]
                - np.median(picture[row - 10 : row + 11, column - 10 : column + 11])
                for row, column in zip(y.round().astype(int), x.round().astype(int), strict=True)
            ]
        )

    np.testing.assert_allclose(
        height_above_surroundings(corrected), height_above_surroundings(image), atol=synthetic.NOISE
    )


def test_an_extended_object_is_not_mistaken_for_background() -> None:
    image = field_with_gradient()
    rows, columns = np.mgrid[:400, :600]
    nebula = 0.01 * np.exp(-((rows - 200) ** 2 + (columns - 300) ** 2) / (2 * 40**2))
    image = image + nebula.astype(np.float32)

    corrected, _, samples = extract_background(image)

    on_nebula = np.hypot(samples.x - 300, samples.y - 200) < 40
    assert on_nebula.any()
    assert not samples.kept[on_nebula].any()
    assert samples.kept.mean() > 0.7
    peak = corrected[190:210, 290:310].mean() - np.median(corrected)
    assert peak == pytest.approx(0.01, rel=0.1)


def frame_filling_galaxy(arms: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A broad galaxy across the whole frame over a tilted sky; returns image and galaxy.

    arms is the depth of a broad ripple across the galaxy, standing in for
    spiral arms and dust lanes several samples wide.
    """
    rows, columns = np.mgrid[:400, :600]
    galaxy = 0.02 * np.exp(-(((rows - 200) / 150) ** 2 + ((columns - 300) / 260) ** 2))
    galaxy = galaxy * (1 + arms * np.sin(columns / 90) * np.sin(rows / 70))
    image, _, _ = synthetic.star_field(SHAPE, 80)
    tilt = (0.006 * columns / 600).astype(np.float32)
    return image + tilt + galaxy.astype(np.float32), galaxy


def test_the_spline_takes_a_frame_filling_target_away() -> None:
    image, galaxy = frame_filling_galaxy()

    corrected, _, _ = extract_background(image)

    left = corrected[150:250, 250:350].mean() - corrected[:40, :40].mean()
    assert left < 0.1 * (galaxy[150:250, 250:350].mean() - galaxy[:40, :40].mean())


def test_samples_on_a_target_with_detail_show_structure() -> None:
    _, _, samples = extract_background(frame_filling_galaxy(arms=0.5)[0])

    assert samples.structure > 1.5


def test_a_perfectly_smooth_target_cannot_be_told_from_a_gradient() -> None:
    # The limit of the warning: nothing in the samples distinguishes a
    # featureless frame-filling glow from sky brightness.
    _, _, samples = extract_background(frame_filling_galaxy()[0])

    assert samples.structure < 1.5


def test_a_plane_removes_the_tilt_and_leaves_the_target() -> None:
    image, galaxy = frame_filling_galaxy()

    corrected, _, _ = extract_background(image, model="plane")

    contrast = corrected[150:250, 250:350].mean() - corrected[:40, :40].mean()
    expected = galaxy[150:250, 250:350].mean() - galaxy[:40, :40].mean()
    assert contrast == pytest.approx(expected, rel=0.1)
    # The tilt across the frame is gone: left and right edges agree.
    edges = corrected[180:220, :30].mean() - corrected[180:220, -30:].mean()
    assert abs(edges) < 0.001


def test_sky_with_a_gradient_shows_no_structure() -> None:
    _, _, samples = extract_background(field_with_gradient())

    assert samples.structure < 1.5


def test_a_quadratic_removes_a_curved_gradient() -> None:
    image = field_with_gradient()

    corrected, _, _ = extract_background(image, model="quadratic")

    assert unevenness(corrected) < 2 * synthetic.NOISE / 10


def test_unknown_models_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown background model"):
        extract_background(field_with_gradient(), model="cubic")


def test_the_black_border_of_a_registered_frame_is_not_sky() -> None:
    image = field_with_gradient()
    image[:60] = 0.0  # no frame covered the top after alignment

    corrected, model, samples = extract_background(image)

    assert not samples.kept[samples.y < 60].any()
    assert samples.kept[samples.y >= 60].mean() > 0.9
    assert samples.structure < 1.5
    # Below the border the result is as if the border were not there.
    levels = np.median(corrected[100:400].reshape(6, 50, 12, 50), axis=(1, 3))
    assert np.ptp(levels) < 2 * synthetic.NOISE / 10
    reference = extract_background(field_with_gradient())[1]
    assert np.abs(model[100:] - reference[100:]).max() < synthetic.NOISE


def test_mono_and_colour_images_keep_their_shape() -> None:
    image = field_with_gradient()
    colour = np.stack([image, 0.8 * image, 0.6 * image], axis=-1)

    assert extract_background(image)[0].shape == image.shape
    corrected, model, samples = extract_background(colour)
    assert corrected.shape == model.shape == colour.shape
    assert samples.values.shape[1] == 3


def test_samples_cover_the_frame_on_a_grid() -> None:
    samples = sample_background(field_with_gradient()[..., None], samples_per_row=10, radius=5)

    assert len(samples.x) == 11 * 8
    assert samples.x.min() == 5
    assert samples.x.max() == 594
    assert samples.y.min() == 5
    assert samples.y.max() == 394


def test_surface_through_pixinsights_samples_matches_its_model() -> None:
    # The samples of a DBE run in 2018 on an M27 stack, and the background
    # PixInsight subtracted, thinned to every eighth pixel.
    with np.load(M27) as reference:
        samples, model = reference["samples"], reference["model"]
        height, width = (int(size) for size in reference["shape"])

    for channel in range(3):
        surface = fit_surface(
            samples[:, 0] * width,
            samples[:, 1] * height,
            samples[:, 2 + 2 * channel],
            (height, width),
        ).astype(np.float64)
        surface -= np.median(surface)

        expected = model[..., channel]
        difference = surface[::8, ::8] - expected
        # 0.07% of the model's range when written.
        assert np.sqrt((difference**2).mean()) < 0.002 * np.ptp(expected)
        assert np.abs(difference).max() < 0.01 * np.ptp(expected)


@pytest.mark.nas
def test_automatic_background_agrees_with_pixinsights_on_m27() -> None:
    before = read_project_image(M27_PROJECT / "ZZF3PSWZT-000001")
    after = read_project_image(M27_PROJECT / "ZZF3PSWZT-000004")
    expected = before.astype(np.float64) - after

    _, model, samples = extract_background(before)

    assert samples.kept.mean() > 0.9  # a star field with a small nebula: little to reject
    for channel in range(3):
        ours = model[..., channel].astype(np.float64)
        ours -= np.median(ours)
        difference = ours - expected[..., channel]
        # About 5% of the model's range, 2% of the pixel noise, when written.
        assert np.sqrt((difference**2).mean()) < 0.08 * np.ptp(expected[..., channel])
        assert np.corrcoef(ours[::8, ::8].ravel(), expected[::8, ::8, channel].ravel())[0, 1] > 0.95
