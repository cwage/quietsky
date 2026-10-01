import numpy as np
import pytest

import m13
import synthetic
from starkiller.register import solve_transformation
from starkiller.resample import interpolate, resample
from starkiller.stars import detect_stars
from starkiller.xisf import read_xisf

IDENTITY = np.eye(3)


def shift(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])


def test_identity_leaves_the_image_unchanged() -> None:
    image = np.random.default_rng(1).random((20, 30), dtype=np.float32)

    np.testing.assert_allclose(resample(image, IDENTITY), image, rtol=1e-6)


def test_a_whole_pixel_shift_moves_pixels_without_changing_them() -> None:
    image = np.random.default_rng(1).random((20, 30), dtype=np.float32)

    result = resample(image, shift(3, -2))

    # Output pixel (x, y) shows input (x + 3, y - 2).
    np.testing.assert_allclose(result[2:, :-3], image[:-2, 3:], rtol=1e-6)


def test_positions_outside_the_input_come_out_black() -> None:
    image = np.ones((20, 30), np.float32)

    result = resample(image, shift(10, 0))

    assert (result[:, -10:] == 0).all()
    assert (result[:, :19] > 0.99).all()


def test_a_smooth_image_is_interpolated_accurately() -> None:
    rows, columns = np.mgrid[:40, :60]

    def smooth(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        value: np.ndarray = 0.5 + 0.3 * np.sin(x / 9) * np.cos(y / 7)
        return value

    image = smooth(columns, rows).astype(np.float32)

    result = resample(image, shift(0.4, 0.7))

    expected = smooth(columns + 0.4, rows + 0.7)
    np.testing.assert_allclose(result[4:-4, 4:-4], expected[4:-4, 4:-4], atol=1e-3)


def test_clamping_holds_back_the_dark_ring_around_a_star() -> None:
    image = np.full((21, 21), 0.01, np.float32)
    image[10, 10] = 1.0

    ringing = resample(image, shift(0.5, 0.5), clamping=None)
    clamped = resample(image, shift(0.5, 0.5))

    assert ringing.min() < -0.05
    assert clamped.min() > -0.005
    assert clamped.max() == pytest.approx(ringing.max(), rel=0.2)


def test_colour_images_keep_their_channels() -> None:
    image = np.random.default_rng(1).random((20, 30, 3), dtype=np.float32)

    result = resample(image, shift(2, 0))

    assert result.shape == image.shape
    np.testing.assert_allclose(result[:, :-2], image[:, 2:], rtol=1e-6)


def test_a_registered_frame_lines_up_with_the_reference() -> None:
    shape = (300, 400)
    _, x, y = synthetic.star_field(shape, 100)
    flux = 10 ** np.random.default_rng(5).uniform(-1.0, 0.3, len(x))
    angle = np.radians(1.2)
    u = np.cos(angle) * x - np.sin(angle) * y + 9.3
    v = np.sin(angle) * x + np.cos(angle) * y - 4.6
    reference = synthetic.render(shape, x, y, flux, noise_seed=1)
    target = synthetic.render(shape, u, v, flux, noise_seed=2)
    stars = detect_stars(reference)

    transformation = solve_transformation(stars, detect_stars(target))
    aligned = detect_stars(resample(target, transformation.matrix))

    separation = np.hypot(
        aligned.x[:, None] - stars.x[None, :], aligned.y[:, None] - stars.y[None, :]
    )
    nearest = separation.min(axis=1)
    assert (nearest < 1).sum() > 70
    assert np.median(nearest[nearest < 1]) < 0.05


def refine(
    matrix: np.ndarray, source: np.ndarray, target: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Adjust a transformation so that resampling source best reproduces target at (x, y)."""

    def render(parameters: np.ndarray) -> np.ndarray:
        u, v, w = np.append(parameters, 1.0).reshape(3, 3) @ np.stack([x, y, np.ones_like(x)])
        return interpolate(source, u / w, v / w, clamping=0.3)

    parameters = (matrix / matrix[2, 2]).ravel()[:8].copy()
    steps = np.array([1e-6, 1e-6, 1e-3, 1e-6, 1e-6, 1e-3, 1e-10, 1e-10])
    for _ in range(3):
        current = render(parameters)
        jacobian = np.stack(
            [(render(parameters + np.eye(8)[k] * steps[k]) - current) / steps[k] for k in range(8)],
            axis=1,
        )
        parameters += np.linalg.lstsq(jacobian, target - current, rcond=None)[0]
    return np.append(parameters, 1.0).reshape(3, 3)


@pytest.mark.nas
def test_interpolation_matches_pixinsights_registered_frame() -> None:
    # PixInsight does not record its transformation precisely enough to reuse,
    # so start from ours and refine it against the registered frame; what is
    # left is the difference between the two interpolations.
    output = m13.SESSION / "output"
    name = "2018-09-12-20_45_50"
    light = read_xisf(output / "calibrated" / "light" / "debayered" / f"{name}_c_cc_d.xisf")[0].data
    registered = read_xisf(output / "registered" / "m13" / f"{name}_c_cc_d_r.xisf")[0].data
    start = solve_transformation(detect_stars(registered, 15), detect_stars(light, 15)).matrix
    rows, columns = np.mgrid[1100:1500, 1800:2200]
    x, y = columns.ravel().astype(np.float64), rows.ravel().astype(np.float64)
    green = np.asarray(light[..., 1])
    target = registered[1100:1500, 1800:2200, 1].ravel().astype(np.float64)

    matrix = refine(start, green, target, x, y)

    u, v, w = matrix @ np.stack([x, y, np.ones_like(x)])
    difference = interpolate(green, u / w, v / w, clamping=0.3) - target
    unclamped = interpolate(green, u / w, v / w, clamping=None) - target
    noise = target.std()
    assert np.sqrt((difference**2).mean()) < 1e-3 * noise  # 7e-5 of the noise when written
    assert (np.abs(difference) < 1e-7).mean() > 0.99
    assert np.sqrt((unclamped**2).mean()) > 20 * np.sqrt((difference**2).mean())
    # The star-based solution was already within a few hundredths of a pixel.
    centre = np.array([2000.0, 1300.0, 1.0])
    moved = (matrix @ centre)[:2] / (matrix @ centre)[2] - (start @ centre)[:2] / (start @ centre)[
        2
    ]
    assert np.hypot(*moved) < 0.1
