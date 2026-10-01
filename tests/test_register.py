import numpy as np
import pytest

import m13
import synthetic
from quietsky.register import (
    RegistrationError,
    fit_homography,
    solve_transformation,
)
from quietsky.stars import Stars, detect_stars
from quietsky.xisf import read_xisf

SHAPE = (400, 600)
CORNERS_X = np.array([0.0, 599.0, 0.0, 599.0, 300.0])
CORNERS_Y = np.array([0.0, 0.0, 399.0, 399.0, 200.0])


def similarity(angle: float, scale: float, shift: tuple[float, float]) -> np.ndarray:
    """Rotation by angle degrees about the image centre, scaling and a shift."""
    cos, sin = scale * np.cos(np.radians(angle)), scale * np.sin(np.radians(angle))
    cx, cy = SHAPE[1] / 2, SHAPE[0] / 2
    return np.array(
        [
            [cos, -sin, cx - cos * cx + sin * cy + shift[0]],
            [sin, cos, cy - sin * cx - cos * cy + shift[1]],
            [0, 0, 1],
        ]
    )


def project(matrix: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, v, w = matrix @ np.stack([x, y, np.ones_like(x)])
    return u / w, v / w


def frames(matrix: np.ndarray, count: int = 120) -> tuple[Stars, Stars]:
    """Stars detected in a synthetic field and in the same field transformed."""
    _, x, y = synthetic.star_field(SHAPE, count)
    flux = 10 ** np.random.default_rng(5).uniform(-1.0, 0.3, count)
    u, v = project(matrix, x, y)
    reference = detect_stars(synthetic.render(SHAPE, x, y, flux, noise_seed=1))
    target = detect_stars(synthetic.render(SHAPE, u, v, flux, noise_seed=2))
    return reference, target


def test_fit_homography_is_exact_for_exact_points() -> None:
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 600, 20), rng.uniform(0, 400, 20)
    matrix = np.array([[0.99, 0.02, 5.0], [-0.02, 1.01, -3.0], [1e-5, -2e-5, 1.0]])
    u, v = project(matrix, x, y)

    np.testing.assert_allclose(fit_homography(x, y, u, v), matrix, atol=1e-9)


@pytest.mark.parametrize(
    ("angle", "scale", "shift"),
    [
        (0.0, 1.0, (12.3, -7.6)),
        (-1.2, 1.0, (-40.0, 15.0)),
        (30.0, 1.02, (5.0, 5.0)),
        (180.0, 1.0, (0.0, 0.0)),
    ],
)
def test_recovers_a_known_transformation(
    angle: float, scale: float, shift: tuple[float, float]
) -> None:
    matrix = similarity(angle, scale, shift)
    reference, target = frames(matrix)

    result = solve_transformation(reference, target)

    u, v = result.apply(CORNERS_X, CORNERS_Y)
    expected_u, expected_v = project(matrix, CORNERS_X, CORNERS_Y)
    assert np.hypot(u - expected_u, v - expected_v).max() < 0.15
    assert result.rms < 0.15
    assert result.pairs > 40


def test_tolerates_stars_missing_from_one_frame_and_spurious_in_the_other() -> None:
    matrix = similarity(-1.2, 1.0, (-40.0, 15.0))
    reference, target = frames(matrix)
    rng = np.random.default_rng(3)
    kept = rng.random(len(reference)) > 0.25  # a quarter of the reference stars go missing
    reference = Stars(reference.x[kept], reference.y[kept], reference.flux[kept])
    extra = 25  # and the target gains detections that are not stars
    target = Stars(
        np.concatenate([target.x, rng.uniform(20, 580, extra)]),
        np.concatenate([target.y, rng.uniform(20, 380, extra)]),
        np.concatenate([target.flux, rng.uniform(target.flux.min(), target.flux.max(), extra)]),
    )
    order = np.argsort(-target.flux)
    target = Stars(target.x[order], target.y[order], target.flux[order])

    result = solve_transformation(reference, target)

    u, v = result.apply(CORNERS_X, CORNERS_Y)
    expected_u, expected_v = project(matrix, CORNERS_X, CORNERS_Y)
    assert np.hypot(u - expected_u, v - expected_v).max() < 0.2


def test_unrelated_star_fields_are_refused() -> None:
    reference = detect_stars(synthetic.star_field(SHAPE, 80, seed=1)[0])
    target = detect_stars(synthetic.star_field(SHAPE, 80, seed=2)[0])

    with pytest.raises(RegistrationError):
        solve_transformation(reference, target)


@pytest.mark.nas
def test_transformations_agree_with_pixinsights_registered_frames() -> None:
    # PixInsight prints its matrices too coarsely to compare with: the
    # perspective terms it rounds to zero move the corners by over a pixel.
    # Its registered frames show the transformation it applied. A star at
    # reference coordinates in a registered frame must map onto the same
    # star in the unregistered light.
    output = m13.SESSION / "output"
    names = sorted(m13.registration_matrices())[1::12]
    for name in names:
        light = read_xisf(output / "calibrated" / "light" / "debayered" / f"{name}_c_cc_d.xisf")
        registered = read_xisf(output / "registered" / "m13" / f"{name}_c_cc_d_r.xisf")
        in_light = detect_stars(light[0].data, threshold=15)
        in_reference_frame = detect_stars(registered[0].data, threshold=15)

        result = solve_transformation(in_reference_frame, in_light)

        # PixInsight's own matrix, as far as it is printed: same rotation.
        printed = m13.registration_matrices()[name]
        rotation = np.degrees(np.arctan2(result.matrix[1, 0], result.matrix[0, 0]))
        assert rotation == pytest.approx(
            np.degrees(np.arctan2(printed[1, 0], printed[0, 0])), abs=0.01
        )
        assert result.pairs > 300
        assert result.rms < 0.4


@pytest.mark.nas
def test_solving_from_the_reference_light_matches_pixinsights_alignment() -> None:
    output = m13.SESSION / "output"
    debayered = output / "calibrated" / "light" / "debayered"
    reference = detect_stars(read_xisf(debayered / "2018-09-12-20_22_50_c_cc_d.xisf")[0].data, 15)
    for name in ("2018-09-12-20_35_20", "2018-09-12-20_56_55"):
        light = detect_stars(read_xisf(debayered / f"{name}_c_cc_d.xisf")[0].data, 15)
        registered = read_xisf(output / "registered" / "m13" / f"{name}_c_cc_d_r.xisf")[0].data
        aligned = detect_stars(registered, 15)

        result = solve_transformation(reference, light)

        # Where PixInsight put each star (its position in the registered
        # frame) our transformation must send to the star in the light.
        u, v = result.apply(aligned.x, aligned.y)
        separation = np.hypot(u[:, None] - light.x[None, :], v[:, None] - light.y[None, :])
        nearest = separation.argmin(axis=1)
        matched = separation[np.arange(len(u)), nearest] < 1.5
        offset_x = (u - light.x[nearest])[matched].mean()
        offset_y = (v - light.y[nearest])[matched].mean()
        # The two agree to about a tenth of a pixel on average.
        assert matched.sum() > 300
        assert abs(offset_x) < 0.15
        assert abs(offset_y) < 0.15
