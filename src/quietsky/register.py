"""Work out how one frame is shifted, rotated and distorted relative to another.

Two frames of the same sky show the same stars in different places. The
star lists are first matched by the shapes of the triangles the brightest
stars form, which do not depend on shift, rotation or scale. The matches
give a rough transformation, which is then refined on all the stars.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quietsky.stars import Stars

Float64 = npt.NDArray[np.float64]
Index = npt.NDArray[np.intp]

# Triangles are formed among this many of the brightest stars of each frame.
TRIANGLE_STARS = 40
# Triangles this flat or this similar in side lengths are ambiguous.
MIN_SHORT_SIDE_RATIO = 0.2
MAX_SIDE_RATIO = 0.95
# Two triangles match when their side ratios agree this closely.
SHAPE_TOLERANCE = 0.003
# Pairs of stars count as the same star within these distances, in pixels,
# on successive refinements.
MATCH_DISTANCES = (4.0, 2.0, 1.5, 1.5)
MIN_PAIRS = 8
# A pair of stars needs this many matching triangles to seed the solution.
MIN_VOTES = 3


class RegistrationError(Exception):
    """The two star lists could not be matched."""


@dataclass(frozen=True)
class Transformation:
    """A projective transformation and how well it fits.

    matrix maps (x, y, 1) in the reference frame to the target frame, up to
    scale. pairs is the number of stars matched and rms their root mean
    square distance, in pixels, after transformation.
    """

    matrix: Float64
    pairs: int
    rms: float

    def apply(self, x: Float64, y: Float64) -> tuple[Float64, Float64]:
        """Target-frame coordinates of reference-frame points."""
        return _project(self.matrix, x, y)


def _project(matrix: Float64, x: Float64, y: Float64) -> tuple[Float64, Float64]:
    u, v, w = matrix @ np.stack([x, y, np.ones_like(x)])
    return u / w, v / w


def _triangles(x: Float64, y: Float64) -> tuple[Float64, Index]:
    """Shapes of the triangles among some stars, and their vertices.

    A shape is the pair of ratios of the two shorter sides to the longest.
    Vertices are ordered by the length of the side opposite, so that matching
    shapes put corresponding stars in the same position.
    """
    vertices = np.array(list(itertools.combinations(range(len(x)), 3)))
    px, py = x[vertices], y[vertices]
    # Side i is opposite vertex i.
    sides = np.hypot(
        np.roll(px, -1, 1) - np.roll(px, -2, 1), np.roll(py, -1, 1) - np.roll(py, -2, 1)
    )
    order = np.argsort(-sides, axis=1)
    sides = np.take_along_axis(sides, order, 1)
    vertices = np.take_along_axis(vertices, order, 1)
    shapes = sides[:, 1:] / sides[:, :1]
    clear = (
        (shapes[:, 1] > MIN_SHORT_SIDE_RATIO)
        & (shapes[:, 0] < MAX_SIDE_RATIO)
        & (shapes[:, 1] < MAX_SIDE_RATIO * shapes[:, 0])
    )
    return shapes[clear], vertices[clear]


def _votes(reference: Stars, target: Stars) -> npt.NDArray[np.int64]:
    """How often each pair of stars sits at the same corner of matching triangles."""
    count_ref = min(len(reference), TRIANGLE_STARS)
    count_tgt = min(len(target), TRIANGLE_STARS)
    shapes_ref, vertices_ref = _triangles(reference.x[:count_ref], reference.y[:count_ref])
    shapes_tgt, vertices_tgt = _triangles(target.x[:count_tgt], target.y[:count_tgt])
    votes = np.zeros((count_ref, count_tgt), np.int64)
    for start in range(0, len(shapes_ref), 1000):
        chunk = shapes_ref[start : start + 1000]
        alike = (np.abs(chunk[:, None, :] - shapes_tgt[None, :, :]) < SHAPE_TOLERANCE).all(axis=2)
        in_ref, in_tgt = np.nonzero(alike)
        np.add.at(votes, (vertices_ref[start + in_ref], vertices_tgt[in_tgt]), 1)
    return votes


def fit_homography(x: Float64, y: Float64, u: Float64, v: Float64) -> Float64:
    """The projective transformation taking points (x, y) closest to (u, v)."""

    def normalizing(a: Float64, b: Float64) -> Float64:
        scale = np.sqrt(2) / np.hypot(a - a.mean(), b - b.mean()).mean()
        return np.array([[scale, 0, -scale * a.mean()], [0, scale, -scale * b.mean()], [0, 0, 1]])

    source, destination = normalizing(x, y), normalizing(u, v)
    sx, sy = _project(source, x, y)
    du, dv = _project(destination, u, v)
    zero, one = np.zeros_like(sx), np.ones_like(sx)
    equations = np.concatenate(
        [
            np.stack([-sx, -sy, -one, zero, zero, zero, du * sx, du * sy, du], axis=1),
            np.stack([zero, zero, zero, -sx, -sy, -one, dv * sx, dv * sy, dv], axis=1),
        ]
    )
    solution = np.linalg.svd(equations)[2][-1].reshape(3, 3)
    matrix: Float64 = np.linalg.inv(destination) @ solution @ source
    matrix /= matrix[2, 2]
    return matrix


def _fit_similarity(x: Float64, y: Float64, u: Float64, v: Float64) -> Float64:
    """Shift, rotation and scale taking (x, y) closest to (u, v), as a 3x3 matrix."""
    one, zero = np.ones_like(x), np.zeros_like(x)
    equations = np.concatenate(
        [np.stack([x, -y, one, zero], axis=1), np.stack([y, x, zero, one], axis=1)]
    )
    a, b, tx, ty = np.linalg.lstsq(equations, np.concatenate([u, v]), rcond=None)[0]
    return np.array([[a, -b, tx], [b, a, ty], [0, 0, 1]])


def _nearest_pairs(
    matrix: Float64, reference: Stars, target: Stars, distance: float
) -> tuple[Index, Index, Float64]:
    """Reference and target stars that land within distance of each other, one to one."""
    u, v = _project(matrix, reference.x, reference.y)
    separation = np.hypot(u[:, None] - target.x[None, :], v[:, None] - target.y[None, :])
    nearest_target = separation.argmin(axis=1)
    nearest_reference = separation.argmin(axis=0)
    indices = np.arange(len(reference))
    mutual = nearest_reference[nearest_target] == indices
    close = separation[indices, nearest_target] < distance
    keep = mutual & close
    return indices[keep], nearest_target[keep], separation[indices, nearest_target][keep]


def solve_transformation(reference: Stars, target: Stars) -> Transformation:
    """Find the transformation from the reference frame to the target frame."""
    votes = _votes(reference, target)
    # Start from the pairs that are each other's best match.
    best_for_ref = votes.argmax(axis=1)
    mutual = [
        (int(i), int(j))
        for i, j in enumerate(best_for_ref)
        if votes[i, j] >= MIN_VOTES and votes[:, j].argmax() == i
    ]
    if len(mutual) < 3:
        raise RegistrationError("too few stars in common between the frames")
    ref_index = np.array([i for i, _ in mutual])
    tgt_index = np.array([j for _, j in mutual])
    # Some of those are chance matches; fit, drop the pairs that disagree
    # with the rest, and fit again.
    for _ in range(3):
        rx, ry = reference.x[ref_index], reference.y[ref_index]
        tx, ty = target.x[tgt_index], target.y[tgt_index]
        matrix = _fit_similarity(rx, ry, tx, ty)
        u, v = _project(matrix, rx, ry)
        error = np.hypot(u - tx, v - ty)
        agree = error <= max(MATCH_DISTANCES[0], 2 * float(np.median(error)))
        if agree.all() or agree.sum() < 3:
            break
        ref_index, tgt_index = ref_index[agree], tgt_index[agree]

    distances = np.empty(0)
    for distance in MATCH_DISTANCES:
        ref_index, tgt_index, distances = _nearest_pairs(matrix, reference, target, distance)
        if len(ref_index) < MIN_PAIRS:
            raise RegistrationError("too few stars in common between the frames")
        matrix = fit_homography(
            reference.x[ref_index], reference.y[ref_index], target.x[tgt_index], target.y[tgt_index]
        )
    ref_index, tgt_index, distances = _nearest_pairs(matrix, reference, target, MATCH_DISTANCES[-1])
    return Transformation(matrix, len(ref_index), float(np.sqrt((distances**2).mean())))
