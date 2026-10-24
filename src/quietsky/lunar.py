"""Lucky imaging: stack the sharpest moments of a video of the Moon.

Turbulence in the air blurs and bends each frame of a video differently and
changes many times a second, so over a few hundred frames every part of the
disc is sharp in some of them, though rarely all at once. The frames are
ranked by sharpness and aligned on the surface detail; then the image is cut
into overlapping alignment boxes, and each box is stacked from the frames in
which that box was sharpest, after its own small shift. Wavelet sharpening
then brings out the detail the stack has lifted above the noise.

Everything works on luminance. The Moon is nearly grey, and a colour mosaic
filtered with the 3x3 binomial kernel gives a quarter red, half green and a
quarter blue at every pixel whatever its colour, so there is no grid to
remove and no demosaicing to do.
"""

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import EarthLocation, get_body
from astropy.time import Time
from astropy.utils import iers

from quietsky.deconvolve import gaussian_blur
from quietsky.frame import normalized
from quietsky.resample import resample
from quietsky.wavelets import b3_layers

Float32 = npt.NDArray[np.float32]
Float64 = npt.NDArray[np.float64]
FrameSource = Callable[[int], Float32]

# Alignment boxes are this many pixels square and placed half a box apart.
BOX = 64
# Sharpness is measured on this wavelet layer: scales of about 2 to 4
# pixels, where seeing blur shows and single-pixel noise does not.
SHARPNESS_LAYER = 1
# A box is used where the reference is at least this fraction of the bright
# end of the disc, which keeps the terminator and leaves the sky out.
BOX_MIN_LEVEL = 0.08
# Part of the frame the global alignment looks at, as a downsampling factor.
GLOBAL_BINNING = 2
# Extra pixels cut around a box so a fractional shift has room to move it.
SHIFT_MARGIN = 4


def mosaic_luminance(mosaic: npt.NDArray[Any]) -> Float32:
    """Luminance of a Bayer mosaic: the 3x3 binomial average at every pixel.

    Reflecting at the edge keeps each mirrored pixel on a site of the right
    colour, so the mixture is the same at the border as inside.
    """
    padded = np.pad(normalized(mosaic), 1, mode="reflect")
    rows = (padded[:-2] + 2 * padded[1:-1] + padded[2:]) / np.float32(4)
    result: Float32 = (rows[:, :-2] + 2 * rows[:, 1:-1] + rows[:, 2:]) / np.float32(4)
    return result


def bayer_balance(mosaic: npt.NDArray[Any]) -> tuple[Float32, Float32]:
    """Per-site sky level and gain that turn a Bayer mosaic of a grey target
    into its luminance.

    Returns two (2, 2) arrays for the four sites of the pattern. Subtracting
    the sky and multiplying by the gain brings the lit pixels of each colour
    to the level mosaic_luminance gives them, less its sky. The Moon is nearly
    grey, so the balanced mosaic is then its luminance at full resolution with
    nothing interpolated, apart from a faint grid where its colour differs.
    mosaic is best an average of several frames, so noise does not set the gains.
    """
    data = normalized(mosaic)
    luminance = mosaic_luminance(data)
    sky_level = float(np.median(luminance))
    above = luminance - sky_level
    lit = above > 0.2 * float(np.percentile(above, 99.5))
    target = float(np.median(above[lit]))
    sky = np.zeros((2, 2), np.float32)
    gain = np.zeros((2, 2), np.float32)
    for row in (0, 1):
        for column in (0, 1):
            site = data[row::2, column::2]
            sky[row, column] = np.median(site)
            gain[row, column] = target / float(
                np.median(site[lit[row::2, column::2]] - sky[row, column])
            )
    return sky, gain


def balanced(mosaic: npt.NDArray[Any], sky: Float32, gain: Float32) -> Float32:
    """A Bayer mosaic with each site's sky taken off and its gain applied."""
    data = normalized(mosaic)
    height, width = data.shape
    reps = ((height + 1) // 2, (width + 1) // 2)
    result: Float32 = (
        (data - np.tile(sky, reps)[:height, :width]) * np.tile(gain, reps)[:height, :width]
    ).astype(np.float32)
    return result


def upsample(image: Float32, scale: int, workers: int | None = None) -> Float32:
    """Lanczos-interpolate an (H, W) or (H, W, C) image onto a grid scale times finer."""
    offset = 0.5 / scale - 0.5
    matrix = np.array([[1 / scale, 0, offset], [0, 1 / scale, offset], [0, 0, 1]])
    height, width = image.shape[:2]
    return resample(image, matrix, workers=workers, shape=(height * scale, width * scale))


def detail(image: Float32) -> Float32:
    """The wavelet layer sharpness is measured on."""
    layers, _ = b3_layers(image, SHARPNESS_LAYER + 1)
    return layers[SHARPNESS_LAYER]


# Sharpness is measured where a frame is at least this fraction of its
# bright end, so the sky's noise does not drown the detail of a small target.
SHARPNESS_MIN_LEVEL = 0.1


def sharpness(image: Float32) -> float:
    """How much fine detail the lit part of a frame has, independent of its
    brightness. The frame should have its background taken off."""
    lit = image >= SHARPNESS_MIN_LEVEL * float(np.percentile(image, 99.9))
    level = float(np.mean(image[lit])) if lit.any() else 0.0
    if level <= 0:
        return 0.0
    return float(np.mean(detail(image)[lit] ** 2)) / level**2


# A frame shows the whole target when its lit area (above this fraction of
# the run's typical bright end) is at least WHOLE_MIN_AREA of the run's median.
WHOLE_LIT_LEVEL = 0.25
WHOLE_MIN_AREA = 0.97
# Lit areas are counted on frames thinned by this factor.
WHOLE_THINNING = 8


def whole_frames(thinned: Sequence[Float32]) -> npt.NDArray[np.bool_]:
    """Which frames show the whole target: not partly out of the frame, and not
    dimmed by cloud. A frame with almost nothing in it can score as the sharpest
    of all, its noise passing for detail, so these must be left out before
    ranking. thinned are the frames, background taken off, thinned the same way."""
    level = float(np.median([np.percentile(frame, 99.9) for frame in thinned]))
    area = np.array([np.count_nonzero(frame > WHOLE_LIT_LEVEL * level) for frame in thinned])
    whole: npt.NDArray[np.bool_] = area >= WHOLE_MIN_AREA * np.median(area)
    return whole


def _hann(size: int) -> Float32:
    """A periodic Hann window: copies half a window apart add up to one."""
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)).astype(np.float32)


def _peak(correlation: Float64) -> tuple[float, float]:
    """Position of the highest value of a periodic correlation surface, to a fraction
    of a pixel, as a signed (dy, dx) shift."""
    height, width = correlation.shape[-2:]
    index = int(np.argmax(correlation))
    row, column = divmod(index, width)

    def refine(minus: float, centre: float, plus: float) -> float:
        curvature = minus - 2 * centre + plus
        return 0.0 if curvature >= 0 else 0.5 * (minus - plus) / curvature

    dy = row + refine(
        correlation[(row - 1) % height, column],
        correlation[row, column],
        correlation[(row + 1) % height, column],
    )
    dx = column + refine(
        correlation[row, (column - 1) % width],
        correlation[row, column],
        correlation[row, (column + 1) % width],
    )
    dy = dy - height if dy > height / 2 else dy
    dx = dx - width if dx > width / 2 else dx
    return dy, dx


def phase_correlate(reference: npt.NDArray[np.complexfloating], image: Float32) -> Float64:
    """The phase correlation surface between a transformed reference and an image.

    Its peak sits at the shift that moves the reference onto the image.
    """
    spectrum = np.fft.rfft2(image)
    cross = spectrum * np.conj(reference)
    cross /= np.maximum(np.abs(cross), 1e-12)
    surface: Float64 = np.fft.irfft2(cross, s=image.shape).astype(np.float64)
    return surface


def fourier_shift(image: Float32, dy: float, dx: float) -> Float32:
    """Move an image by a fraction of a pixel, wrapping at the edges."""
    height, width = image.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    ramp = np.exp(-2j * np.pi * (fy * dy + fx * dx))
    shifted: Float32 = np.fft.irfft2(np.fft.rfft2(image) * ramp, s=image.shape).astype(np.float32)
    return shifted


def shift_image(image: Float32, dy: float, dx: float) -> Float32:
    """Move an image by (dy, dx) pixels, filling what comes in from outside with zero."""
    height, width = image.shape
    whole_y, whole_x = int(np.floor(dy)), int(np.floor(dx))
    result = np.zeros_like(image)
    top, bottom = max(0, whole_y), min(height, height + whole_y)
    left, right = max(0, whole_x), min(width, width + whole_x)
    if top < bottom and left < right:
        result[top:bottom, left:right] = image[
            top - whole_y : bottom - whole_y, left - whole_x : right - whole_x
        ]
    return fourier_shift(result, dy - whole_y, dx - whole_x)


def _binned(image: Float32, factor: int) -> Float32:
    height, width = (s // factor * factor for s in image.shape)
    blocks = image[:height, :width].reshape(height // factor, factor, width // factor, factor)
    result: Float32 = blocks.mean(axis=(1, 3))
    return result


@dataclass(frozen=True)
class Boxes:
    """Alignment boxes: the top left corner of each in the reference."""

    top: npt.NDArray[np.intp]
    left: npt.NDArray[np.intp]
    size: int

    def __len__(self) -> int:
        return len(self.top)


def place_boxes(reference: Float32, size: int = BOX) -> Boxes:
    """Overlapping boxes, half a box apart, over the lit part of the reference."""
    height, width = reference.shape
    bright = float(np.percentile(reference, 99.5))
    step = size // 2
    tops, lefts = [], []
    for top in range(0, height - size + 1, step):
        for left in range(0, width - size + 1, step):
            box = reference[top : top + size, left : left + size]
            if float(box.mean()) >= BOX_MIN_LEVEL * bright:
                tops.append(top)
                lefts.append(left)
    return Boxes(np.array(tops, np.intp), np.array(lefts, np.intp), size)


def _box_sums(values: Float32, boxes: Boxes) -> Float64:
    """The sum of an image over each box, from its summed-area table."""
    table = np.zeros((values.shape[0] + 1, values.shape[1] + 1))
    table[1:, 1:] = values.cumsum(0).cumsum(1)
    t, left, s = boxes.top, boxes.left, boxes.size
    sums: Float64 = (
        table[t + s, left + s] - table[t, left + s] - table[t + s, left] + table[t, left]
    )
    return sums


def _cut(image: Float32, boxes: Boxes, dy: Float64, dx: Float64, margin: int) -> Float32:
    """Cut each box, moved by its whole-pixel shift, with a margin, from an image.

    Boxes that would reach outside the image come out as zeros.
    """
    size = boxes.size + 2 * margin
    height, width = image.shape
    tops = boxes.top + np.floor(dy).astype(np.intp) - margin
    lefts = boxes.left + np.floor(dx).astype(np.intp) - margin
    result = np.zeros((len(boxes), size, size), np.float32)
    inside = (tops >= 0) & (lefts >= 0) & (tops + size <= height) & (lefts + size <= width)
    for index in np.flatnonzero(inside):
        result[index] = image[tops[index] : tops[index] + size, lefts[index] : lefts[index] + size]
    return result


@dataclass(frozen=True)
class FrameMeasures:
    """What the first passes learn about every frame."""

    quality: Float64  # (frames,) whole-frame sharpness
    shift: Float64  # (frames, 2) global (dy, dx) relative to the reference
    box_quality: Float64  # (frames, boxes) sharpness of each box
    box_shift: Float64  # (frames, boxes, 2) shift of each box beyond the global shift


def _parallel_map[T](function: Callable[[int], T], count: int, workers: int | None) -> list[T]:
    with ThreadPoolExecutor(workers or os.process_cpu_count()) as pool:
        return list(pool.map(function, range(count)))


def global_alignment(
    frames: FrameSource, count: int, reference: Float32, workers: int | None = None
) -> Float64:
    """The (dy, dx) of every frame relative to the reference, by phase correlation."""
    window = np.outer(*(_hann(s // GLOBAL_BINNING) for s in reference.shape)).astype(np.float32)
    target = np.fft.rfft2(_binned(reference, GLOBAL_BINNING) * window)

    def measure(index: int) -> tuple[float, float]:
        surface = phase_correlate(target, _binned(frames(index), GLOBAL_BINNING) * window)
        dy, dx = _peak(surface)
        return dy * GLOBAL_BINNING, dx * GLOBAL_BINNING

    return np.array(_parallel_map(measure, count, workers))


def aligned_mean(
    frames: FrameSource, indices: Sequence[int], shift: Float64, shape: tuple[int, int]
) -> Float32:
    """The average of some frames, each moved back onto the reference."""
    total = np.zeros(shape, np.float64)
    for index in indices:
        dy, dx = shift[index]
        total += shift_image(frames(index), -dy, -dx)
    result: Float32 = (total / len(indices)).astype(np.float32)
    return result


def measure_boxes(
    frames: FrameSource,
    count: int,
    reference: Float32,
    shift: Float64,
    boxes: Boxes,
    workers: int | None = None,
) -> tuple[Float64, Float64]:
    """Per frame and box: the box's sharpness, and its shift beyond the global one."""
    window = np.outer(_hann(boxes.size), _hann(boxes.size)).astype(np.float32)
    zero = np.zeros(len(boxes))
    plain_reference = _cut(reference, boxes, zero, zero, 0)
    targets = np.fft.rfft2(plain_reference * window)
    # For refining the shifts: the reference's gradient in each box and the
    # matrix of the least-squares problem, which is the same for every frame.
    gradient_y, gradient_x = np.gradient(plain_reference.astype(np.float64), axis=(1, 2))
    syy = (window * gradient_y**2).sum(axis=(1, 2))
    sxx = (window * gradient_x**2).sum(axis=(1, 2))
    sxy = (window * gradient_y * gradient_x).sum(axis=(1, 2))
    determinant = syy * sxx - sxy**2
    solvable = determinant > 1e-12 * np.maximum(syy * sxx, 1e-30)
    frequency_y = np.fft.fftfreq(boxes.size)[None, :, None]
    frequency_x = np.fft.rfftfreq(boxes.size)[None, None, :]

    def measure(index: int) -> tuple[Float64, Float64]:
        image = frames(index)
        dy, dx = shift[index]
        whole_y, whole_x = np.floor(dy), np.floor(dx)
        # Detail and brightness are measured where the box sits in this frame.
        moved = Boxes(
            np.clip(boxes.top + int(whole_y), 0, image.shape[0] - boxes.size),
            np.clip(boxes.left + int(whole_x), 0, image.shape[1] - boxes.size),
            boxes.size,
        )
        energy = _box_sums(detail(image) ** 2, moved) / boxes.size**2
        level = _box_sums(image, moved) / boxes.size**2
        quality = energy / np.maximum(level, 1e-6) ** 2
        plain = _cut(image, boxes, np.full(len(boxes), dy), np.full(len(boxes), dx), 0)
        cross = np.fft.rfft2(plain * window) * np.conj(targets)
        cross /= np.maximum(np.abs(cross), 1e-12)
        surfaces = np.fft.irfft2(cross, s=(boxes.size, boxes.size))
        # Boxes were cut at the whole-pixel global shift, so what is measured
        # here is the rest: its fraction, plus the box's own movement.
        local = np.array([_peak(surface) for surface in surfaces])
        # The phase correlation peak is a narrow spike on sharp frames, and its
        # fitted position can be a few tenths of a pixel out: too coarse for
        # drizzle. Two Lucas-Kanade steps correct it: each box is moved back by
        # its shift so far, and the shift that remains is solved for from the
        # difference with the reference and the reference's gradient.
        spectrum = np.fft.rfft2(plain)
        for _ in range(2):
            ramp = np.exp(
                2j
                * np.pi
                * (frequency_y * local[:, 0, None, None] + frequency_x * local[:, 1, None, None])
            )
            moved_back = np.fft.irfft2(spectrum * ramp, s=(boxes.size, boxes.size))
            difference = window * (plain_reference - moved_back)
            by = (gradient_y * difference).sum(axis=(1, 2))
            bx = (gradient_x * difference).sum(axis=(1, 2))
            with np.errstate(divide="ignore", invalid="ignore"):
                rest_y = np.where(solvable, (sxx * by - sxy * bx) / determinant, 0.0)
                rest_x = np.where(solvable, (syy * bx - sxy * by) / determinant, 0.0)
            # A step of more than a pixel means the linear model does not hold.
            trusted = (np.abs(rest_y) < 1) & (np.abs(rest_x) < 1)
            local[:, 0] += np.where(trusted, rest_y, 0.0)
            local[:, 1] += np.where(trusted, rest_x, 0.0)
        return quality, local

    results = _parallel_map(measure, count, workers)
    return np.array([r[0] for r in results]), np.array([r[1] for r in results])


# A box's own shift is trusted up to this many pixels; beyond that the
# correlation has most likely locked onto the wrong feature.
MAX_BOX_SHIFT = BOX / 4


def _drizzle(
    total: Float64,
    weight: Float64,
    cut: Float32,
    fraction_y: float,
    fraction_x: float,
    scale: int,
) -> None:
    """Drop the pixels of one cut box onto a grid scale times finer, in place.

    The cut has SHIFT_MARGIN pixels on each side and starts a whole number of
    pixels from the box; fraction_y and fraction_x are the rest of its shift.
    Each pixel lands where its centre falls on the fine grid and is shared
    between the two nearest fine pixels along each axis. Frames shifted by
    different fractions of a pixel land between each other's pixels, so
    together they fill in detail finer than one of them holds (drizzle).
    """
    count = cut.shape[0]
    fine = total.shape[0]
    placements = []
    for fraction in (fraction_y, fraction_x):
        # Pixel i of the cut is at box coordinate i - margin - fraction, and a box
        # coordinate p is at (p + 0.5) * scale - 0.5 on the fine grid.
        start = scale * (0.5 - SHIFT_MARGIN - fraction) - 0.5
        first = int(np.floor(start))
        share = start - first
        axis = []
        for step, part in ((0, 1 - share), (1, share)):
            index = scale * np.arange(count) + first + step
            inside = (index >= 0) & (index < fine)
            axis.append((index[inside], np.flatnonzero(inside), part))
        placements.append(axis)
    for rows, source_rows, part_y in placements[0]:
        for columns, source_columns, part_x in placements[1]:
            part = part_y * part_x
            if part <= 0:
                continue
            total[np.ix_(rows, columns)] += part * cut[np.ix_(source_rows, source_columns)]
            weight[np.ix_(rows, columns)] += part


def stack_boxes(
    frames: FrameSource,
    count: int,
    background: Float32,
    shift: Float64,
    boxes: Boxes,
    box_quality: Float64,
    box_shift: Float64,
    keep: float,
    scale: int = 1,
    sharp: FrameSource | None = None,
) -> tuple[Float32, Float64]:
    """Stack each box from its sharpest frames and blend the boxes together.

    Returns the image and the number of frames each box used. Where the boxes
    do not reach, the background (the globally aligned average) shows
    through, faded in under the edge of the boxes so there is no seam.

    Boxes are chosen and placed from frames, but their pixels are taken from
    sharp when it is given. With scale above 1 they are drizzled onto a grid
    scale times finer, and background must be on that grid: see _drizzle.
    """
    per_box = max(1, round(keep * count))
    usable = np.all(np.abs(box_shift) <= MAX_BOX_SHIFT, axis=2)
    ranked = np.where(usable, box_quality, -np.inf)
    chosen = np.argsort(-ranked, axis=0)[:per_box]  # (per_box, boxes)
    chosen_mask = np.zeros((count, len(boxes)), bool)
    for box in range(len(boxes)):
        picks = chosen[:, box]
        chosen_mask[picks[np.isfinite(ranked[picks, box])], box] = True

    size = boxes.size
    fine = size * scale
    source = sharp or frames
    totals = np.zeros((len(boxes), fine, fine))
    weights = np.zeros((len(boxes), fine, fine))
    used = np.zeros(len(boxes))
    for index in np.flatnonzero(chosen_mask.any(axis=1)).tolist():
        selected = np.flatnonzero(chosen_mask[index])
        dy, dx = shift[index]
        local = box_shift[index, selected]
        subset = Boxes(boxes.top[selected], boxes.left[selected], size)
        # The total shift of each box: global whole pixels, then what remains.
        total_y = np.floor(dy) + local[:, 0]
        total_x = np.floor(dx) + local[:, 1]
        cut = _cut(source(index), subset, total_y, total_x, SHIFT_MARGIN)
        for k, chosen_box in enumerate(selected.tolist()):
            if not cut[k].any():
                continue
            fraction_y = float(total_y[k] - np.floor(total_y[k]))
            fraction_x = float(total_x[k] - np.floor(total_x[k]))
            if scale == 1:
                moved = fourier_shift(cut[k], -fraction_y, -fraction_x)
                totals[chosen_box] += moved[SHIFT_MARGIN:-SHIFT_MARGIN, SHIFT_MARGIN:-SHIFT_MARGIN]
                weights[chosen_box] += 1
            else:
                _drizzle(
                    totals[chosen_box], weights[chosen_box], cut[k], fraction_y, fraction_x, scale
                )
            used[chosen_box] += 1

    window = np.outer(_hann(fine), _hann(fine))
    image = np.zeros(background.shape)
    weight = np.zeros(background.shape)
    for filled in np.flatnonzero(used).tolist():
        t, left = boxes.top[filled] * scale, boxes.left[filled] * scale
        under = background[t : t + fine, left : left + fine]
        # A fine pixel that few frames reached takes the background instead.
        enough = weights[filled] >= 0.02 * used[filled]
        mean = np.where(enough, totals[filled] / np.maximum(weights[filled], 1e-12), under)
        image[t : t + fine, left : left + fine] += window * mean
        weight[t : t + fine, left : left + fine] += window
    weight = np.minimum(weight, 1.0)
    result: Float32 = (image + (1 - weight) * background).astype(np.float32)
    return result, used


@dataclass(frozen=True)
class LuckyStack:
    image: Float32
    background: float  # the level subtracted from every frame
    reference_index: int
    shift: Float64  # (frames, 2) global (dy, dx) of every frame
    frames_per_box: Float64
    boxes: Boxes
    quality: Float64  # 0 for the frames left out
    whole: npt.NDArray[np.bool_]  # the frames that show the whole target


def lucky_stack(
    frames: FrameSource,
    count: int,
    shape: tuple[int, int],
    keep: float = 0.1,
    box_size: int = BOX,
    workers: int | None = None,
    log: Callable[[str], None] = lambda message: None,
    sharp: FrameSource | None = None,
    scale: int = 1,
) -> LuckyStack:
    """Stack a run of frames of the same object, keeping the best of each part.

    The background (the sensor's black level and any sky glow), taken as the
    median of the first frame, is subtracted from every frame, so the sky of
    the stack is black and the strips that shifting brings in at the edges
    match it.

    frames are what the frames are ranked and aligned on. If sharp is given,
    the stack is made from it instead: the same frames with nothing smoothed,
    such as a balanced Bayer mosaic. With scale above 1 the stack is drizzled
    onto a grid scale times finer.
    """
    background = np.float32(np.median(frames(0)))

    def cleaned(index: int) -> Float32:
        result: Float32 = frames(index) - background
        return result

    def measure(index: int) -> tuple[float, Float32]:
        image = cleaned(index)
        return sharpness(image), image[::WHOLE_THINNING, ::WHOLE_THINNING].copy()

    measured = _parallel_map(measure, count, workers)
    quality = np.array([m[0] for m in measured])
    whole = whole_frames([m[1] for m in measured])
    del measured
    if not whole.all():
        log(f"{count - int(whole.sum())} frames left out: the target is not whole in them")
    # Frames left out are never the reference nor among the best.
    quality = np.where(whole, quality, 0.0)
    best = int(np.argmax(quality))
    log(f"sharpest frame {best}, sharpness {quality[whole].min():.3g} to {quality.max():.3g}")

    shift = global_alignment(cleaned, count, cleaned(best), workers)
    ranked = np.argsort(-quality)
    top = ranked[whole[ranked]][: max(1, round(keep * int(whole.sum())))]
    reference = aligned_mean(cleaned, top.tolist(), shift, shape)
    # Measure again against the average of the best frames, which is less
    # noisy than any one of them.
    shift = global_alignment(cleaned, count, reference, workers)
    reference = aligned_mean(cleaned, top.tolist(), shift, shape)
    log(f"global shifts up to {np.abs(shift[whole]).max():.1f} px")

    boxes = place_boxes(reference, box_size)
    box_quality, box_shift = measure_boxes(cleaned, count, reference, shift, boxes, workers)
    box_quality[~whole] = -np.inf
    log(f"{len(boxes)} alignment boxes")
    cleaned_sharp: FrameSource | None = None
    if sharp is not None:
        sharp_background = np.float32(np.median(sharp(0)))

        def cleaned_sharp(index: int) -> Float32:
            result: Float32 = sharp(index) - sharp_background
            return result

    under = upsample(reference, scale, workers) if scale > 1 else reference
    image, used = stack_boxes(
        cleaned,
        count,
        under,
        shift,
        boxes,
        box_quality,
        box_shift,
        keep,
        scale=scale,
        sharp=cleaned_sharp,
    )
    return LuckyStack(image, float(background), best, shift, used, boxes, quality, whole)


def rotation_matrix(angle: float, shape: tuple[int, int]) -> Float64:
    """A turn by angle (radians) about the centre of a frame, on (x, y, 1)."""
    cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [cos, -sin, cx - cos * cx + sin * cy],
            [sin, cos, cy - sin * cx - cos * cy],
            [0, 0, 1],
        ]
    )


def alignment_matrix(angle: float, dy: float, dx: float, shape: tuple[int, int]) -> Float64:
    """The transformation from reference to image coordinates found by find_rotation:
    the shift seen after turning the image back, then the turn."""
    shift = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
    matrix: Float64 = rotation_matrix(angle, shape) @ shift
    return matrix


def moon_parallactic_angle(time: str, latitude: float, longitude: float) -> float:
    """The Moon's parallactic angle, in radians, at a UTC time (ISO) from a site.

    On an alt-azimuth mount the sky turns in the frame by the change in this
    angle, so the difference between two clips' angles is how far one is
    turned against the other: about 95 degrees over a long night.
    """
    # UT1 from UTC without the IERS tables (no download, any age): the error
    # is under a second, far below what matters here.
    with iers.conf.set_temp("auto_download", False), iers.conf.set_temp("auto_max_age", None):
        when = Time(time, scale="utc")
        site = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg)
        moon = get_body("moon", when, site)
        hour_angle = (when.sidereal_time("apparent", longitude=site.lon) - moon.ra).radian
    declination, phi = moon.dec.radian, np.radians(latitude)
    return float(
        np.arctan2(
            np.sin(hour_angle),
            np.tan(phi) * np.cos(declination) - np.sin(declination) * np.cos(hour_angle),
        )
    )


def find_rotation(
    reference: Float32,
    image: Float32,
    max_angle: float = np.radians(15),
    workers: int | None = None,
    expected: float = 0.0,
) -> tuple[float, float, float]:
    """The turn and shift that best line an image up with the reference.

    An alt-azimuth mount lets the field turn slowly while it tracks, so
    stacks made minutes apart differ by a rotation as well as a shift. Each
    trial angle is undone and the remaining shift found by phase correlation;
    the angle with the sharpest correlation peak wins, first on a coarse
    grid and then on a finer one around the best. Angles are tried within
    max_angle of expected; over a whole night the field turns much further
    than any search should reach, so pass the predicted turn as expected
    (see moon_parallactic_angle).
    """

    def best_angle(angles: Float64, size: int) -> float:
        """The trial angle with the sharpest correlation, on images binned to about size."""
        factor = max(1, min(reference.shape) // size)
        small_reference = _binned(reference, factor)
        small_image = _binned(image, factor)
        shape = small_reference.shape
        window = np.outer(*(_hann(s) for s in shape)).astype(np.float32)
        target = np.fft.rfft2(small_reference * window)

        def score(angle: float) -> float:
            turned = resample(small_image, rotation_matrix(angle, shape), workers=1)
            return float(phase_correlate(target, turned * window).max())

        with ThreadPoolExecutor(workers or os.process_cpu_count()) as pool:
            scores = list(pool.map(score, angles))
        return float(angles[int(np.argmax(scores))])

    # Coarse steps on a small image, finer ones on larger images, since a
    # small turn moves the edge of a small image by less than a pixel.
    trials = np.arange(-max_angle, max_angle + 1e-9, np.radians(0.5))
    angle = best_angle(expected + trials, 256)
    for step, size in ((np.radians(0.1), 256), (np.radians(0.02), 512)):
        angle = best_angle(np.arange(angle - 5 * step, angle + 5 * step + 1e-9, step), size)
    # The shift found at the coarse scale, then refined at full resolution.
    turned = resample(image, rotation_matrix(angle, reference.shape), workers=workers)
    full_window = np.outer(*(_hann(s) for s in reference.shape)).astype(np.float32)
    dy, dx = _peak(phase_correlate(np.fft.rfft2(reference * full_window), turned * full_window))
    return angle, dy, dx


def sharpen(image: Float32, gains: Sequence[float], noise_threshold: float = 0.0) -> Float32:
    """Multiply each wavelet detail layer by its gain, finest first.

    With a threshold, values of the finest layer smaller than that many
    times its noise are left out before amplifying, so noise is not
    sharpened along with the detail.
    """
    layers, residual = b3_layers(image, len(gains))
    if noise_threshold > 0:
        finest = layers[0]
        noise = 1.4826 * float(np.median(np.abs(finest - np.median(finest))))
        limit = noise_threshold * noise
        layers[0] = (np.sign(finest) * np.maximum(np.abs(finest) - limit, 0)).astype(np.float32)
    result = residual.astype(np.float64)
    for layer, gain in zip(layers, gains, strict=True):
        result += gain * layer
    output: Float32 = result.astype(np.float32)
    return output


def colour_mean(
    frames: Callable[[int], Float32], indices: Sequence[int], shift: Float64
) -> Float32:
    """The average of some (H, W, 3) colour frames, each moved back onto the reference."""
    total: Float64 | None = None
    for index in indices:
        rgb = frames(index)
        dy, dx = shift[index]
        moved = np.stack([shift_image(rgb[..., c], -dy, -dx) for c in range(3)], axis=-1)
        total = moved.astype(np.float64) if total is None else total + moved
    if total is None:
        raise ValueError("no frames to average")
    result: Float32 = (total / len(indices)).astype(np.float32)
    return result


# Where the luminance is below this fraction of the bright end, the colour of
# the sky is noise; it fades to grey there.
COLOUR_MIN_LEVEL = 0.05


def mineral_colour(
    luminance: Float32, colour: Float32, saturation: float, smoothing: float = 2.0
) -> Float32:
    """Colour a sharp luminance image with the hue of a softer colour image.

    The colour is balanced so the median of the lit disc is grey, smoothed,
    and its differences from grey multiplied by saturation; the Moon's colours
    are faint, so a strong boost is what makes them visible at all.
    """
    rgb = colour.astype(np.float64) - np.median(colour, axis=(0, 1))
    grey = rgb.mean(axis=2)
    bright = float(np.percentile(grey, 99.5))
    lit = grey > 0.1 * bright
    balance = np.median(rgb[lit], axis=0)
    rgb = rgb / balance * balance.mean()
    grey = rgb.mean(axis=2)
    # How far each pixel's colour is from grey, counted only where the disc is
    # lit; the sky's colour is undefined and must not bleed into the limb.
    weight = np.clip(grey / (COLOUR_MIN_LEVEL * bright), 0, 1)
    deviation = np.where(weight[..., None] > 0, rgb / np.maximum(grey, 1e-9)[..., None] - 1, 0)
    reach = gaussian_blur(weight.astype(np.float32), smoothing)
    smooth = np.stack(
        [
            gaussian_blur((deviation[..., c] * weight).astype(np.float32), smoothing)
            / np.maximum(reach, 1e-3)
            for c in range(3)
        ],
        axis=-1,
    )
    factor = 1 + smooth * saturation * np.clip(reach, 0, 1)[..., None]
    result: Float32 = (np.clip(luminance, 0, None)[..., None] * np.clip(factor, 0, None)).astype(
        np.float32
    )
    return result


def align_channels(colour: Float32) -> tuple[Float32, list[tuple[float, float]]]:
    """Move the red and blue channels onto the green one.

    Low in the sky the atmosphere bends blue light more than red, so the
    channels of a colour image are offset by a pixel or two along the
    vertical; uncorrected, the limb gets a red fringe on one side and a blue
    one on the other. Returns the corrected image and the (dy, dx) applied to
    red and to blue.
    """
    window = np.outer(*(_hann(s) for s in colour.shape[:2])).astype(np.float32)
    green = colour[..., 1] - np.float32(np.median(colour[..., 1]))
    target = np.fft.rfft2(green * window)
    channels = [colour[..., 0], colour[..., 1], colour[..., 2]]
    moves = []
    for c in (0, 2):
        channel = colour[..., c] - np.float32(np.median(colour[..., c]))
        # A peak fit is only exact for small offsets, so the channel is moved
        # by the estimate and what remains is measured again.
        dy = dx = 0.0
        for _ in range(4):
            moved = shift_image(channel, -dy, -dx)
            surface = np.fft.irfft2(np.fft.rfft2(moved * window) * np.conj(target), s=channel.shape)
            rest_y, rest_x = _peak(surface)
            dy, dx = dy + rest_y, dx + rest_x
        channels[c] = shift_image(colour[..., c], -dy, -dx)
        moves.append((-dy, -dx))
    result: Float32 = np.stack(channels, axis=-1).astype(np.float32)
    return result, moves
