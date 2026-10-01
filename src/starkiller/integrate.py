"""Combine a stack of frames into one, rejecting outlier pixels.

The rejection algorithm reproduces PixInsight's ImageIntegration "Winsorized
sigma clipping" closely enough to give the same rejection decisions, which
tests/test_reference_m13.py checks against a 2018 PixInsight run.
"""

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from starkiller.estimators import ikss
from starkiller.frame import DiskStack

Float = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]
Index = npt.NDArray[np.int64]
# Frames to integrate: an (N, H, W) array, or the same read from files.
Stack = npt.NDArray[Any] | DiskStack

# Constants of the Winsorization step: clip at 1.5 sigma, then correct the
# standard deviation of the clipped sample back to that of a normal one.
WINSOR_CUTOFF = 1.5
WINSOR_SIGMA_CORRECTION = 1.134
WINSOR_TOLERANCE = 0.0005
WINSOR_MAX_ITERATIONS = 10_000


@dataclass(frozen=True)
class Integration:
    """Result of integrating N frames of shape (H, W)."""

    image: npt.NDArray[np.float32]
    # Fraction of the N frames rejected at each pixel, as PixInsight's
    # rejection maps store it.
    rejection_low: npt.NDArray[np.float32]
    rejection_high: npt.NDArray[np.float32]
    # Number of pixels rejected in each frame.
    frame_rejected_low: Index
    frame_rejected_high: Index


def _std(values: Float, mask: Bool, n: Index) -> Float:
    """Sample standard deviation down each column over the masked rows."""
    mean = np.where(mask, values, 0.0).sum(axis=0) / n
    dev = np.where(mask, values - mean, 0.0)
    var = ((dev * dev).sum(axis=0) - dev.sum(axis=0) ** 2 / n) / (n - 1)
    return np.sqrt(np.maximum(var, 0.0))


def _median(sorted_values: Float, lo: Index, n: Index) -> Float:
    """Median down each column of the n sorted values starting at row lo."""
    a = np.take_along_axis(sorted_values, (lo + (n - 1) // 2)[None], axis=0)[0]
    b = np.take_along_axis(sorted_values, (lo + n // 2)[None], axis=0)[0]
    median: Float = 0.5 * (a + b)
    return median


def _winsorized_sigma(values: Float, mask: Bool, lo: Index, n: Index) -> tuple[Float, Float, Bool]:
    """Robust (median, sigma) of each sorted column, by iterated Winsorization.

    Returns median, sigma and a flag for columns where a sigma exists at all
    (at least three values, not all equal). A sigma of exactly zero with the
    flag set means the estimate collapsed: every value that differs from the
    median is an outlier.
    """
    cols = values.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = _std(values, mask, n)
        median = _median(values, lo, n)
        valid = (n >= 3) & (1.0 + sigma != 1.0)

        # Work on the columns still iterating, dropping each one as it settles.
        cur = np.flatnonzero(valid)
        w = values[:, cur]
        w_mask, w_lo, w_n = mask[:, cur], lo[cur], n[cur]
        w_sigma, w_median = sigma[cur], median[cur]
        out_sigma = np.zeros(cols)
        out_median = median.copy()

        for iteration in range(1, WINSOR_MAX_ITERATIONS + 1):
            if not cur.size:
                break
            t0 = w_median - WINSOR_CUTOFF * w_sigma
            t1 = w_median + WINSOR_CUTOFF * w_sigma
            # PixInsight holds the stack as 32-bit samples, so the clipped
            # values are rounded to that precision.
            t0 = t0.astype(np.float32).astype(np.float64)
            t1 = t1.astype(np.float32).astype(np.float64)
            w = np.clip(w, t0, t1)
            previous = w_sigma
            w_sigma = WINSOR_SIGMA_CORRECTION * _std(w, w_mask, w_n)
            w_median = _median(w, w_lo, w_n)

            change = np.abs(previous - w_sigma) / previous
            converged = change < WINSOR_TOLERANCE if iteration > 1 else np.zeros(cur.size, np.bool_)
            # Once every value off the median is clipped, sigma shrinks by the
            # same ratio each iteration and never converges. This is common in
            # quantised data (bias frames). PixInsight iterates it down to the
            # resolution of its 32-bit samples, thousands of iterations, where
            # rounding decides the outcome: a ratio above one half sticks at a
            # tiny non-zero sigma, so everything off the median is rejected;
            # below one half sigma reaches zero and the column is left alone.
            pinned = ((w == t0) | (w == t1) | (w == w_median) | ~w_mask).all(axis=0)
            runaway = pinned & (w_sigma < previous) & (change >= WINSOR_TOLERANCE)
            vanished = (runaway & (w_sigma < 0.5 * previous)) | (1.0 + w_sigma == 1.0)
            collapsed = runaway & ~vanished
            w_sigma = np.where(collapsed, 0.0, w_sigma)
            valid[cur[vanished]] = False

            done = converged | collapsed | vanished
            out_sigma[cur[done]] = w_sigma[done]
            out_median[cur[done]] = w_median[done]
            keep = ~done
            cur, w, w_mask = cur[keep], w[:, keep], w_mask[:, keep]
            w_lo, w_n, w_sigma, w_median = w_lo[keep], w_n[keep], w_sigma[keep], w_median[keep]
        else:
            # Not settled within the iteration budget: use the last estimate.
            out_sigma[cur] = w_sigma
            out_median[cur] = w_median

    return out_median, out_sigma, valid


def winsorized_sigma_clip(
    stack: Float,
    sigma_low: float = 4.0,
    sigma_high: float = 3.0,
    excluded_low: Bool | None = None,
    excluded_high: Bool | None = None,
) -> tuple[Bool, Bool]:
    """Find outliers down each column of an (N, P) stack.

    Returns two (N, P) masks, in the stack's own row order: values rejected
    for being too low and too high. A value is rejected when it lies at least
    sigma_low (or sigma_high) robust standard deviations from the median of
    the values not yet rejected; this repeats until nothing more is rejected.

    Values marked in excluded_low and excluded_high take no part and come
    back as rejected on their side.
    """
    frames, cols = stack.shape
    lo = np.zeros(cols, np.int64)
    hi = np.full(cols, frames, np.int64)
    if excluded_low is not None:
        stack = np.where(excluded_low, -np.inf, stack)
        lo = excluded_low.sum(axis=0)
    if excluded_high is not None:
        stack = np.where(excluded_high, np.inf, stack)
        hi = frames - excluded_high.sum(axis=0)
    order = np.argsort(stack, axis=0, kind="stable")
    values = np.take_along_axis(stack, order, axis=0)
    row = np.arange(frames)[:, None]

    # Outliers come off the two ends of each sorted column, so the survivors
    # are always the rows lo <= row < hi. Fewer than three cannot be judged.
    active = np.flatnonzero(hi - lo >= 3)
    while active.size:
        v, a_lo, a_hi = values[:, active], lo[active], hi[active]
        mask = (row >= a_lo) & (row < a_hi)
        median, sigma, valid = _winsorized_sigma(np.where(mask, v, 0.0), mask, a_lo, a_hi - a_lo)
        usable = mask & valid
        low = usable & (v < median) & (median - v >= sigma_low * sigma)
        high = usable & (v > median) & (v - median >= sigma_high * sigma)
        n_low, n_high = low.sum(axis=0), high.sum(axis=0)
        lo[active] = a_lo + n_low
        hi[active] = a_hi - n_high
        active = active[(n_low + n_high) > 0]

    rejected_low = np.zeros((frames, cols), np.bool_)
    rejected_high = np.zeros((frames, cols), np.bool_)
    np.put_along_axis(rejected_low, order, row < lo, axis=0)
    np.put_along_axis(rejected_high, order, row >= hi, axis=0)
    return rejected_low, rejected_high


@dataclass(frozen=True)
class Normalization:
    """How each frame of a stack is brought into line with the first.

    A pixel of frame i becomes (value - subtract[i]) * multiply[i] + add.
    """

    subtract: Float
    multiply: Float
    add: float


def frame_estimates(stack: Stack, workers: int | None = None) -> tuple[Float, Float]:
    """Location and scale of every frame of an (N, H, W) stack."""
    with ThreadPoolExecutor(workers or os.process_cpu_count()) as pool:
        # One frame per worker at a time, so a stack on disk is never
        # loaded whole.
        estimates = np.array(list(pool.map(lambda index: ikss(stack[index]), range(len(stack)))))
    return estimates[:, 0], estimates[:, 1]


def flux_normalization(locations: Float) -> Normalization:
    """Match the brightness of frames by multiplication.

    Flat frames taken against a changing light source differ in brightness
    but not in shape.
    """
    return Normalization(np.zeros(len(locations)), locations[0] / locations, 0.0)


def level_and_scale_normalization(locations: Float, scales: Float) -> Normalization:
    """Match both the background level and the spread of frames.

    Light frames differ in sky brightness, which adds, and in transparency
    and exposure, which scale the signal.
    """
    return Normalization(locations, scales[0] / scales, float(locations[0]))


def noise_weights(noise: Float, normalization: Normalization) -> Float:
    """Weights that favour the frames with less noise once normalised.

    A frame's weight is the inverse of its noise variance after scaling,
    relative to the first frame.
    """
    weights: Float = (noise[0] / (noise * normalization.multiply)) ** 2
    return weights


def integrate(
    stack: Stack,
    sigma_low: float = 4.0,
    sigma_high: float = 3.0,
    normalization: Normalization | None = None,
    weights: Float | None = None,
    valid_range: tuple[float, float] | None = None,
    chunk_rows: int = 16,
    workers: int | None = None,
) -> Integration:
    """Average an (N, H, W) stack of frames, ignoring rejected pixels.

    Bias and dark frames are combined as they are. Flats and lights are
    first normalised (see flux_normalization, level_and_scale_normalization)
    so that frames can be compared, and lights are weighted by their noise.
    With valid_range, pixels at or outside its limits, such as the black
    borders of registered frames and saturated stars, are rejected outright,
    except that a pixel saturated in every frame that covers it stays
    saturated.

    The stack is processed chunk_rows rows at a time, on workers threads
    (one per CPU by default), to bound memory; it may be a DiskStack, which
    reads those rows from files as they are needed.
    """
    frames, height, width = stack.shape
    image = np.empty((height, width), np.float32)
    rejection_low = np.empty((height, width), np.float32)
    rejection_high = np.empty((height, width), np.float32)
    frame_low = np.zeros(frames, np.int64)
    frame_high = np.zeros(frames, np.int64)
    frame_weights = np.ones(frames) if weights is None else weights

    def integrate_rows(top: int) -> tuple[Index, Index]:
        rows = slice(top, min(top + chunk_rows, height))
        chunk = np.asarray(stack[:, rows, :], dtype=np.float64).reshape(frames, -1)
        too_low = too_high = None
        if valid_range is not None:
            too_low, too_high = chunk <= valid_range[0], chunk >= valid_range[1]
            # Where every frame that covers a pixel is saturated, as in the
            # core of a bright star, saturated is the right answer; rejecting
            # them all would leave a black hole.
            too_high &= (~too_low & ~too_high).any(axis=0)
        if normalization is not None:
            chunk = (chunk - normalization.subtract[:, None]) * normalization.multiply[:, None]
            # Rounded to 32 bits, as PixInsight holds the normalised samples.
            chunk = (chunk + normalization.add).astype(np.float32).astype(np.float64)
        low, high = winsorized_sigma_clip(chunk, sigma_low, sigma_high, too_low, too_high)
        kept = ~(low | high) * frame_weights[:, None]
        total = kept.sum(axis=0)
        with np.errstate(invalid="ignore"):
            mean = np.where(total > 0, (kept * chunk).sum(axis=0) / total, 0.0)
        shape = image[rows].shape
        image[rows] = mean.reshape(shape)
        rejection_low[rows] = (low.sum(axis=0) / frames).reshape(shape)
        rejection_high[rows] = (high.sum(axis=0) / frames).reshape(shape)
        return low.sum(axis=1), high.sum(axis=1)

    with ThreadPoolExecutor(workers or os.process_cpu_count()) as pool:
        for low_counts, high_counts in pool.map(integrate_rows, range(0, height, chunk_rows)):
            frame_low += low_counts
            frame_high += high_counts

    return Integration(image, rejection_low, rejection_high, frame_low, frame_high)


# Pixels at or beyond these limits are black registration borders or
# saturated, and are left out when lights are combined.
LIGHT_RANGE = (0.0, 0.98)


def integrate_lights(
    channels: Sequence[Stack],
    noise: Float,
    sigma_low: float = 4.0,
    sigma_high: float = 3.0,
) -> list[Integration]:
    """Integrate registered lights, one result per colour channel.

    channels holds an (N, H, W) stack per colour channel and noise the
    (N, C) noise of each frame's channels, ideally measured before
    registration, which smooths it. Each channel is normalised in level and
    scale to the first frame, weighted by noise and cleared of black and
    saturated pixels.
    """
    results = []
    for channel, frames in enumerate(channels):
        normalization = level_and_scale_normalization(*frame_estimates(frames))
        weights = noise_weights(noise[:, channel], normalization)
        results.append(
            integrate(
                frames, sigma_low, sigma_high, normalization, weights, valid_range=LIGHT_RANGE
            )
        )
    return results
