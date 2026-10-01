import numpy as np

from starkiller.integrate import (
    flux_normalization,
    frame_estimates,
    integrate,
    level_and_scale_normalization,
    noise_weights,
    winsorized_sigma_clip,
)


def noise(frames: int, pixels: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.1, 0.001, (frames, pixels))


def test_rejects_an_outlier_in_the_frame_it_came_from() -> None:
    stack = noise(12, 50)
    stack[7, 20] += 0.5  # a cosmic ray hit
    stack[3, 31] -= 0.05  # a dead pixel

    low, high = winsorized_sigma_clip(stack, sigma_low=6, sigma_high=6)

    assert np.argwhere(high).tolist() == [[7, 20]]
    assert np.argwhere(low).tolist() == [[3, 31]]


def test_rejects_nothing_from_identical_values() -> None:
    low, high = winsorized_sigma_clip(np.full((10, 4), 0.25))

    assert not low.any()
    assert not high.any()


def test_rejects_nothing_from_fewer_than_three_frames() -> None:
    low, high = winsorized_sigma_clip(np.array([[0.1], [0.9]]))

    assert not low.any()
    assert not high.any()


def test_quantised_stack_keeps_only_the_median_value() -> None:
    # Bias frames are mostly one value per pixel with a few a step either
    # side. PixInsight's sigma estimate collapses on such a stack and it
    # rejects every value off the median; we follow it.
    step = 1 / 65535
    stack = np.array([128 - 1] * 5 + [128] * 13 + [128 + 1] * 2)[:, None] * step

    low, high = winsorized_sigma_clip(stack)

    assert low[:, 0].tolist() == [True] * 5 + [False] * 15
    assert high[:, 0].tolist() == [False] * 18 + [True] * 2


def test_lone_outlier_among_identical_values_is_kept() -> None:
    # The other side of the same PixInsight behaviour: with a single value
    # off the median, sigma reaches zero and nothing is rejected at all.
    stack = np.array([128] * 19 + [129])[:, None] / 65535

    low, high = winsorized_sigma_clip(stack)

    assert not low.any()
    assert not high.any()


def test_integrate_averages_the_values_it_keeps() -> None:
    stack = noise(12, 6 * 8).reshape(12, 6, 8)
    stack[7, 2, 3] += 0.5

    result = integrate(stack, sigma_low=6, sigma_high=6)

    expected = stack.mean(axis=0)
    expected[2, 3] = np.delete(stack[:, 2, 3], 7).mean()
    np.testing.assert_allclose(result.image, expected, rtol=1e-6)
    assert result.rejection_high[2, 3] == np.float32(1 / 12)
    assert result.rejection_high.sum() == np.float32(1 / 12)
    assert result.frame_rejected_high.tolist() == [0] * 7 + [1] + [0] * 4
    assert result.frame_rejected_low.sum() == 0


def test_integrate_does_not_depend_on_chunk_size() -> None:
    stack = np.rint(noise(15, 10 * 12, seed=2).reshape(15, 10, 12) * 2000) / 2000

    whole = integrate(stack, chunk_rows=64)
    by_row = integrate(stack, chunk_rows=1)

    np.testing.assert_array_equal(whole.image, by_row.image)
    np.testing.assert_array_equal(whole.rejection_low, by_row.rejection_low)
    np.testing.assert_array_equal(whole.rejection_high, by_row.rejection_high)
    assert whole.frame_rejected_low.tolist() == by_row.frame_rejected_low.tolist()
    assert whole.frame_rejected_high.tolist() == by_row.frame_rejected_high.tolist()


def flats_of_varying_brightness() -> tuple[np.ndarray, np.ndarray]:
    """Noisy copies of one vignetted field under a light that slowly fades."""
    rows, columns = np.mgrid[:32, :48]
    field = 0.5 - ((rows - 16) ** 2 + (columns - 24) ** 2) / 4000
    brightness = np.linspace(1.0, 0.8, 10)
    noise = np.random.default_rng(3).normal(1, 0.002, (10, 32, 48))
    return field * brightness[:, None, None] * noise, brightness


def test_flux_normalization_undoes_the_change_in_brightness() -> None:
    stack, brightness = flats_of_varying_brightness()

    normalization = flux_normalization(frame_estimates(stack)[0])

    np.testing.assert_allclose(normalization.multiply, brightness[0] / brightness, rtol=2e-3)
    assert not normalization.subtract.any()
    assert normalization.add == 0


def test_integrate_with_gains_combines_frames_at_the_level_of_the_first() -> None:
    stack, brightness = flats_of_varying_brightness()
    stack[6, 10, 10] *= 1.2  # a cosmic ray hit in one of the dimmer frames

    result = integrate(stack, normalization=flux_normalization(brightness))

    rows, columns = np.mgrid[:32, :48]
    field = 0.5 - ((rows - 16) ** 2 + (columns - 24) ** 2) / 4000
    np.testing.assert_allclose(result.image, field, rtol=3e-3)
    assert result.frame_rejected_high[6] >= 1
    assert result.rejection_high[10, 10] > 0


def test_without_gains_an_outlier_in_a_dim_frame_goes_unnoticed() -> None:
    stack, _ = flats_of_varying_brightness()
    stack[6, 10, 10] *= 1.2  # no brighter than the same pixel in the first frame

    result = integrate(stack)

    assert result.rejection_high[10, 10] == 0


def test_excluded_values_take_no_part_and_count_as_rejected() -> None:
    stack = noise(12, 5)
    excluded_low = np.zeros(stack.shape, np.bool_)
    excluded_high = np.zeros(stack.shape, np.bool_)
    stack[2, 1], excluded_low[2, 1] = 0.0, True  # black border of a registered frame
    stack[9, 3], excluded_high[9, 3] = 1.0, True  # a saturated pixel

    low, high = winsorized_sigma_clip(stack, 6, 6, excluded_low, excluded_high)

    assert np.argwhere(low).tolist() == [[2, 1]]
    assert np.argwhere(high).tolist() == [[9, 3]]


def test_valid_range_keeps_black_borders_out_of_the_average() -> None:
    stack = noise(12, 6 * 8).reshape(12, 6, 8)
    stack[4, :, :2] = 0.0  # one frame shifted off this edge by registration
    stack[:, 5, 7] = 0.0  # and a corner no frame covers

    result = integrate(stack, 10, 10, valid_range=(0.0, 0.98))

    expected = np.delete(stack, 4, axis=0)[:, :, :2].mean(axis=0)
    np.testing.assert_allclose(result.image[:, :2], expected, rtol=1e-6)
    assert result.image[5, 7] == 0
    assert result.frame_rejected_low[4] >= 12


def test_level_and_scale_normalization_matches_frames_to_the_first() -> None:
    rng = np.random.default_rng(4)
    sky = rng.normal(0, 1, (8, 20, 30))
    levels = np.linspace(0.10, 0.14, 8)
    scales = np.linspace(0.001, 0.002, 8)
    stack = levels[:, None, None] + scales[:, None, None] * sky

    normalization = level_and_scale_normalization(levels, scales)

    result = integrate(stack, 1000, 1000, normalization=normalization)  # no rejection

    np.testing.assert_allclose(result.image, 0.10 + 0.001 * sky.mean(axis=0), atol=1e-6)


def test_noise_weights_favour_the_quieter_frames() -> None:
    normalization = level_and_scale_normalization(np.array([0.1, 0.1, 0.1]), np.ones(3))

    weights = noise_weights(np.array([2e-5, 1e-5, 4e-5]), normalization)

    np.testing.assert_allclose(weights, [1.0, 4.0, 0.25])


def test_weights_set_each_frames_share_of_the_average() -> None:
    stack = np.stack([np.full((2, 2), 0.1), np.full((2, 2), 0.4)])

    result = integrate(stack, weights=np.array([3.0, 1.0]))

    np.testing.assert_allclose(result.image, 0.175, rtol=1e-6)
