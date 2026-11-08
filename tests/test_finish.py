import os
from pathlib import Path

import numpy as np
import pytest

from quietsky.cli import main
from quietsky.finish import Settings, finish, to_8bit
from quietsky.frame import Frame, load, save_fits
from quietsky.wavelets import b3_layers

HEART = Path(os.environ.get("QUIETSKY_DATA", "/data")) / "projects" / "heart" / "three-nights"


# The channels' share of the light: stars are white before this, so the
# balance should find gains proportional to its inverse.
CHANNEL_SCALE = (1.0, 0.9, 0.8)
STARS = (
    (60, 50, 0.3),
    (250, 190, 0.2),
    (200, 60, 0.1),
    (90, 200, 0.05),
    (300, 30, 0.08),
    (30, 120, 0.06),
    (150, 220, 0.12),
    (280, 100, 0.04),
)


def master(seed: int = 0, red_sigma: float = 1.5) -> np.ndarray:
    """A linear colour master: sky with a gradient, red nebula, white stars, noise.

    The stars' light is the same in each channel before CHANNEL_SCALE;
    red_sigma widens their profile in red, as a camera whose red channel
    is softer does, without changing their light.
    """
    rows, columns = np.mgrid[:240, :320].astype(np.float32)
    nebula = 0.01 * np.exp(-((rows - 120) ** 2 + (columns - 160) ** 2) / (2 * 40.0**2))
    rng = np.random.default_rng(seed)
    channels = []
    for colour, tint in ((0, 1.0), (1, 0.35), (2, 0.45)):
        sky = 0.010 + 0.0004 * colour + 0.00001 * rows
        channel = sky + tint * nebula
        sigma = red_sigma if colour == 0 else 1.5
        for x, y, flux in STARS:
            profile = np.exp(-((rows - y) ** 2 + (columns - x) ** 2) / (2 * sigma**2))
            channel += flux * profile * (1.5 / sigma) ** 2
        channel *= CHANNEL_SCALE[colour]
        channels.append(channel + rng.normal(0, 0.0003, channel.shape))
    result: np.ndarray = np.clip(np.stack(channels, axis=-1), 0, 1).astype(np.float32)
    return result


def bright_pixel_gains(src: np.ndarray) -> np.ndarray:
    """The Heart script's colour balance: the brightest unsaturated pixels average white."""
    rgb = src.astype(np.float64)
    sky = np.median(rgb, axis=(0, 1))
    rgb -= sky
    lum = rgb.mean(axis=2)
    bright = (lum > np.percentile(lum, 99.7)) & (rgb.max(axis=2) < 0.9 - sky.max())
    star_colour = np.median(rgb[bright] / rgb[bright].mean(axis=1, keepdims=True), axis=0)
    result: np.ndarray = 1 / star_colour
    return result


def by_hand(src: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """The finish as the Heart script did it, line for line, given the colour gains."""
    rgb = src.astype(np.float64)
    sky = np.median(rgb, axis=(0, 1))
    rgb -= sky
    lum = rgb.mean(axis=2)
    noise = 1.4826 * np.median(np.abs(lum - np.median(lum)))
    rgb *= gains
    lum = rgb.mean(axis=2)
    black = -2.5 * noise
    top = float(np.percentile(lum, 99.99)) - black
    level = float(np.median(lum)) - black

    def stretched(values: np.ndarray, a: float) -> np.ndarray:
        result: np.ndarray = np.arcsinh(a * values) / np.arcsinh(a * top)
        return result

    low, high = 1.0, 1e9
    for _ in range(100):
        mid = np.sqrt(low * high)
        low, high = (mid, high) if stretched(np.array(level), mid) < 0.13 else (low, mid)
    stretch = np.sqrt(low * high)
    chan = np.stack(
        [stretched(np.clip(rgb[..., c] - black, 0, None), stretch) for c in range(3)], axis=-1
    )
    chan -= np.median(chan, axis=(0, 1)) - np.median(chan)
    grey = chan.mean(axis=2, keepdims=True)
    out = np.clip(grey + (chan - grey) * 1.3, 0, 1)
    result: np.ndarray = (out * 255 + 0.5).astype(np.uint8)
    return result


def test_defaults_reproduce_the_finish_by_hand() -> None:
    image = master()
    out, tone = finish(image)
    assert np.array_equal(to_8bit(out), by_hand(image, tone.gains))
    assert tone.balance == 0.5
    assert np.median(out.mean(axis=2)) == pytest.approx(0.13, abs=0.01)


def test_balance_makes_the_stars_white() -> None:
    _, tone = finish(master())
    expected = 1 / np.array(CHANNEL_SCALE)
    assert tone.gains / tone.gains.mean() == pytest.approx(expected / expected.mean(), rel=0.02)


def test_balance_is_on_the_whole_star_not_its_peak() -> None:
    # Red half as wide again: the peaks hold less red than the whole stars, which are still white.
    image = master(red_sigma=2.2)
    _, tone = finish(image)
    expected = 1 / np.array(CHANNEL_SCALE)
    assert tone.gains / tone.gains.mean() == pytest.approx(expected / expected.mean(), rel=0.03)
    peak = bright_pixel_gains(image)
    assert peak[0] / peak[1] > 1.1 * tone.gains[0] / tone.gains[1]


def blob_colour(out: np.ndarray, x: int, y: int) -> np.ndarray:
    """A star's colour in the output as a blob: each channel's share of its light less the sky."""
    window = out[y - 6 : y + 7, x - 6 : x + 7]
    sky = np.median(out[y - 20 : y - 10, x - 20 : x + 20], axis=(0, 1))
    light = (window - sky).sum(axis=(0, 1))
    result: np.ndarray = light / light.mean()
    return result


def test_matching_the_widths_makes_every_pixel_of_a_star_white() -> None:
    image = master(red_sigma=2.2)
    plain, tone_plain = finish(image)
    matched, tone = finish(image, Settings(match_psf=True))
    assert tone.widths is not None
    assert tone.widths[0] > tone.widths[1] + 0.5
    x, y, _ = STARS[1]  # well away from the nebula, whose gradient a blob's sky does not cancel
    assert blob_colour(plain, x, y)[0] > 1.08
    assert blob_colour(matched, x, y) == pytest.approx(np.ones(3), abs=0.03)
    # The stretch is still set on the master as it came.
    assert (tone.stretch, tone.black, tone.gains.tolist()) == (
        tone_plain.stretch,
        tone_plain.black,
        tone_plain.gains.tolist(),
    )


def test_gains_can_be_given() -> None:
    image = master()
    out, tone = finish(image, Settings(gains=(1.0, 1.1, 1.2)))
    assert tuple(tone.gains) == (1.0, 1.1, 1.2)
    assert np.array_equal(to_8bit(out), by_hand(image, np.array([1.0, 1.1, 1.2])))


def test_too_few_stars_falls_back_to_the_brightest_pixels() -> None:
    rows, columns = np.mgrid[:120, :160].astype(np.float32)
    rng = np.random.default_rng(3)
    channels = []
    for colour in range(3):
        channel = 0.01 + 0.02 * np.exp(-((rows - 60) ** 2 + (columns - 80) ** 2) / (2 * 30.0**2))
        channel *= CHANNEL_SCALE[colour]
        channels.append(channel + rng.normal(0, 0.0003, channel.shape))
    starless = np.clip(np.stack(channels, axis=-1), 0, 1).astype(np.float32)
    _, tone = finish(starless)
    assert np.allclose(tone.gains, bright_pixel_gains(starless))


def test_range_puts_the_nebula_where_asked() -> None:
    image = master()
    out, tone = finish(image, Settings(sky_target=0.16, range_target=0.05))
    lum = out.mean(axis=2)
    assert np.median(lum) == pytest.approx(0.16, abs=0.01)
    assert np.percentile(lum, 99) - np.median(lum) == pytest.approx(0.05, abs=0.01)
    assert tone.balance != 0.5
    # A range no curve can give, on this faint a nebula, gets the most there is.
    most, _ = finish(image, Settings(sky_target=0.16, range_target=0.5))
    lum = most.mean(axis=2)
    assert 0.05 < np.percentile(lum, 99) - np.median(lum) < 0.5
    assert np.median(lum) == pytest.approx(0.16, abs=0.01)


def test_local_contrast_changes_only_the_coarse_layers() -> None:
    image = master()
    plain, _ = finish(image)
    boosted, _ = finish(image, Settings(local_contrast=1.5))
    change = (boosted - plain).mean(axis=2).astype(np.float32)
    layers, _ = b3_layers(change, 7)
    fine = sum(np.abs(layer).mean() for layer in layers[:3])
    coarse = sum(np.abs(layer).mean() for layer in layers[4:7])
    assert coarse > 5 * fine
    assert np.abs(change).max() > 0.01


def test_sky_chroma_keeps_the_sky_grey_and_the_nebula_coloured() -> None:
    image = master()
    plain, _ = finish(image, Settings(saturation=1.6))
    spared, _ = finish(image, Settings(saturation=1.6, sky_chroma=0.0))
    unsaturated, _ = finish(image, Settings(saturation=1.0))

    def chroma(out: np.ndarray, window: tuple[slice, slice]) -> float:
        patch = out[window]
        return float(np.abs(patch - patch.mean(axis=2, keepdims=True)).mean())

    sky = (slice(0, 40), slice(0, 60))
    nebula = (slice(100, 140), slice(140, 180))
    # The sky is left near what it would be with no saturation; the nebula gets it all.
    assert chroma(unsaturated, sky) <= chroma(spared, sky) < 0.8 * chroma(plain, sky)
    assert chroma(spared, nebula) == pytest.approx(chroma(plain, nebula), rel=0.1)


def test_star_reduction_shrinks_the_stars_and_leaves_the_rest() -> None:
    image = master()
    plain, _ = finish(image)
    reduced, _ = finish(image, Settings(star_reduction=0.8))
    star = (slice(45, 56), slice(55, 66))
    assert reduced[star].sum() < 0.9 * plain[star].sum()
    sky = (slice(0, 30), slice(0, 30))
    assert np.allclose(reduced[sky], plain[sky])


def test_denoise_is_shown_under_the_same_stretch() -> None:
    image = master()
    plain, tone = finish(image)
    cleaned, tone_cleaned = finish(image, Settings(denoise="wavelet"))
    assert (tone_cleaned.stretch, tone_cleaned.black, tone_cleaned.balance) == (
        tone.stretch,
        tone.black,
        tone.balance,
    )
    sky = (slice(0, 40), slice(0, 60))
    assert cleaned[sky].std() < 0.7 * plain[sky].std()
    assert np.median(cleaned.mean(axis=2)) == pytest.approx(np.median(plain.mean(axis=2)), abs=0.01)


def test_finish_command_writes_the_png(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "light_bg.fits"
    save_fits(source, Frame(master()))
    output, half = tmp_path / "heart.png", tmp_path / "heart-half.png"
    main(["finish", str(source), "-o", str(output), "--half", str(half), "--range", "0.3"])
    assert output.exists() and half.exists()
    out = capsys.readouterr().out
    assert "stretch factor" in out and "wrote" in out
    main(["finish", str(source), "-o", str(output), "--gains", "1,1.1,1.2", "--match-psf"])
    out = capsys.readouterr().out
    assert "colour gains [1.  1.1 1.2]" in out and "star width per channel" in out
    with pytest.raises(SystemExit):
        main(["finish", str(source), "-o", str(output), "--gains", "1,1"])


@pytest.mark.nas
def test_defaults_reproduce_the_heart_given_its_gains() -> None:
    """The Heart was balanced on its brightest pixels; with those gains the rest is the same."""
    from PIL import Image

    if not (HEART / "heart.png").exists():
        pytest.skip("the Heart project is not mounted")
    image = load(HEART / "light_bg_plane.fits").data
    out, _ = finish(image, Settings(gains=tuple(bright_pixel_gains(image))))
    expected = np.asarray(Image.open(HEART / "heart.png"))
    assert np.array_equal(to_8bit(out), expected)


def test_pixels_no_light_reached_come_out_black_and_change_nothing_else() -> None:
    image = master()
    whole, tone_whole = finish(image)
    cut = image.copy()
    cut[:70, :90] = 0  # a corner no light reached, as a union canvas has

    out, tone = finish(cut)

    assert (out[:70, :90] == 0).all()
    # The synthetic master's "noise" is mostly its sky gradient, which the
    # missing corner shifts a little; the sky and the balance hold.
    assert tone.noise == pytest.approx(tone_whole.noise, rel=0.15)
    assert tone.sky == pytest.approx(tone_whole.sky, rel=0.02)
    assert tone.gains == pytest.approx(tone_whole.gains, rel=0.02)
    # The stretch moves a little with the noise estimate; the picture does not.
    inside = (slice(90, None), slice(110, None))
    levels = np.abs(to_8bit(out[inside]).astype(int) - to_8bit(whole[inside]).astype(int))
    assert np.median(levels) <= 3 and np.percentile(levels, 99) <= 12
    # No edge is drawn where the fill meets the sky: the seam differs from
    # the whole no more than the interior does.
    difference = out - whole
    seam = (slice(60, 80), slice(100, 200))
    assert difference[seam].std() <= 1.5 * difference[inside].std() + 0.002


def test_finish_and_denoise_commands_take_the_coverage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image = master()
    coverage = np.full(image.shape[:2], 8, np.int32)
    coverage[:, 200:] = 2
    source, cover = tmp_path / "light.fits", tmp_path / "coverage.fits"
    save_fits(source, Frame(image))
    save_fits(cover, Frame(coverage))
    main(
        ["finish", str(source), "-o", str(tmp_path / "out.png"), "--denoise", "wavelet"]
        + ["--coverage", str(cover)]
    )
    assert "wrote" in capsys.readouterr().out
    main(["denoise", str(source), "-o", str(tmp_path / "dn.fits"), "--coverage", str(cover)])
    assert "wrote" in capsys.readouterr().out


def test_a_nearer_black_point_clips_the_dark_side_of_the_sky() -> None:
    image = master()
    deep, tone_deep = finish(image)
    near, tone_near = finish(image, Settings(black_point=-1.0))
    assert tone_near.black == pytest.approx(-1.0 * tone_near.noise)
    assert np.median(near.mean(axis=2)) == pytest.approx(0.13, abs=0.01)
    # Some of the sky's dark side is clipped to black now; none was before.
    sky = (slice(0, 40), slice(0, 60))
    assert (near[sky] == 0).mean() > 0.02
    assert (deep[sky] == 0).mean() < 0.001


def test_a_lower_white_point_gives_the_nebula_and_stars_the_range() -> None:
    image = master()
    cores, tone_cores = finish(image)
    wide, tone_wide = finish(image, Settings(top_percentile=99.9))
    assert tone_wide.top < tone_cores.top
    assert np.median(wide.mean(axis=2)) == pytest.approx(0.13, abs=0.01)
    # The stretch has less far to go, so the mid-tones are less compressed.
    assert tone_wide.stretch < tone_cores.stretch
    nebula = (slice(100, 140), slice(140, 180))
    assert wide[nebula].mean() > cores[nebula].mean()
    # The brightest star cores clip to white, which they did not before.
    assert (wide == 1).mean() > (cores == 1).mean()


def test_a_saturated_star_core_comes_out_white() -> None:
    image = master()
    x, y, _ = STARS[0]
    image[y - 1 : y + 2, x - 1 : x + 2] = 1.0  # clipped in the camera, in every channel
    out, tone = finish(image)
    assert tone.gains.max() / tone.gains.min() > 1.1
    core = out[y, x]
    assert core.max() == pytest.approx(core.min(), abs=1e-6)
