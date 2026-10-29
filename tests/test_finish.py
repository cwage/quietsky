import os
from pathlib import Path

import numpy as np
import pytest

from quietsky.cli import main
from quietsky.finish import Settings, finish, to_8bit
from quietsky.frame import Frame, load, save_fits
from quietsky.wavelets import b3_layers

HEART = Path(os.environ.get("QUIETSKY_DATA", "/data")) / "projects" / "heart" / "three-nights"


def master(seed: int = 0) -> np.ndarray:
    """A linear colour master: sky with a gradient, red nebula, white stars, noise."""
    rows, columns = np.mgrid[:240, :320].astype(np.float32)
    nebula = 0.01 * np.exp(-((rows - 120) ** 2 + (columns - 160) ** 2) / (2 * 40.0**2))
    rng = np.random.default_rng(seed)
    channels = []
    for colour, tint in ((0, 1.0), (1, 0.35), (2, 0.45)):
        sky = 0.010 + 0.0004 * colour + 0.00001 * rows
        channel = sky + tint * nebula
        for x, y, flux in ((60, 50, 0.3), (250, 190, 0.2), (200, 60, 0.1), (90, 200, 0.05)):
            channel += flux * np.exp(-((rows - y) ** 2 + (columns - x) ** 2) / (2 * 1.5**2))
        channel *= 1 - 0.1 * colour
        channels.append(channel + rng.normal(0, 0.0003, channel.shape))
    result: np.ndarray = np.clip(np.stack(channels, axis=-1), 0, 1).astype(np.float32)
    return result


def by_hand(src: np.ndarray) -> np.ndarray:
    """The finish as the Heart script did it, line for line, as the reference."""
    rgb = src.astype(np.float64)
    sky = np.median(rgb, axis=(0, 1))
    rgb -= sky
    lum = rgb.mean(axis=2)
    noise = 1.4826 * np.median(np.abs(lum - np.median(lum)))
    bright = (lum > np.percentile(lum, 99.7)) & (rgb.max(axis=2) < 0.9 - sky.max())
    star_colour = np.median(rgb[bright] / rgb[bright].mean(axis=1, keepdims=True), axis=0)
    rgb *= 1 / star_colour
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
    assert np.array_equal(to_8bit(out), by_hand(image))
    assert tone.balance == 0.5
    assert np.median(out.mean(axis=2)) == pytest.approx(0.13, abs=0.01)


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


@pytest.mark.nas
def test_defaults_reproduce_the_heart() -> None:
    from PIL import Image

    if not (HEART / "heart.png").exists():
        pytest.skip("the Heart project is not mounted")
    image = load(HEART / "light_bg_plane.fits").data
    out, _ = finish(image)
    expected = np.asarray(Image.open(HEART / "heart.png"))
    assert np.array_equal(to_8bit(out), expected)
