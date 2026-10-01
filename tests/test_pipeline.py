import shutil
from pathlib import Path

import numpy as np
import pytest

import m13
from quietsky.cli import main
from quietsky.frame import Frame, load, load_fits
from quietsky.pipeline import Session, preprocess
from quietsky.stars import detect_stars

# Every light covers this part of the fixture crop; nearer the edge some are
# black after registration.
INNER = (slice(40, -40), slice(40, -40))


def frames(kind: str) -> list[Path]:
    return sorted((m13.DATA / kind).glob("*.fits"))


def test_preprocess_command_runs_a_session_from_raw_frames_to_master_light(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lights = tmp_path / "light"
    lights.mkdir()
    for path in frames("light")[:5]:
        shutil.copy(path, lights)
    output = tmp_path / "out"

    main(
        ["preprocess", "--bias", str(m13.DATA / "bias"), "--dark", str(m13.DATA / "dark")]
        + ["--flat", str(m13.DATA / "flat"), "--light", str(lights), "-o", str(output)]
    )

    for name in ("bias.fits", "dark.fits", "flat.fits", "light.fits", "light.png"):
        assert (output / "master" / name).exists()
    registered = sorted((output / "registered").glob("*_r.fits"))
    assert len(registered) == 5
    master = load(output / "master" / "light.fits").data
    assert master.shape == (256, 256, 3)
    assert master[INNER].min() > 0
    # The aligned frames put the stars where the reference frame has them.
    reference = detect_stars(load(registered[0]).data)
    aligned = detect_stars(load(registered[4]).data)
    separation = np.hypot(
        aligned.x[:, None] - reference.x[None, :], aligned.y[:, None] - reference.y[None, :]
    ).min(axis=1)
    assert (separation < 1).sum() >= 25
    assert np.median(separation[separation < 1]) < 0.3
    out = capsys.readouterr().out
    assert "master light from 5 frames" in out
    assert "reference" in out


@pytest.mark.slow
def test_master_light_from_raw_crops_resembles_pixinsights(tmp_path: Path) -> None:
    # The whole pipeline on the crops of all 103 raw frames, loaded as
    # PixInsight loaded them. It cannot equal PixInsight's master: our dark
    # scaling and registration differ slightly, hot pixels are not corrected,
    # and a crop knows less than the full frame did. It should be the same
    # picture.
    def as_pixinsight_loaded(path: Path) -> Frame:
        frame = load_fits(path)
        return Frame(m13.as_pixinsight_loaded(frame.data), frame.header)

    session = Session(frames("bias"), frames("dark"), frames("flat"), frames("light"))
    reports: list[str] = []

    master = preprocess(session, tmp_path, as_pixinsight_loaded, reports.append)

    assert reports[-1].startswith("master light from 43 frames")
    ours = load(master).data[INNER]
    theirs = m13.master("light")["integration"][INNER]
    # The flat is scaled by its mean, which over the crop (the bright centre
    # of the field) is higher than over the full frame.
    flat = load(tmp_path / "master" / "flat.fits").data
    expected_ratio = flat.mean() / m13.master_flat_mean()
    for channel in range(3):
        a, b = ours[..., channel].astype(np.float64), theirs[..., channel].astype(np.float64)
        assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.999
        assert np.median(a / b) == pytest.approx(expected_ratio, rel=0.02)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("kinds", "said"),
    [
        (("dark", "flat"), ["dark scaled by 1.000", "lights calibrated with: dark, flat"]),
        (("bias", "flat"), ["bias subtracted", "lights calibrated with: bias, flat"]),
        (("bias", "dark"), ["lights calibrated with: bias, dark"]),
        ((), ["lights calibrated with: nothing"]),
    ],
)
def test_preprocess_works_without_some_calibration_frames(
    tmp_path: Path, kinds: tuple[str, ...], said: list[str]
) -> None:
    session = Session(
        frames("bias") if "bias" in kinds else [],
        frames("dark") if "dark" in kinds else [],
        frames("flat") if "flat" in kinds else [],
        frames("light")[:4],
    )
    reports: list[str] = []

    master = preprocess(session, tmp_path, report=reports.append)

    assert load(master).data.shape == (256, 256, 3)
    for kind in ("bias", "dark", "flat"):
        assert (tmp_path / "master" / f"{kind}.fits").exists() == (kind in kinds)
    for phrase in said:
        assert any(phrase in line for line in reports)
    assert reports[-1].startswith("master light from 4 frames")
    # Without a bias the dark cannot be scaled.
    if "bias" not in kinds:
        per_light = [line for line in reports if "dark scale " in line]
        assert len(per_light) == 4
        assert all("dark scale 1.000" in line for line in per_light)


@pytest.mark.slow
def test_reference_aligns_a_second_run_to_the_first(tmp_path: Path) -> None:
    # Two runs over different lights, as for two filters of a mono camera.
    calibration = (frames("bias"), frames("dark"), frames("flat"))
    quiet: list[str] = []
    first = preprocess(
        Session(*calibration, frames("light")[:3]), tmp_path / "a", report=quiet.append
    )
    reference = sorted((tmp_path / "a" / "registered").glob("*_r.fits"))[0]
    reports: list[str] = []

    second = preprocess(
        Session(*calibration, frames("light")[20:23]),
        tmp_path / "b",
        report=reports.append,
        reference=reference,
    )

    assert any(line.startswith(f"aligning to {reference.name}") for line in reports)
    assert not any("reference," in line for line in reports)
    one, two = detect_stars(load(first).data), detect_stars(load(second).data)
    separation = np.hypot(two.x[:, None] - one.x[None, :], two.y[:, None] - one.y[None, :]).min(1)
    assert (separation < 1).sum() >= 25
    assert np.median(separation[separation < 1]) < 0.3


@pytest.mark.slow
def test_masters_made_elsewhere_are_used_as_they_are(tmp_path: Path) -> None:
    lights = frames("light")[:3]
    quiet: list[str] = []
    first = preprocess(
        Session(frames("bias"), frames("dark"), frames("flat"), lights),
        tmp_path / "a",
        report=quiet.append,
    )
    masters = tmp_path / "a" / "master"
    reports: list[str] = []

    second = preprocess(
        Session(
            [],
            [],
            [],
            lights,
            master_bias=masters / "bias.fits",
            master_dark=masters / "dark.fits",
            master_flat=masters / "flat.fits",
        ),
        tmp_path / "b",
        report=reports.append,
    )

    assert any(line == "master flat from flat.fits" for line in reports)
    np.testing.assert_array_equal(load(second).data, load(first).data)
    np.testing.assert_array_equal(
        load(tmp_path / "b" / "master" / "flat.fits").data, load(masters / "flat.fits").data
    )
