from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import m13
from starkiller.calibrate import calibrate, unit_flat
from starkiller.cli import main
from starkiller.frame import load, normalized
from starkiller.integrate import flux_gains, integrate


def test_stack_writes_the_integration_of_its_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = sorted((m13.DATA / "bias").glob("*.fits"))
    output = tmp_path / "master_bias.fits"

    main(["stack", *map(str, inputs), "-o", str(output)])

    expected = integrate(normalized(m13.frames("bias")))
    np.testing.assert_array_equal(load(output).data, expected.image)
    assert "integrated 20 frames" in capsys.readouterr().out


def test_stack_flat_matches_brightness_before_integrating(tmp_path: Path) -> None:
    inputs = sorted((m13.DATA / "flat").glob("*.fits"))
    output = tmp_path / "master_flat.fits"

    main(["stack", "--flat", *map(str, inputs), "-o", str(output)])

    stack = normalized(m13.frames("flat"))
    expected = integrate(stack, gains=flux_gains(stack))
    np.testing.assert_array_equal(load(output).data, expected.image)


def test_calibrate_writes_each_frame_minus_bias_and_scaled_dark(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bias, dark = tmp_path / "bias.fits", tmp_path / "dark.fits"
    main(["stack", *map(str, sorted((m13.DATA / "bias").glob("*.fits"))), "-o", str(bias)])
    main(["stack", *map(str, sorted((m13.DATA / "dark").glob("*.fits"))), "-o", str(dark)])
    flat = sorted((m13.DATA / "flat").glob("*.fits"))[0]

    main(
        ["calibrate", str(flat), "--bias", str(bias), "--dark", str(dark)]
        + ["--dark-scale", "0.5", "-o", str(tmp_path / "calibrated")]
    )

    result = load(tmp_path / "calibrated" / f"{flat.stem}_c.fits")
    expected = calibrate(normalized(load(flat).data), load(bias).data, load(dark).data, 0.5)
    np.testing.assert_array_equal(result.data, expected)
    assert result.header["DARKSCAL"] == 0.5
    assert result.header["BAYERPAT"] == "RGGB"
    assert "dark scale 0.500" in capsys.readouterr().out


def test_calibrate_divides_by_a_flat_scaled_to_unit_mean(tmp_path: Path) -> None:
    bias, dark, flat = tmp_path / "bias.fits", tmp_path / "dark.fits", tmp_path / "flat.fits"
    main(["stack", *map(str, sorted((m13.DATA / "bias").glob("*.fits"))), "-o", str(bias)])
    main(["stack", *map(str, sorted((m13.DATA / "dark").glob("*.fits"))), "-o", str(dark)])
    main(
        ["stack", "--flat", *map(str, sorted((m13.DATA / "flat").glob("*.fits"))), "-o", str(flat)]
    )
    light = sorted((m13.DATA / "light").glob("*.fits"))[5]

    main(
        ["calibrate", str(light), "--bias", str(bias), "--dark", str(dark), "--flat", str(flat)]
        + ["--dark-scale", "1.0", "-o", str(tmp_path)]
    )

    expected = calibrate(
        normalized(load(light).data),
        load(bias).data,
        load(dark).data,
        1.0,
        unit_flat(load(flat).data),
    )
    np.testing.assert_array_equal(load(tmp_path / f"{light.stem}_c.fits").data, expected)


def test_calibrate_optimises_the_dark_scale_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bias, dark = tmp_path / "bias.fits", tmp_path / "dark.fits"
    main(["stack", *map(str, sorted((m13.DATA / "bias").glob("*.fits"))), "-o", str(bias)])
    main(["stack", *map(str, sorted((m13.DATA / "dark").glob("*.fits"))), "-o", str(dark)])
    light = sorted((m13.DATA / "light").glob("*.fits"))[5]

    main(["calibrate", str(light), "--bias", str(bias), "--dark", str(dark), "-o", str(tmp_path)])

    scale = load(tmp_path / f"{light.stem}_c.fits").header["DARKSCAL"]
    assert isinstance(scale, float)
    assert 0.5 < scale < 2.5  # the lights ran somewhat warmer than the darks


def test_preview_writes_a_stretched_png(tmp_path: Path) -> None:
    light = sorted((m13.DATA / "light").glob("*.fits"))[0]
    output = tmp_path / "light.png"

    main(["preview", str(light), "-o", str(output)])

    with Image.open(output) as image:
        assert image.size == (256, 256)
        assert 40 < np.median(np.asarray(image)) < 90  # dark grey sky


def test_info_reports_size_and_header(capsys: pytest.CaptureFixture[str]) -> None:
    light = sorted((m13.DATA / "light").glob("*.fits"))[0]

    main(["info", str(light)])

    out = capsys.readouterr().out
    assert "256x256 uint16" in out
    assert "BAYERPAT RGGB" in out
