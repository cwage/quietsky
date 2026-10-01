from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import m13
import synthetic
from starkiller.calibrate import calibrate, unit_flat
from starkiller.cli import main
from starkiller.debayer import debayer_vng
from starkiller.frame import Frame, load, normalized, save_fits
from starkiller.integrate import flux_normalization, frame_estimates, integrate
from starkiller.stars import detect_stars


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
    expected = integrate(stack, normalization=flux_normalization(frame_estimates(stack)[0]))
    np.testing.assert_array_equal(load(output).data, expected.image)


def test_stack_light_combines_colour_frames_at_the_level_of_the_first(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    scene = rng.uniform(0.0, 0.01, (40, 50, 3)).astype(np.float32)
    paths = []
    for index, sky in enumerate([0.10, 0.12, 0.15, 0.11, 0.13, 0.14, 0.12, 0.16]):
        frame = sky + scene + rng.normal(0, 0.0005, scene.shape).astype(np.float32)
        if index == 3:
            frame[20, 25] += 0.3  # a satellite trail pixel
        if index == 5:
            frame[:, :4] = 0.0  # shifted off this edge by registration
        paths.append(tmp_path / f"light{index}.fits")
        save_fits(paths[-1], Frame(frame.astype(np.float32)))
    output = tmp_path / "master_light.fits"

    main(["stack", "--light", *map(str, paths), "-o", str(output)])

    master = load(output).data
    assert master.shape == (40, 50, 3)
    np.testing.assert_allclose(master, 0.10 + scene, atol=0.002)
    assert master[:, :4].min() > 0.09  # the black border did not drag the average down


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


def test_debayer_writes_rgb_frames_using_the_pattern_in_the_header(tmp_path: Path) -> None:
    light = sorted((m13.DATA / "light").glob("*.fits"))[5]

    main(["debayer", str(light), "-o", str(tmp_path)])

    result = load(tmp_path / f"{light.stem}_d.fits")
    np.testing.assert_array_equal(result.data, debayer_vng(normalized(load(light).data), "RGGB"))
    assert "BAYERPAT" not in result.header
    assert result.header["EXPTIME"] == 30.0
    noise = [result.header[f"NOISE{channel:02d}"] for channel in range(3)]
    assert all(isinstance(value, float) and 0 < value < 0.001 for value in noise)


def test_register_aligns_frames_to_the_reference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shape = (300, 400)
    _, x, y = synthetic.star_field(shape, 100)
    flux = 10 ** np.random.default_rng(5).uniform(-1.0, 0.3, len(x))
    reference, shifted = tmp_path / "reference.fits", tmp_path / "shifted.fits"
    save_fits(reference, Frame(synthetic.render(shape, x, y, flux, noise_seed=1)))
    save_fits(shifted, Frame(synthetic.render(shape, x + 6.5, y - 3.25, flux, noise_seed=2)))

    main(["register", str(shifted), "--reference", str(reference), "-o", str(tmp_path / "out")])

    aligned = detect_stars(load(tmp_path / "out" / "shifted_r.fits").data)
    stars = detect_stars(load(reference).data)
    separation = np.hypot(
        aligned.x[:, None] - stars.x[None, :], aligned.y[:, None] - stars.y[None, :]
    )
    assert np.median(separation.min(axis=1)) < 0.05
    assert "stars matched" in capsys.readouterr().out


def test_register_reports_frames_it_cannot_align(tmp_path: Path) -> None:
    reference, other = tmp_path / "reference.fits", tmp_path / "other.fits"
    save_fits(reference, Frame(synthetic.star_field((300, 400), 80, seed=1)[0]))
    save_fits(other, Frame(synthetic.star_field((300, 400), 80, seed=2)[0]))

    with pytest.raises(SystemExit, match="1 frame"):
        main(["register", str(other), "--reference", str(reference), "-o", str(tmp_path / "out")])


def test_background_writes_the_corrected_image_and_the_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows, columns = np.mgrid[:300, :400]
    image = synthetic.star_field((300, 400), 60)[0] + (0.005 * columns / 400).astype(np.float32)
    source, output, model = tmp_path / "in.fits", tmp_path / "out.fits", tmp_path / "model.fits"
    save_fits(source, Frame(image))

    main(["background", str(source), "-o", str(output), "--model", str(model)])

    corrected = load(output).data
    left, right = np.median(corrected[:, :50]), np.median(corrected[:, -50:])
    assert abs(right - left) < 0.0002
    np.testing.assert_allclose(
        load(model).data + corrected, image + np.median(load(model).data), atol=1e-6
    )
    assert "kept" in capsys.readouterr().out


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
