from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import m13
from starkiller.cli import main
from starkiller.frame import load, normalized
from starkiller.integrate import integrate


def test_stack_writes_the_integration_of_its_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = sorted((m13.DATA / "bias").glob("*.fits"))
    output = tmp_path / "master_bias.fits"

    main(["stack", *map(str, inputs), "-o", str(output)])

    expected = integrate(normalized(m13.frames("bias")))
    np.testing.assert_array_equal(load(output).data, expected.image)
    assert "integrated 20 frames" in capsys.readouterr().out


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
