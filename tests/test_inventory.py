import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import m13
from quietsky.cli import main
from quietsky.inventory import Inventory, describe_frame, take_inventory


def write(path: Path, header: dict[str, object] | None = None, floating: bool = False) -> Path:
    """A small FITS frame with the given header keywords."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((4, 6), np.float32 if floating else np.uint16)
    hdu = fits.PrimaryHDU(data)
    for key, value in (header or {}).items():
        hdu.header[key] = value
    hdu.writeto(path)
    return path


def counts(inventory: Inventory) -> dict[tuple[str, bool, str | None, float | None], int]:
    return {
        (g.key.kind, g.key.master, g.key.filter, g.key.exposure): len(g.frames)
        for g in inventory.groups
    }


def test_the_frame_type_keyword_decides_over_the_folder(tmp_path: Path) -> None:
    path = write(tmp_path / "lights" / "a.fits", {"IMAGETYP": "Dark Frame", "EXPTIME": 60.0})

    frame = describe_frame(path, tmp_path)

    assert frame.kind == "dark"
    assert frame.kind_from == "IMAGETYP 'Dark Frame'"


@pytest.mark.parametrize(
    ("folder", "kind"),
    [
        ("offset", "bias"),
        ("BIAS", "bias"),
        ("DARKFLAT", "darkflat"),
        ("Flats", "flat"),
        ("light", "light"),
        ("darks", "dark"),
    ],
)
def test_folder_names_decide_when_the_header_is_silent(
    tmp_path: Path, folder: str, kind: str
) -> None:
    path = write(tmp_path / folder / "x" / "a.fits")

    frame = describe_frame(path, tmp_path)

    assert frame.kind == kind
    assert frame.kind_from == f"folder '{folder}'"


def test_file_names_decide_last(tmp_path: Path) -> None:
    path = write(tmp_path / "masterDark_BIN-1_EXPOSURE-60.00s.fits", floating=True)

    frame = describe_frame(path, tmp_path)

    assert (frame.kind, frame.master, frame.processed) == ("dark", True, False)


def test_masters_are_told_from_processed_frames(tmp_path: Path) -> None:
    master = write(tmp_path / "a.fits", {"IMAGETYP": "Master Flat", "FILTER": "Red"}, True)
    calibrated = write(tmp_path / "b.fits", {"IMAGETYP": "LIGHT", "FILTER": "Red"}, True)

    inventory = take_inventory(tmp_path)

    assert counts(inventory) == {("flat", True, "R", None): 1}
    assert [frame.path for frame in inventory.processed] == [calibrated]
    assert master.exists()


def test_filter_names_are_made_common(tmp_path: Path) -> None:
    for name in ("Lum", "Red", "green", "B", "H-alpha", "OIII"):
        write(tmp_path / f"{name}.fits", {"IMAGETYP": "LIGHT", "FILTER": name, "EXPTIME": 60})

    filters = sorted(str(group.key.filter) for group in take_inventory(tmp_path).groups)

    assert filters == ["B", "G", "Ha", "L", "OIII", "R"]


def test_a_blank_filter_is_taken_from_the_folder(tmp_path: Path) -> None:
    path = write(tmp_path / "FLAT" / "L" / "a.fits", {"IMAGETYP": "FLAT", "FILTER": ""})

    frame = describe_frame(path, tmp_path)

    assert (frame.filter, frame.filter_from) == ("L", "'L'")


def test_filter_wheel_slots_are_named_from_folders(tmp_path: Path) -> None:
    # As N.I.N.A. 1.9 wrote it: the slot number, with folders naming the filter.
    header = {"IMAGETYP": "LIGHT", "FILTER": "4", "EXPOSURE": 120}
    write(tmp_path / "LIGHT" / "B" / "a.fits", header)
    write(tmp_path / "LIGHT" / "B" / "b.fits", header)
    older = write(tmp_path / "LIGHT" / "Bold" / "c.fits", header)

    inventory = take_inventory(tmp_path)

    assert counts(inventory) == {("light", False, "B", 120.0): 3}
    frame = next(f for g in inventory.groups for f in g.frames if f.path == older)
    assert frame.filter_from == "FILTER '4', which other frames' folders call B"


def test_flats_of_different_exposures_stay_together(tmp_path: Path) -> None:
    for index, exposure in enumerate((1.1, 1.3, 0.9)):
        write(
            tmp_path / f"flat{index}.fits", {"IMAGETYP": "FLAT", "FILTER": "L", "EXPTIME": exposure}
        )
    for index, exposure in enumerate((60, 60, 120)):
        write(
            tmp_path / f"light{index}.fits",
            {"IMAGETYP": "LIGHT", "FILTER": "L", "EXPTIME": exposure},
        )

    assert counts(take_inventory(tmp_path)) == {
        ("light", False, "L", 60.0): 2,
        ("light", False, "L", 120.0): 1,
        ("flat", False, "L", None): 3,
    }


def test_darks_are_split_by_temperature_and_lights_are_not(tmp_path: Path) -> None:
    for index, temperature in enumerate((-20.0, -20.2, -10.0)):
        dark = {"IMAGETYP": "DARK", "EXPTIME": 60, "CCD-TEMP": temperature}
        write(tmp_path / f"dark{index}.fits", dark)
        light = {"IMAGETYP": "LIGHT", "EXPTIME": 60, "FILTER": "R", "CCD-TEMP": temperature}
        write(tmp_path / f"light{index}.fits", light)

    groups = take_inventory(tmp_path).groups

    assert sorted(len(g.frames) for g in groups if g.key.kind == "dark") == [1, 2]
    assert [len(g.frames) for g in groups if g.key.kind == "light"] == [3]


def test_unreadable_and_unrecognised_files_are_reported(tmp_path: Path) -> None:
    (tmp_path / "broken.fits").write_bytes(b"not a FITS file")
    write(tmp_path / "misc" / "a.fits")
    (tmp_path / "notes.txt").write_text("not a frame")
    write(tmp_path / ".hidden" / "light" / "b.fits")

    inventory = take_inventory(tmp_path)

    assert [path.name for path, _ in inventory.unreadable] == ["broken.fits"]
    assert [frame.path.name for frame in inventory.unknown] == ["a.fits"]
    assert inventory.groups == []


def test_the_m13_fixtures_are_grouped_as_shot() -> None:
    inventory = take_inventory(m13.DATA)

    groups = {
        (g.key.kind, g.key.exposure, g.key.gain, g.key.bayer): len(g.frames)
        for g in inventory.groups
    }
    assert groups == {
        ("light", 30.0, 640.0, "RGGB"): 42,
        ("light", 10.0, 1600.0, "RGGB"): 1,  # the stray test exposure
        ("flat", None, 640.0, "RGGB"): 20,
        ("dark", 30.0, 640.0, "RGGB"): 20,
        ("bias", None, 640.0, "RGGB"): 20,
    }
    assert not inventory.processed


def test_inventory_command_prints_the_groups(capsys: pytest.CaptureFixture[str]) -> None:
    main(["inventory", str(m13.DATA)])

    out = capsys.readouterr().out
    assert "  colour RGGB, 30 s, gain 640, 256x256: 42 frames, 2018-09-12\n" in out
    assert "bias\n  gain 640, 256x256: 20 frames, 2018-09-12\n" in out
    assert "kinds decided by IMAGETYP (103)" in out


@pytest.mark.nas
def test_the_raw_sony_session_is_sorted_by_its_folders() -> None:
    # The raw files say nothing about what they are; only the folders do.
    # PixInsight's output is in the same tree.
    inventory = take_inventory(m13.SESSION)

    raw = {
        (g.key.kind, g.key.exposure, g.key.gain): len(g.frames)
        for g in inventory.groups
        if not g.key.master
    }
    assert raw == {
        ("light", 30.0, 640.0): 42,
        ("light", 10.0, 1600.0): 1,
        ("flat", None, 640.0): 20,
        ("dark", 30.0, 640.0): 20,
        ("bias", None, 640.0): 20,
    }
    masters = sorted(g.key.kind for g in inventory.groups if g.key.master)
    assert masters == ["bias", "dark", "flat", "light"]
    assert len(inventory.processed) == 192  # calibrated, cosmetized, debayered, registered


MONO = os.environ.get("QUIETSKY_MONO_SESSION")


@pytest.mark.skipif(MONO is None, reason="QUIETSKY_MONO_SESSION is not set")
def test_a_mono_session_with_slot_numbered_filters() -> None:
    # An ASI1600MM session written by N.I.N.A. 1.9: FILTER is the wheel slot
    # in the lights and blank in the flats; folders name the filters.
    assert MONO is not None
    inventory = take_inventory(Path(MONO))

    groups = counts(inventory)
    for name in "LRG":
        assert groups[("light", False, name, 120.0)] == 20
        assert groups[("flat", False, name, None)] == 10
    assert groups[("light", False, "B", 120.0)] == 40  # B and an earlier set in Bold
    assert groups[("dark", False, None, 120.0)] == 20


MOANA = os.environ.get("QUIETSKY_MOANA_SESSION")


@pytest.mark.skipif(MOANA is None, reason="QUIETSKY_MOANA_SESSION is not set")
def test_a_download_with_masters_made_elsewhere() -> None:
    # Moana project M13: raw lights with PixInsight's master darks and flats.
    assert MOANA is not None
    inventory = take_inventory(Path(MOANA))

    groups = counts(inventory)
    assert groups[("light", False, "R", 60.0)] == 86
    assert groups[("light", False, "G", 60.0)] == 87
    assert groups[("light", False, "B", 60.0)] == 79
    assert groups[("dark", True, None, 60.0)] == 1
    for name in ("L", "R", "G", "B", "Ha", "OIII", "SII"):
        assert groups[("flat", True, name, None)] == 1
