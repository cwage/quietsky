"""Find out what a folder of frames holds.

Reads the header of every frame under a directory, without loading pixels,
decides what each frame is (bias, dark, dark flat, flat or light, raw or a
master), which filter it was taken through and how, and groups frames that
belong together. Headers are trusted first; where they are silent, folder
and file names are used, and the reason for every decision is kept.
"""

import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

import rawpy
from astropy.io import fits

from quietsky.frame import FITS_SUFFIXES, RAW_SUFFIXES
from quietsky.xisf import read_xisf

Value = str | int | float | bool

KINDS = ("light", "flat", "darkflat", "dark", "bias")
# Words that name a kind of frame, in folder and file names.
KIND_WORDS = {
    "darkflat": {"darkflat", "darkflats", "flatdark", "flatdarks"},
    "bias": {"bias", "biases", "offset", "offsets", "zero", "zeros"},
    "dark": {"dark", "darks"},
    "flat": {"flat", "flats", "flatfield", "flatfields"},
    "light": {"light", "lights", "science"},
}
# Names of common filters, and what each is called here.
FILTER_WORDS = {
    "L": {"l", "lum", "luminance", "clear"},
    "R": {"r", "red"},
    "G": {"g", "green"},
    "B": {"b", "blue"},
    "Ha": {"ha", "halpha", "hydrogenalpha"},
    "OIII": {"oiii", "o3"},
    "SII": {"sii", "s2"},
}
SKIPPED_FOLDERS = ("#", ".", "@")


@dataclass(frozen=True)
class FrameInfo:
    """What one file is, and how that was decided."""

    path: Path
    kind: str | None
    master: bool
    # Floating point data that is not a master: the output of some earlier
    # processing, such as a calibrated or registered frame.
    processed: bool
    filter: str | None
    exposure: float | None
    gain: float | None
    offset: float | None
    binning: int
    width: int
    height: int
    bayer: str | None
    temperature: float | None
    date: str | None
    kind_from: str
    filter_from: str


@dataclass(frozen=True)
class GroupKey:
    """What frames must share to be used together."""

    kind: str
    master: bool
    filter: str | None
    exposure: float | None
    gain: float | None
    offset: float | None
    binning: int
    width: int
    height: int
    bayer: str | None
    temperature: float | None


@dataclass
class Group:
    key: GroupKey
    frames: list[FrameInfo] = field(default_factory=list)


@dataclass
class Inventory:
    groups: list[Group]
    processed: list[FrameInfo]
    unknown: list[FrameInfo]
    unreadable: list[tuple[Path, str]]


def _value(text: str) -> Value:
    """A header value given as text, as a number where it is one."""
    text = text.strip().strip("'").strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _header(path: Path) -> tuple[dict[str, Value], bool, int, int]:
    """Header keywords, whether the pixels are integers, and the size of a frame."""
    suffix = path.suffix.lower()
    if suffix in RAW_SUFFIXES:
        with rawpy.imread(str(path)) as raw:
            pattern = raw.raw_pattern
            colours = raw.color_desc.decode()
            header: dict[str, Value] = {
                "EXPTIME": float(raw.other.shutter_speed),
                "GAIN": float(raw.other.iso_speed),
                "DATE-LOC": raw.other.timestamp.isoformat(),
            }
            if pattern is not None:
                header["BAYERPAT"] = "".join(colours[i] for i in pattern.ravel())
            return header, True, raw.sizes.width, raw.sizes.height
    if suffix in FITS_SUFFIXES:
        with fits.open(path, memmap=True) as hdus:
            hdu = next(h for h in hdus if h.header.get("NAXIS", 0) > 0)
            cards = hdu.header
            header = {key: value for key, value in cards.items() if key}
            return header, cards["BITPIX"] > 0, cards["NAXIS1"], cards["NAXIS2"]
    image = read_xisf(path)[0]
    header = {key: _value(value) for key, value in image.keywords}
    return header, image.data.dtype.kind in "iu", image.data.shape[1], image.data.shape[0]


def _words(text: str) -> list[str]:
    """Lower-case words of a name, split at anything that is not a letter or digit."""
    return [word for word in re.split(r"[^a-z0-9]+", text.lower()) if word]


def _kind_from_text(text: str) -> tuple[str | None, bool]:
    """Kind and master flag named by a frame-type keyword such as IMAGETYP."""
    compact = re.sub(r"[^a-z]", "", text.lower())
    master = "master" in compact
    if "darkflat" in compact or "flatdark" in compact:
        return "darkflat", master
    for kind, words in (
        ("bias", ("bias", "offset", "zero")),
        ("dark", ("dark",)),
        ("flat", ("flat",)),
        ("light", ("light", "object", "science")),
    ):
        if any(word in compact for word in words):
            return kind, master
    return None, master


def _kind_from_name(name: str) -> tuple[str | None, bool]:
    """Kind and master flag named by a folder or file name, word by word."""
    master = False
    for word in _words(name):
        if word.startswith("master"):
            master = True
            word = word.removeprefix("master")
        for kind, names in KIND_WORDS.items():
            if word in names:
                return kind, master
    return None, master


def canonical_filter(text: str) -> str | None:
    """The common name of a filter, or None if the text does not name one."""
    compact = re.sub(r"[^a-z0-9]", "", text.lower())
    for name, words in FILTER_WORDS.items():
        if compact in words:
            return name
    return None


def _filter_from_names(names: Iterable[str]) -> tuple[str | None, str]:
    for name in names:
        for word in [name, *_words(name)]:
            found = canonical_filter(word)
            if found is not None:
                return found, name
    return None, ""


def _number(header: dict[str, Value], *keys: str) -> float | None:
    for key in keys:
        value = header.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def describe_frame(path: Path, root: Path) -> FrameInfo:
    """Read a frame's header and decide what it is."""
    header, integer, width, height = _header(path)
    folders = [part for part in reversed(path.relative_to(root).parent.parts)]
    folders.append(root.name)

    kind, master, kind_from = None, False, ""
    for key in ("IMAGETYP", "FRAME", "FRAMETYP"):
        value = header.get(key)
        if isinstance(value, str) and value.strip():
            kind, master = _kind_from_text(value)
            if kind is not None:
                kind_from = f"{key} '{value.strip()}'"
                break
    if kind is None:
        for folder in folders:
            kind, master = _kind_from_name(folder)
            if kind is not None:
                kind_from = f"folder '{folder}'"
                break
    if kind is None:
        kind, master = _kind_from_name(path.stem)
        kind_from = f"file name '{path.name}'" if kind is not None else "nothing says"
    if not master and "NCOMBINE" in header and not integer:
        master = True

    filter_name, filter_from = None, "no filter"
    raw_filter = header.get("FILTER", header.get("FILTERS"))
    raw_filter = str(raw_filter).strip() if raw_filter is not None else ""
    from_names, name = _filter_from_names([*folders, path.stem])
    if raw_filter:
        filter_name = canonical_filter(raw_filter)
        filter_from = f"FILTER '{raw_filter}'"
        if filter_name is None and from_names is not None:
            filter_name, filter_from = from_names, f"'{name}' (FILTER says '{raw_filter}')"
        elif filter_name is None:
            filter_name = raw_filter
    elif from_names is not None:
        filter_name, filter_from = from_names, f"'{name}'"

    date = header.get("DATE-OBS", header.get("DATE-LOC"))
    bayer = header.get("BAYERPAT")
    return FrameInfo(
        path=path,
        kind=kind,
        master=master,
        processed=not integer and not master,
        filter=filter_name,
        exposure=_number(header, "EXPTIME", "EXPOSURE"),
        gain=_number(header, "GAIN", "ISOSPEED"),
        offset=_number(header, "OFFSET"),
        binning=int(_number(header, "XBINNING") or 1),
        width=width,
        height=height,
        bayer=bayer.strip() if isinstance(bayer, str) and bayer.strip() else None,
        temperature=_number(header, "SET-TEMP", "CCD-TEMP"),
        date=str(date)[:10] if date else None,
        kind_from=kind_from,
        filter_from=filter_from,
    )


def _resolve_filter_slots(frames: list[FrameInfo]) -> list[FrameInfo]:
    """Name filters that headers give only as wheel slots, from frames that name them.

    A camera may record the filter as a slot number. If other frames with
    that same number are in a folder named for a filter, the number means
    that filter.
    """
    meaning: dict[str, Counter[str]] = defaultdict(Counter)
    for frame in frames:
        said = re.search(r"FILTER says '([^']*)'", frame.filter_from)
        if said and frame.filter is not None:
            meaning[said.group(1)][frame.filter] += 1
    resolved = []
    for frame in frames:
        names = meaning.get(frame.filter or "")
        if frame.filter_from.startswith("FILTER '") and names and len(names) == 1:
            name = next(iter(names))
            frame = replace(
                frame,
                filter=name,
                filter_from=f"FILTER '{frame.filter}', which other frames' folders call {name}",
            )
        resolved.append(frame)
    return resolved


def group_key(frame: FrameInfo) -> GroupKey:
    """What a frame shares with the frames it can be combined with.

    Flats are not split by exposure, which can vary from frame to frame,
    and only darks and bias are split by sensor temperature.
    """
    assert frame.kind is not None
    timed = frame.kind in ("light", "dark", "darkflat")
    thermal = frame.kind in ("dark", "darkflat", "bias")
    filtered = frame.kind in ("light", "flat", "darkflat")
    return GroupKey(
        kind=frame.kind,
        master=frame.master,
        filter=frame.filter if filtered else None,
        exposure=round(frame.exposure, 2) if timed and frame.exposure is not None else None,
        gain=frame.gain,
        offset=frame.offset,
        binning=frame.binning,
        width=frame.width,
        height=frame.height,
        bayer=frame.bayer,
        temperature=(
            round(frame.temperature) if thermal and frame.temperature is not None else None
        ),
    )


def frame_files(root: Path) -> list[Path]:
    """Every camera raw, FITS and XISF file under a directory, skipping hidden folders."""
    suffixes = RAW_SUFFIXES | FITS_SUFFIXES | {".xisf"}
    found = []
    for directory, folders, files in os.walk(root, followlinks=True):
        folders[:] = sorted(name for name in folders if not name.startswith(SKIPPED_FOLDERS))
        found += [
            Path(directory) / name
            for name in sorted(files)
            if Path(name).suffix.lower() in suffixes
        ]
    return found


def take_inventory(root: Path) -> Inventory:
    """Describe and group every frame under a directory."""
    frames, unreadable = [], []
    for path in frame_files(root):
        try:
            frames.append(describe_frame(path, root))
        except Exception as error:  # noqa: BLE001 - any unreadable file is reported, not fatal
            unreadable.append((path, f"{type(error).__name__}: {error}"))
    frames = _resolve_filter_slots(frames)
    groups: dict[GroupKey, Group] = {}
    processed, unknown = [], []
    for frame in frames:
        if frame.processed:
            processed.append(frame)
        elif frame.kind is None:
            unknown.append(frame)
        else:
            key = group_key(frame)
            groups.setdefault(key, Group(key)).frames.append(frame)

    def order(group: Group) -> tuple[int, bool, str, float]:
        key = group.key
        return (KINDS.index(key.kind), key.master, key.filter or "", key.exposure or 0.0)

    return Inventory(sorted(groups.values(), key=order), processed, unknown, unreadable)


def _summary(key: GroupKey) -> str:
    parts = []
    if key.filter:
        parts.append(key.filter)
    elif key.kind in ("light", "flat"):
        parts.append("no filter" if key.bayer is None else f"colour {key.bayer}")
    if key.exposure is not None:
        parts.append(f"{key.exposure:g} s")
    if key.gain is not None:
        parts.append(f"gain {key.gain:g}")
    if key.offset is not None:
        parts.append(f"offset {key.offset:g}")
    if key.binning != 1:
        parts.append(f"bin {key.binning}")
    parts.append(f"{key.width}x{key.height}")
    if key.temperature is not None:
        parts.append(f"{key.temperature:g} C")
    return ", ".join(parts)


def describe(inventory: Inventory, root: Path, frames: bool = False) -> str:
    """The inventory as text for people."""
    lines = []
    for kind in KINDS:
        for master in (False, True):
            groups = [g for g in inventory.groups if g.key.kind == kind and g.key.master == master]
            if not groups:
                continue
            label = {"darkflat": "dark flats", "bias": "bias"}.get(kind, f"{kind}s")
            lines.append(f"{'master ' if master else ''}{label}")
            for group in groups:
                count = len(group.frames)
                dates = sorted({f.date for f in group.frames if f.date})
                when = (
                    f", {dates[0]}" + (f" to {dates[-1]}" if len(dates) > 1 else "")
                    if dates
                    else ""
                )
                lines.append(f"  {_summary(group.key)}: {count} frame{'s' * (count != 1)}{when}")
                if frames:
                    for frame in group.frames:
                        lines.append(
                            f"      {frame.path.relative_to(root)}  "
                            f"[kind from {frame.kind_from}; filter from {frame.filter_from}]"
                        )
    how = Counter(f.kind_from.split(" '")[0] for g in inventory.groups for f in g.frames)
    if how:
        lines.append("kinds decided by " + ", ".join(f"{k} ({n})" for k, n in how.most_common()))
    if inventory.processed:
        folders = Counter(str(f.path.parent.relative_to(root)) for f in inventory.processed)
        lines.append(
            f"left out {len(inventory.processed)} processed frames "
            "(floating point, not masters) in " + ", ".join(sorted(folders))
        )
    for frame in inventory.unknown:
        lines.append(f"not recognised: {frame.path.relative_to(root)}")
    for path, error in inventory.unreadable:
        lines.append(f"unreadable: {path.relative_to(root)} ({error})")
    return "\n".join(lines)
