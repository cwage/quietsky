"""Judge denoised versions of a master against each other, and against a deeper stack.

    docker compose run --rm quietsky python tools/evaluate_denoise.py ORIGINAL -o OUTDIR \\
        [--deeper DEEPER] [--crop NAME=X,Y,W,H ...] [--zoom N] NAME=FILE ...

ORIGINAL is the master the versions were made from, each NAME=FILE a denoised
version of it. For each version the script reports the noise by wavelet layer
(1 to 16 pixels), the noise at 1x1 and binned 8x8, the star profile width and
how many faint stars survive, all on the same stars, and writes the residual
(original minus version) as a stretched PNG: grain means noise was removed,
shapes mean signal was.

With --deeper, a master of the same field from more frames, the deeper image
is registered onto the original, its flux matched, and each version's
distance from it is measured over the scales a denoise acts on (wavelet
layers 1 to 4): the robust spread of the difference, stars left out, since
the two stacks' star profiles differ. The version nearest the deeper stack
is the one that removed noise rather than signal, whatever it looks like.
The deeper stack has noise of its own, which no version can get under; the
error column is the distance with that floor taken out in quadrature, so it
is each version's own departure from the truth: the original's is its noise,
a perfect denoise's would be zero, and anything left is noise that survived
or signal that did not.

Each --crop is cut 1:1 (or --zoom times larger) from every image under the
original's stretch and laid side by side. OUTDIR gets index.html with it all.
"""

import argparse
import html
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from quietsky.deconvolve import measure_psf
from quietsky.denoise import layer_noise, star_footprint, unclipped
from quietsky.frame import load, normalized
from quietsky.noise import evaluate_noise
from quietsky.register import solve_transformation
from quietsky.resample import resample
from quietsky.stars import Stars, detect_stars
from quietsky.stretch import autostretch
from quietsky.wavelets import b3_layers

Float32 = npt.NDArray[np.float32]
Bool = npt.NDArray[np.bool_]

LAYERS = 5
DISTANCE_LAYERS = 4
FAINT = 5.0
STRONG = 8.0
RESIDUAL_STRETCH = 6.0
MAD_TO_SIGMA = 1.4826


def luminance(image: Float32) -> Float32:
    return image if image.ndim == 2 else image.mean(axis=-1).astype(np.float32)


def binned(image: Float32, size: int = 8) -> Float32:
    height, width = (image.shape[0] // size) * size, (image.shape[1] // size) * size
    result: Float32 = (
        image[:height, :width].reshape(height // size, size, width // size, size).mean((1, 3))
    ).astype(np.float32)
    return result


def surviving(reference: Stars, found: Stars) -> int:
    """How many of the reference stars have a found star within a pixel."""
    if len(found) == 0:
        return 0
    distance = np.hypot(
        reference.x[:, None] - found.x[None, :], reference.y[:, None] - found.y[None, :]
    ).min(axis=1)
    return int((distance < 1).sum())


def spread(values: npt.NDArray[np.floating]) -> float:
    """A robust standard deviation: the median absolute deviation, scaled."""
    return MAD_TO_SIGMA * float(np.median(np.abs(values - np.median(values))))


def detail(image: Float32, layers: int) -> Float32:
    """The image less its smooth residual beyond the given layers."""
    _, smooth = b3_layers(image, layers)
    result: Float32 = (image - smooth).astype(np.float32)
    return result


@dataclass
class Row:
    """One version's measurements."""

    name: str
    layers: list[float]
    fine: float
    coarse: float
    psf: float
    kept: int
    distance: float | None = None
    error: float | None = None


def to_png(path: Path, data: npt.NDArray[np.floating]) -> None:
    Image.fromarray((np.clip(data, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def register_deeper(original: Float32, deeper: Float32) -> tuple[Float32, Bool, float, float]:
    """The deeper image in the original's frame, flux-matched; and where it has data.

    Returns the matched image, a mask of covered pixels, and the scale and
    offset that took the original's flux onto the deeper's.
    """
    transformation = solve_transformation(
        detect_stars(luminance(original)), detect_stars(luminance(deeper))
    )
    shape = original.shape[:2]
    if deeper.ndim == 2:
        moved = resample(deeper, transformation.matrix, shape=shape)
    else:
        moved = np.stack(
            [
                resample(deeper[..., c], transformation.matrix, shape=shape)
                for c in range(deeper.shape[-1])
            ],
            axis=-1,
        )
    covered = unclipped(luminance(moved)) & unclipped(luminance(original))
    # The deeper stack is on another flux scale (more frames, another night's
    # sky); a line fitted through the pixels in common takes it out.
    x, y = (
        luminance(original)[covered].astype(np.float64),
        luminance(moved)[covered].astype(np.float64),
    )
    scale, offset = 1.0, 0.0
    keep = np.ones(len(x), bool)
    for _ in range(3):
        scale, offset = np.polyfit(x[keep], y[keep], 1)
        error = y - (scale * x + offset)
        keep = np.abs(error) < 3 * np.std(error[keep])
    return moved.astype(np.float32), covered, float(scale), float(offset)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("original", type=Path)
    parser.add_argument("versions", nargs="+", help="NAME=FILE, a denoised version of the original")
    parser.add_argument("-o", "--output", type=Path, required=True, help="directory for the page")
    parser.add_argument("--deeper", type=Path, help="a master of the same field from more frames")
    parser.add_argument(
        "--crop", action="append", default=[], help="NAME=X,Y,W,H, a 1:1 crop to show"
    )
    parser.add_argument("--zoom", type=int, default=1, help="enlarge the crops this many times")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    original = normalized(load(args.original).data)
    images: dict[str, Float32] = {"original": original}
    for spec in args.versions:
        name, _, file = spec.partition("=")
        image = normalized(load(Path(file)).data)
        if image.shape != original.shape:
            sys.exit(f"{name}: shape {image.shape} differs from the original's {original.shape}")
        images[name] = image

    lum = {name: luminance(image) for name, image in images.items()}
    valid = unclipped(lum["original"])
    stars = detect_stars(lum["original"], FAINT)
    strong = detect_stars(lum["original"], STRONG)
    footprint = star_footprint(lum["original"], stars) > 0.1

    deeper = None
    floor = 0.0
    if args.deeper:
        moved, covered, scale, offset = register_deeper(
            original, normalized(load(args.deeper).data)
        )
        # Shown and compared on the original's flux scale.
        matched = ((moved - offset) / scale).astype(np.float32)
        lum_matched = luminance(matched)
        footprint |= star_footprint(lum_matched, detect_stars(lum_matched, FAINT)) > 0.1
        where = covered & ~footprint
        deeper = lum_matched, where
        floor = float(
            np.sqrt(np.sum(np.square(layer_noise(lum_matched, DISTANCE_LAYERS, valid=where))))
        )
        images["deeper"] = matched
        lum["deeper"] = lum_matched
        print(f"deeper master registered; flux scale {scale:.4f}, offset {offset:.3g}")
        print(
            f"covered pixels {covered.mean():.0%}; floor (the deeper stack's own noise) {floor:.3g}"
        )

    rows: list[Row] = []
    for name, image in images.items():
        if name == "deeper":
            continue
        layers = layer_noise(lum[name], LAYERS, valid=valid)
        fine = evaluate_noise(lum[name]).sigma
        coarse = evaluate_noise(binned(lum[name])).sigma
        psf = measure_psf(lum[name], strong)
        kept = surviving(stars, detect_stars(lum[name], FAINT))
        row = Row(name, layers, fine, coarse, psf, kept)
        if deeper is not None:
            lum_deep, where = deeper
            difference = detail((lum[name] - lum_deep).astype(np.float32), DISTANCE_LAYERS)
            row.distance = spread(difference[where])
            row.error = float(np.sqrt(max(row.distance**2 - floor**2, 0)))
        rows.append(row)
        if name != "original":
            residual = original - image
            sigma = residual.std(axis=(0, 1)) if residual.ndim == 3 else residual.std()
            to_png(
                args.output / f"{name}-residual.png", residual / (RESIDUAL_STRETCH * sigma) + 0.5
            )

    header = [
        "version",
        *[f"{2**i} px" for i in range(LAYERS)],
        "1x1",
        "8x8",
        "psf px",
        "faint stars",
    ]
    if deeper is not None:
        header += ["distance from deeper", "error above floor"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [
            row.name,
            *[f"{s:.3g}" for s in row.layers],
            f"{row.fine:.3g}",
            f"{row.coarse:.3g}",
            f"{row.psf:.3f}",
            f"{row.kept} of {len(stars)}",
        ]
        if row.distance is not None and row.error is not None:
            cells += [f"{row.distance:.3g}", f"{row.error:.3g}"]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)

    crops = []
    stretched = {name: autostretch(image, reference=original) for name, image in images.items()}
    for spec in args.crop:
        name, _, box = spec.partition("=")
        x, y, w, h = (int(v) for v in box.split(","))
        window = (slice(y, y + h), slice(x, x + w))
        tiles = [stretched[n][window] for n in images]
        sheet = np.concatenate(tiles, axis=1)
        if sheet.ndim == 2:
            sheet = np.repeat(sheet[..., None], 3, axis=-1)
        sheet = np.repeat(np.repeat(sheet, args.zoom, axis=0), args.zoom, axis=1)
        to_png(args.output / f"crop-{name}.png", sheet)
        crops.append((name, list(images)))

    page = [
        "<!doctype html><html><head><meta charset='utf-8'><title>denoise evaluation</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#ddd;margin:2em}"
        "table{border-collapse:collapse}"
        "td,th{border:1px solid #444;padding:4px 8px;text-align:right}"
        "td:first-child,th:first-child{text-align:left}img{max-width:100%;image-rendering:pixelated}"
        "figure{margin:1em 0}figcaption{color:#999}</style></head><body>",
        f"<h1>{html.escape(str(args.original))}</h1>",
        "<p>Noise by wavelet layer, then at 1x1 and binned 8x8; star profile sigma and faint stars "
        "found again, on the original's stars."
        + (
            f" Distance: robust spread of the difference from the deeper master over layers 1 "
            f"to 4, stars left out. The deeper stack's own noise, {floor:.3g}, is taken out of "
            "it in quadrature to give the error: each version's own departure from the truth, "
            "lower is better."
            if deeper is not None
            else ""
        )
        + "</p>",
        "<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in header) + "</tr>",
    ]
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        page.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    page.append("</table>")
    for name, order in crops:
        page.append(
            f"<figure><img src='crop-{html.escape(name)}.png'><figcaption>{html.escape(name)}: "
            + " | ".join(html.escape(n) for n in order)
            + "</figcaption></figure>"
        )
    for row in rows:
        if row.name != "original":
            page.append(
                f"<figure><img src='{html.escape(row.name)}-residual.png'>"
                f"<figcaption>residual of {html.escape(row.name)}, "
                f"&plusmn;{RESIDUAL_STRETCH:g}&sigma;</figcaption></figure>"
            )
    page.append("</body></html>")
    (args.output / "index.html").write_text("\n".join(page))
    print(f"wrote {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
