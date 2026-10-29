"""Command line interface."""

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from quietsky.avi import Video, read_video
from quietsky.background import MODELS, STRUCTURE_WARNING, extract_background
from quietsky.calibrate import calibrate, optimize_dark, unit_flat
from quietsky.debayer import BAYER_PATTERNS, debayer_bilinear, debayer_vng
from quietsky.deconvolve import (
    deconvolve,
    denoise,
    gaussian_blur,
    measure_psf,
    richardson_lucy,
)
from quietsky.denoise import (
    CHROMA_THRESHOLDS,
    LUMINANCE_THRESHOLDS,
    denoise_image,
)
from quietsky.frame import (
    FITS_SUFFIXES,
    RAW_SUFFIXES,
    DiskStack,
    Frame,
    load,
    load_header,
    normalized,
    save_fits,
)
from quietsky.integrate import (
    flux_normalization,
    frame_estimates,
    integrate,
    integrate_lights,
)
from quietsky.inventory import describe, take_inventory
from quietsky.lunar import (
    align_channels,
    alignment_matrix,
    balanced,
    bayer_balance,
    colour_mean,
    find_rotation,
    lucky_stack,
    mineral_colour,
    moon_parallactic_angle,
    mosaic_luminance,
    sharpen,
    sharpness,
    upsample,
)
from quietsky.noise import evaluate_noise
from quietsky.pipeline import Session, frame_noise, noise_keywords, preprocess
from quietsky.register import RegistrationError, solve_transformation
from quietsky.resample import resample
from quietsky.stars import detect_stars
from quietsky.stretch import autostretch


def _info(args: argparse.Namespace) -> None:
    for path in args.files:
        frame = load(path)
        data = frame.data
        height, width = data.shape[:2]
        print(f"{path}: {width}x{height} {data.dtype}", end="")
        print(f" min {data.min():g} median {np.median(data):g} max {data.max():g}")
        for key, value in frame.header.items():
            print(f"  {key:8} {value}")


def _stack(args: argparse.Namespace) -> None:
    if args.light:
        # Lights are read from disk a few rows at a time: there can be
        # hundreds of them.
        first = load(args.files[0]).data
        count = 1 if first.ndim == 2 else first.shape[-1]
        channels = [DiskStack(args.files, channel) for channel in range(count)]
        headers = [load_header(path) for path in args.files]
        results = integrate_lights(
            channels, frame_noise(headers, channels), args.sigma_low, args.sigma_high
        )
        image = np.squeeze(np.stack([result.image for result in results], axis=-1))
        total = first.size * len(args.files)
    else:
        stack = np.stack([normalized(load(path).data) for path in args.files])
        normalization = flux_normalization(frame_estimates(stack)[0]) if args.flat else None
        results = [integrate(stack, args.sigma_low, args.sigma_high, normalization)]
        image = results[0].image
        total = stack.size
    low = sum(result.frame_rejected_low.sum() for result in results)
    high = sum(result.frame_rejected_high.sum() for result in results)
    print(f"integrated {len(args.files)} frames")
    print(f"rejected low {low} ({low / total:.3%}), high {high} ({high / total:.3%})")
    save_fits(args.output, Frame(image, {"NCOMBINE": len(args.files)}))
    print(f"wrote {args.output}")


def _calibrate(args: argparse.Namespace) -> None:
    bias = normalized(load(args.bias).data) if args.bias else None
    dark = normalized(load(args.dark).data) if args.dark else None
    flat = unit_flat(normalized(load(args.flat).data)) if args.flat else None
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        frame = load(path)
        data = normalized(frame.data)
        scale = args.dark_scale
        if scale is None:
            # Without a bias the dark cannot be scaled and is subtracted whole.
            scale = 1.0
            if bias is not None and dark is not None:
                scale = optimize_dark(data, bias, dark, mosaic="BAYERPAT" in frame.header)
        calibrated = calibrate(data, bias, dark, scale, flat)
        output = args.output / f"{path.stem}_c.fits"
        save_fits(output, Frame(calibrated, frame.header | {"DARKSCAL": round(scale, 4)}))
        print(f"{output}: dark scale {scale:.3f}")


def _debayer(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        frame = load(path)
        pattern = args.pattern or frame.header.get("BAYERPAT")
        if not isinstance(pattern, str):
            raise SystemExit(f"{path}: no BAYERPAT in the header; give --pattern")
        header = {key: value for key, value in frame.header.items() if key != "BAYERPAT"}
        rgb = debayer_vng(normalized(frame.data), pattern)
        header |= noise_keywords(rgb)
        output = args.output / f"{path.stem}_d.fits"
        save_fits(output, Frame(rgb, header))
        print(f"{output}: {pattern}")


def _register(args: argparse.Namespace) -> None:
    reference = detect_stars(normalized(load(args.reference).data))
    print(f"{args.reference}: reference, {len(reference)} stars")
    args.output.mkdir(parents=True, exist_ok=True)
    failed = 0
    for path in args.files:
        frame = load(path)
        data = normalized(frame.data)
        try:
            transformation = solve_transformation(reference, detect_stars(data))
        except RegistrationError as error:
            print(f"{path}: skipped, {error}")
            failed += 1
            continue
        output = args.output / f"{path.stem}_r.fits"
        save_fits(output, Frame(resample(data, transformation.matrix), frame.header))
        print(f"{output}: {transformation.pairs} stars matched, rms {transformation.rms:.2f} px")
    if failed:
        raise SystemExit(f"{failed} frame(s) could not be registered")


def _background(args: argparse.Namespace) -> None:
    frame = load(args.file)
    corrected, model, samples = extract_background(
        normalized(frame.data), args.samples_per_row, smoothing=args.smoothing, model=args.model
    )
    save_fits(args.output, Frame(corrected, frame.header))
    print(f"{args.model} model from {int(samples.kept.sum())} of {len(samples.x)} samples")
    print(f"background varied by {float(np.ptp(model)):.3g} across the frame")
    if args.model == "spline" and samples.structure > STRUCTURE_WARNING:
        print(
            f"warning: the samples vary {samples.structure:.1f} times more than sky would. "
            "If the target fills the frame the spline will have removed some of it; "
            "try --model plane"
        )
    print(f"wrote {args.output}")
    if args.write_model:
        save_fits(args.write_model, Frame(model))
        print(f"wrote {args.write_model}")


def _inventory(args: argparse.Namespace) -> None:
    print(describe(take_inventory(args.directory), args.directory, frames=args.frames))


def _frames_in(directory: Path | None) -> list[Path]:
    if directory is None:
        return []
    suffixes = RAW_SUFFIXES | FITS_SUFFIXES | {".xisf"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in suffixes)


def _preprocess(args: argparse.Namespace) -> None:
    session = Session(
        _frames_in(args.bias),
        _frames_in(args.dark),
        _frames_in(args.flat),
        _frames_in(args.light),
        master_bias=args.master_bias,
        master_dark=args.master_dark,
        master_flat=args.master_flat,
    )
    if not session.light:
        raise SystemExit(f"no light frames in {args.light}")
    preprocess(session, args.output, reference=args.reference)


def _png16(path: Path, image: npt.NDArray[np.float32]) -> None:
    """Write an image in [0, 1] as a 16-bit PNG, greyscale or RGB."""
    data = (np.clip(image, 0, 1) * 65535 + 0.5).astype(np.uint16)
    if data.ndim == 2:
        Image.fromarray(data).save(path)
    else:
        # Pillow has no 16-bit RGB mode, so the PNG is written directly.
        _write_png_rgb16(path, data)


def _write_png_rgb16(path: Path, data: npt.NDArray[np.uint16]) -> None:
    height, width, _ = data.shape
    raw = b"".join(b"\x00" + data[row].astype(">u2").tobytes() for row in range(height))

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _lunar(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    clips = args.output / "clips"
    clips.mkdir(exist_ok=True)
    stacks: list[npt.NDArray[np.float32]] = []
    colours: list[npt.NDArray[np.float32]] = []
    names = []
    # When and where each stack was taken, if its video says: for predicting the turn.
    sites: list[tuple[str, float, float] | None] = []
    scale = args.drizzle
    suffix = "" if scale == 1 else f"-x{scale}"
    for path in args.videos:
        video = read_video(path)
        pattern = str(video.header.get("BAYERPAT", "GRBG"))
        print(f"{path.name}: {len(video)} frames of {video.width}x{video.height}")
        when, latitude, longitude = (
            video.header.get(key) for key in ("DATE-OBS", "SITELAT", "SITELONG")
        )
        site = (
            (when, float(latitude), float(longitude))
            if isinstance(when, str)
            and isinstance(latitude, float | int)
            and isinstance(longitude, float | int)
            else None
        )
        # A long clip is stacked in pieces: on an alt-azimuth mount the field
        # turns while it records, and each piece is turned back on its own.
        pieces = max(1, round(len(video) / args.chunk))
        bounds = np.linspace(0, len(video), pieces + 1).round().astype(int).tolist()
        for piece, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):

            def luminance(
                index: int, video: Video = video, start: int = start
            ) -> npt.NDArray[np.float32]:
                return mosaic_luminance(video.frame(start + index))

            def colour(
                index: int, video: Video = video, start: int = start, pattern: str = pattern
            ) -> npt.NDArray[np.float32]:
                return debayer_bilinear(normalized(video.frame(start + index)), pattern)

            name = path.stem if pieces == 1 else f"{path.stem}-{piece + 1}"
            saved = clips / f"{name}{suffix}.fits"
            saved_colour = clips / f"{name}-colour.fits"
            if saved.exists() and saved_colour.exists():
                # Stacks from an earlier run are used as they are.
                print(f" {name}: using {saved}")
                stacks.append(load(saved).data.astype(np.float32))
                colours.append(load(saved_colour).data.astype(np.float32))
                names.append(name)
                sites.append(site)
                continue
            print(f" {name}: frames {start} to {end - 1}")
            # The frames are ranked and aligned on their smoothed luminance, but
            # stacked from the mosaic itself, each colour scaled to match: for a
            # target as grey as the Moon that is its luminance at full resolution.
            sample = np.linspace(start, end - 1, 16).round().astype(int)
            sky, gain = bayer_balance(
                np.mean([normalized(video.frame(int(i))) for i in sample], axis=0)
            )

            def sharp(
                index: int,
                video: Video = video,
                start: int = start,
                sky: npt.NDArray[np.float32] = sky,
                gain: npt.NDArray[np.float32] = gain,
            ) -> npt.NDArray[np.float32]:
                return balanced(video.frame(start + index), sky, gain)

            result = lucky_stack(
                luminance,
                end - start,
                (video.height, video.width),
                keep=args.keep,
                box_size=args.box,
                log=lambda message: print(f"  {message}"),
                sharp=sharp,
                scale=scale,
            )
            print(f"  stacked {np.median(result.frames_per_box):.0f} frames per box")
            best = np.argsort(-result.quality)[: max(1, round(args.keep * (end - start)))]
            colour_stack = colour_mean(colour, best.tolist(), result.shift)
            save_fits(saved, Frame(result.image, dict(video.header)))
            save_fits(saved_colour, Frame(colour_stack, dict(video.header)))
            stacks.append(result.image)
            colours.append(colour_stack)
            names.append(name)
            sites.append(site)

    if len(stacks) == 1:
        combined, colour_combined = stacks[0], colours[0]
    else:
        # The sharpest stack is the reference; the others are turned and moved onto it.
        order = np.argsort([-sharpness(stack) for stack in stacks])
        reference = stacks[order[0]]
        coarse_shape = colours[order[0]].shape[:2]
        aligned = [reference]
        aligned_colours = [colours[order[0]]]
        # On an alt-azimuth mount the field turns by the change in the Moon's
        # parallactic angle, about 95 degrees over a night; predict each turn
        # from that and search only near it. Without times, search near zero.
        known = [site for site in sites if site is not None]
        parallactic = (
            [moon_parallactic_angle(*site) for site in known] if len(known) == len(sites) else None
        )
        for index in order[1:]:
            if parallactic is None:
                angle, dy, dx = find_rotation(reference, stacks[index])
                print(f"{names[index]}: turned {np.degrees(angle):+.2f} degrees")
            else:
                turn = parallactic[index] - parallactic[order[0]]
                expected = float(np.angle(np.exp(1j * turn)))  # wrapped to +-180 degrees
                angle, dy, dx = find_rotation(
                    reference, stacks[index], max_angle=np.radians(3), expected=expected
                )
                print(
                    f"{names[index]}: turned {np.degrees(angle):+.2f} degrees"
                    f" (predicted {np.degrees(expected):+.2f})"
                )
            matrix = alignment_matrix(angle, dy, dx, reference.shape)
            aligned.append(resample(stacks[index], matrix))
            # The colour stacks are on the coarse grid, where shifts are smaller.
            coarse = alignment_matrix(angle, dy / scale, dx / scale, coarse_shape)
            aligned_colours.append(resample(colours[index], coarse))
        result = lucky_stack(
            lambda index: aligned[index],
            len(aligned),
            reference.shape,
            keep=args.combine_keep,
            box_size=args.box * scale,
            log=lambda message: print(f"  {message}"),
        )
        combined = result.image
        # Colour is smooth, so a plain average of the aligned colour stacks is enough.
        colour_combined = np.mean(aligned_colours, axis=0).astype(np.float32)
    if scale > 1:
        colour_combined = upsample(colour_combined, scale)
    save_fits(args.output / "stack.fits", Frame(combined))
    save_fits(args.output / "colour.fits", Frame(colour_combined))

    sharpened = combined
    if args.deconvolve:
        sigma, iterations = args.deconvolve
        # sigma is in the frames' pixels; the stack's are scale times smaller.
        sharpened = richardson_lucy(sharpened, sigma * scale, int(iterations))
    if args.gains:
        sharpened = sharpen(sharpened, args.gains, args.noise_threshold)
    # The bright end of the disc, not its brightest pixel, goes to near white.
    white = float(np.percentile(sharpened, 99.9))
    scaled = np.clip(sharpened / white * 0.95, 0, 1) ** (1 / args.gamma)
    save_fits(args.output / "sharpened.fits", Frame(sharpened))
    _png16(args.output / "moon.png", scaled.astype(np.float32))
    print(f"wrote {args.output / 'moon.png'}")
    if args.saturation > 0:
        colour_aligned, moves = align_channels(colour_combined)
        print(
            "colour channels moved onto green: "
            + ", ".join(
                f"{n} ({dy:+.2f}, {dx:+.2f}) px" for n, (dy, dx) in zip("RB", moves, strict=True)
            )
        )
        coloured = mineral_colour(sharpened, colour_aligned, args.saturation)
        scaled = np.clip(coloured / white * 0.95, 0, 1) ** (1 / args.gamma)
        _png16(args.output / "moon-colour.png", scaled.astype(np.float32))
        print(f"wrote {args.output / 'moon-colour.png'}")


def _sharpen(args: argparse.Namespace) -> None:
    frame = load(args.file)
    data = normalized(frame.data)
    colour = data.ndim == 3
    channels = [data[..., c] for c in range(data.shape[-1])] if colour else [data]
    names = "RGB" if colour and len(channels) == 3 else [str(c) for c in range(len(channels))]
    noises = [evaluate_noise(channel).sigma for channel in channels]
    sigmas = [args.psf if args.psf else measure_psf(channel) for channel in channels]
    # The star profile differs between the colours (focus does), and a star
    # left wider in one colour than another is ringed with that colour. The
    # narrower colours are blurred to the widest profile first, so that all
    # are deconvolved alike and end alike.
    target = max(sigmas)
    results = []
    for name, channel, noise, sigma in zip(names, channels, noises, sigmas, strict=True):
        result = channel
        if args.denoise > 0:
            result = denoise(result, noise, args.denoise, args.layers)
        if sigma < target:
            result = gaussian_blur(result, float(np.sqrt(target**2 - sigma**2)))
        if args.iterations > 0:
            result = deconvolve(result, target, args.iterations, noise, args.support)
        if args.gains:
            result = sharpen(result, args.gains)
        print(
            f"{name}: stars sigma {sigma:.2f} px (FWHM {2.3548 * sigma:.1f}) -> "
            f"{measure_psf(result):.2f} px, noise {noise:.3g}"
        )
        results.append(result)
    output = np.stack(results, axis=-1) if colour else results[0]
    save_fits(args.output, Frame(output, frame.header))
    print(f"wrote {args.output}")
    if args.png:
        stretched = autostretch(output)
        Image.fromarray((stretched * 255 + 0.5).astype(np.uint8)).save(args.png)
        print(f"wrote {args.png}")


def _denoise(args: argparse.Namespace) -> None:
    frame = load(args.file)
    data = normalized(frame.data)
    mask = normalized(load(args.mask).data) if args.mask else None
    cleaned, used_mask, components = denoise_image(
        data, tuple(args.threshold), tuple(args.chroma_threshold), mask, args.sky, args.amount
    )
    for component in components:
        before = ", ".join(f"{sigma:.3g}" for sigma in component.before)
        after = ", ".join(f"{sigma:.3g}" for sigma in component.after)
        print(f"{component.name}: noise by layer {before} -> {after}")
    save_fits(args.output, Frame(cleaned, frame.header))
    print(f"wrote {args.output}")
    if args.mask_output:
        save_fits(args.mask_output, Frame(used_mask))
        print(f"wrote {args.mask_output}")
    if args.residual:
        save_fits(args.residual, Frame((data - cleaned).astype(np.float32), frame.header))
        print(f"wrote {args.residual}")
    if args.png:
        stretched = autostretch(cleaned)
        Image.fromarray((stretched * 255 + 0.5).astype(np.uint8)).save(args.png)
        print(f"wrote {args.png}")


def _preview(args: argparse.Namespace) -> None:
    stretched = autostretch(normalized(load(args.file).data))
    Image.fromarray((stretched * 255 + 0.5).astype(np.uint8)).save(args.output)
    print(f"wrote {args.output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="quietsky", description=__doc__)
    commands = parser.add_subparsers(required=True)

    info = commands.add_parser("info", help="show size, statistics and header of frames")
    info.add_argument("files", nargs="+", type=Path)
    info.set_defaults(run=_info)

    stack = commands.add_parser(
        "stack", help="average bias, dark, flat or registered light frames with outlier rejection"
    )
    stack.add_argument("files", nargs="+", type=Path)
    stack.add_argument("-o", "--output", type=Path, required=True, help="FITS file to write")
    stack.add_argument("--sigma-low", type=float, default=4.0)
    stack.add_argument("--sigma-high", type=float, default=3.0)
    kind = stack.add_mutually_exclusive_group()
    kind.add_argument(
        "--flat",
        action="store_true",
        help="match the brightness of the frames before combining them, as flats need",
    )
    kind.add_argument(
        "--light",
        action="store_true",
        help="match level and scale, weight frames by noise and ignore black or saturated "
        "pixels, as registered lights need",
    )
    stack.set_defaults(run=_stack)

    calibration = commands.add_parser(
        "calibrate",
        help="subtract a master bias and a scaled master dark from frames, and divide by a "
        "flat; each master is optional",
    )
    calibration.add_argument("files", nargs="+", type=Path)
    calibration.add_argument("--bias", type=Path, help="master bias")
    calibration.add_argument("--dark", type=Path, help="master dark")
    calibration.add_argument("--flat", type=Path, help="master flat to divide by (for lights)")
    calibration.add_argument(
        "--dark-scale",
        type=float,
        help="factor to scale the dark by (default: the factor that minimises noise)",
    )
    calibration.add_argument(
        "-o", "--output", type=Path, required=True, help="directory for the calibrated frames"
    )
    calibration.set_defaults(run=_calibrate)

    debayer = commands.add_parser("debayer", help="turn colour mosaics into RGB images (VNG)")
    debayer.add_argument("files", nargs="+", type=Path)
    debayer.add_argument(
        "--pattern", choices=BAYER_PATTERNS, help="Bayer pattern (default: BAYERPAT in the header)"
    )
    debayer.add_argument(
        "-o", "--output", type=Path, required=True, help="directory for the RGB frames"
    )
    debayer.set_defaults(run=_debayer)

    register = commands.add_parser(
        "register", help="align frames to a reference frame by their stars"
    )
    register.add_argument("files", nargs="+", type=Path)
    register.add_argument("--reference", type=Path, required=True, help="frame to align to")
    register.add_argument(
        "-o", "--output", type=Path, required=True, help="directory for the aligned frames"
    )
    register.set_defaults(run=_register)

    whole = commands.add_parser(
        "preprocess",
        help="run the whole pipeline on a session: masters, calibration, debayer, "
        "registration and integration",
    )
    for name in ("bias", "dark", "flat"):
        whole.add_argument(f"--{name}", type=Path, help=f"directory of {name} frames, if any")
    whole.add_argument("--light", type=Path, required=True, help="directory of light frames")
    for name in ("bias", "dark", "flat"):
        whole.add_argument(
            f"--master-{name}",
            type=Path,
            help=f"a master {name} made elsewhere, used as it is instead of --{name}",
        )
    whole.add_argument(
        "--reference",
        type=Path,
        help="calibrated frame to align the lights to (default: the first light)",
    )
    whole.add_argument(
        "-o", "--output", type=Path, required=True, help="directory for masters and aligned lights"
    )
    whole.set_defaults(run=_preprocess)

    inventory = commands.add_parser(
        "inventory", help="sort the frames under a directory into groups by kind and filter"
    )
    inventory.add_argument("directory", type=Path)
    inventory.add_argument(
        "--frames", action="store_true", help="list every frame and how it was classified"
    )
    inventory.set_defaults(run=_inventory)

    background = commands.add_parser(
        "background", help="subtract the sky background gradient from a linear image"
    )
    background.add_argument("file", type=Path)
    background.add_argument("-o", "--output", type=Path, required=True, help="FITS file to write")
    background.add_argument(
        "--model",
        choices=MODELS,
        default="spline",
        help="shape of the background: a spline follows the samples closely, a quadratic or "
        "a plane only a gradient (for targets that fill the frame)",
    )
    background.add_argument(
        "--write-model", type=Path, help="also write the background model to this FITS file"
    )
    background.add_argument(
        "--samples-per-row", type=int, default=16, help="background samples across the frame"
    )
    background.add_argument(
        "--smoothing", type=float, default=0.25, help="how loosely the model follows the samples"
    )
    background.set_defaults(run=_background)

    lunar = commands.add_parser(
        "lunar",
        help="lucky-image videos of the Moon: stack the sharpest frames of each part, "
        "combine the clips and sharpen",
    )
    lunar.add_argument("videos", nargs="+", type=Path, help="uncompressed 8-bit AVI files")
    lunar.add_argument(
        "-o", "--output", type=Path, required=True, help="directory for the stacks and image"
    )
    lunar.add_argument(
        "--keep", type=float, default=0.1, help="fraction of frames stacked in each box"
    )
    lunar.add_argument(
        "--combine-keep",
        type=float,
        default=0.5,
        help="fraction of the clip stacks combined in each box",
    )
    lunar.add_argument(
        "--chunk",
        type=int,
        default=750,
        help="stack long clips in pieces of about this many frames",
    )
    lunar.add_argument("--box", type=int, default=64, help="alignment box size in pixels")
    lunar.add_argument(
        "--drizzle",
        type=int,
        default=2,
        help="stack onto a grid this many times finer than the frames (1 for none)",
    )
    lunar.add_argument(
        "--deconvolve",
        type=lambda text: None if text == "none" else tuple(float(v) for v in text.split(":")),
        default=(1.5, 30),
        help="Richardson-Lucy deconvolution as SIGMA:ITERATIONS, or none (default 1.5:30)",
    )
    lunar.add_argument(
        "--gains",
        type=lambda text: [float(value) for value in text.split(",")],
        help="wavelet layer gains, finest first, applied after any deconvolution "
        "(e.g. 1.3,1.8,1.6,1.3,1.1 without deconvolution)",
    )
    lunar.add_argument(
        "--saturation",
        type=float,
        default=3.0,
        help="colour boost for moon-colour.png; 0 for no colour image",
    )
    lunar.add_argument(
        "--noise-threshold",
        type=float,
        default=1.0,
        help="leave out finest-layer values within this many noise deviations before sharpening",
    )
    lunar.add_argument("--gamma", type=float, default=1.0, help="gamma applied to the PNG")
    lunar.set_defaults(run=_lunar)

    preview = commands.add_parser("preview", help="write an auto-stretched PNG of a frame")
    preview.add_argument("file", type=Path)
    preview.add_argument("-o", "--output", type=Path, required=True, help="PNG file to write")
    preview.set_defaults(run=_preview)

    def floats(text: str) -> list[float]:
        return [float(value) for value in text.split(",")]

    denoise_cmd = commands.add_parser(
        "denoise",
        help="hide the noise of a linear master before it is stretched: shrink the wavelet "
        "layers within the noise, under a mask that spares bright signal, colour harder "
        "than brightness",
    )
    denoise_cmd.add_argument(
        "file", type=Path, help="a background-subtracted master, mono or colour"
    )
    denoise_cmd.add_argument("-o", "--output", type=Path, required=True, help="FITS to write")
    denoise_cmd.add_argument(
        "--threshold",
        type=floats,
        default=list(LUMINANCE_THRESHOLDS),
        help="luminance thresholds in noise deviations, one per wavelet layer, finest first "
        f"(default {','.join(f'{t:g}' for t in LUMINANCE_THRESHOLDS)}); 0 for none",
    )
    denoise_cmd.add_argument(
        "--chroma-threshold",
        type=floats,
        default=list(CHROMA_THRESHOLDS),
        help="the same for the colour, each channel less the luminance "
        f"(default {','.join(f'{t:g}' for t in CHROMA_THRESHOLDS)})",
    )
    denoise_cmd.add_argument(
        "--mask",
        type=Path,
        help="FITS mask: 1 where the luminance is denoised, 0 where it is kept (default: "
        "built from the smoothed, stretched, inverted luminance)",
    )
    denoise_cmd.add_argument(
        "--sky",
        type=lambda text: tuple(int(value) for value in text.split(",")),
        help="X,Y,W,H of a patch of empty sky to measure the noise in (default: the pixels "
        "within the noise at every scale)",
    )
    denoise_cmd.add_argument(
        "--amount",
        type=float,
        default=1.0,
        help="how much of the luminance denoising to take, 0 to 1; less keeps some grain",
    )
    denoise_cmd.add_argument("--mask-output", type=Path, help="also write the mask used")
    denoise_cmd.add_argument("--residual", type=Path, help="also write input minus output")
    denoise_cmd.add_argument("--png", type=Path, help="also write an auto-stretched PNG")
    denoise_cmd.set_defaults(run=_denoise)

    sharpen_cmd = commands.add_parser(
        "sharpen",
        help="tighten the stars of a deep-sky master by deconvolution, after dropping its "
        "finest noise",
    )
    sharpen_cmd.add_argument("file", type=Path, help="a master light, mono or colour")
    sharpen_cmd.add_argument("-o", "--output", type=Path, required=True, help="FITS to write")
    sharpen_cmd.add_argument(
        "--psf",
        type=float,
        help="Gaussian sigma of the star profile in pixels (default: measured from the stars, "
        "per channel)",
    )
    sharpen_cmd.add_argument(
        "--iterations", type=int, default=5, help="Richardson-Lucy iterations; 0 for none"
    )
    sharpen_cmd.add_argument(
        "--denoise",
        type=float,
        default=3.0,
        help="threshold, in noise deviations, below which the finest wavelet layer is dropped "
        "(halved for each coarser layer); 0 for none",
    )
    sharpen_cmd.add_argument("--layers", type=int, default=1, help="wavelet layers to denoise")
    sharpen_cmd.add_argument(
        "--support",
        type=float,
        default=3.0,
        help="deconvolve only where the smoothed signal stands this many noise deviations "
        "above the sky",
    )
    sharpen_cmd.add_argument(
        "--gains",
        type=lambda text: [float(value) for value in text.split(",")],
        help="wavelet layer gains applied afterwards, finest first (e.g. 1.0,1.3,1.2)",
    )
    sharpen_cmd.add_argument("--png", type=Path, help="also write an auto-stretched PNG")
    sharpen_cmd.set_defaults(run=_sharpen)

    args = parser.parse_args(argv)
    args.run(args)
