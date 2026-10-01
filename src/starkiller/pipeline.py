"""Run the whole preprocessing pipeline on one session's frames.

Bias, dark and flat frames become master frames; each light is calibrated
with them, demosaiced, aligned to the first light and written out; the
aligned lights are integrated into the master light.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from starkiller.calibrate import calibrate, optimize_dark, unit_flat
from starkiller.debayer import debayer_vng
from starkiller.frame import Frame, HeaderValue, load, normalized, save_fits
from starkiller.integrate import (
    flux_normalization,
    frame_estimates,
    integrate,
    integrate_lights,
)
from starkiller.noise import evaluate_noise
from starkiller.register import RegistrationError, solve_transformation
from starkiller.resample import resample
from starkiller.stars import detect_stars
from starkiller.stretch import autostretch

Float32 = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Session:
    """The frames of one session, by kind."""

    bias: Sequence[Path]
    dark: Sequence[Path]
    flat: Sequence[Path]
    light: Sequence[Path]


def load_normalized(path: Path) -> Frame:
    """Load a frame with its pixel values as float32, integers mapped onto [0, 1]."""
    frame = load(path)
    return Frame(normalized(frame.data), frame.header)


def noise_keywords(image: Float32) -> dict[str, HeaderValue]:
    """Header keywords recording the noise of each channel of a colour image.

    Written before registration, which smooths the noise; light integration
    weights frames by these.
    """
    return {
        f"NOISE{channel:02d}": evaluate_noise(image[..., channel]).sigma
        for channel in range(image.shape[-1])
    }


def frame_noise(
    frames: Sequence[Frame], stack: npt.NDArray[np.floating]
) -> npt.NDArray[np.float64]:
    """Noise of each channel of each frame of an (N, H, W, C) stack, as (N, C).

    Taken from the frames' NOISE keywords where they all have them, and
    measured on the stack otherwise.
    """
    noise = np.empty((len(frames), stack.shape[-1]))
    for channel in range(stack.shape[-1]):
        key = f"NOISE{channel:02d}"
        if all(key in frame.header for frame in frames):
            noise[:, channel] = [float(frame.header[key]) for frame in frames]
        else:
            noise[:, channel] = [evaluate_noise(image[..., channel]).sigma for image in stack]
    return noise


def _exposure(frames: Sequence[Frame]) -> float | None:
    """Exposure time shared by some frames, if their headers give one."""
    exposure = frames[0].header.get("EXPTIME")
    return float(exposure) if isinstance(exposure, int | float) else None


def preprocess(
    session: Session,
    output: Path,
    load_frame: Callable[[Path], Frame] = load_normalized,
    report: Callable[[str], None] = print,
) -> Path:
    """Preprocess a session and return the path of the master light.

    Writes master/bias.fits, dark.fits, flat.fits and light.fits, a preview
    master/light.png, and the aligned lights in registered/. Lights are
    aligned to the first one; a light whose stars cannot be matched to it is
    left out. Hot pixels are not corrected.
    """
    master = output / "master"
    registered = output / "registered"
    master.mkdir(parents=True, exist_ok=True)
    registered.mkdir(exist_ok=True)

    bias = integrate(np.stack([load_frame(path).data for path in session.bias])).image
    save_fits(master / "bias.fits", Frame(bias, {"NCOMBINE": len(session.bias)}))
    report(f"master bias from {len(session.bias)} frames")

    darks = [load_frame(path) for path in session.dark]
    dark = integrate(np.stack([frame.data for frame in darks])).image
    save_fits(master / "dark.fits", Frame(dark, {"NCOMBINE": len(darks)}))
    report(f"master dark from {len(darks)} frames")

    # A flat's exposure is too short for the dark scale to be found from its
    # noise, so the dark is scaled by exposure time, or left out without one.
    flats = [load_frame(path) for path in session.flat]
    flat_exposure, dark_exposure = _exposure(flats), _exposure(darks)
    flat_dark_scale = flat_exposure / dark_exposure if flat_exposure and dark_exposure else 0.0
    calibrated = np.stack([calibrate(frame.data, bias, dark, flat_dark_scale) for frame in flats])
    flat = integrate(calibrated, normalization=flux_normalization(frame_estimates(calibrated)[0]))
    save_fits(master / "flat.fits", Frame(flat.image, {"NCOMBINE": len(flats)}))
    report(f"master flat from {len(flats)} frames, dark scaled by {flat_dark_scale:.3f}")
    flat_unit = unit_flat(flat.image)

    aligned: list[Float32] = []
    frames: list[Frame] = []
    reference = None
    for path in session.light:
        frame = load_frame(path)
        pattern = frame.header.get("BAYERPAT")
        dark_scale = optimize_dark(frame.data, bias, dark, mosaic=isinstance(pattern, str))
        image = calibrate(frame.data, bias, dark, dark_scale, flat_unit)
        header = {key: value for key, value in frame.header.items() if key != "BAYERPAT"}
        header["DARKSCAL"] = round(dark_scale, 4)
        image = debayer_vng(image, pattern) if isinstance(pattern, str) else image[..., None]
        header |= noise_keywords(image)
        stars = detect_stars(image)
        if reference is None:
            reference = stars
            report(f"{path.name}: reference, {len(stars)} stars, dark scale {dark_scale:.3f}")
        else:
            try:
                transformation = solve_transformation(reference, stars)
            except RegistrationError as error:
                report(f"{path.name}: left out, {error}")
                continue
            image = resample(image, transformation.matrix)
            report(
                f"{path.name}: {transformation.pairs} stars matched, "
                f"rms {transformation.rms:.2f} px, dark scale {dark_scale:.3f}"
            )
        save_fits(registered / f"{path.stem}_r.fits", Frame(np.squeeze(image), header))
        aligned.append(image)
        frames.append(Frame(image, header))

    stack = np.stack(aligned)
    del aligned
    results = integrate_lights(stack, frame_noise(frames, stack))
    light = np.squeeze(np.stack([result.image for result in results], axis=-1))
    total = stack.size
    low = sum(int(result.frame_rejected_low.sum()) for result in results)
    high = sum(int(result.frame_rejected_high.sum()) for result in results)
    report(
        f"master light from {len(frames)} frames, "
        f"rejected low {low / total:.3%}, high {high / total:.3%}"
    )
    path = master / "light.fits"
    save_fits(path, Frame(light, {"NCOMBINE": len(frames)}))
    Image.fromarray((autostretch(light) * 255 + 0.5).astype(np.uint8)).save(master / "light.png")
    return path
