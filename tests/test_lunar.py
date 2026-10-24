import struct
from pathlib import Path

import numpy as np
import numpy.typing as npt

from quietsky.avi import read_video
from quietsky.cli import _png16
from quietsky.deconvolve import gaussian_blur, richardson_lucy
from quietsky.lunar import (
    align_channels,
    alignment_matrix,
    balanced,
    bayer_balance,
    colour_mean,
    find_rotation,
    fourier_shift,
    global_alignment,
    lucky_stack,
    mineral_colour,
    moon_parallactic_angle,
    mosaic_luminance,
    sharpen,
    sharpness,
    shift_image,
    upsample,
    whole_frames,
)
from quietsky.resample import resample

Float32 = npt.NDArray[np.float32]
SHAPE = (256, 320)


def moon(shape: tuple[int, int] = SHAPE, seed: int = 1) -> Float32:
    """A bright disc covered in craters of many sizes, on black."""
    rng = np.random.default_rng(seed)
    rows, columns = np.mgrid[: shape[0], : shape[1]].astype(np.float64)
    cy, cx, radius = shape[0] / 2, shape[1] / 2, min(shape) * 0.4
    image = np.where(np.hypot(rows - cy, columns - cx) < radius, 0.6, 0.0)
    for _ in range(300):
        y, x = rng.uniform(cy - radius, cy + radius), rng.uniform(cx - radius, cx + radius)
        size = rng.uniform(1.5, 12)
        depth = rng.uniform(-0.25, 0.25)
        image += depth * np.exp(-((rows - y) ** 2 + (columns - x) ** 2) / (2 * size**2))
    image *= np.hypot(rows - cy, columns - cx) < radius
    return np.clip(image, 0, 1).astype(np.float32)


def blurred(image: Float32, sigma: float) -> Float32:
    fy = np.fft.fftfreq(image.shape[0])[:, None]
    fx = np.fft.rfftfreq(image.shape[1])[None, :]
    kernel = np.exp(-2 * (np.pi * sigma) ** 2 * (fy**2 + fx**2))
    return np.fft.irfft2(np.fft.rfft2(image) * kernel, s=image.shape).astype(np.float32)


def video(
    count: int = 40, seed: int = 2
) -> tuple[Float32, list[Float32], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Frames of the synthetic moon, each shifted and blurred by a different amount."""
    rng = np.random.default_rng(seed)
    truth = moon()
    shifts = rng.uniform(-6, 6, (count, 2))
    blurs = rng.uniform(0.6, 3.0, count)
    frames = [
        shift_image(blurred(truth, blur), dy, dx) + rng.normal(0, 0.01, SHAPE).astype(np.float32)
        for (dy, dx), blur in zip(shifts, blurs, strict=True)
    ]
    return truth, frames, shifts, blurs


def test_mosaic_luminance_is_flat_on_a_flat_colour() -> None:
    # A grey seen through a GRBG mosaic with unbalanced channels.
    mosaic = np.empty((8, 10), np.float32)
    mosaic[0::2, 0::2] = mosaic[1::2, 1::2] = 0.5  # green
    mosaic[0::2, 1::2] = 0.2  # red
    mosaic[1::2, 0::2] = 0.8  # blue

    luminance = mosaic_luminance(mosaic)

    np.testing.assert_allclose(luminance, 0.25 * 0.2 + 0.5 * 0.5 + 0.25 * 0.8, rtol=1e-6)


def test_mosaic_luminance_scales_integers_to_one() -> None:
    np.testing.assert_allclose(mosaic_luminance(np.full((4, 4), 255, np.uint8)), 1.0)


def test_fourier_shift_moves_by_a_fraction_of_a_pixel() -> None:
    image = blurred(moon(), 2.0)

    there_and_back = fourier_shift(fourier_shift(image, 0.3, -0.7), -0.3, 0.7)

    np.testing.assert_allclose(there_and_back, image, atol=1e-5)


def test_shift_image_moves_content_the_right_way() -> None:
    image = np.zeros((20, 20), np.float32)
    image[5, 7] = 1

    moved = shift_image(image, 3, -2)

    assert tuple(int(i) for i in np.unravel_index(np.argmax(moved), moved.shape)) == (8, 5)


def test_blur_lowers_sharpness() -> None:
    truth = moon()

    assert (
        sharpness(blurred(truth, 0.8))
        > sharpness(blurred(truth, 2.0))
        > sharpness(blurred(truth, 4.0))
    )


def test_global_alignment_finds_the_shifts() -> None:
    truth, frames, shifts, _ = video(count=8)

    found = global_alignment(lambda i: frames[i], len(frames), truth, workers=2)

    np.testing.assert_allclose(found, shifts, atol=0.25)


def test_lucky_stack_beats_the_plain_average() -> None:
    truth, frames, shifts, blurs = video()

    result = lucky_stack(lambda i: frames[i], len(frames), SHAPE, keep=0.25, box_size=32, workers=2)

    # The stack sits on the sharpest frame; put the truth there to compare.
    dy, dx = shifts[result.reference_index]
    reference = shift_image(truth, dy, dx)
    average = np.mean(
        [shift_image(frame, -sy, -sx) for frame, (sy, sx) in zip(frames, shifts, strict=True)],
        axis=0,
    )
    average = shift_image(average.astype(np.float32), dy, dx)
    inner = (slice(40, -40), slice(40, -40))
    stack_error = np.abs(result.image + result.background - reference)[inner].mean()
    average_error = np.abs(average - reference)[inner].mean()
    assert stack_error < 0.7 * average_error
    # Frames this close in blur can swap places once noise is added.
    assert result.reference_index in np.argsort(blurs)[:3]


def test_lucky_stack_leaves_out_frames_without_the_whole_moon() -> None:
    truth, frames, shifts, blurs = video()
    rng = np.random.default_rng(3)
    # The moon slides mostly out of frame, then the frame is left with noise
    # alone, whose fine grain looks sharper than any real frame.
    frames[5] = shift_image(frames[5], 0, 150)
    frames[9] = rng.normal(0, 0.05, SHAPE).astype(np.float32)

    result = lucky_stack(lambda i: frames[i], len(frames), SHAPE, keep=0.25, box_size=32, workers=2)

    assert not result.whole[5] and not result.whole[9]
    assert result.whole.sum() == len(frames) - 2
    assert result.reference_index not in (5, 9)
    assert result.quality[5] == result.quality[9] == 0


def test_whole_frames_keeps_every_frame_of_a_steady_run() -> None:
    _, frames, _, _ = video()
    assert whole_frames([frame[::4, ::4] for frame in frames]).all()


def test_find_rotation_undoes_a_turn_and_a_shift() -> None:
    reference = blurred(moon((256, 256)), 1.0)
    matrix = alignment_matrix(np.radians(3.0), 4.0, -2.5, reference.shape)
    # An image that, put through matrix, gives back the reference.
    image = resample(reference, np.linalg.inv(matrix))

    angle, dy, dx = find_rotation(reference, image, workers=2)

    assert abs(np.degrees(angle) - 3.0) < 0.1
    aligned = resample(image, alignment_matrix(angle, dy, dx, reference.shape))
    inner = (slice(60, -60), slice(60, -60))
    assert np.abs(aligned - reference)[inner].mean() < 0.01


def test_find_rotation_reaches_a_large_turn_near_the_expected_one() -> None:
    reference = blurred(moon((256, 256)), 1.0)
    matrix = alignment_matrix(np.radians(60.0), 3.0, 1.5, reference.shape)
    image = resample(reference, np.linalg.inv(matrix))

    angle, _, _ = find_rotation(
        reference, image, max_angle=np.radians(3), workers=2, expected=np.radians(58.5)
    )

    assert abs(np.degrees(angle) - 60.0) < 0.1


def test_moon_parallactic_angle_turns_as_the_field_did_on_2026_10_23() -> None:
    # Nashville, first and last clips of the night: the clip stacks fitted
    # within 0.1 degrees of this where the old +-15 degree search reached.
    site = (36.1817, -86.7322)
    first = moon_parallactic_angle("2026-10-24T00:40:47", *site)
    last = moon_parallactic_angle("2026-10-24T07:22:58", *site)
    assert abs(np.degrees(last - first) - 95.5) < 0.5


def test_sharpen_with_unit_gains_changes_nothing() -> None:
    image = blurred(moon(), 1.5)

    np.testing.assert_allclose(sharpen(image, [1, 1, 1, 1]), image, atol=1e-5)


def test_sharpen_brings_back_fine_detail_without_changing_the_level() -> None:
    truth = blurred(moon(), 0.7)
    soft = blurred(truth, 1.5)

    sharpened = sharpen(soft, [1.8, 1.6, 1.3, 1.0])

    assert sharpness(soft) < sharpness(sharpened) <= 1.5 * sharpness(truth)
    assert abs(float(sharpened.mean()) - float(soft.mean())) < 1e-4


def write_avi(path: Path, frames: list[npt.NDArray[np.uint8]], segment_frames: int) -> None:
    """A minimal uncompressed 8-bit AVI, split into OpenDML segments."""
    height, width = frames[0].shape

    def chunk(chunk_id: bytes, body: bytes) -> bytes:
        return chunk_id + struct.pack("<I", len(body)) + body + b"\0" * (len(body) & 1)

    def riff(kind: bytes, body: bytes) -> bytes:
        return chunk(b"RIFF", kind + body)

    def listed(kind: bytes, body: bytes) -> bytes:
        return chunk(b"LIST", kind + body)

    avih = struct.pack("<10I", 100000, 0, 0, 0, len(frames), 0, 1, 0, width, height) + bytes(16)
    strf = struct.pack("<IiiHHIIiiII", 40, width, -height, 1, 8, 0, 0, 0, 0, 256, 0)
    header = listed(
        b"hdrl",
        chunk(b"avih", avih)
        + listed(b"strl", chunk(b"strh", b"vidsDIB " + bytes(48)) + chunk(b"strf", strf)),
    )
    segments = [frames[i : i + segment_frames] for i in range(0, len(frames), segment_frames)]
    data = b""
    for number, segment in enumerate(segments):
        movi = b"".join(chunk(b"JUNK", bytes(6)) + chunk(b"00db", f.tobytes()) for f in segment)
        body = (header if number == 0 else b"") + listed(b"movi", movi)
        data += riff(b"AVI " if number == 0 else b"AVIX", body)
    path.write_bytes(data)


def test_read_video_survives_oversized_segment_lengths(tmp_path: Path) -> None:
    # The Seestar writes each AVIX segment's size 8 bytes larger than it is.
    rng = np.random.default_rng(4)
    frames = [rng.integers(0, 256, (6, 4), dtype=np.uint8) for _ in range(6)]
    path = tmp_path / "clip.avi"
    write_avi(path, frames, segment_frames=2)
    data = bytearray(path.read_bytes())
    offset = 0
    while offset + 12 <= len(data):
        size = struct.unpack("<I", data[offset + 4 : offset + 8])[0]
        if data[offset + 8 : offset + 12] == b"AVIX":
            data[offset + 4 : offset + 8] = struct.pack("<I", size + 8)
        offset += 8 + size
    path.write_bytes(bytes(data))

    video = read_video(path)

    assert len(video) == 6
    np.testing.assert_array_equal(video.frame(5), frames[5])


def test_read_video_finds_frames_in_every_segment(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 256, (6, 4), dtype=np.uint8) for _ in range(5)]
    path = tmp_path / "clip.avi"
    write_avi(path, frames, segment_frames=2)
    (tmp_path / "clip.avi.txt").write_text("[Seestar S50]\nBayer = GR\nOBJECT = Lunar\n")

    video = read_video(path)

    assert (len(video), video.width, video.height, video.top_down) == (5, 4, 6, True)
    for index, frame in enumerate(frames):
        np.testing.assert_array_equal(video.frame(index), frame)
    assert video.header == {"BAYERPAT": "GRBG", "OBJECT": "Lunar"}


def test_richardson_lucy_sharpens_and_keeps_the_brightness() -> None:
    truth = blurred(moon(), 0.6)
    soft = blurred(truth, 1.5)

    restored = richardson_lucy(soft, 1.5, 30)

    assert np.abs(restored - truth).mean() < 0.8 * np.abs(soft - truth).mean()
    assert abs(float(restored.sum()) / float(soft.sum()) - 1) < 0.01


def test_gaussian_blur_keeps_the_total() -> None:
    image = moon()

    assert abs(float(gaussian_blur(image, 3.0).sum()) - float(image.sum())) < 1e-2 * image.sum()


def test_colour_mean_lines_frames_up() -> None:
    rgb = np.stack([moon(), 0.5 * moon(), 0.25 * moon()], axis=-1)
    shifts = np.array([[0.0, 0.0], [2.5, -1.5], [-3.0, 4.0]])
    frames = [
        np.stack([shift_image(rgb[..., c], dy, dx) for c in range(3)], axis=-1) for dy, dx in shifts
    ]

    mean = colour_mean(lambda i: frames[i], [0, 1, 2], shifts)

    inner = (slice(20, -20), slice(20, -20))
    np.testing.assert_allclose(mean[inner], rgb[inner], atol=0.02)


def test_mineral_colour_leaves_a_grey_moon_grey() -> None:
    luminance = moon()
    colour = np.stack([luminance] * 3, axis=-1)

    result = mineral_colour(luminance, colour, saturation=4.0)

    np.testing.assert_allclose(result[..., 0], result[..., 2], atol=1e-3)
    np.testing.assert_allclose(result.mean(axis=2), luminance, atol=1e-3)


def test_mineral_colour_strengthens_a_tint() -> None:
    luminance = moon()
    rows = np.arange(SHAPE[0])[:, None] < SHAPE[0] // 2
    # The top half slightly blue, the bottom slightly red.
    tint = np.where(rows[..., None], [0.98, 1.0, 1.02], [1.02, 1.0, 0.98])
    colour = (luminance[..., None] * tint).astype(np.float32)

    result = mineral_colour(luminance, colour, saturation=4.0)

    top = result[90:110, 120:200].reshape(-1, 3).mean(axis=0)
    bottom = result[150:170, 120:200].reshape(-1, 3).mean(axis=0)
    assert top[2] / top[0] > 1.1 and bottom[0] / bottom[2] > 1.1


def test_png16_writes_rgb(tmp_path: Path) -> None:
    from PIL import Image

    image = np.random.default_rng(5).random((5, 7, 3)).astype(np.float32)
    path = tmp_path / "rgb.png"

    _png16(path, image)

    with Image.open(path) as written:
        assert written.size == (7, 5)
        assert written.mode in ("RGB", "RGB;16", "I;16")


def test_align_channels_undoes_dispersion() -> None:
    green = blurred(moon(), 1.0)
    colour = np.stack(
        [shift_image(green, 1.5, 0.0), green, shift_image(green, -1.2, 0.3)], axis=-1
    ).astype(np.float32)

    aligned, moves = align_channels(colour)

    np.testing.assert_allclose(moves[0], (-1.5, 0.0), atol=0.1)
    np.testing.assert_allclose(moves[1], (1.2, -0.3), atol=0.1)
    inner = (slice(30, -30), slice(30, -30))
    assert np.abs(aligned[..., 0] - green)[inner].mean() < 0.005


def test_bayer_balance_turns_a_grey_mosaic_into_its_luminance() -> None:
    truth = blurred(moon(), 0.8)
    # A GRBG camera seeing a grey moon: each site with its own response and black level.
    response = np.array([[0.9, 0.6], [1.4, 0.95]], np.float32)
    black = np.array([[0.05, 0.06], [0.04, 0.05]], np.float32)
    reps = (SHAPE[0] // 2, SHAPE[1] // 2)
    mosaic = truth * np.tile(response, reps) + np.tile(black, reps)

    sky, gain = bayer_balance(mosaic)
    flat = balanced(mosaic, sky, gain)

    # Every site now has the same response, so no grid is left.
    lit = truth > 0.3
    ratio = flat[lit] / truth[lit]
    assert ratio.std() < 0.02 * ratio.mean()


def test_upsample_keeps_a_smooth_image() -> None:
    image = blurred(moon(), 3.0)

    fine = upsample(image, 2, workers=2)

    assert fine.shape == (2 * SHAPE[0], 2 * SHAPE[1])
    # Fine pixels 2i and 2i+1 straddle coarse pixel i.
    paired = 0.5 * (fine[:, 0::2] + fine[:, 1::2])
    paired = 0.5 * (paired[0::2] + paired[1::2])
    inner = (slice(10, -10), slice(10, -10))
    np.testing.assert_allclose(paired[inner], image[inner], atol=0.01)


def test_drizzle_recovers_detail_finer_than_a_pixel() -> None:
    rng = np.random.default_rng(7)
    fine_shape = (2 * SHAPE[0], 2 * SHAPE[1])
    base = blurred(moon(fine_shape, seed=3), 1.5)
    # Faint stripes three fine pixels apart: finer than the coarse frames can hold.
    columns = np.arange(fine_shape[1])
    stripes = 0.04 * np.cos(2 * np.pi * columns / 3)[None, :] * (base > 0.2)
    truth = (base + stripes).astype(np.float32)
    shifts = rng.uniform(-6, 6, (40, 2))  # in fine pixels

    def bin2(image: Float32) -> Float32:
        h, w = image.shape
        return image.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).astype(np.float32)

    frames = [
        bin2(shift_image(truth, dy, dx)) + rng.normal(0, 0.002, SHAPE).astype(np.float32)
        for dy, dx in shifts
    ]

    def source(i: int) -> Float32:
        return frames[i]

    def stripe_amplitude(image: Float32, reference_index: int) -> float:
        dy, dx = shifts[reference_index]
        x = columns - dx
        lit = shift_image(base, dy, dx) > 0.3
        wave = np.exp(-2j * np.pi * x / 3)[None, :]
        return float(2 * np.abs((image * wave)[lit].mean()) / lit.mean() * lit.mean())

    drizzled = lucky_stack(source, len(frames), SHAPE, keep=1.0, box_size=32, workers=2, scale=2)
    plain = lucky_stack(source, len(frames), SHAPE, keep=1.0, box_size=32, workers=2)
    upsampled = upsample(plain.image, 2, workers=2)
    assert drizzled.image.shape == fine_shape
    found = stripe_amplitude(drizzled.image, drizzled.reference_index)
    lost = stripe_amplitude(upsampled, plain.reference_index)
    # The stripes reach the frames weakened by the pixel footprint (to 0.41 of
    # their 0.04) and drizzle's own sharing between fine pixels (to 0.68 of
    # that), so about 0.011 can come back. Plain averaging loses them.
    assert found > 0.008
    assert found > 3 * lost
