"""Separate Hα and OIII in a dual-band colour master, and show the OIII.

    python tools/oiii.py light_bg.fits -o boosted.fits [--boost 2] [--oiii MAP] [--hoo HOO]

Through a dual-band filter on a colour camera the red channel holds Hα
and starlight, the blue channel OIII and starlight, and the green
channel OIII, starlight and a leak of Hα through the green filter. The
channels are balanced on the stars first (quietsky.finish.measure), so
starlight is the same in all three. On nebula pixels, stars masked, the
leak k and the green-to-blue ratio of OIII q are fitted by least squares
in G = k R + q B; the OIII map is the inverse-variance mean of B and
(G - k R) / q, smoothed a little as the noisier line, and made starless
by filling every star's footprint from its surroundings.

The output is the master with the OIII excess, what stands above the
noise in the starless map, added to green and blue boost times over, so
the shell comes out teal where the OIII is and nothing else changes:
stars keep their colour and the sky its noise. --hoo writes a bicolour
instead, Hα in red and OIII in green and blue, with the stars from the
balanced RGB. Either goes through the finish as any master.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

from quietsky.deconvolve import gaussian_blur, star_light, star_profile
from quietsky.denoise import DEFAULT_PSF, noise_factor, star_footprint
from quietsky.finish import measure
from quietsky.frame import Frame, load, normalized, save_fits
from quietsky.stars import detect_stars
from quietsky.wavelets import b3_layers

Float32 = npt.NDArray[np.float32]
Float64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]

MAD_TO_SIGMA = 1.4826
# Nebula pixels for the fit: this many luminance noise deviations above the sky.
NEBULA_SIGNAL = 5.0
# Stars are masked where their modelled light (peak spread by the measured
# profile) stands this many noise deviations above the sky, halos and all,
# the mask grown by this sigma.
STAR_MASK_SIGMA = 1.0
STAR_GROW = 2.0
# The surroundings a star's footprint is filled from, in pixels.
FILL_SIGMA = 8.0
# Pixels this far from the fit, in noise deviations, are dropped and the fit repeated.
FIT_CLIP = 3.0
# The OIII excess starts this many of its noise deviations above zero, and
# is taken from the scales of this many wavelet layers up (4 px and
# coarser), where a star too faint to be detected and masked hardly shows.
EXCESS_SIGMA = 2.0
EXCESS_LAYERS = 2
# Stars are detected this many noise deviations up, for the mask; and as
# the detector keeps the brightest few thousand only, every point-like
# thing is masked as well: where the luminance's two finest wavelet
# layers stand this many of their noise deviations up.
STAR_THRESHOLD = 3.0
POINT_SIGMA = 3.0


def noise_of(values: Float64) -> float:
    return MAD_TO_SIGMA * float(np.median(np.abs(values - np.median(values))))


def filled(image: Float64, mask: Bool, sigma: float) -> Float64:
    """The image with the masked pixels replaced by a blur of the unmasked ones around them."""
    keep = (~mask).astype(np.float32)
    light = gaussian_blur((image * keep).astype(np.float32), sigma)
    weight = gaussian_blur(keep, sigma)
    fill = light / np.maximum(weight, 1e-6)
    return np.where(mask, fill, image).astype(np.float64)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("master", type=Path, help="a linear colour master, background removed")
    parser.add_argument("-o", "--output", type=Path, required=True, help="the boosted FITS")
    parser.add_argument(
        "--boost", type=float, default=2.0, help="how many times over the OIII excess is added"
    )
    parser.add_argument("--oiii", type=Path, help="also write the starless OIII map")
    parser.add_argument(
        "--coverage",
        type=Path,
        help="the master's coverage.fits: the fit and the noise are taken on the deepest "
        "half of the frame, as a union canvas's shallow edges would swamp them",
    )
    parser.add_argument("--hoo", type=Path, help="also write a bicolour, Hα red and OIII teal")
    parser.add_argument(
        "--smooth", type=float, default=1.0, help="sigma, px, the OIII map is smoothed by"
    )
    args = parser.parse_args(argv)

    header = load(args.master).header
    image = normalized(load(args.master).data)
    rgb = image.astype(np.float64)
    covered = rgb.max(axis=2) > 0
    deep = covered
    if args.coverage:
        coverage = load(args.coverage).data
        deep = covered & (coverage >= coverage.max() / 2)
    sky, noise, gains = measure(rgb, valid=deep)
    balanced = (rgb - sky) * gains
    r, g, b = (balanced[..., c] for c in range(3))
    lum = balanced.mean(axis=2).astype(np.float32)
    stars = detect_stars(lum, STAR_THRESHOLD)
    try:
        profile = star_profile(lum, stars)
    except ValueError:
        offsets = np.arange(-10, 11, dtype=np.float64)
        profile = np.exp(-(offsets[:, None] ** 2 + offsets[None, :] ** 2) / (2 * DEFAULT_PSF**2))
    rows = np.clip(np.rint(stars.y).astype(np.intp), 0, lum.shape[0] - 1)
    columns = np.clip(np.rint(stars.x).astype(np.intp), 0, lum.shape[1] - 1)
    model = star_light(lum.shape, stars, lum[rows, columns].astype(np.float64), profile)
    core = star_footprint(lum, stars) > 0.02
    layers, _ = b3_layers(lum, 2)
    fine = (layers[0] + layers[1]).astype(np.float64)
    fine_noise = noise_of(fine[deep & ~core])
    point = fine > POINT_SIGMA * fine_noise
    starry = (
        gaussian_blur(
            (core | point | (model > STAR_MASK_SIGMA * noise)).astype(np.float32), STAR_GROW
        )
        > 0.05
    )
    print(f"{len(stars)} stars; {100 * starry.mean():.1f}% of the frame masked as star light")
    nebula = deep & ~starry & (lum > NEBULA_SIGNAL * noise)
    print(f"balanced on the stars, gains {gains.round(3)}; {int(nebula.sum())} nebula pixels")

    # G = k R + q B on the nebula, outliers (OIII knots, star halos) clipped.
    use = nebula.copy()
    k = q = 0.0
    for _ in range(5):
        terms = np.stack([r[use], b[use]], axis=1)
        k, q = np.linalg.lstsq(terms, g[use], rcond=None)[0]
        residual = g - (k * r + q * b)
        use = nebula & (np.abs(residual) < FIT_CLIP * noise_of(residual[use]))
    print(f"Hα leak into green k = {k:.3f}; OIII green over blue q = {q:.3f}")

    # OIII from blue and from green less the leak, weighted by their noise.
    oiii_b = b
    oiii_g = (g - k * r) / max(q, 1e-6)
    empty = deep & ~starry & (np.abs(lum) < 2 * noise)
    var_b, var_g = noise_of(oiii_b[empty]) ** 2, noise_of(oiii_g[empty]) ** 2
    oiii = (oiii_b / var_b + oiii_g / var_g) / (1 / var_b + 1 / var_g)
    oiii = filled(oiii, starry | ~covered, FILL_SIGMA)
    if args.smooth > 0:
        oiii = gaussian_blur(oiii.astype(np.float32), args.smooth).astype(np.float64)
    ha = filled(r, starry | ~covered, FILL_SIGMA)
    ha_noise, oiii_noise = noise_of(ha[empty]), noise_of(oiii[empty])
    ratio = oiii[nebula] / np.maximum(ha[nebula], 1e-9)
    print(
        f"sky noise: Hα {ha_noise:.3g}, OIII {oiii_noise:.3g} (blue alone {np.sqrt(var_b):.3g}); "
        f"OIII over Hα on the nebula, 10th/50th/90th percentile: "
        f"{np.percentile(ratio, 10):.2f} / {np.percentile(ratio, 50):.2f} / "
        f"{np.percentile(ratio, 90):.2f}; above 3 sigma on "
        f"{100 * (oiii[nebula] > 3 * oiii_noise).mean():.0f}%"
    )

    # The excess: what stands above the noise, added to green and blue in
    # the measure that a pixel's light is the nebula's rather than a star's,
    # so a bright star's halo is not painted and a faint star's
    # surroundings are not left out.
    starlight = np.clip(gaussian_blur(starry.astype(np.float32), 1.0), 0, 1)
    nebular = np.clip(ha, 0, None) / (np.clip(ha, 0, None) + np.clip(model, 0, None) + 1e-12)
    _, extended = b3_layers(oiii.astype(np.float32), EXCESS_LAYERS)
    floor: float | Float64 = EXCESS_SIGMA * oiii_noise
    if args.coverage:
        floor = floor * noise_factor(coverage)
    excess = np.clip(extended.astype(np.float64) - floor, 0, None) * nebular
    boosted = balanced.copy()
    boosted[..., 1] += args.boost * excess * q
    boosted[..., 2] += args.boost * excess
    out = (boosted / gains + sky).astype(np.float32)
    out[~covered] = 0
    keywords: dict[str, str | int | float | bool] = {
        "OIIILEAK": round(float(k), 4),
        "OIIIQ": round(float(q), 4),
    }
    save_fits(args.output, Frame(out, header | keywords))
    print(f"wrote {args.output}")
    if args.oiii:
        save_fits(args.oiii, Frame((oiii + sky[2]).astype(np.float32), header))
        print(f"wrote {args.oiii}")
    if args.hoo:
        scale = ha_noise / max(oiii_noise, 1e-12)
        bicolour = np.stack([ha, oiii * scale, oiii * scale], axis=-1)
        weight = starlight[..., None]
        bicolour = bicolour * (1 - weight) + balanced * weight
        hoo = (bicolour / gains + sky).astype(np.float32)
        hoo[~covered] = 0
        save_fits(args.hoo, Frame(hoo, header | keywords))
        print(f"wrote {args.hoo}")


if __name__ == "__main__":
    main(sys.argv[1:])
