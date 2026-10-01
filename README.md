# quietsky

Astronomical image calibration, stacking and processing from the command
line: the PixInsight workflow without the GUI.

## Status

| Stage | State |
|---|---|
| Read camera raw, FITS, XISF | done |
| Integrate bias and dark frames (Winsorized sigma clipping) | done, matches PixInsight exactly |
| Integrate flat frames (brightness matched) | done, bit-identical to PixInsight on all but 8 of 12 million pixels |
| Location, scale and noise estimators | done, match PixInsight's recorded values |
| Calibrate with master bias, scaled master dark and master flat | done; arithmetic matches, the automatic dark scale is about 6% above PixInsight's |
| Auto-stretched previews | done |
| One command for a whole session (`preprocess`) | done |
| Cosmetic correction | partly: PixInsight's replacement value is matched; hot pixels are found from the master dark, not by PixInsight's automatic rule |
| Debayer (VNG) | done, identical to PixInsight at all but about 30 pixels of a frame |
| Star detection | done; our own detector, positions good to a tenth of a pixel on synthetic fields |
| Star matching and transformation | done; agrees with PixInsight's alignment to about 0.1 px |
| Resampling (Lanczos-3 with clamping) | done; reproduces PixInsight's registered pixels to float precision once given its transformation |
| Integrate registered lights (normalised, noise-weighted) | done; identical to PixInsight on the parts of the frame tested, apart from a rare difference of one unit in the last place |
| Background extraction | done; our surface reproduces PixInsight's DBE model from its samples, and the automatic model is within about 5% of it |

Open questions are tracked as issues in the repository.

## Running

Everything runs in Docker. A whole session in one go:

    make build
    docker compose run --rm quietsky quietsky preprocess --bias BIAS_DIR --dark DARK_DIR --flat FLAT_DIR --light LIGHT_DIR -o out/session

That writes the four master frames and a stretched preview of the master
light to `out/session/master`, and the aligned lights to
`out/session/registered`. Lights are aligned to the first one. Hot pixels
are not corrected. `--bias`, `--dark` and `--flat` may each be left out; the
corrections that cannot be made are skipped, and without bias frames the
dark is subtracted unscaled. A master made elsewhere goes in as
`--master-bias`, `--master-dark` or `--master-flat` and is used as it is,
so a master flat that is already calibrated is not calibrated again.

For the M13 session in the archive, at the camera's full 14-bit depth (about
20 minutes for its 103 frames):

    docker compose run --rm quietsky quietsky preprocess --bias /data/2018-09-12/m13/offset --dark /data/2018-09-12/m13/dark --flat /data/2018-09-12/m13/flat --light /data/2018-09-12/m13/light -o out/m13

Then take the sky gradient out of the master light:

    docker compose run --rm quietsky quietsky background out/session/master/light.fits -o out/session/master/light_bg.fits

That fits a spline through samples of the sky, which is right when most of
the frame is sky. If the target fills the frame, add `--model plane`: the
spline would follow the target and remove it. The command warns when the
samples look like more than sky, but it cannot tell a featureless glow from
a gradient, and a strong gradient with a slightly non-quadratic shape can
set it off (the blue M81 master does), so this is yours to judge.

The preprocessing steps are also available one at a time:

    docker compose run --rm quietsky quietsky info FRAME...
    docker compose run --rm quietsky quietsky stack BIAS... -o master_bias.fits
    docker compose run --rm quietsky quietsky calibrate FLAT... --bias master_bias.fits --dark master_dark.fits -o flats/
    docker compose run --rm quietsky quietsky stack --flat flats/*.fits -o master_flat.fits
    docker compose run --rm quietsky quietsky calibrate LIGHT... --bias master_bias.fits --dark master_dark.fits --flat master_flat.fits -o lights/
    docker compose run --rm quietsky quietsky debayer lights/*.fits -o rgb/
    docker compose run --rm quietsky quietsky register rgb/*.fits --reference rgb/FIRST_d.fits -o registered/
    docker compose run --rm quietsky quietsky stack --light registered/*.fits -o master_light.fits
    docker compose run --rm quietsky quietsky preview master_light.fits -o master_light.png

The container sees the project at `/app` and your frames read-only at
`/data`. That is the `data` directory of the checkout unless
`QUIETSKY_DATA_DIR` names another; `data` is ignored by git and may be a
symbolic link to wherever your pictures live.

## Checks

    make check      # lint, type check, fast tests
    make test-all   # also the slow tests and the full-frame comparisons, which read the archive

## Test data

The reference set is an M13 session from 2018-09-12 (Sony A7S II, 200mm):
43 lights, 20 darks, 20 flats and 20 bias frames, together with everything
PixInsight 1.8.5's BatchPreprocessing script made from them. It lives in the
archive at `2018-09-12/m13`.

`tests/data/m13` holds a 256x256 crop around the cluster of every raw frame,
as FITS, and the same crop of PixInsight's four master frames, of its
calibrated flats and of three sample lights after each stage, a 96x96 crop
of all 43 registered lights, and the per-frame numbers from its log. `make fixtures`
regenerates it. Bias and dark integration works one pixel stack at a time, so
the fast tests can compare our result on the crops with the crop of
PixInsight's master. The tests marked `nas` repeat the comparison on the full
frames and check the per-frame rejection counts from PixInsight's log.

The reference for background extraction is an M27 session of 2018-09-14,
whose PixInsight project (`2018-09-14/m27/m27.xosm`) records the
DynamicBackgroundExtraction that was run: its 74 samples in the project
file, and the image before and after in the project's data directory.
`tests/data/m27` holds the samples and the background PixInsight subtracted;
`tools/make_m27_fixtures.py` regenerates it.

## End to end

`tools/compare_m13_end_to_end.py` runs `preprocess` on the 103 raw frames of
the M13 session, loaded as PixInsight loaded them, and compares the master
light with PixInsight's over the part of the frame every light covers:

| Channel | Correlation | Level, ours / PixInsight | RMS difference |
|---|---|---|---|
| R | 0.99955 | 1.0002 | 0.21% of the pixel value, 0.21 of the background noise |
| G | 0.99873 | 1.0000 | 0.23% of the pixel value, 0.27 of the background noise |
| B | 0.99919 | 1.0001 | 0.23% of the pixel value, 0.28 of the background noise |

The remaining difference comes from the steps that do not match PixInsight
exactly: the dark scaling factor, the transformation of each light (about a
tenth of a pixel), and the uncorrected hot pixels. The run took 19 minutes
on 16 cores while sharing them with another run; PixInsight took 15 and a
half minutes in 2018.

## Other data

Everything above was developed against one colour camera. Two other
sessions have been run to see what that left untested. Neither has
PixInsight output, so they show that the pipeline works, not that it
matches anything.

**M81, 2019-04-16**: a cooled mono ZWO
ASI1600MM Pro at 430 mm, 16-megapixel FITS files from N.I.N.A., 20 lights of
120 s in each of L, R, G and B, 20 darks, 10 flats per filter, no bias. Each
filter went through `preprocess` on its own, the colour filters with
`--reference` set to the first registered luminance frame:

    docker compose run --rm quietsky quietsky preprocess --dark D/DARK --flat D/FLAT/L --light D/LIGHT/L -o out/m81/L
    docker compose run --rm quietsky quietsky preprocess --dark D/DARK --flat D/FLAT/R --light D/LIGHT/R --reference out/m81/L/registered/FIRST_r.fits -o out/m81/R

All 80 lights registered, with about 1,950 stars matched per luminance
frame at 0.38 px rms. The R, G and B masters sit within 0.01 px of the L
master on average. A luminance run takes about four and a half minutes.
Aligning the colour frames to the luminance reference leaves a black strip
at the top of their masters where no colour frame covered the sky, which
background extraction has to leave out of its samples.

It found one bug: the cores of bright stars, saturated in every frame, came
out black, because saturated pixels were rejected and nothing was left. A
pixel saturated in every frame that covers it now stays saturated. The M13
frames never saturate, so nothing there could have shown it.

**M31, 2025-09-26** (`2025-09-26/M31`): a Seestar S50, 2-megapixel GRBG FITS
files of 10 s, no calibration frames, with field rotation. Twelve frames
went through `debayer`, `register`, `stack --light` and `background`. All
registered. Background extraction with the default spline removed most of
the galaxy's outer disk, because the galaxy fills the frame; `--model plane`
keeps it, and the command now warns in this case.

All 730 frames of that session then went through `preprocess` with no
calibration frames: every one registered, in 34 minutes, and integrating
them took under 10 minutes at a peak of 2.7 GB of memory. Lights are
integrated from the registered files on disk, so their number is limited by
disk space, not memory: the 730 registered frames take 18 GB.

Still untested: raw files from cameras other than the Sony, long focal
lengths where stars span many pixels, mosaics and frames at different
scales, and combining L, R, G and B into one image.

## Matching PixInsight

Things that had to be found out to get an exact match, and are easy to lose:

- **Raw values.** PixInsight 1.8.5 decoded Sony raw files with dcraw, which
  drops the two low bits of the 14-bit samples. Its pixel values are
  `(raw >> 2) / 65535`. We load the full 14 bits; the tests apply the shift
  (`tests/m13.py`) only to compare with the reference.
- **Frame area.** PixInsight's frames are the 4256 columns that LibRaw calls
  the visible area, starting at column 0 of the 4288 on the sensor.
- **Sigma collapse.** On quantised data the Winsorized sigma estimate can
  shrink without converging. PixInsight iterates until 32-bit rounding stops
  it, then either rejects every value off the median or none. This is why its
  2018 run rejected 5% of all bias pixels. `quietsky.integrate` reaches the
  same decisions directly; the comment there has the rule.
- **Input order.** The per-frame entries in a master's history follow the
  order PixInsight integrated the files in, which for the flats is not the
  order of their names. The fixtures record it (`m13.master_inputs`).
- **Dark optimisation.** For a colour mosaic PixInsight bins the frame and
  the bias-subtracted master dark 2x2, keeps the dark pixels above median +
  3 x 1.4826 x MAD, and looks for the dark scaling factor that minimises the
  noise. We reproduce its threshold and pixel count exactly, but not the
  factor: the noise changes by less than 0.1% across a wide range of factors,
  in small steps, so the minimum is poorly defined. On the lights we come out
  about 6% above PixInsight (11% at most). On the flats, whose exposure is a
  fiftieth of the dark's, the factor is arbitrary in both programs. Tests
  that compare calibrated frames pass PixInsight's factor in.
- **Noise evaluation.** Our MRS estimate agrees with PixInsight's within
  0.02%; the last of its four printed digits differs by one in about one
  channel in nine.
- **32-bit samples.** PixInsight keeps each pixel stack as 32-bit floats,
  including after scaling a frame and after clipping during Winsorization.
  Rounding at the same points is what makes the master flat bit-identical;
  without it about one pixel in a million is rejected differently.
- **Flat division.** PixInsight divides by the master flat scaled to a mean
  of one, the mean being taken over the whole mosaic, not per colour.
- **Cosmetic correction.** The 2018 run used CosmeticCorrection's automatic
  detection, which changed only 20 to 60 pixels per light. A corrected hot
  pixel becomes the mean of its eight same-colour neighbours two pixels
  away, hot neighbours included; that much we reproduce exactly. Every
  corrected hot pixel exceeds the median of those neighbours by at least 3
  times the image's average absolute deviation, but so do some 10,000 other
  pixels per frame that PixInsight left alone, and no rule built from the
  5x5 neighbourhood or from the calibration masters tells the two groups
  apart. The few pixels it raised, beside saturated stars, follow no
  replacement rule we could find either.
- **Debayer.** PixInsight's VNG is dcraw's, in floating point. The gradient
  table in `debayer.py` was written from memory and then fitted to
  PixInsight's output, which showed two weights to be wrong; it has not been
  compared with dcraw's source. The outermost two pixels repeat the nearest
  pixel inside them.
- **Registration matrices.** PixInsight fits a projective transformation
  and prints it with six decimals. That rounds the two perspective terms to
  zero, yet terms of that size still move the frame corners by more than a
  pixel, so the printed matrix cannot be used to check a solution or to
  reproduce a registered frame. The registered frames themselves show what
  was applied; by our star positions they agree with our transformations to
  about a tenth of a pixel.
- **Interpolation.** Lanczos-3 with clamping threshold c = 0.3: the negative
  lobes' contribution is left alone while it is under c times the positive
  lobes', and scaled by 1 - ((r - c) / (1 - c))^2 for a ratio r above that.
- **Light integration.** Each channel is integrated on its own. A frame is
  normalised as (value - level) x (reference scale / scale) + reference
  level, with level and scale from the iterative k-sigma estimator over the
  whole registered frame, black borders included. Its weight is (reference
  noise / (noise x scale factor)) squared, where the noise is the estimate
  made after demosaicing and carried in the file, not one made on the
  registered frame. Pixels at 0 or at 0.98 and above are rejected before
  clipping and count as rejected low or high.
- **PixInsight projects.** The `.data` directory beside a project file
  holds each image, and each earlier state in its history, as a file
  starting `egamiwar`: width, height and channel count at byte 28, then per
  channel a count of blocks, each block being two 64-bit sizes (stored and
  original), a 16-byte digest and the data, zlib-compressed unless the sizes
  are equal, with the four bytes of each sample stored apart.
  `quietsky.pixinsight_project` reads them. The file saved as
  `..._integration_DBE.xisf` in the M27 session is not the plain result of
  DBE; it has further processing applied.
- **Background extraction.** DBE's sample value is the plain mean of its
  box, and its generator nudges samples off stars. Its model is a
  thin-plate spline through the samples, in coordinates scaled to the unit
  square, less its own median. Our kernel is r^2 ln r^2 with 0.05 x
  smoothing added to the diagonal, which fits PixInsight's model for its
  smoothing of 0.25 to 0.07% of the model's range; other smoothing values
  are untested. Sample placement and rejection are our own: medians of
  larger boxes on a fixed grid, dropping samples that stand above a
  quadratic fit through all of them.
