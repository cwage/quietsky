# starkiller

Astronomical image calibration, stacking and processing from the command
line: the PixInsight workflow without the GUI.

## Status

| Stage | State |
|---|---|
| Read camera raw, FITS, XISF | done |
| Integrate bias and dark frames (Winsorized sigma clipping) | done, matches PixInsight exactly |
| Location, scale and noise estimators | done, match PixInsight's recorded values |
| Calibrate with master bias and scaled master dark | done; arithmetic matches, the automatic dark scale is about 6% above PixInsight's |
| Auto-stretched previews | done |
| Flat integration, flat division, cosmetic correction, debayer, registration, light integration | not started |

The remaining steps are tracked as issues in the repository.

## Running

Everything runs in Docker.

    make build
    docker compose run --rm starkiller starkiller info FRAME...
    docker compose run --rm starkiller starkiller stack BIAS... -o master_bias.fits
    docker compose run --rm starkiller starkiller calibrate FRAME... --bias master_bias.fits --dark master_dark.fits -o calibrated/
    docker compose run --rm starkiller starkiller preview master_bias.fits -o master_bias.png

The container sees the project at `/app` and the picture archive read-only at
`/data` (`/mnt/nas/Pictures` unless `STARKILLER_DATA_DIR` says otherwise).

## Checks

    make check      # lint, type check, fast tests
    make test-all   # also the full-frame comparisons, which read the archive

## Test data

The reference set is an M13 session from 2018-09-12 (Sony A7S II, 200mm):
43 lights, 20 darks, 20 flats and 20 bias frames, together with everything
PixInsight 1.8.5's BatchPreprocessing script made from them. It lives in the
archive at `2018-09-12/m13`.

`tests/data/m13` holds a 256x256 crop around the cluster of every raw frame,
as FITS, and the same crop of PixInsight's four master frames and of its
calibrated flats, with the per-frame numbers from its log. `make fixtures`
regenerates it. Bias and dark integration works one pixel stack at a time, so
the fast tests can compare our result on the crops with the crop of
PixInsight's master. The tests marked `nas` repeat the comparison on the full
frames and check the per-frame rejection counts from PixInsight's log.

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
  2018 run rejected 5% of all bias pixels. `starkiller.integrate` reaches the
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
