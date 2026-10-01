"""Extract the reference for background extraction from the M27 PixInsight project.

The project in $STARKILLER_DATA/2018-09-14/m27 records a
DynamicBackgroundExtraction: its 74 samples in the project file, and the
image before and after it in the project's data directory. This writes the
samples and the background model PixInsight subtracted, thinned to every
eighth pixel, to tests/data/m27.

    docker compose run --rm starkiller python tools/make_m27_fixtures.py
"""

import os
import re
from pathlib import Path

import numpy as np

from starkiller.pixinsight_project import read_project_image

SESSION = Path(os.environ["STARKILLER_DATA"]) / "2018-09-14" / "m27"
OUTPUT = Path(__file__).parent.parent / "tests" / "data" / "m27"
# The image DBE was applied to, and the result, among the project's images.
BEFORE, AFTER = "ZZF3PSWZT-000001", "ZZF3PSWZT-000004"
STEP = 8


def samples(project: str) -> np.ndarray:
    """The DBE samples, one row each.

    Columns are x and y as fractions of the width and height, then the value
    and the weight for each of the three channels.
    """
    start = project.index('<instance class="DynamicBackgroundExtraction"')
    instance = project[start : project.index("</instance>", start)]
    table = re.search(r'<table id="data" rows="\d+">(.*?)</table>', instance, re.DOTALL)
    assert table is not None
    columns = ("x", "y", "z0", "w0", "z1", "w1", "z2", "w2")
    rows = []
    for row in re.findall(r"<tr>(.*?)</tr>", table.group(1), re.DOTALL):
        cells = dict(re.findall(r'<td id="(\w+)" value="([^"]*)"', row))
        rows.append([float(cells[column]) for column in columns])
    return np.array(rows)


def main() -> None:
    before = read_project_image(SESSION / "m27.data" / BEFORE)
    after = read_project_image(SESSION / "m27.data" / AFTER)
    model = before.astype(np.float64) - after
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "dbe.npz",
        samples=samples((SESSION / "m27.xosm").read_text(errors="replace")),
        shape=np.array(before.shape[:2]),
        model=model[::STEP, ::STEP].astype(np.float32),
    )
    print(f"M27 DBE reference: image {before.shape}, model range {np.ptp(model):.2e}")


if __name__ == "__main__":
    main()
