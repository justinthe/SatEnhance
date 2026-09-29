"""Runs INSIDE the enhance container: asserts the enhanced output is geometrically sane.

usage: python - <run_id>
"""

import json
import sys
from pathlib import Path

import rasterio

run_id = sys.argv[1]
raw = Path("/data/rawdata") / run_id
out = Path("/data/output") / run_id
man = json.loads((raw / "manifest.json").read_text())
scene = man["scene"]["id"]

with rasterio.open(raw / man["scene"]["bands"]["B04"]["file"]) as src, \
        rasterio.open(next(out.glob("enhanced_*_2p5m.tif"))) as dst:
    assert dst.res == (2.5, 2.5), dst.res
    assert dst.crs == src.crs
    assert dst.bounds == src.bounds, (dst.bounds, src.bounds)
    assert (dst.height, dst.width) == (src.height * 4, src.width * 4)
    assert dst.count == 4 and dst.read().min() >= 1
for name in ("preview_before_after.png", "enhance_report.json"):
    assert (out / name).stat().st_size > 0, name
rep = json.loads((out / "enhance_report.json").read_text())
assert rep["stub_model"] is True and rep["output_pixel_size_m"] == 2.5
assert scene == "SMOKE_B", scene  # same cloud as A, more recent
print(f"OK {run_id}: {dst.width}x{dst.height} @ 2.5 m")
