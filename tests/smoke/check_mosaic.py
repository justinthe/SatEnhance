"""Runs INSIDE the acquire container: the straddling AOI must have produced a 2-tile mosaic.

usage: python - <run_id>
"""

import json
import sys
from pathlib import Path

import rasterio

raw = Path("/data/rawdata") / sys.argv[1]
man = json.loads((raw / "manifest.json").read_text())
scene = man["scene"]
assert man["schema_version"] == "1.1", man["schema_version"]
assert scene["mosaic"] is True and scene["id"].startswith("MOSAIC_"), scene["id"]
assert sorted(t["id"] for t in scene["tiles"]) == ["MOSAIC_A", "MOSAIC_B"], scene["tiles"]
assert all(t["resampled"] is False for t in scene["tiles"])
assert scene["aoi_coverage"] > 99, scene["aoi_coverage"]
for band, info in scene["bands"].items():
    with rasterio.open(raw / info["file"]) as d:
        arr = d.read(1)
        assert (arr > 0).all(), f"{band} has holes at the tile seam"
print(f"OK mosaic {scene['id']} tiles={[t['id'] for t in scene['tiles']]}")
