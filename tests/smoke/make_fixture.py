"""Runs INSIDE the acquire container: writes an offline scene fixture and a matching AOI file."""

from pathlib import Path

from satenhance_acquire.aoi import from_geometry, write_aoi
from satenhance_acquire.providers.fixture import build_fixture, fixture_aoi

SCENES = [
    {"id": "SMOKE_A", "datetime": "2026-01-05T02:00:00Z", "tile_cloud": 4.0},
    {"id": "SMOKE_B", "datetime": "2026-01-15T02:00:00Z", "tile_cloud": 4.0},
    {"id": "SMOKE_CLOUDY", "datetime": "2026-01-20T02:00:00Z", "tile_cloud": 3.0,
     "scl_cloud_frac": 0.9},
]
build_fixture(Path("/data/cache/fixture"), SCENES)
Path("/data/cache/smoke").mkdir(parents=True, exist_ok=True)
write_aoi(from_geometry(fixture_aoi(), label="smoke"), "/data/cache/smoke/site.geojson")
print("fixture ready")

# --- two tiles from one pass + an AOI that straddles their seam (mosaic scenario) -------------
from pyproj import Transformer  # noqa: E402
from shapely.geometry import box  # noqa: E402
from shapely.ops import transform  # noqa: E402

WHEN = "2026-01-10T02:31:00Z"


def _tile(id_, origin):
    return {"id": id_, "datetime": WHEN, "tile_cloud": 4.0, "platform": "sentinel-2a",
            "relative_orbit": 74, "noise": 0, "origin": origin, "size_10m": 300}


build_fixture(Path("/data/cache/fixture_mosaic"),
              [_tile("MOSAIC_A", (400000.0, 6470000.0)), _tile("MOSAIC_B", (402500.0, 6470000.0))])
seam = transform(Transformer.from_crs("EPSG:32750", "EPSG:4326", always_xy=True).transform,
                 box(401500, 6468500, 403500, 6469500))
write_aoi(from_geometry(seam, label="straddle"), "/data/cache/smoke/straddle.geojson")
print("mosaic fixture ready")
