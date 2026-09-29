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
