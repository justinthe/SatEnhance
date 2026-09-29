"""Generate a synthetic System-1 rawdata run (for tests and the container smoke test).

    python -m satenhance_enhance.synthetic --out /data/rawdata [--sensor rgb] [--size 160]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from satenhance_common import manifest as mf
from satenhance_common.paths import write_latest

BANDS_10 = ["B02", "B03", "B04", "B08"]
BANDS_20 = ["B05", "B06", "B07", "B8A", "B11", "B12"]


def make_rawdata(
    root: Path, *, sensor: str = "rgb", size: int = 160, run_id: str = "20260101T000000_synthetic",
    origin=(400000.0, 6470000.0), crs="EPSG:32750", seed: int = 0, offset: float = -0.1,
    nodata_corner: bool = False,
) -> Path:
    """size = pixels per side on the 10 m grid (must be even)."""
    assert size % 2 == 0
    root = Path(root)
    run_dir = root / run_id
    scene_dir = run_dir / "S2_SYNTH"
    scene_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = BANDS_10 + (BANDS_20 if sensor == "multispectral" else [])
    infos = {}
    for b in [*names, "SCL"]:
        res = 10 if b in BANDS_10 else 20
        n = size * 10 // res
        yy, xx = np.mgrid[0:n, 0:n].astype("float32")
        arr = (1500 + 3000 * (xx + yy) / (2 * n) + rng.normal(0, 20, (n, n))).clip(1, 10000)
        arr = arr.astype("uint16")
        if b == "SCL":
            arr = np.full((n, n), 4, dtype="uint8")
        if nodata_corner and b != "SCL":
            arr[: n // 10, : n // 10] = 0
        with rasterio.open(
            scene_dir / f"{b}.tif", "w", driver="GTiff", height=n, width=n, count=1,
            dtype=str(arr.dtype), crs=crs, transform=from_origin(origin[0], origin[1], res, res),
            nodata=0,
        ) as dst:
            dst.write(arr, 1)
        is_scl = b == "SCL"
        infos[b] = mf.BandInfo(file=f"S2_SYNTH/{b}.tif", res_m=res,
                               scale=1.0 if is_scl else 0.0001, offset=0.0 if is_scl else offset)
    # AOI polygon covering the raster, in WGS84
    from pyproj import Transformer
    from shapely.geometry import box, mapping
    from shapely.ops import transform

    span = size * 10
    poly = transform(Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform,
                     box(origin[0], origin[1] - span, origin[0] + span, origin[1]))
    (run_dir / "aoi.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(poly)}]}))
    man = mf.Manifest(
        run_id=run_id, created_utc="2026-01-01T00:00:00Z",
        query=mf.Query(aoi_source="synthetic", start="2026-01-01", end="2026-01-31",
                       max_cloud=20, sensor=sensor),
        aoi=mf.Aoi(bbox=list(poly.bounds), area_km2=(span / 1000) ** 2),
        scene=mf.Scene(id="SYNTH", datetime="2026-01-05T00:00:00Z", aoi_cloud_fraction=0.0,
                       aoi_coverage=100.0, processing_baseline="05.11", bands=infos),
    )
    mf.save(man, run_dir)
    write_latest(root, run_id)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sensor", default="rgb")
    ap.add_argument("--size", type=int, default=160)
    a = ap.parse_args()
    print(make_rawdata(a.out, sensor=a.sensor, size=a.size))


if __name__ == "__main__":
    main()
