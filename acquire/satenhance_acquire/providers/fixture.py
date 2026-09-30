"""Offline provider serving synthetic scenes from a local directory.

Used by tests and the container smoke test (SATENHANCE_PROVIDER=fixture). It performs
no network access and ignores geometry when searching.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry.base import BaseGeometry

from .base import Candidate

INDEX = "scenes.json"
ALL_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "SCL"]


def build_fixture(
    root: Path,
    scenes: list[dict],
    *,
    origin: tuple[float, float] = (400000.0, 6470000.0),
    crs: str = "EPSG:32750",
    size_10m: int = 400,
    seed: int = 0,
) -> Path:
    """Create scene rasters + index.

    Each scene dict: id, datetime, tile_cloud; optional baseline, scl_cloud_frac (fraction of
    SCL columns flagged cloud, counted from `scl_cloud_side`), origin / crs / size_10m to place a
    tile elsewhere (multi-tile scenarios), noise (default 30; 0 gives values that depend only on
    world coordinates, so tiles agree exactly where they overlap), and platform / relative_orbit /
    tile_id to make tiles groupable into one pass.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    index = []
    for sc in scenes:
        o = tuple(sc.get("origin", origin))
        c = sc.get("crs", crs)
        n10 = sc.get("size_10m", size_10m)
        noise = sc.get("noise", 30)
        bands = {}
        for bi, b in enumerate(ALL_BANDS):
            res = 10 if b in ("B02", "B03", "B04", "B08") else 20
            n = n10 * 10 // res
            t = from_origin(o[0], o[1], res, res)
            if b == "SCL":
                arr = np.full((n, n), 4, dtype="uint8")  # vegetation
                k = int(sc.get("scl_cloud_frac", 0.0) * n)
                if k:
                    if sc.get("scl_cloud_side", "left") == "left":
                        arr[:, :k] = 9
                    else:
                        arr[:, n - k:] = 9
                dtype, nodata = "uint8", 0
            else:
                # value depends on the world pixel index, so overlapping tiles agree
                cols = int(round(o[0] / res)) + np.arange(n)
                rows = int(round(o[1] / res)) - np.arange(n)
                base = 1000 + ((cols[None, :] * 7 + rows[:, None] * 13 + bi * 101) % 4000)
                arr = (base + (rng.normal(0, noise, (n, n)) if noise else 0)).clip(1, 10000)
                arr = arr.astype("uint16")
                dtype, nodata = "uint16", 0
            path = root / f"{sc['id']}_{b}.tif"
            with rasterio.open(
                path, "w", driver="GTiff", height=n, width=n, count=1, dtype=dtype,
                crs=c, transform=t, nodata=nodata,
            ) as dst:
                dst.write(arr, 1)
            bands[b] = str(path)
        index.append({
            **{k: sc[k] for k in ("id", "datetime", "tile_cloud")},
            "baseline": sc.get("baseline", "05.11"), "bands": bands,
            **{k: sc[k] for k in ("platform", "relative_orbit", "tile_id") if k in sc},
        })
    (root / INDEX).write_text(json.dumps(index, indent=2))
    return root


class FixtureProvider:
    name = "fixture"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def check_auth(self) -> None:
        return None

    def rasterio_env(self) -> rasterio.Env:
        return rasterio.Env()

    def href(self, candidate: Candidate, band: str) -> str:
        return candidate.assets[band]

    def search(
        self, aoi: BaseGeometry, start: date, end: date, max_tile_cloud: float, limit: int = 50
    ) -> list[Candidate]:
        out = []
        for sc in json.loads((self.root / INDEX).read_text()):
            d = datetime.fromisoformat(sc["datetime"].replace("Z", "+00:00")).date()
            if start <= d <= end and sc["tile_cloud"] <= max_tile_cloud:
                out.append(Candidate(
                    id=sc["id"], datetime=sc["datetime"], tile_cloud=sc["tile_cloud"],
                    processing_baseline=sc.get("baseline"), assets=sc["bands"],
                    platform=sc.get("platform"),
                    relative_orbit=None if sc.get("relative_orbit") is None
                    else str(sc["relative_orbit"]),
                    tile_id=sc.get("tile_id"),
                ))
        return out[:limit]


def fixture_aoi(
    *,
    origin: tuple[float, float] = (400000.0, 6470000.0),
    crs: str = "EPSG:32750",
    size_10m: int = 400,
    inset: float = 0.25,
):
    """WGS84 polygon covering the central part of a fixture scene (in-bounds AOI)."""
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform

    span = size_10m * 10
    x0, y1 = origin
    x0, x1 = x0 + span * inset, x0 + span * (1 - inset)
    y0, y1 = y1 - span * (1 - inset), y1 - span * inset
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    return transform(tr, box(x0, y0, x1, y1))
