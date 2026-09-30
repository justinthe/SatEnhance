"""Mosaic-aware band download + manifest creation (PRD section 8.3, FIX_PLAN C5).

Every band is written as one GeoTIFF on the scene grid. For a single tile that is just a windowed
copy; for an AOI spanning tiles from one pass, each pixel comes from the tile the mosaic plan
chose (the same tile for every band).
"""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from satenhance_common import manifest as mf
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from shapely.geometry.base import BaseGeometry

from .mosaic import BAND_RES, Acquisition, MosaicPlan, _open, plan_mosaic, read_into_grid
from .select import Assessment
from .sizing import geodesic_area_km2

log = logging.getLogger(__name__)

BAND_SETS = {
    "rgb": ["B02", "B03", "B04", "B08"],
    "multispectral": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
}
# Baseline 04.00 (25 Jan 2022) added a +1000 DN offset to L2A reflectance. [VERIFY]
BASELINE_OFFSET_DATE = datetime(2022, 1, 25, tzinfo=UTC)
SCALE = 0.0001


def reflectance_offset(baseline: str | None, acquired_iso: str) -> float:
    if baseline:
        nums = re.findall(r"\d+", baseline)
        if nums:
            major = int(nums[0])
            return -0.1 if major >= 4 else 0.0
    acquired = datetime.fromisoformat(acquired_iso.replace("Z", "+00:00"))
    return -0.1 if acquired >= BASELINE_OFFSET_DATE else 0.0


def _write_tif(dest: Path, arr: np.ndarray, grid, dtype: str) -> None:
    part = dest.with_name(dest.name + ".part")
    with rasterio.open(
        part, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
        dtype=dtype, crs=grid.crs, transform=grid.transform, nodata=0, tiled=True,
        blockxsize=256, blockysize=256, compress="deflate",
        predictor=2 if np.dtype(dtype).kind in "ui" else 1,
    ) as out:
        out.write(arr, 1)
    os.replace(part, dest)


def _already_done(dest: Path, grid) -> bool:
    if not dest.exists():
        return False
    with rasterio.open(dest) as d:
        return (d.width, d.height) == (grid.width, grid.height)


def download_scene(
    provider, acq: Acquisition, aoi: BaseGeometry, sensor: str, scene_dir: Path,
    plan: MosaicPlan | None = None,
) -> tuple[dict[str, mf.BandInfo], list[mf.TileInfo]]:
    """Write every band of the scene. Returns (band infos, tile infos) for the manifest."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    plan = plan or plan_mosaic(provider, aoi, acq)
    if not plan.inside.any() or plan.coverage <= 0:
        raise SatEnhanceError(ExitCode.NO_DATA, f"AOI does not overlap scene {acq.id}")

    tile_offsets = [reflectance_offset(t.processing_baseline, t.datetime) for t in plan.tiles]
    target_offset = min(tile_offsets)  # -0.1 if any tile carries the +1000 DN offset
    if len(set(tile_offsets)) > 1:
        log.info("Tiles use different reflectance offsets %s; harmonising to %s",
                 sorted(set(tile_offsets)), target_offset)

    resampled_tiles: set[str] = {s.id for s in plan.stats if s.resampled}
    infos: dict[str, mf.BandInfo] = {}
    for band in [*BAND_SETS[sensor], "SCL"]:
        res = BAND_RES[band]
        grid = plan.grid10 if res == plan.grid10.res else plan.grid20
        is_scl = band == "SCL"
        dest = scene_dir / f"{band}.tif"
        infos[band] = mf.BandInfo(
            file=f"{scene_dir.name}/{dest.name}", res_m=res,
            scale=1.0 if is_scl else SCALE, offset=0.0 if is_scl else target_offset,
        )
        if _already_done(dest, grid):
            log.info("Skipping %s (already downloaded)", dest.name)
            continue
        log.info("Downloading %s %s%s", acq.id, band,
                 f" from {len(plan.tiles)} tiles" if len(plan.tiles) > 1 else "")
        choice = plan.choice_at(res)
        out = None
        try:
            with rasterio.Env(**provider.gdal_env()):
                for i, tile in enumerate(plan.tiles):
                    m = choice == i
                    if not m.any():
                        continue  # this tile supplies no pixel: don't read it
                    with _open(provider, tile, band) as src:
                        arr, resampled = read_into_grid(
                            src, grid, Resampling.nearest if is_scl else Resampling.bilinear
                        )
                        if resampled:
                            resampled_tiles.add(tile.id)
                        if out is None:
                            out = np.zeros(grid.shape, dtype=src.dtypes[0])
                    if not is_scl and tile_offsets[i] != target_offset:
                        # same reflectance, expressed with the target offset convention
                        delta = int(round((tile_offsets[i] - target_offset) / SCALE))
                        info = np.iinfo(out.dtype)
                        shifted = np.clip(arr.astype("int64") + delta, 1, info.max)
                        arr = np.where(arr > 0, shifted, 0).astype(out.dtype)
                    out[m] = arr[m]
        except rasterio.errors.RasterioIOError as e:
            raise SatEnhanceError(
                ExitCode.NETWORK_FAILURE, f"Download failed for {acq.id} {band}: {e}"
            ) from e
        if out is None:
            raise SatEnhanceError(ExitCode.NO_DATA, f"No tile supplies pixels for {band}")
        _write_tif(dest, out, grid, str(out.dtype))

    tiles = [
        mf.TileInfo(id=s.id, crs=s.crs, resampled=s.id in resampled_tiles,
                    coverage_pct=round(s.coverage_pct, 2))
        for s in plan.stats
        if s.coverage_pct > 0  # tiles that supplied no pixel are not part of the scene
    ]
    return infos, tiles


def build_manifest(
    *, run_id: str, aoi_source: str, aoi: BaseGeometry, start: str, end: str,
    max_cloud: float, sensor: str, acq: Acquisition, best: Assessment,
    alternates: list[Assessment], bands: dict[str, mf.BandInfo], tiles: list[mf.TileInfo],
) -> mf.Manifest:
    first = min(acq.tiles, key=lambda t: t.datetime)
    return mf.Manifest(
        run_id=run_id,
        created_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        query=mf.Query(aoi_source=aoi_source, start=start, end=end, max_cloud=max_cloud,
                       sensor=sensor),
        aoi=mf.Aoi(bbox=[round(v, 6) for v in aoi.bounds],
                   area_km2=round(geodesic_area_km2(aoi), 3)),
        scene=mf.Scene(
            id=acq.id, datetime=acq.datetime, tile_cloud_cover=acq.tile_cloud,
            aoi_cloud_fraction=best.aoi_cloud_fraction, aoi_coverage=best.aoi_coverage,
            processing_baseline=first.processing_baseline, bands=bands, tiles=tiles,
            mosaic=len(tiles) > 1,
        ),
        alternates=[
            mf.Alternate(id=a.id, datetime=a.datetime, aoi_cloud_fraction=a.aoi_cloud_fraction,
                         reason=a.reason or a.status)
            for a in alternates
        ],
    )
