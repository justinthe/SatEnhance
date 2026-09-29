"""Windowed band download + manifest creation (PRD section 8.3)."""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import rasterio
from rasterio.windows import bounds as win_bounds
from rasterio.windows import transform as win_transform
from satenhance_common import manifest as mf
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from shapely.geometry.base import BaseGeometry

from .providers.base import Candidate, Provider
from .raster import aoi_in_crs, snapped_window
from .select import Assessment
from .sizing import geodesic_area_km2

log = logging.getLogger(__name__)

BAND_SETS = {
    "rgb": ["B02", "B03", "B04", "B08"],
    "multispectral": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
}
REFERENCE_BAND = "B04"  # 10 m grid that all other windows derive from
# Baseline 04.00 (25 Jan 2022) added a +1000 DN offset to L2A reflectance. [VERIFY]
BASELINE_OFFSET_DATE = datetime(2022, 1, 25, tzinfo=UTC)


def reflectance_offset(baseline: str | None, acquired_iso: str) -> float:
    if baseline:
        nums = re.findall(r"\d+", baseline)
        if nums:
            major = int(nums[0])
            return -0.1 if major >= 4 else 0.0
    acquired = datetime.fromisoformat(acquired_iso.replace("Z", "+00:00"))
    return -0.1 if acquired >= BASELINE_OFFSET_DATE else 0.0


def _write_window(provider: Provider, cand: Candidate, band: str, bounds, dest: Path,
                  even: bool = False) -> tuple[int, int, float]:
    """Read the window covering `bounds` from `band`, write dest atomically. Returns
    (width, height, res_m)."""
    with rasterio.Env(**provider.gdal_env()), rasterio.open(provider.href(cand, band)) as src:
        win = snapped_window(src, bounds, even=even)
        if win.width == 0 or win.height == 0:
            raise SatEnhanceError(
                ExitCode.NO_DATA, f"AOI does not overlap scene {cand.id} for band {band}"
            )
        res = abs(src.transform.a)
        if dest.exists():
            with rasterio.open(dest) as d:
                if (d.width, d.height) == (int(win.width), int(win.height)):
                    log.info("Skipping %s (already downloaded)", dest.name)
                    return d.width, d.height, res
        data = src.read(window=win)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff", width=int(win.width), height=int(win.height),
            transform=win_transform(win, src.transform), tiled=True,
            blockxsize=256, blockysize=256, compress="deflate",
            predictor=2 if data.dtype.kind in "ui" else 1,
        )
        part = dest.with_name(dest.name + ".part")
        with rasterio.open(part, "w", **profile) as out:
            out.write(data)
            if src.descriptions and src.descriptions[0]:
                out.set_band_description(1, src.descriptions[0])
        os.replace(part, dest)
        return int(win.width), int(win.height), res


def download_scene(
    provider: Provider, cand: Candidate, aoi: BaseGeometry, sensor: str, scene_dir: Path
) -> dict[str, mf.BandInfo]:
    scene_dir.mkdir(parents=True, exist_ok=True)
    offset = reflectance_offset(cand.processing_baseline, cand.datetime)
    try:
        with rasterio.Env(**provider.gdal_env()), rasterio.open(
            provider.href(cand, REFERENCE_BAND)
        ) as ref:
            geom = aoi_in_crs(aoi, ref.crs)
            win = snapped_window(ref, geom.bounds, even=True)
            if win.width == 0 or win.height == 0:
                raise SatEnhanceError(ExitCode.NO_DATA, f"AOI does not overlap scene {cand.id}")
            bounds = win_bounds(win, ref.transform)
    except rasterio.errors.RasterioIOError as e:
        raise SatEnhanceError(ExitCode.NETWORK_FAILURE, f"Could not open {cand.id}: {e}") from e

    infos: dict[str, mf.BandInfo] = {}
    for band in [*BAND_SETS[sensor], "SCL"]:
        dest = scene_dir / f"{band}.tif"
        log.info("Downloading %s %s", cand.id, band)
        try:
            _, _, res = _write_window(provider, cand, band, bounds, dest, even=False)
        except rasterio.errors.RasterioIOError as e:
            raise SatEnhanceError(
                ExitCode.NETWORK_FAILURE, f"Download failed for {cand.id} {band}: {e}"
            ) from e
        is_scl = band == "SCL"
        infos[band] = mf.BandInfo(
            file=f"{scene_dir.name}/{dest.name}", res_m=res,
            scale=1.0 if is_scl else 0.0001, offset=0.0 if is_scl else offset,
        )
    return infos


def build_manifest(
    *, run_id: str, aoi_source: str, aoi: BaseGeometry, start: str, end: str,
    max_cloud: float, sensor: str, cand: Candidate, best: Assessment,
    alternates: list[Assessment], bands: dict[str, mf.BandInfo],
) -> mf.Manifest:
    return mf.Manifest(
        run_id=run_id,
        created_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        query=mf.Query(aoi_source=aoi_source, start=start, end=end, max_cloud=max_cloud,
                       sensor=sensor),
        aoi=mf.Aoi(bbox=[round(v, 6) for v in aoi.bounds],
                   area_km2=round(geodesic_area_km2(aoi), 3)),
        scene=mf.Scene(
            id=cand.id, datetime=cand.datetime, tile_cloud_cover=cand.tile_cloud,
            aoi_cloud_fraction=best.aoi_cloud_fraction, aoi_coverage=best.aoi_coverage,
            processing_baseline=cand.processing_baseline, bands=bands,
        ),
        alternates=[
            mf.Alternate(id=a.id, datetime=a.datetime, aoi_cloud_fraction=a.aoi_cloud_fraction,
                         reason=a.reason or a.status)
            for a in alternates
        ],
    )
