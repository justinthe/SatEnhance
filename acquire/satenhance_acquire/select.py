"""Scene selection using AOI-local cloud fraction from the SCL band (PRD section 8.2)."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from rasterio.windows import transform as win_transform
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from shapely.geometry.base import BaseGeometry

from .providers.base import Candidate, Provider
from .raster import aoi_in_crs

log = logging.getLogger(__name__)

# SCL classes: 3 cloud shadow, 8 cloud medium prob, 9 cloud high prob, 10 thin cirrus
CLOUD_CLASSES = (3, 8, 9, 10)
SHORTLIST = 5
MAX_ASSESS = 15
# Tile-level cloud is only a coarse prefilter; look a bit wider than the AOI limit.
SEARCH_CLOUD_MARGIN = 30.0


@dataclass
class Assessment:
    id: str
    datetime: str
    tile_cloud: float | None
    aoi_coverage: float | None = None  # percent of AOI pixels with data
    aoi_cloud_fraction: float | None = None  # percent of covered AOI pixels that are cloud
    status: str = "not_assessed"  # ok | rejected | not_assessed
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def measure_scl(provider: Provider, cand: Candidate, aoi: BaseGeometry) -> tuple[float, float]:
    """Return (coverage %, cloud %) of the AOI from the SCL band."""
    try:
        with rasterio.Env(**provider.gdal_env()), rasterio.open(provider.href(cand, "SCL")) as src:
            geom = aoi_in_crs(aoi, src.crs)
            b = geom.bounds
            win = rasterio.windows.from_bounds(*b, transform=src.transform)
            win = Window(
                int(np.floor(win.col_off)), int(np.floor(win.row_off)),
                max(int(np.ceil(win.width)), 1), max(int(np.ceil(win.height)), 1),
            )
            scl = src.read(1, window=win, boundless=True, fill_value=0)
            inside = ~geometry_mask(
                [geom], out_shape=scl.shape, transform=win_transform(win, src.transform),
                all_touched=True,
            )
    except SatEnhanceError:
        raise
    except rasterio.errors.RasterioIOError as e:
        text = str(e)
        if any(s in text for s in ("403", "401", "AccessDenied", "InvalidAccessKeyId", "Forbidden")):
            raise SatEnhanceError(ExitCode.AUTH_FAILURE, f"Access denied reading {cand.id}: {e}") from e
        raise SatEnhanceError(ExitCode.NETWORK_FAILURE, f"Could not read SCL for {cand.id}: {e}") from e
    n_aoi = int(inside.sum())
    if n_aoi == 0:
        return 0.0, 100.0
    covered = inside & (scl != 0)
    n_cov = int(covered.sum())
    coverage = 100.0 * n_cov / n_aoi
    if n_cov == 0:
        return 0.0, 100.0
    cloudy = covered & np.isin(scl, CLOUD_CLASSES)
    return coverage, 100.0 * int(cloudy.sum()) / n_cov


def select_best(
    provider: Provider,
    aoi: BaseGeometry,
    candidates: list[Candidate],
    *,
    max_cloud: float,
    min_coverage: float,
    shortlist: int = SHORTLIST,
    max_assess: int = MAX_ASSESS,
) -> tuple[Candidate | None, list[Assessment], list[Assessment]]:
    """Returns (best candidate | None, all assessments, passing assessments ranked)."""
    ordered = sorted(
        candidates,
        key=lambda c: (c.tile_cloud if c.tile_cloud is not None else 100.0, -_ts(c.datetime)),
    )
    by_id = {c.id: c for c in candidates}
    assessments = [
        Assessment(c.id, c.datetime, c.tile_cloud) for c in ordered
    ]
    passing: list[Assessment] = []
    done = 0
    for a in assessments:
        if done >= max_assess or (done >= shortlist and passing):
            break
        cand = by_id[a.id]
        cov, cloud = measure_scl(provider, cand, aoi)
        a.aoi_coverage, a.aoi_cloud_fraction = round(cov, 2), round(cloud, 2)
        done += 1
        if cov < min_coverage:
            a.status, a.reason = "rejected", f"AOI coverage {cov:.1f}% < {min_coverage:g}%"
        elif cloud > max_cloud:
            a.status, a.reason = "rejected", f"AOI cloud {cloud:.1f}% > {max_cloud:g}%"
        else:
            a.status = "ok"
            if cov < 100:
                log.warning("Scene %s covers only %.1f%% of the AOI", a.id, cov)
            passing.append(a)
    passing.sort(key=lambda a: (round(a.aoi_cloud_fraction, 1), -_ts(a.datetime)))
    best = by_id[passing[0].id] if passing else None
    return best, assessments, passing
