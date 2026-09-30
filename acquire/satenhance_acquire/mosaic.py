"""Mosaic Sentinel-2 tiles from one satellite pass onto a common grid (FIX_PLAN C5).

An AOI that straddles a tile edge is covered by several tiles captured at (almost) the same
moment. They are grouped into an *acquisition*, joined pixel by pixel, and scored as one scene.
This is not a cloud-free composite across dates.

Rules
-----
* Grouping: same platform + relative orbit, sensing times within ``GROUP_GAP`` of each other.
  A single tile is just a group of one.
* Grid: the CRS most tiles use (ties: first tile id); the 10 m pixel lattice of that CRS,
  snapped to even offsets so the 20 m grid lines up exactly.
* Per pixel: the tile with the best quality wins (clear beats cloudy beats no data); ties go to
  the tile whose AOI is clearer overall. All bands use the same tile for a given pixel.
* Tiles on the same lattice are copied without resampling. Anything else is reprojected
  (bilinear for reflectance, nearest for the class map).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.io import DatasetReader
from rasterio.warp import reproject
from rasterio.windows import Window
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from .providers.base import Candidate, Provider
from .raster import aoi_in_crs

log = logging.getLogger(__name__)

GROUP_GAP = timedelta(minutes=10)
MAX_TILES = 4  # an AOI under a few hundred km2 touches at most 4 tiles (at a corner)
REFERENCE_BAND = "B04"  # 10 m band used to read each tile's pixel lattice
CLOUD_CLASSES = (3, 8, 9, 10)  # SCL: cloud shadow, medium/high cloud, thin cirrus
BAND_RES = {b: 10 for b in ("B02", "B03", "B04", "B08")}
BAND_RES.update({b: 20 for b in ("B05", "B06", "B07", "B8A", "B11", "B12", "SCL")})
ALIGN_TOL = 1e-3  # pixels


def _ts(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


# --------------------------------------------------------------------------- grouping
@dataclass
class Acquisition:
    """Tiles from one satellite pass. ``id`` is the tile id for a single tile."""

    id: str
    tiles: list[Candidate]

    @property
    def datetime(self) -> str:
        return min(self.tiles, key=lambda t: _ts(t.datetime)).datetime

    @property
    def tile_cloud(self) -> float | None:
        vals = [t.tile_cloud for t in self.tiles if t.tile_cloud is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def is_mosaic(self) -> bool:
        return len(self.tiles) > 1


def _intersection_area(c: Candidate, aoi: BaseGeometry | None) -> float:
    if aoi is None or not c.footprint:
        return 0.0
    try:
        return shape(c.footprint).intersection(aoi).area
    except Exception:  # noqa: BLE001  (malformed footprint: treat as unknown)
        return 0.0


def group_candidates(
    cands: list[Candidate], aoi: BaseGeometry | None = None, max_tiles: int = MAX_TILES,
) -> list[Acquisition]:
    """Group tiles captured in the same pass; drop tiles beyond ``max_tiles`` per group."""
    by_track: dict[tuple, list[Candidate]] = {}
    for c in cands:
        by_track.setdefault((c.platform, c.relative_orbit), []).append(c)

    groups: list[list[Candidate]] = []
    for track in by_track.values():
        track.sort(key=lambda c: _ts(c.datetime))
        current = [track[0]]
        for c in track[1:]:
            if _ts(c.datetime) - _ts(current[-1].datetime) <= GROUP_GAP:
                current.append(c)
            else:
                groups.append(current)
                current = [c]
        groups.append(current)

    out = []
    for g in groups:
        if len(g) > max_tiles:
            g = sorted(g, key=lambda c: -_intersection_area(c, aoi))[:max_tiles]
        out.append(Acquisition(id=_acquisition_id(g), tiles=g))
    return out


def _acquisition_id(tiles: list[Candidate]) -> str:
    if len(tiles) == 1:
        return tiles[0].id
    first = min(tiles, key=lambda t: _ts(t.datetime))
    plat = (first.platform or "S2").replace("sentinel-", "S").upper()
    orbit = f"R{first.relative_orbit}" if first.relative_orbit else "R"
    return f"MOSAIC_{plat}_{_ts(first.datetime):%Y%m%dT%H%M%S}_{orbit}_{len(tiles)}T"


# --------------------------------------------------------------------------- grid
@dataclass(frozen=True)
class Grid:
    crs: object
    transform: Affine
    width: int
    height: int

    @property
    def res(self) -> float:
        return abs(self.transform.a)

    def at(self, res: float) -> Grid:
        """Same extent at another resolution (must divide the extent exactly)."""
        factor = res / self.res
        w, h = self.width / factor, self.height / factor
        if abs(w - round(w)) > 1e-9 or abs(h - round(h)) > 1e-9:
            raise ValueError("grid extent is not a multiple of the requested resolution")
        t = Affine(res, 0, self.transform.c, 0, -res, self.transform.f)
        return Grid(self.crs, t, int(round(w)), int(round(h)))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


def _open(provider: Provider, cand: Candidate, band: str) -> DatasetReader:
    try:
        return rasterio.open(provider.href(cand, band))
    except rasterio.errors.RasterioIOError as e:
        text = str(e)
        if any(s in text for s in ("403", "401", "AccessDenied", "InvalidAccessKeyId", "Forbidden")):
            raise SatEnhanceError(
                ExitCode.AUTH_FAILURE, f"Access denied reading {cand.id} {band}: {e}"
            ) from e
        raise SatEnhanceError(
            ExitCode.NETWORK_FAILURE, f"Could not open {cand.id} {band}: {e}"
        ) from e


def plan_grid(provider: Provider, aoi: BaseGeometry, tiles: list[Candidate]) -> Grid:
    """10 m grid covering the AOI bbox, on the lattice of the most common tile CRS."""
    heads = []
    with rasterio.Env(**provider.gdal_env()):
        for t in sorted(tiles, key=lambda c: c.id):
            with _open(provider, t, REFERENCE_BAND) as src:
                heads.append((src.crs, src.transform))
    counts: dict[str, int] = {}
    for crs, _ in heads:
        counts[str(crs)] = counts.get(str(crs), 0) + 1
    best = max(counts.values())
    crs, ref = next((c, t) for c, t in heads if counts[str(c)] == best)
    res = abs(ref.a)
    minx, miny, maxx, maxy = aoi_in_crs(aoi, crs).bounds
    ox, oy = ref.c, ref.f
    col0 = math.floor(round((minx - ox) / res, 6))
    col1 = math.ceil(round((maxx - ox) / res, 6))
    row0 = math.floor(round((oy - maxy) / res, 6))
    row1 = math.ceil(round((oy - miny) / res, 6))
    col0, row0 = col0 - col0 % 2, row0 - row0 % 2  # floor to even (works for negatives)
    col1, row1 = col1 + col1 % 2, row1 + row1 % 2
    transform = Affine(res, 0, ox + col0 * res, 0, -res, oy - row0 * res)
    return Grid(crs, transform, col1 - col0, row1 - row0)


def read_into_grid(src: DatasetReader, grid: Grid, resampling: Resampling) -> tuple[np.ndarray, bool]:
    """Read ``src`` onto ``grid`` (0 = no data). Returns (array, was_resampled)."""
    aligned = (
        src.crs == grid.crs
        and abs(abs(src.transform.a) - grid.res) < 1e-9
        and abs(abs(src.transform.e) - grid.res) < 1e-9
    )
    if aligned:
        col = (grid.transform.c - src.transform.c) / grid.res
        row = (src.transform.f - grid.transform.f) / grid.res
        aligned = abs(col - round(col)) < ALIGN_TOL and abs(row - round(row)) < ALIGN_TOL
    if aligned:
        win = Window(round(col), round(row), grid.width, grid.height)
        arr = src.read(1, window=win, boundless=True, fill_value=0)
        return arr, False
    if src.crs == grid.crs:
        log.warning("Tile pixel lattice differs from the scene grid; reprojecting (unexpected)")
    dst = np.zeros(grid.shape, dtype=src.dtypes[0])
    reproject(
        source=rasterio.band(src, 1), destination=dst, dst_transform=grid.transform,
        dst_crs=grid.crs, resampling=resampling, src_nodata=0, dst_nodata=0,
    )
    return dst, True


# --------------------------------------------------------------------------- planning
@dataclass
class TileStats:
    id: str
    crs: str  # the tile's own CRS
    resampled: bool  # True if the tile had to be reprojected onto the scene grid
    coverage_pct: float  # share of the AOI where this tile supplies pixels
    own_cloud_pct: float  # cloud share over the AOI pixels this tile could supply on its own


@dataclass
class MosaicPlan:
    tiles: list[Candidate]  # in priority order (clearest first)
    grid10: Grid
    grid20: Grid
    choice20: np.ndarray  # (H20, W20) index into ``tiles`` of the tile supplying each pixel
    scl: np.ndarray  # mosaicked SCL on grid20
    inside: np.ndarray  # AOI mask on grid20
    coverage: float  # % of AOI pixels with data
    cloud: float  # % of covered AOI pixels that are cloud/shadow/cirrus
    stats: list[TileStats] = field(default_factory=list)

    def choice_at(self, res: float) -> np.ndarray:
        if res == self.grid20.res:
            return self.choice20
        k = int(round(self.grid20.res / res))
        return np.repeat(np.repeat(self.choice20, k, axis=0), k, axis=1)


def _quality(scl: np.ndarray) -> np.ndarray:
    """0 = no data, 1 = cloudy, 2 = clear."""
    q = np.full(scl.shape, 2, dtype=np.uint8)
    q[np.isin(scl, CLOUD_CLASSES)] = 1
    q[scl == 0] = 0
    return q


def plan_mosaic(provider: Provider, aoi: BaseGeometry, acq: Acquisition) -> MosaicPlan:
    """Read every tile's SCL over the AOI, decide which tile supplies each pixel, and score it."""
    grid10 = plan_grid(provider, aoi, acq.tiles)
    grid20 = grid10.at(20)
    aoi_geom = aoi_in_crs(aoi, grid20.crs)
    inside = ~geometry_mask([aoi_geom], out_shape=grid20.shape, transform=grid20.transform,
                            all_touched=True)
    n_inside = int(inside.sum())

    scls, meta = [], []
    with rasterio.Env(**provider.gdal_env()):
        for t in acq.tiles:
            with _open(provider, t, "SCL") as src:
                arr, resampled = read_into_grid(src, grid20, Resampling.nearest)
                meta.append((str(src.crs), resampled))
            scls.append(arr)

    def own_cloud(scl: np.ndarray) -> float:
        valid = inside & (scl != 0)
        return 100.0 * (valid & np.isin(scl, CLOUD_CLASSES)).sum() / max(int(valid.sum()), 1)

    # Priority: tiles whose own AOI pixels are clearest come first (they win ties per pixel).
    order = sorted(range(len(acq.tiles)), key=lambda i: (own_cloud(scls[i]), acq.tiles[i].id))
    tiles = [acq.tiles[i] for i in order]
    scls = [scls[i] for i in order]
    meta = [meta[i] for i in order]

    quality = np.stack([_quality(s) for s in scls])  # (n, H, W)
    choice = quality.argmax(axis=0).astype(np.uint8)  # first (=clearest tile) wins ties
    best_q = quality.max(axis=0)
    scl_m = np.zeros(grid20.shape, dtype=scls[0].dtype)
    for i, s in enumerate(scls):
        m = (choice == i) & (best_q > 0)
        scl_m[m] = s[m]

    covered = inside & (best_q > 0)
    coverage = 100.0 * int(covered.sum()) / n_inside if n_inside else 0.0
    cloud = (100.0 * int((covered & np.isin(scl_m, CLOUD_CLASSES)).sum()) / int(covered.sum())
             if covered.any() else 100.0)
    stats = [
        TileStats(
            id=t.id, crs=meta[i][0], resampled=meta[i][1],
            coverage_pct=100.0 * int((covered & (choice == i)).sum()) / max(n_inside, 1),
            own_cloud_pct=own_cloud(scls[i]),
        )
        for i, t in enumerate(tiles)
    ]
    return MosaicPlan(tiles, grid10, grid20, choice, scl_m, inside, coverage, cloud, stats)
