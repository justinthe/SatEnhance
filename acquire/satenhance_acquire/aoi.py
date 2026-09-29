"""Read any supported vector source into one normalised WGS84 polygon (PRD section 7)."""

from __future__ import annotations

import json
import logging
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

log = logging.getLogger(__name__)

_GEOJSON_EXT = {".geojson", ".json"}
_MULTILAYER_EXT = {".kml", ".gpkg"}


@dataclass
class AoiResult:
    geometry: BaseGeometry  # EPSG:4326, polygonal
    source: str
    feature_count: int


def _fail(msg: str) -> SatEnhanceError:
    return SatEnhanceError(ExitCode.INVALID_INPUT, msg)


def _read_layers(path: Path) -> gpd.GeoDataFrame:
    layers = [name for name, _ in pyogrio.list_layers(path)]
    frames = [gpd.read_file(path, layer=name, engine="pyogrio") for name in layers]
    frames = [f for f in frames if len(f)]
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    crs = frames[0].crs
    frames = [f.to_crs(crs) if f.crs is not None and crs is not None else f for f in frames]
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=crs)


def _read_file(path: Path) -> gpd.GeoDataFrame:
    ext = path.suffix.lower()
    try:
        if ext == ".kmz":
            with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path) as z:
                kmls = [n for n in z.namelist() if n.lower().endswith(".kml")]
                if not kmls:
                    raise _fail(f"{path.name}: KMZ contains no .kml file")
                z.extract(kmls[0], tmp)
                return _read_layers(Path(tmp) / kmls[0])
        if ext == ".zip":
            with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path) as z:
                shps = [n for n in z.namelist() if n.lower().endswith(".shp")]
                if not shps:
                    raise _fail(f"{path.name}: zip contains no .shp file")
                z.extractall(tmp)
                return gpd.read_file(Path(tmp) / shps[0], engine="pyogrio")
        if ext in _MULTILAYER_EXT:
            return _read_layers(path)
        return gpd.read_file(path, engine="pyogrio")
    except SatEnhanceError:
        raise
    except zipfile.BadZipFile as e:
        raise _fail(f"{path.name}: not a valid zip archive ({e})") from e
    except Exception as e:  # GDAL/driver errors come in many types
        raise _fail(f"Could not read AOI file {path.name}: {e}") from e


def normalise(gdf: gpd.GeoDataFrame, *, label: str, assume_wgs84: bool = False) -> AoiResult:
    """PRD 7.2: polygons only, WGS84, valid, dissolved into one geometry."""
    if gdf.crs is None:
        if assume_wgs84:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            raise _fail(f"{label}: no coordinate reference system defined; refusing to guess")
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        raise _fail(f"{label}: no geometries found")

    is_poly = gdf.geom_type.isin(["Polygon", "MultiPolygon"])
    is_collection = gdf.geom_type == "GeometryCollection"
    keep = gdf[is_poly | is_collection]
    dropped = len(gdf) - len(keep)
    if keep.empty:
        raise _fail(
            f"{label}: only points/lines found ({', '.join(sorted(set(gdf.geom_type)))}); "
            "an AOI needs polygons"
        )
    if dropped:
        log.warning("%s: ignored %d non-polygon feature(s)", label, dropped)

    geoms = keep.to_crs("EPSG:4326").geometry
    geoms = shapely.force_2d(geoms.values)
    polys: list[BaseGeometry] = []
    for g in geoms:
        g = shapely.make_valid(g)
        for part in shapely.get_parts(g):
            if part.geom_type in ("Polygon", "MultiPolygon"):
                polys.extend(shapely.get_parts(part))
    polys = [p for p in polys if not p.is_empty]
    if not polys:
        raise _fail(f"{label}: no valid polygon area after repair")
    geom = shapely.union_all(polys)
    return AoiResult(geometry=geom, source=label, feature_count=len(keep))


def load_aoi(path: str | Path) -> AoiResult:
    p = Path(path)
    if not p.exists():
        raise _fail(f"AOI file not found: {p}")
    gdf = _read_file(p)
    return normalise(gdf, label=f"file:{p.name}", assume_wgs84=p.suffix.lower() in _GEOJSON_EXT)


def from_geometry(geom: BaseGeometry, *, label: str) -> AoiResult:
    """Wrap an already-WGS84 geometry (e.g. a geocoder result) through the same rules."""
    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    return normalise(gdf, label=label)


def write_aoi(result: AoiResult, path: str | Path) -> None:
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"source": result.source, "feature_count": result.feature_count},
                "geometry": mapping(result.geometry),
            }
        ],
    }
    Path(path).write_text(json.dumps(fc))
