"""Geodesic area, size estimates and the AOI area cap (PRD section 7.4)."""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import Geod
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

_GEOD = Geod(ellps="WGS84")

# Number of 10 m-grid-equivalent bands fetched per sensor product.
BANDS_PER_SENSOR = {"rgb": 4, "multispectral": 10}
ENHANCE_SCALE = 4  # 10 m -> 2.5 m per axis


def geodesic_area_km2(geom: BaseGeometry) -> float:
    """Area in km2 of a lon/lat (EPSG:4326) geometry."""
    area, _ = _GEOD.geometry_area_perimeter(geom)
    return abs(area) / 1e6


def bbox_area_km2(geom: BaseGeometry) -> float:
    return geodesic_area_km2(box(*geom.bounds))


@dataclass(frozen=True)
class SizeEstimate:
    polygon_km2: float
    bbox_km2: float
    pixels_10m: int
    rawdata_mb: float
    output_mb: float


def estimate(geom: BaseGeometry, sensor: str) -> SizeEstimate:
    poly = geodesic_area_km2(geom)
    bb = bbox_area_km2(geom)
    bands = BANDS_PER_SENSOR[sensor]
    pixels = int(bb * 1e6 / 100.0)  # 10 m pixels = 100 m2
    raw = pixels * bands * 2 / 1e6
    out = pixels * ENHANCE_SCALE**2 * bands * 2 / 1e6
    return SizeEstimate(poly, bb, pixels, raw, out)


def check_size(geom: BaseGeometry, sensor: str, max_area_km2: float) -> SizeEstimate:
    """Reject AOIs whose bounding box exceeds the cap.

    The bounding box is what is downloaded and enhanced, so it (not the polygon
    area) is what the cap applies to.
    """
    est = estimate(geom, sensor)
    if est.bbox_km2 > max_area_km2:
        raise SatEnhanceError(
            ExitCode.AOI_TOO_LARGE,
            f"AOI bounding box is {est.bbox_km2:.1f} km2 (polygon {est.polygon_km2:.1f} km2), "
            f"above the {max_area_km2:g} km2 cap. Estimated raw data {est.rawdata_mb:.0f} MB, "
            f"enhanced output {est.output_mb:.0f} MB. Use a smaller AOI or raise "
            f"--max-area-km2 / SATENHANCE_MAX_AREA_KM2.",
        )
    return est
