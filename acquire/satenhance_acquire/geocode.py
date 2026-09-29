"""Place-name -> AOI via Nominatim (PRD section 7.3)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

from .prompts import ask, say
from .sizing import geodesic_area_km2

log = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
DEFAULT_UA = "SatEnhance/0.1 (set NOMINATIM_USER_AGENT with a contact)"
MIN_INTERVAL_S = 1.0  # Nominatim usage policy: at most ~1 request/second [VERIFY]

_last_request = 0.0


@dataclass
class GeocodeCandidate:
    display_name: str
    kind: str
    bbox: tuple[float, float, float, float]  # minx, miny, maxx, maxy
    geometry: BaseGeometry
    has_boundary: bool

    @property
    def area_km2(self) -> float:
        return geodesic_area_km2(self.geometry)


def _cache_path(cache_dir: Path, query: str) -> Path:
    return cache_dir / "geocode" / (hashlib.sha1(query.strip().lower().encode()).hexdigest() + ".json")


def _fetch(query: str, user_agent: str, session: requests.Session, limit: int) -> list[dict]:
    global _last_request
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    try:
        resp = session.get(
            NOMINATIM_URL,
            params={"q": query, "format": "jsonv2", "polygon_geojson": 1, "limit": limit},
            headers={"User-Agent": user_agent},
            timeout=30,
        )
        _last_request = time.monotonic()
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise SatEnhanceError(ExitCode.NETWORK_FAILURE, f"Geocoding request failed: {e}") from e


def _to_candidate(item: dict) -> GeocodeCandidate:
    south, north, west, east = (float(v) for v in item["boundingbox"])
    bbox = (west, south, east, north)
    gj = item.get("geojson")
    geom, has_boundary = None, False
    if gj and gj.get("type") in ("Polygon", "MultiPolygon"):
        geom, has_boundary = shape(gj), True
    if geom is None:
        geom = box(*bbox)
    return GeocodeCandidate(
        display_name=item.get("display_name", "?"),
        kind=f"{item.get('category', '')}/{item.get('type', '')}".strip("/"),
        bbox=bbox,
        geometry=geom,
        has_boundary=has_boundary,
    )


def search(
    query: str,
    *,
    cache_dir: Path,
    user_agent: str | None = None,
    session: requests.Session | None = None,
    limit: int = 5,
) -> list[GeocodeCandidate]:
    user_agent = user_agent or os.environ.get("NOMINATIM_USER_AGENT") or DEFAULT_UA
    cache = _cache_path(Path(cache_dir), query)
    if cache.exists():
        raw = json.loads(cache.read_text())
    else:
        raw = _fetch(query, user_agent, session or requests.Session(), limit)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(raw))
    if not raw:
        raise SatEnhanceError(ExitCode.INVALID_INPUT, f"No place found for '{query}'")
    return [_to_candidate(i) for i in raw]


def choose(
    candidates: list[GeocodeCandidate],
    *,
    yes: bool,
    interactive: bool,
    input_fn: Callable[[str], str] | None = None,
) -> GeocodeCandidate:
    if yes:
        return candidates[0]
    if not interactive:
        raise SatEnhanceError(
            ExitCode.GEOCODE_NEEDS_CONFIRMATION,
            f"Geocode match needs confirmation (top result: {candidates[0].display_name}). "
            "Re-run with --yes to accept the top result.",
        )
    for i, c in enumerate(candidates, 1):
        say(f"Match {i}/{len(candidates)}: {c.display_name} [{c.kind}]")
        say(
            f"  bbox (W,S,E,N): {c.bbox[0]:.4f}, {c.bbox[1]:.4f}, {c.bbox[2]:.4f}, {c.bbox[3]:.4f}"
            f"  area ~{c.area_km2:.1f} km2  ({'boundary' if c.has_boundary else 'bbox only'})"
        )
        while True:
            ans = ask("Use this? [Y/n/next] ", input_fn).lower()
            if ans in ("", "y", "yes"):
                return c
            if ans in ("n", "no"):
                raise SatEnhanceError(ExitCode.INVALID_INPUT, "Geocode match declined")
            if ans in ("next", "x"):
                break
    raise SatEnhanceError(ExitCode.INVALID_INPUT, "No more geocode candidates")
