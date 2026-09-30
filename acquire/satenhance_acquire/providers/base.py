"""Provider interface: catalogue search + band access (PRD section 8.1)."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

from shapely.geometry.base import BaseGeometry


@dataclass
class Candidate:
    id: str
    datetime: str  # ISO 8601
    tile_cloud: float | None
    processing_baseline: str | None
    assets: dict[str, str]  # band name (B04, SCL, ...) -> href
    footprint: dict | None = None
    properties: dict = field(default_factory=dict)
    # Used to group tiles from the same satellite pass into one mosaic (all optional)
    platform: str | None = None  # e.g. "sentinel-2a"
    relative_orbit: str | None = None
    tile_id: str | None = None  # e.g. MGRS tile "50HMK"


class Provider(Protocol):
    name: str

    def check_auth(self) -> None:
        """Raise SatEnhanceError(AUTH_FAILURE) if credentials are missing/invalid."""

    def search(
        self, aoi: BaseGeometry, start: date, end: date, max_tile_cloud: float, limit: int = 500
    ) -> list[Candidate]:
        """Candidates intersecting the AOI in [start, end] with tile cloud <= max_tile_cloud."""

    def href(self, candidate: Candidate, band: str) -> str:
        """A GDAL-openable path for a band."""

    def rasterio_env(self) -> AbstractContextManager:
        """A `rasterio.Env` configured to read this provider's hrefs (credentials included).
        Use as `with provider.rasterio_env(): rasterio.open(provider.href(...))`."""
