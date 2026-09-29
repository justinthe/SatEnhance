"""The System 1 -> System 2 contract (PRD section 10)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .exit_codes import ExitCode, SatEnhanceError

SCHEMA_VERSION = "1.0"
MANIFEST_NAME = "manifest.json"

Sensor = Literal["rgb", "multispectral"]


class BandInfo(BaseModel):
    file: str
    res_m: float
    scale: float = 0.0001
    offset: float = 0.0


class Query(BaseModel):
    aoi_source: str
    aoi_file: str = "aoi.geojson"
    start: str
    end: str
    max_cloud: float
    sensor: Sensor


class Aoi(BaseModel):
    crs: str = "EPSG:4326"
    bbox: list[float] = Field(min_length=4, max_length=4)
    area_km2: float


class Scene(BaseModel):
    mission: str = "sentinel-2"
    product: str = "L2A"
    id: str
    datetime: str
    tile_cloud_cover: float | None = None
    aoi_cloud_fraction: float
    aoi_coverage: float
    processing_baseline: str | None = None
    bands: dict[str, BandInfo]


class Alternate(BaseModel):
    id: str
    datetime: str
    aoi_cloud_fraction: float | None = None
    reason: str | None = None


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    run_id: str
    created_utc: str
    query: Query
    aoi: Aoi
    scene: Scene
    alternates: list[Alternate] = Field(default_factory=list)
    status: Literal["complete"] = "complete"
    tool_version: str = "0.1.0"


def json_schema() -> dict:
    return Manifest.model_json_schema()


def save(manifest: Manifest, run_dir: Path) -> Path:
    path = Path(run_dir) / MANIFEST_NAME
    path.write_text(manifest.model_dump_json(indent=2))
    return path


def load(run_dir_or_file: Path, *, code: ExitCode = ExitCode.ENHANCE_INPUT_INVALID) -> Manifest:
    p = Path(run_dir_or_file)
    if p.is_dir():
        p = p / MANIFEST_NAME
    if not p.exists():
        raise SatEnhanceError(code, f"Manifest not found: {p}")
    try:
        return Manifest.model_validate(json.loads(p.read_text()))
    except (json.JSONDecodeError, ValidationError) as e:
        raise SatEnhanceError(code, f"Invalid manifest {p}: {e}") from e
