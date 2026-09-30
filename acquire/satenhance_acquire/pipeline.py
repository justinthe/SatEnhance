"""System 1 orchestration: AOI -> search -> select -> download -> manifest."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from satenhance_common import manifest as mf
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from satenhance_common.paths import make_run_id, write_latest

from . import aoi as aoi_mod
from . import geocode, nodata, sizing
from .dates import parse_date, today_utc  # noqa: F401  (parse_date re-exported)
from .download import BAND_SETS, build_manifest, download_scene
from .providers.base import Provider
from .select import SEARCH_CLOUD_MARGIN, select_best

log = logging.getLogger(__name__)

SUPPORTED_SENSORS = tuple(BAND_SETS)
UNSUPPORTED_MESSAGES = {
    "sar": "Sentinel-1 (SAR) is not supported in v1; use --sensor rgb or multispectral.",
    "lidar": "LiDAR is not provided by Sentinel satellites and is not supported.",
    "hyperspectral": "Hyperspectral data is not provided by Sentinel-2 (it is multispectral). "
    "Use --sensor multispectral.",
}


@dataclass
class AcquireParams:
    start: date
    end: date
    max_cloud: float = 20.0
    sensor: str = "rgb"
    out_dir: Path = Path("/data/rawdata")
    cache_dir: Path = Path("/data/cache")
    aoi_file: Path | None = None
    aoi_text: str | None = None
    max_area_km2: float = 100.0
    min_coverage: float = 95.0
    yes: bool = False
    interactive: bool = False


def validate(p: AcquireParams, today: date | None = None) -> None:
    today = today or today_utc()
    bad = lambda m: SatEnhanceError(ExitCode.INVALID_INPUT, m)  # noqa: E731
    if p.sensor not in SUPPORTED_SENSORS:
        raise bad(UNSUPPORTED_MESSAGES.get(p.sensor, f"Unknown sensor '{p.sensor}'. "
                                           f"Choose from {', '.join(SUPPORTED_SENSORS)}."))
    if bool(p.aoi_file) == bool(p.aoi_text):
        raise bad("Provide exactly one of --aoi-file or --aoi-text")
    if p.start > p.end:
        raise bad(f"--start ({p.start}) is after --end ({p.end})")
    if p.end > today:
        raise bad(f"--end ({p.end}) is in the future")
    if not 0 <= p.max_cloud <= 100:
        raise bad("--max-cloud must be between 0 and 100")
    if not 0 <= p.min_coverage <= 100:
        raise bad("--min-coverage must be between 0 and 100")


def resolve_aoi(p: AcquireParams, input_fn: Callable[[str], str] | None = None
                ) -> tuple[aoi_mod.AoiResult, str]:
    if p.aoi_file:
        res = aoi_mod.load_aoi(p.aoi_file)
        return res, Path(p.aoi_file).stem
    cands = geocode.search(p.aoi_text, cache_dir=p.cache_dir)
    chosen = geocode.choose(cands, yes=p.yes, interactive=p.interactive, input_fn=input_fn)
    res = aoi_mod.from_geometry(chosen.geometry, label=f"text:{p.aoi_text}")
    return res, p.aoi_text


def acquire(
    p: AcquireParams, provider: Provider, input_fn: Callable[[str], str] | None = None
) -> Path:
    """Run System 1. Returns the run directory."""
    validate(p)
    provider.check_auth()
    aoi_res, label = resolve_aoi(p, input_fn)
    est = sizing.check_size(aoi_res.geometry, p.sensor, p.max_area_km2)
    log.info(
        "AOI %s: bbox %.1f km2 (polygon %.1f km2); est. raw %.0f MB, enhanced %.0f MB",
        label, est.bbox_km2, est.polygon_km2, est.rawdata_mb, est.output_mb,
    )
    run_id = make_run_id(label)
    run_dir = Path(p.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    aoi_mod.write_aoi(aoi_res, run_dir / "aoi.geojson")

    start, end, max_cloud, min_coverage = p.start, p.end, p.max_cloud, p.min_coverage
    while True:
        cands = provider.search(
            aoi_res.geometry, start, end, min(100.0, max_cloud + SEARCH_CLOUD_MARGIN)
        )
        log.info("%d candidate scene(s) in catalogue", len(cands))
        best, assessments, passing = select_best(
            provider, aoi_res.geometry, cands, max_cloud=max_cloud, min_coverage=min_coverage
        )
        (run_dir / "search_results.json").write_text(
            json.dumps([a.to_dict() for a in assessments], indent=2)
        )
        if best is not None:
            break
        diag = nodata.diagnose(
            provider, aoi_res.geometry, start, end, max_cloud, assessments, min_coverage
        )
        if not p.interactive:
            msg = nodata.explain(diag, label)
            path = nodata.write_report(diag, run_dir, msg)
            raise SatEnhanceError(ExitCode.NO_DATA, f"{msg} Report: {path}")
        max_cloud, start, end, min_coverage = nodata.retry_menu(
            diag, start, end, max_cloud, min_coverage, label, input_fn
        )

    log.info("Selected %s (AOI cloud %.1f%%, coverage %.1f%%)", best.id,
             passing[0].aoi_cloud_fraction, passing[0].aoi_coverage)
    scene_dir = run_dir / f"S2_{best.id}"
    bands, tiles = download_scene(
        provider, best, aoi_res.geometry, p.sensor, scene_dir, plan=passing[0].plan
    )
    man = build_manifest(
        run_id=run_id, aoi_source=aoi_res.source, aoi=aoi_res.geometry,
        start=start.isoformat(), end=end.isoformat(), max_cloud=max_cloud, sensor=p.sensor,
        acq=best, best=passing[0], alternates=passing[1:] + [
            a for a in assessments if a.status == "rejected"
        ], bands=bands, tiles=tiles,
    )
    mf.save(man, run_dir)
    write_latest(Path(p.out_dir), run_id)
    return run_dir
