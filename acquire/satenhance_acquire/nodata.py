"""No-data diagnosis, suggestions, interactive retry menu and report (PRD section 8.4)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from shapely.geometry.base import BaseGeometry

from .dates import today_utc
from .prompts import ask, say
from .providers.base import Provider
from .select import Assessment

REPORT_NAME = "no_data_report.json"


@dataclass
class Diagnosis:
    query: dict
    searched_candidates: int
    nearest: list[dict] = field(default_factory=list)
    suggested_max_cloud: float | None = None
    rejected_for_cloud: int = 0
    rejected_for_coverage: int = 0
    best_coverage: float | None = None  # % of the AOI the best-covering pass supplied
    suggested_min_coverage: float | None = None
    suggested_start: str | None = None
    suggested_end: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _d(iso: str) -> date:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).date()


def _distance_days(d: date, start: date, end: date) -> int:
    if d < start:
        return (start - d).days
    if d > end:
        return (d - end).days
    return 0


def diagnose(
    provider: Provider, aoi: BaseGeometry, start: date, end: date, max_cloud: float,
    assessments: list[Assessment], min_coverage: float = 95.0, today: date | None = None,
) -> Diagnosis:
    today = today or today_utc()
    pad = max(30, (end - start).days)
    wide_start, wide_end = start - timedelta(days=pad), min(end + timedelta(days=pad), today)
    wide = provider.search(aoi, wide_start, wide_end, 100.0, limit=20)
    rows = sorted(
        wide, key=lambda c: (_distance_days(_d(c.datetime), start, end), c.tile_cloud or 100.0)
    )
    nearest = [
        {"id": c.id, "date": _d(c.datetime).isoformat(), "tile_cloud": c.tile_cloud,
         "in_range": _distance_days(_d(c.datetime), start, end) == 0}
        for c in rows[:5]
    ]
    diag = Diagnosis(
        query={"start": start.isoformat(), "end": end.isoformat(), "max_cloud": max_cloud},
        searched_candidates=len(wide), nearest=nearest,
    )

    cloud_rejects = [a for a in assessments if a.reason_kind == "cloud"]
    coverage_rejects = [a for a in assessments if a.reason_kind == "coverage"]
    diag.rejected_for_cloud, diag.rejected_for_coverage = len(cloud_rejects), len(coverage_rejects)

    # Cloud suggestion: best in-range evidence (AOI-local if assessed, else tile-level).
    evidence = [a.aoi_cloud_fraction for a in cloud_rejects if a.aoi_cloud_fraction is not None]
    if not evidence and not coverage_rejects:
        evidence = [c.tile_cloud for c in rows
                    if c.tile_cloud is not None and _distance_days(_d(c.datetime), start, end) == 0]
    if evidence and min(evidence) > max_cloud:
        diag.suggested_max_cloud = float(min(100, math.ceil(min(evidence)) + 2))

    # Coverage suggestion: the AOI straddles a tile/swath edge that even mosaicking can't fill.
    covs = [a.aoi_coverage for a in coverage_rejects if a.aoi_coverage is not None]
    if covs:
        diag.best_coverage = max(covs)
        if 1 <= diag.best_coverage < min_coverage:
            diag.suggested_min_coverage = float(math.floor(diag.best_coverage))

    # Date suggestion: nearest out-of-range scene that already meets the cloud limit.
    ok_outside = [c for c in rows if _distance_days(_d(c.datetime), start, end) > 0
                  and c.tile_cloud is not None and c.tile_cloud <= max_cloud]
    if ok_outside:
        d = _d(ok_outside[0].datetime)
        diag.suggested_start = min(start, d).isoformat()
        diag.suggested_end = max(end, d).isoformat()
    return diag


def write_report(diag: Diagnosis, run_dir: Path, message: str) -> Path:
    path = Path(run_dir) / REPORT_NAME
    path.write_text(json.dumps({"message": message, **diag.to_dict()}, indent=2))
    return path


def explain(diag: Diagnosis, aoi_label: str) -> str:
    q = diag.query
    base = (
        f"No Sentinel-2 scene found for {aoi_label} between {q['start']} and {q['end']} "
        f"with cloud cover <= {q['max_cloud']:g}%."
    )
    if diag.rejected_for_coverage and not diag.rejected_for_cloud:
        return (
            f"{base} The problem is coverage, not cloud: the best pass supplies only "
            f"{diag.best_coverage:.0f}% of the AOI, even after joining tiles from the same pass "
            "(the AOI probably sits on the edge of the satellite's swath)."
        )
    if diag.rejected_for_coverage:
        base += f" {diag.rejected_for_coverage} pass(es) were also rejected for partial coverage."
    return base


def retry_menu(
    diag: Diagnosis, start: date, end: date, max_cloud: float, min_coverage: float,
    aoi_label: str, input_fn: Callable[[str], str] | None = None, today: date | None = None,
) -> tuple[float, date, date, float]:
    """Ask the user how to relax the query. Returns new (max_cloud, start, end, min_coverage)."""
    today = today or today_utc()
    say(explain(diag, aoi_label))
    if diag.nearest:
        say("Closest candidates:")
        for n in diag.nearest:
            say(f"  {n['date']}  tile cloud {n['tile_cloud']}%{'  (in range)' if n['in_range'] else ''}")
    new_cloud = diag.suggested_max_cloud or float(min(100, max_cloud + 10))
    if diag.suggested_start:
        new_start, new_end = date.fromisoformat(diag.suggested_start), date.fromisoformat(diag.suggested_end)
    else:
        pad = timedelta(days=max(30, (end - start).days))
        new_start, new_end = start - pad, min(end + pad, today)
    say("What would you like to do?")
    say(f"  [1] Raise max cloud cover (suggest {new_cloud:g}%)")
    say(f"  [2] Widen date range (suggest {new_start} -> {new_end})")
    say("  [3] Both")
    say("  [4] Enter values manually")
    say("  [5] Quit")
    if diag.suggested_min_coverage is not None:
        say(f"  [6] Accept partial coverage (lower --min-coverage to {diag.suggested_min_coverage:g}%)")
    while True:
        choice = ask("Choice [1-6]: " if diag.suggested_min_coverage is not None else "Choice [1-5]: ",
                     input_fn)
        if choice == "1":
            return new_cloud, start, end, min_coverage
        if choice == "2":
            return max_cloud, new_start, new_end, min_coverage
        if choice == "3":
            return new_cloud, new_start, new_end, min_coverage
        if choice == "6" and diag.suggested_min_coverage is not None:
            return max_cloud, start, end, diag.suggested_min_coverage
        if choice == "4":
            try:
                c = float(ask(f"Max cloud cover % [{max_cloud:g}]: ", input_fn) or max_cloud)
                s = date.fromisoformat(ask(f"Start date [{start}]: ", input_fn) or start.isoformat())
                e = date.fromisoformat(ask(f"End date [{end}]: ", input_fn) or end.isoformat())
                mc = float(ask(f"Min AOI coverage % [{min_coverage:g}]: ", input_fn) or min_coverage)
            except ValueError as err:
                say(f"Invalid value: {err}")
                continue
            if not (0 <= c <= 100) or not (0 <= mc <= 100) or s > e or e > today:
                say("Need 0<=cloud<=100, 0<=coverage<=100 and start<=end<=today")
                continue
            return c, s, e, mc
        if choice == "5":
            raise SatEnhanceError(ExitCode.NO_DATA, "No data found and user chose to quit")
        say("Please enter one of the listed numbers")
