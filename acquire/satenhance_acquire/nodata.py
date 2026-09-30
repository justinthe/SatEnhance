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
    assessments: list[Assessment], today: date | None = None,
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

    # Cloud suggestion: best in-range evidence (AOI-local if assessed, else tile-level).
    evidence = [
        a.aoi_cloud_fraction for a in assessments
        if a.aoi_cloud_fraction is not None and (a.aoi_coverage or 0) > 0
    ] or [c.tile_cloud for c in rows
          if c.tile_cloud is not None and _distance_days(_d(c.datetime), start, end) == 0]
    if evidence and min(evidence) > max_cloud:
        diag.suggested_max_cloud = float(min(100, math.ceil(min(evidence)) + 2))

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
    return (
        f"No Sentinel-2 scene found for {aoi_label} between {q['start']} and {q['end']} "
        f"with cloud cover <= {q['max_cloud']:g}%."
    )


def retry_menu(
    diag: Diagnosis, start: date, end: date, max_cloud: float, aoi_label: str,
    input_fn: Callable[[str], str] | None = None, today: date | None = None,
) -> tuple[float, date, date]:
    """Ask the user how to relax the query. Returns new (max_cloud, start, end)."""
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
    while True:
        choice = ask("Choice [1-5]: ", input_fn)
        if choice == "1":
            return new_cloud, start, end
        if choice == "2":
            return max_cloud, new_start, new_end
        if choice == "3":
            return new_cloud, new_start, new_end
        if choice == "4":
            try:
                c = float(ask(f"Max cloud cover % [{max_cloud:g}]: ", input_fn) or max_cloud)
                s = date.fromisoformat(ask(f"Start date [{start}]: ", input_fn) or start.isoformat())
                e = date.fromisoformat(ask(f"End date [{end}]: ", input_fn) or end.isoformat())
            except ValueError as err:
                say(f"Invalid value: {err}")
                continue
            if not (0 <= c <= 100) or s > e or e > today:
                say("Need 0<=cloud<=100 and start<=end<=today")
                continue
            return c, s, e
        if choice == "5":
            raise SatEnhanceError(ExitCode.NO_DATA, "No data found and user chose to quit")
        say("Please enter 1-5")
