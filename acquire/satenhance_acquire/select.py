"""Scene selection using AOI-local cloud fraction from the SCL band (PRD section 8.2).

Candidates are first grouped into acquisitions (one or more tiles from the same satellite
pass, see mosaic.py). Each acquisition is scored *as a mosaic*, so an AOI split across two
tiles is judged on the joined result rather than rejected tile by tile.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from shapely.geometry.base import BaseGeometry

from .mosaic import Acquisition, MosaicPlan, group_candidates, plan_mosaic
from .providers.base import Candidate, Provider

log = logging.getLogger(__name__)

SHORTLIST = 5
MAX_ASSESS = 15
# Tile-level cloud is only a coarse prefilter; look a bit wider than the AOI limit.
SEARCH_CLOUD_MARGIN = 30.0

REASON_CLOUD = "cloud"
REASON_COVERAGE = "coverage"


@dataclass
class Assessment:
    id: str
    datetime: str
    tile_cloud: float | None
    aoi_coverage: float | None = None  # percent of AOI pixels with data
    aoi_cloud_fraction: float | None = None  # percent of covered AOI pixels that are cloud
    status: str = "not_assessed"  # ok | rejected | not_assessed
    reason: str | None = None
    reason_kind: str | None = None  # "cloud" | "coverage" when rejected
    tiles: list[str] = field(default_factory=list)
    # Not serialised: kept so the download step does not re-read the SCL bands.
    acq: Acquisition | None = field(default=None, repr=False, compare=False)
    plan: MosaicPlan | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "datetime": self.datetime, "tile_cloud": self.tile_cloud,
            "aoi_coverage": self.aoi_coverage, "aoi_cloud_fraction": self.aoi_cloud_fraction,
            "status": self.status, "reason": self.reason, "reason_kind": self.reason_kind,
            "tiles": self.tiles,
        }


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def select_best(
    provider: Provider,
    aoi: BaseGeometry,
    candidates: list[Candidate],
    *,
    max_cloud: float,
    min_coverage: float,
    shortlist: int = SHORTLIST,
    max_assess: int = MAX_ASSESS,
) -> tuple[Acquisition | None, list[Assessment], list[Assessment]]:
    """Returns (best acquisition | None, all assessments, passing assessments ranked)."""
    groups = group_candidates(candidates, aoi)
    ordered = sorted(
        groups,
        key=lambda g: (g.tile_cloud if g.tile_cloud is not None else 100.0, -_ts(g.datetime)),
    )
    assessments = [
        Assessment(g.id, g.datetime, g.tile_cloud, tiles=[t.id for t in g.tiles], acq=g)
        for g in ordered
    ]
    passing: list[Assessment] = []
    done = 0
    for a in assessments:
        if done >= max_assess or (done >= shortlist and passing):
            break
        plan = plan_mosaic(provider, aoi, a.acq)
        contributing = [st.id for st in plan.stats if st.coverage_pct > 0]
        if len(contributing) == 1 and len(a.acq.tiles) > 1:
            # Only one tile of the pass actually supplies pixels: it is an ordinary scene.
            tile = next(t for t in a.acq.tiles if t.id == contributing[0])
            a.acq = Acquisition(tile.id, [tile])
            a.id, a.tiles = tile.id, [tile.id]
        a.plan = plan
        a.aoi_coverage, a.aoi_cloud_fraction = round(plan.coverage, 2), round(plan.cloud, 2)
        done += 1
        n = len(a.acq.tiles)
        if plan.coverage < min_coverage:
            a.status, a.reason_kind = "rejected", REASON_COVERAGE
            a.reason = (
                f"AOI coverage {plan.coverage:.1f}% < {min_coverage:g}%"
                + (f" even after mosaicking {n} tiles" if n > 1 else "")
            )
        elif plan.cloud > max_cloud:
            a.status, a.reason_kind = "rejected", REASON_CLOUD
            a.reason = f"AOI cloud {plan.cloud:.1f}% > {max_cloud:g}%"
        else:
            a.status = "ok"
            if plan.coverage < 100:
                log.warning("Scene %s covers only %.1f%% of the AOI", a.id, plan.coverage)
            if n > 1:
                log.info("Scene %s is a mosaic of %d tiles", a.id, n)
            passing.append(a)
    passing.sort(key=lambda a: (round(a.aoi_cloud_fraction, 1), -_ts(a.datetime)))
    return (passing[0].acq if passing else None), assessments, passing
