"""satenhance-acquire command line (System 1)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import typer
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from satenhance_common.logging import setup_logging

from .dates import DEFAULT_DAYS, parse_date, resolve_window
from .pipeline import AcquireParams, acquire
from .prompts import is_interactive
from .providers import get_provider

log = logging.getLogger("satenhance")
app = typer.Typer(add_completion=False, help="SatEnhance System 1: download Sentinel-2 data.")


@app.command()
def run(
    start: str | None = typer.Option(None, "--start", help="Start date YYYY-MM-DD (default: end minus --days)"),
    end: str | None = typer.Option(None, "--end", help="End date YYYY-MM-DD (default: today, UTC)"),
    days: int = typer.Option(DEFAULT_DAYS, "--days", help="Length of the default window when --start/--end are omitted"),
    aoi_file: Path | None = typer.Option(None, "--aoi-file", help="Vector file (geojson, shp/zip, kml, kmz, gpkg, ...)"),
    aoi_text: str | None = typer.Option(None, "--aoi-text", help='Place name, e.g. "Perth City, Western Australia"'),
    max_cloud: float = typer.Option(20.0, "--max-cloud", help="Max AOI cloud cover %"),
    sensor: str = typer.Option("rgb", "--sensor", help="rgb | multispectral"),
    out: Path = typer.Option(Path("/data/rawdata"), "--out"),
    cache: Path = typer.Option(Path("/data/cache"), "--cache"),
    max_area_km2: float = typer.Option(
        float(os.environ.get("SATENHANCE_MAX_AREA_KM2", 100)), "--max-area-km2"),
    min_coverage: float = typer.Option(95.0, "--min-coverage", help="Min AOI coverage % by the scene"),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Never prompt"),
    yes: bool = typer.Option(False, "--yes", help="Accept the top geocode match"),
    log_json: bool = typer.Option(False, "--log-json"),
) -> None:
    setup_logging(log_json)
    try:
        d_start, d_end, defaulted = resolve_window(
            parse_date(start) if start else None, parse_date(end) if end else None, days
        )
        log.info(
            "Searching %s -> %s%s", d_start, d_end,
            f" (default window: {days} days)" if defaulted else "",
        )
        params = AcquireParams(
            start=d_start, end=d_end, max_cloud=max_cloud, sensor=sensor,
            out_dir=out, cache_dir=cache, aoi_file=aoi_file, aoi_text=aoi_text,
            max_area_km2=max_area_km2, min_coverage=min_coverage, yes=yes,
            interactive=is_interactive(non_interactive),
        )
        run_dir = acquire(params, get_provider())
    except SatEnhanceError as e:
        log.error("%s", e.message)
        raise typer.Exit(int(e.code)) from e
    except (KeyboardInterrupt, EOFError):
        log.error("Aborted")
        raise typer.Exit(int(ExitCode.UNEXPECTED)) from None
    except Exception:  # noqa: BLE001
        log.exception("Unexpected error")
        raise typer.Exit(int(ExitCode.UNEXPECTED)) from None
    print(f"Rawdata written to {run_dir}", file=sys.stderr)
    print(f"SATENHANCE_RUN_ID={run_dir.name}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
