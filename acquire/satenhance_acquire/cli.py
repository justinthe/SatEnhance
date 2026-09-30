"""satenhance-acquire command line (System 1)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import typer
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from satenhance_common.logging import setup_logging
from satenhance_common.report import write_error_report

from .dates import DEFAULT_DAYS, parse_date, resolve_window
from .pipeline import AcquireParams, acquire, resolve_aoi
from .probe import probe as run_probe
from .prompts import is_interactive
from .providers import get_provider

log = logging.getLogger("satenhance")
app = typer.Typer(add_completion=False, help="SatEnhance System 1: download Sentinel-2 data.")
COMMANDS = ("run", "probe")


def _fail(e: BaseException, out: Path, stage: str | None = None) -> typer.Exit:
    """Log, write error_report.json (next to the run if it exists), and pick the exit code."""
    if isinstance(e, SatEnhanceError):
        log.error("%s", e.message)
        code = int(e.code)
    else:
        log.error("Unexpected error: %s: %s", type(e).__name__, e)
        code = int(ExitCode.UNEXPECTED)
    where = Path(e.context["run_dir"]) if isinstance(e, SatEnhanceError) and "run_dir" in e.context else Path(out)
    path = write_error_report(where, tool="satenhance-acquire", exc=e, stage=stage)
    if path:
        log.error("Details: %s", path)
    return typer.Exit(code)


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
    """Download the best Sentinel-2 scene for an AOI into the rawdata folder."""
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
    except (KeyboardInterrupt, EOFError):
        log.error("Aborted")
        raise typer.Exit(int(ExitCode.UNEXPECTED)) from None
    except Exception as e:  # noqa: BLE001  (SatEnhanceError and anything unexpected)
        raise _fail(e, out) from e
    print(f"Rawdata written to {run_dir}", file=sys.stderr)
    print(f"SATENHANCE_RUN_ID={run_dir.name}")


@app.command()
def probe(
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
    days: int = typer.Option(DEFAULT_DAYS, "--days"),
    aoi_file: Path | None = typer.Option(None, "--aoi-file"),
    aoi_text: str | None = typer.Option(None, "--aoi-text"),
    cache: Path = typer.Option(Path("/data/cache"), "--cache"),
    yes: bool = typer.Option(False, "--yes"),
    non_interactive: bool = typer.Option(False, "--non-interactive"),
    log_json: bool = typer.Option(False, "--log-json"),
) -> None:
    """Check catalogue and pixel access once (needs credentials); prints what the first scene looks like."""
    setup_logging(log_json)
    try:
        d_start, d_end, _ = resolve_window(
            parse_date(start) if start else None, parse_date(end) if end else None, days
        )
        params = AcquireParams(
            start=d_start, end=d_end, cache_dir=cache, aoi_file=aoi_file, aoi_text=aoi_text,
            yes=yes, interactive=is_interactive(non_interactive),
        )
        aoi_res, _ = resolve_aoi(params)
        run_probe(get_provider(), aoi_res.geometry, d_start, d_end)
    except (KeyboardInterrupt, EOFError):
        raise typer.Exit(int(ExitCode.UNEXPECTED)) from None
    except SatEnhanceError as e:
        print(f"PROBE FAILED: {e.message}")
        raise typer.Exit(int(e.code)) from e
    except Exception as e:  # noqa: BLE001
        print(f"PROBE FAILED: {type(e).__name__}: {e}")
        raise typer.Exit(int(ExitCode.UNEXPECTED)) from e


def main() -> None:
    args = sys.argv[1:]
    # `satenhance-acquire --start ...` is shorthand for `satenhance-acquire run --start ...`
    if not args or (args[0] not in COMMANDS and args[0] not in ("--help", "-h")):
        sys.argv.insert(1, "run")
    app()


if __name__ == "__main__":
    main()
