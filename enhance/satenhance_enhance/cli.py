"""satenhance-enhance command line (System 2)."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import typer
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from satenhance_common.logging import setup_logging
from satenhance_common.report import host_path, write_error_report

app = typer.Typer(add_completion=False, help="SatEnhance System 2: super-resolve Sentinel-2 rawdata.")
log = logging.getLogger("satenhance")


COMMANDS = ("run", "prefetch", "selftest")


def _fail(e: BaseException, report_dir: Path | None = None) -> typer.Exit:
    """Log, write error_report.json next to the outputs, return the right exit."""
    if isinstance(e, SatEnhanceError):
        log.error("%s", e.message)
        code = int(e.code)
    else:
        log.error("Unexpected error: %s: %s", type(e).__name__, e)
        code = int(ExitCode.UNEXPECTED)
    if report_dir is not None:
        path = write_error_report(report_dir, tool="satenhance-enhance", exc=e)
        if path:
            log.error("Details: %s", host_path(path))
    return typer.Exit(code)


@app.command()
def run(
    run_id: str | None = typer.Option(None, "--run-id", help="Run in rawdata (default: LATEST)"),
    in_dir: Path | None = typer.Option(None, "--in", help="Explicit rawdata run directory"),
    rawdata: Path = typer.Option(Path("/data/rawdata"), "--rawdata"),
    out: Path = typer.Option(Path("/data/output"), "--out"),
    cache: Path = typer.Option(Path("/data/cache"), "--cache"),
    model: str = typer.Option("lite", "--model", help="lite | full (full needs a GPU)"),
    variant: str = typer.Option("auto", "--variant", help="auto | rgbn_x4 | multispectral_x4"),
    overlap: int = typer.Option(32, "--overlap"),
    block: int = typer.Option(512, "--block", help="Input pixels per processing block"),
    device: str = typer.Option("auto", "--device", help="auto | cpu | cuda"),
    cog: bool = typer.Option(False, "--cog", help="Write Cloud-Optimised GeoTIFFs"),
    clip_to_polygon: bool = typer.Option(False, "--clip-to-polygon"),
    reflectance: str = typer.Option("offset-corrected", "--reflectance",
                                    help="offset-corrected | raw-div10000"),
    log_json: bool = typer.Option(False, "--log-json"),
) -> None:
    setup_logging(log_json)
    from .runner import EnhanceParams, enhance

    try:
        if model not in ("lite", "full") or device not in ("auto", "cpu", "cuda"):
            raise SatEnhanceError(ExitCode.INVALID_INPUT, "Bad --model or --device value")
        params = EnhanceParams(
            run_id=run_id, in_dir=in_dir, rawdata=rawdata, out_dir=out, cache_dir=cache,
            family=model, variant=variant, overlap=overlap, block=block, device=device,
            cog=cog, clip_to_polygon=clip_to_polygon, reflectance=reflectance,
        )
        result = enhance(params)
    except (KeyboardInterrupt, EOFError):
        log.error("Aborted")
        raise typer.Exit(int(ExitCode.UNEXPECTED)) from None
    except Exception as e:  # noqa: BLE001  (SatEnhanceError and anything unexpected)
        raise _fail(e, Path(out) / run_id if run_id else Path(out)) from e
    print(f"Output written to {result}", file=sys.stderr)
    print(f"SATENHANCE_OUTPUT={result}")


@app.command()
def prefetch(
    cache: Path = typer.Option(Path("/data/cache"), "--cache"),
    model: str = typer.Option("lite", "--model"),
    no_selftest: bool = typer.Option(False, "--no-selftest", help="Skip the load-and-run check"),
    log_json: bool = typer.Option(False, "--log-json"),
) -> None:
    """Download model weights into the cache volume so later runs can be offline, then verify them."""
    setup_logging(log_json)
    from .models import prefetch as do_prefetch
    from .selftest import selftest as do_selftest
    from .variants import VARIANTS

    t0 = time.time()
    try:
        names = [n for n, v in VARIANTS.items() if model == "lite" or v.full_url]
        done = do_prefetch(names, model, cache)
    except Exception as e:  # noqa: BLE001
        raise _fail(e, Path(cache)) from e
    print(f"Prefetched {done} in {time.time() - t0:.0f}s", file=sys.stderr)
    if not no_selftest:
        code = do_selftest(model, "auto" if model == "full" else "cpu", cache, names)
        if code:
            raise typer.Exit(code)


@app.command()
def selftest(
    cache: Path = typer.Option(Path("/data/cache"), "--cache"),
    model: str = typer.Option("lite", "--model", help="lite | full"),
    variant: str = typer.Option("all", "--variant", help="all | rgbn_x4 | multispectral_x4"),
    device: str = typer.Option("auto", "--device"),
    compare_reflectance: str | None = typer.Option(
        None, "--compare-reflectance", metavar="RUN_ID",
        help="Enhance one crop of this run in both reflectance modes and save a side-by-side PNG"),
    rawdata: Path = typer.Option(Path("/data/rawdata"), "--rawdata"),
    out: Path = typer.Option(Path("/data/output"), "--out"),
    log_json: bool = typer.Option(False, "--log-json"),
) -> None:
    """Load each model variant and run a test patch; optionally compare reflectance conventions."""
    setup_logging(log_json)
    from .selftest import compare_reflectance as do_compare
    from .selftest import selftest as do_selftest
    from .variants import VARIANTS

    try:
        if variant != "all" and variant not in VARIANTS:
            raise SatEnhanceError(ExitCode.INVALID_INPUT,
                                  f"Unknown variant '{variant}'. Choose all or {sorted(VARIANTS)}")
        if compare_reflectance:
            path = do_compare(compare_reflectance, model, "auto" if variant == "all" else variant,
                              device, rawdata, out, cache)
            print(f"Wrote {path}  (left to right: input, offset-corrected, raw-div10000)")
            print("Pick the panel whose colours look natural and use --reflectance with that name.")
            return
        names = None if variant == "all" else [variant]
        code = do_selftest(model, device, cache, names)
    except Exception as e:  # noqa: BLE001
        raise _fail(e, Path(cache)) from e
    if code:
        raise typer.Exit(code)


def main() -> None:
    args = sys.argv[1:]
    # Allow `satenhance-enhance --run-id X` as shorthand for `... run --run-id X`.
    if not args or (args[0] not in COMMANDS and args[0] not in ("--help", "-h")):
        sys.argv.insert(1, "run")
    app()


if __name__ == "__main__":
    main()
