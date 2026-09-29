"""satenhance-enhance command line (System 2)."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import typer
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from satenhance_common.logging import setup_logging

app = typer.Typer(add_completion=False, help="SatEnhance System 2: super-resolve Sentinel-2 rawdata.")
log = logging.getLogger("satenhance")


def _fail(e: SatEnhanceError) -> typer.Exit:
    log.error("%s", e.message)
    return typer.Exit(int(e.code))


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
    except SatEnhanceError as e:
        raise _fail(e) from e
    except (KeyboardInterrupt, EOFError):
        log.error("Aborted")
        raise typer.Exit(int(ExitCode.UNEXPECTED)) from None
    except Exception:  # noqa: BLE001
        log.exception("Unexpected error")
        raise typer.Exit(int(ExitCode.UNEXPECTED)) from None
    print(f"Output written to {result}", file=sys.stderr)
    print(f"SATENHANCE_OUTPUT={result}")


@app.command()
def prefetch(
    cache: Path = typer.Option(Path("/data/cache"), "--cache"),
    model: str = typer.Option("lite", "--model"),
    log_json: bool = typer.Option(False, "--log-json"),
) -> None:
    """Download model weights into the cache volume so later runs can be offline."""
    setup_logging(log_json)
    from .models import prefetch as do_prefetch
    from .variants import VARIANTS

    t0 = time.time()
    try:
        names = [n for n, v in VARIANTS.items() if model == "lite" or v.full_url]
        done = do_prefetch(names, model, cache)
    except SatEnhanceError as e:
        raise _fail(e) from e
    print(f"Prefetched {done} in {time.time() - t0:.0f}s", file=sys.stderr)


def main() -> None:
    args = sys.argv[1:]
    # Allow `satenhance-enhance --run-id X` as shorthand for `... run --run-id X`.
    if not args or (args[0] not in ("run", "prefetch") and args[0] not in ("--help", "-h")):
        sys.argv.insert(1, "run")
    app()


if __name__ == "__main__":
    main()
