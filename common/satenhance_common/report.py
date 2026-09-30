"""error_report.json: what failed, where, and with which versions (FIX_PLAN D2)."""

from __future__ import annotations

import json
import platform
import sys
import traceback
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from .exit_codes import ExitCode, SatEnhanceError

FILENAME = "error_report.json"
PACKAGES = ("satenhance-common", "satenhance-acquire", "satenhance-enhance", "torch",
            "sen2sr", "mlstac", "rasterio", "geopandas", "pystac-client", "numpy")


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def host_path(path: Path | str) -> str:
    """Show a container path the way the user sees it: /data/rawdata/x -> ./rawdata/x.

    The scripts mount ./rawdata, ./output and ./cache of the SatEnhance folder at /data/...
    """
    text = str(path)
    if text.startswith("/data/"):
        return "./" + text[len("/data/"):] + "  (relative to your SatEnhance folder)"
    return text


def write_error_report(
    directory: Path, *, tool: str, exc: BaseException, stage: str | None = None,
    argv: list[str] | None = None,
) -> Path | None:
    """Write `<directory>/error_report.json`. Never raises: a failed report must not hide the
    original error. Returns the path, or None if it could not be written."""
    try:
        if isinstance(exc, SatEnhanceError):
            code, message, context = int(exc.code), exc.message, dict(exc.context)
        else:
            code, message, context = int(ExitCode.UNEXPECTED), f"{type(exc).__name__}: {exc}", {}
        cause = exc.__cause__ if exc.__cause__ is not None else exc
        tb = None
        if not isinstance(exc, SatEnhanceError) or exc.__cause__ is not None:
            tb = "".join(traceback.format_exception(type(cause), cause, cause.__traceback__))
            tb = "\n".join(tb.splitlines()[-40:])  # the useful end of it
        report = {
            "tool": tool,
            "time_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "exit_code": code,
            "exit_name": ExitCode(code).name,
            "message": message,
            "stage": stage or context.get("stage"),
            "context": {k: v for k, v in context.items() if k != "stage"},
            "argv": _redact(argv if argv is not None else sys.argv[1:]),
            "versions": _versions(),
            **({"traceback": tb} if tb else {}),
        }
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / FILENAME
        path.write_text(json.dumps(report, indent=2, default=str))
        return path
    except Exception:  # noqa: BLE001
        return None


def _redact(argv: list[str]) -> list[str]:
    hidden = ("key", "secret", "password", "token")
    return ["***" if any(h in a.lower() for h in hidden) else a for a in argv]
