"""Date-window resolution (--start / --end are optional; default window is the last 30 days)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from satenhance_common.exit_codes import ExitCode, SatEnhanceError

DEFAULT_DAYS = 30
MAX_DAYS = 3650


def today_utc() -> date:
    return datetime.now(UTC).date()


def parse_date(text: str) -> date:
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as e:
        raise SatEnhanceError(
            ExitCode.INVALID_INPUT, f"Invalid date '{text}' (want YYYY-MM-DD)"
        ) from e


def resolve_window(
    start: date | None, end: date | None, days: int = DEFAULT_DAYS, today: date | None = None
) -> tuple[date, date, bool]:
    """Fill in whichever of start/end is missing. Returns (start, end, used_default).

    neither      -> last `days` days ending today (UTC)
    only --end   -> `days` days before it
    only --start -> `days` days after it, but never later than today
    both         -> unchanged
    """
    today = today or today_utc()
    if not 1 <= days <= MAX_DAYS:
        raise SatEnhanceError(ExitCode.INVALID_INPUT, f"--days must be between 1 and {MAX_DAYS}")
    if start is not None and end is not None:
        return start, end, False
    if start is None and end is None:
        return today - timedelta(days=days), today, True
    if end is not None:
        return end - timedelta(days=days), end, True
    if start > today:
        raise SatEnhanceError(ExitCode.INVALID_INPUT, f"--start ({start}) is in the future")
    return start, min(start + timedelta(days=days), today), True
