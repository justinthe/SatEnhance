from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

LATEST_FILE = "LATEST"


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug or "aoi")[:max_len].strip("-") or "aoi"


def make_run_id(label: str, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%dT%H%M%S')}_{slugify(label)}"


def write_latest(rawdata_dir: Path, run_id: str) -> None:
    (Path(rawdata_dir) / LATEST_FILE).write_text(run_id + "\n")


def read_latest(rawdata_dir: Path) -> str | None:
    f = Path(rawdata_dir) / LATEST_FILE
    if not f.exists():
        return None
    return f.read_text().strip() or None
