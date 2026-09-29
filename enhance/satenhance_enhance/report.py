from __future__ import annotations

import json
from pathlib import Path

from .write import DISCLAIMER


def write_report(path: Path, **fields) -> Path:
    path.write_text(json.dumps({"disclaimer": DISCLAIMER, **fields}, indent=2, default=str))
    return path
