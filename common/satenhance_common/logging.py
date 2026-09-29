"""Logging with secret redaction and optional JSON output."""

from __future__ import annotations

import json
import logging
import os
import sys

_SECRET_HINTS = ("KEY", "SECRET", "PASSWORD", "TOKEN")


def _secret_values() -> list[str]:
    vals = []
    for name, value in os.environ.items():
        if value and len(value) >= 4 and any(h in name.upper() for h in _SECRET_HINTS):
            vals.append(value)
    return vals


class RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for v in _secret_values():
            msg = msg.replace(v, "***")
        record.msg, record.args = msg, ()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
        )


def setup_logging(json_logs: bool = False, level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(RedactFilter())
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(levelname)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
