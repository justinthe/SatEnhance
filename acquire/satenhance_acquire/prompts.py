"""Thin wrappers over stdin/stdout so interactive flows are testable."""

from __future__ import annotations

import sys
from collections.abc import Callable


def is_interactive(non_interactive: bool) -> bool:
    return (not non_interactive) and sys.stdin.isatty()


def say(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def ask(prompt: str, input_fn: Callable[[str], str] | None = None) -> str:
    fn = input_fn or input
    return fn(prompt).strip()
