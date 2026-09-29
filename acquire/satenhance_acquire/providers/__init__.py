from __future__ import annotations

import os

from .base import Candidate, Provider


def get_provider(name: str | None = None) -> Provider:
    name = (name or os.environ.get("SATENHANCE_PROVIDER") or "cdse").lower()
    if name == "cdse":
        from .cdse import CdseProvider

        return CdseProvider()
    if name == "fixture":
        from .fixture import FixtureProvider

        return FixtureProvider(os.environ["SATENHANCE_FIXTURE_DIR"])
    raise ValueError(f"Unknown provider '{name}'")


__all__ = ["Candidate", "Provider", "get_provider"]
