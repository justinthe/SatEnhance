"""Host-memory guard: pick a block size that fits (PRD section 9.5).

`sen2sr.predict_large` builds the whole output for one padded block in host RAM (float32) even
when the model runs on a GPU, so that is what has to fit.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

SAFETY_FRACTION = 0.7  # use at most this share of available memory
OVERHEAD = 1.5  # activations, copies, torch allocator slack
MIN_BLOCK = 128


def _read_int(path: str) -> int | None:
    try:
        text = Path(path).read_text().strip()
        return None if text == "max" else int(text)
    except (OSError, ValueError):
        return None


def available_bytes() -> int | None:
    """Available memory: MemAvailable, capped by the container (cgroup) limit if any."""
    avail = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    limit = _read_int("/sys/fs/cgroup/memory.max") or _read_int(
        "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    )
    used = _read_int("/sys/fs/cgroup/memory.current") or _read_int(
        "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    )
    if limit and limit < (1 << 60):  # cgroup v1 reports a huge number for "no limit"
        cg_avail = max(limit - (used or 0), 0)
        avail = cg_avail if avail is None else min(avail, cg_avail)
    return avail


def estimate_block_bytes(channels: int, block: int, overlap: int, scale: int) -> int:
    side = block + 2 * overlap  # block plus halo on both sides
    return int(channels * (side * scale) ** 2 * 4 * OVERHEAD)


def choose_block(requested: int, channels: int, overlap: int, scale: int,
                 avail: int | None = None) -> tuple[int, str | None]:
    """Largest block <= requested whose estimated peak fits. Returns (block, warning|None)."""
    avail = available_bytes() if avail is None else avail
    if avail is None:
        return requested, None
    budget = avail * SAFETY_FRACTION
    block = requested
    while block > MIN_BLOCK and estimate_block_bytes(channels, block, overlap, scale) > budget:
        block = max(MIN_BLOCK, (block // 2) // 64 * 64)
    if estimate_block_bytes(channels, block, overlap, scale) > budget:
        return block, (
            f"Even the smallest block ({block}px) may need more than {budget / 1e9:.1f} GB of "
            f"the {avail / 1e9:.1f} GB available; expect an out-of-memory error. Give Docker "
            "more memory or use a smaller AOI."
        )
    if block != requested:
        return block, (
            f"Lowered --block from {requested} to {block} to fit in "
            f"{avail / 1e9:.1f} GB of available memory"
        )
    return block, None
