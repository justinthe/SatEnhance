"""Blocked inference with bounded memory (PRD section 9.5).

`sen2sr.predict_large` tiles internally but (a) allocates the whole output in RAM and
(b) assumes a square input of at least 129 px (see the notes in `_predict_square`). We
therefore process the image in outer blocks with a halo, feed each block to it as a
padded square, and write finished blocks straight to disk.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field

import numpy as np
from rasterio.windows import Window
from satenhance_common.exit_codes import ExitCode, SatEnhanceError

from .cube import Cube
from .models import LoadedModel
from .write import EnhancedWriter

log = logging.getLogger(__name__)

PATCH = 128  # patch size hard-coded in sen2sr.predict_large


@dataclass
class InferStats:
    blocks: int = 0
    oom_splits: int = 0
    warnings: list[str] = field(default_factory=list)


def _silence_predict_large_progress() -> None:
    import sen2sr.utils as u

    u.tqdm = lambda it, **_: it  # per-block bars would flood logs; we log per outer block


def _is_oom(exc: BaseException) -> bool:
    import torch

    oom_types = tuple(
        t for t in (getattr(torch, "OutOfMemoryError", None),
                    getattr(torch.cuda, "OutOfMemoryError", None)) if t
    )
    return isinstance(exc, oom_types) if oom_types else False


def _predict_square(model: LoadedModel, sub: np.ndarray, overlap: int) -> np.ndarray:
    """(C,h,w) float32 reflectance -> (C,h*s,w*s).

    predict_large allocates its output as X.shape[1] x X.shape[1] and, for an input of
    exactly 128 px, leaves the last 64 output px unwritten. So we reflect-pad to a square
    of side >= 129 and crop afterwards.
    """
    import sen2sr
    import torch

    _, h, w = sub.shape
    side = max(h, w, PATCH + 1)
    if (h, w) != (side, side):
        sub = np.pad(sub, ((0, 0), (0, side - h), (0, side - w)), mode="reflect")
    x = torch.from_numpy(np.ascontiguousarray(sub)).to(model.device)
    with torch.inference_mode():
        y = sen2sr.predict_large(X=x, model=model.module, overlap=overlap)
    y = y.numpy()
    if y.shape[1] != side * model.scale or y.shape[2] != side * model.scale:
        raise SatEnhanceError(
            ExitCode.INFERENCE_FAILURE,
            f"Model returned {y.shape[1:]} for a {side}x{side} input; expected x{model.scale}",
        )
    return y[:, : h * model.scale, : w * model.scale]


def run_inference(
    model: LoadedModel, cube: Cube, writer: EnhancedWriter, *, block: int = 512,
    overlap: int = 32, show_progress: bool | None = None,
) -> InferStats:
    import torch

    _silence_predict_large_progress()
    _, height, width = cube.data.shape
    s = model.scale
    halo = overlap
    stats = InferStats()

    def process(r0: int, c0: int, h: int, w: int, depth: int) -> None:
        rs, cs = max(r0 - halo, 0), max(c0 - halo, 0)
        re, ce = min(r0 + h + halo, height), min(c0 + w + halo, width)
        try:
            y = _predict_square(model, cube.data[:, rs:re, cs:ce], overlap)
        except Exception as e:  # noqa: BLE001
            if not _is_oom(e):
                if isinstance(e, SatEnhanceError):
                    raise
                raise SatEnhanceError(ExitCode.INFERENCE_FAILURE, f"Inference failed: {e}") from e
            if model.device == "cuda":
                torch.cuda.empty_cache()
            if depth >= 1 or min(h, w) < 2 * PATCH:
                raise SatEnhanceError(
                    ExitCode.INFERENCE_FAILURE,
                    "Out of memory even after halving the block; use a smaller --block "
                    "or a smaller AOI",
                ) from e
            stats.oom_splits += 1
            hh, ww = h // 2, w // 2
            for dr, dc, bh, bw in ((0, 0, hh, ww), (0, ww, hh, w - ww),
                                   (hh, 0, h - hh, ww), (hh, ww, h - hh, w - ww)):
                process(r0 + dr, c0 + dc, bh, bw, depth + 1)
            return
        crop = y[:, (r0 - rs) * s : (r0 - rs + h) * s, (c0 - cs) * s : (c0 - cs + w) * s]
        valid = cube.valid[r0 : r0 + h, c0 : c0 + w].repeat(s, axis=0).repeat(s, axis=1)
        writer.write_block(crop, valid, Window(c0 * s, r0 * s, w * s, h * s))
        stats.blocks += 1

    origins = [(r, c) for r in range(0, height, block) for c in range(0, width, block)]
    it = origins
    if show_progress is None:
        show_progress = sys.stderr.isatty()
    if show_progress:
        from tqdm import tqdm

        it = tqdm(origins, desc="enhance", unit="block")
    for i, (r, c) in enumerate(it):
        if not show_progress:
            log.info("Block %d/%d", i + 1, len(origins))
        process(r, c, min(block, height - r), min(block, width - c), 0)
    return stats
