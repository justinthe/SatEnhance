"""selftest / prefetch check / reflectance comparison (FIX_PLAN B2, B3, B6).

The model weights and the loader code that ships with them can only be inspected after a real
download, so this loads each variant, pushes one random patch through the same code path the
pipeline uses, and reports exactly what is wrong (missing package, incompatible torch, wrong
output shape) instead of failing deep inside a real run.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from satenhance_common import manifest as mf
from satenhance_common.exit_codes import ExitCode, SatEnhanceError

from .cube import REFLECTANCE_MODES, build_cube
from .infer import _predict_square
from .inputs import load_input, resolve_run_dir
from .models import _torch_version, load_model
from .variants import VARIANTS, default_variant

log = logging.getLogger(__name__)
PATCH = 128


@dataclass
class VariantResult:
    variant: str
    family: str
    ok: bool
    code: int
    seconds: float
    device: str
    message: str


def selftest_variant(variant: str, family: str, device: str, cache: Path) -> VariantResult:
    t0 = time.time()
    dev = device
    try:
        model = load_model(variant, family, device, cache)
        dev = model.device
        rng = np.random.default_rng(0)
        patch = rng.uniform(0.02, 0.35, (len(VARIANTS[variant].bands), PATCH, PATCH)).astype("float32")
        y = _predict_square(model, patch, 32)
        expect = (len(VARIANTS[variant].bands), PATCH * model.scale, PATCH * model.scale)
        if y.shape != expect:
            raise SatEnhanceError(ExitCode.INFERENCE_FAILURE, f"output shape {y.shape}, expected {expect}")
        if not np.isfinite(y).all():
            raise SatEnhanceError(ExitCode.INFERENCE_FAILURE, "output contains NaN/inf")
        if float(y.std()) == 0.0:
            raise SatEnhanceError(ExitCode.INFERENCE_FAILURE, "output is constant (model not applied?)")
        msg = f"x{model.scale}, {y.shape[0]} bands, mean {float(y.mean()):.3f}" + (
            " [STUB model]" if model.stub else "")
        return VariantResult(variant, family, True, 0, time.time() - t0, dev, msg)
    except SatEnhanceError as e:
        return VariantResult(variant, family, False, int(e.code), time.time() - t0, dev, e.message)
    except Exception as e:  # noqa: BLE001
        return VariantResult(variant, family, False, int(ExitCode.INFERENCE_FAILURE),
                             time.time() - t0, dev, f"{type(e).__name__}: {e}")


def selftest(family: str, device: str, cache: Path, variants: list[str] | None = None,
             out: Callable[[str], None] = print) -> int:
    """Run the check for each variant. Returns the process exit code (0 = all passed)."""
    names = variants or [n for n, v in VARIANTS.items() if family == "lite" or v.full_url]
    out(f"torch {_torch_version()}  family={family}  device={device}")
    worst = 0
    for name in names:
        r = selftest_variant(name, family, device, cache)
        out(f"{'PASS' if r.ok else 'FAIL'}  {r.variant:<18} {r.family:<5} on {r.device:<5} "
            f"{r.seconds:5.1f}s  {r.message}")
        worst = max(worst, r.code)
    out("SELFTEST OK" if worst == 0 else f"SELFTEST FAILED (exit {worst})")
    return worst


# ------------------------------------------------------------------ reflectance comparison
def _bounds(arr: np.ndarray) -> list[tuple[float, float]]:
    out = []
    for band in arr:
        vals = band[band > 0]
        lo, hi = np.percentile(vals, [2, 98]) if vals.size else (0.0, 1.0)
        out.append((float(lo), float(max(hi, lo + 1))))
    return out


def compare_reflectance(run_id: str | None, family: str, variant: str, device: str,
                        rawdata: Path, out_dir: Path, cache: Path, size: int = 256) -> Path:
    """Enhance one centre crop in both reflectance modes and save a side-by-side PNG."""
    from PIL import Image, ImageDraw

    from .write import apply_stretch

    run_dir = resolve_run_dir(run_id, None, rawdata)
    man = mf.load(run_dir)
    name = default_variant(man.query.sensor) if variant == "auto" else variant
    var = VARIANTS[name]
    inp = load_input(run_dir, name)
    model = load_model(name, family, device, cache)

    panels, labels = [], []
    for mode in REFLECTANCE_MODES:
        cube = build_cube(inp, var, mode)
        _, h, w = cube.data.shape
        s = min(size, h, w)
        r0, c0 = (h - s) // 2, (w - s) // 2
        crop = cube.data[:, r0:r0 + s, c0:c0 + s]
        y = _predict_square(model, crop, 32)
        panels.append((crop, y))
        labels.append(mode)

    ridx = list(var.rgb_idx)
    y_dn = [(p[1][ridx] * 10000).astype("float32") for p in panels]
    bounds = _bounds(y_dn[0])  # same stretch for every panel so colours are comparable
    before = (panels[0][0][ridx] * 10000).astype("float32")
    k = model.scale
    tiles = [np.repeat(np.repeat(apply_stretch(before, bounds), k, axis=1), k, axis=2)]
    tiles += [apply_stretch(a, bounds) for a in y_dn]
    titles = [f"input (nearest x{k})", *(f"enhanced, {m}" for m in labels)]
    h_px, w_px = tiles[0].shape[1:]
    pad = 24
    canvas = Image.new("RGB", (w_px * len(tiles) + 8 * (len(tiles) - 1), h_px + pad), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (t, title) in enumerate(zip(tiles, titles, strict=True)):
        x0 = i * (w_px + 8)
        canvas.paste(Image.fromarray(np.moveaxis(t, 0, -1)), (x0, pad))
        draw.text((x0 + 4, 6), title, fill=(0, 0, 0))
    out = Path(out_dir) / man.run_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / "reflectance_compare.png"
    canvas.save(path)
    return path
