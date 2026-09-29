"""Before/after PNG (PRD section 9.6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from .write import apply_stretch

MAX_WIDTH = 2000


def write_preview(
    out_png: Path, before_rgb_refl: np.ndarray, after_rgb_tif: Path,
) -> Path:
    """before_rgb_refl: (3,h,w) uint8 already stretched (input resolution)."""
    with rasterio.open(after_rgb_tif) as ds:
        step = max(1, int(np.ceil(ds.width * 2 / MAX_WIDTH)))
        after = ds.read(out_shape=(3, max(1, ds.height // step), max(1, ds.width // step)),
                        resampling=rasterio.enums.Resampling.nearest)
    h, w = after.shape[1:]
    before_img = Image.fromarray(np.moveaxis(before_rgb_refl, 0, -1)).resize(
        (w, h), Image.NEAREST)
    after_img = Image.fromarray(np.moveaxis(after, 0, -1))
    canvas = Image.new("RGB", (w * 2 + 8, h), (255, 255, 255))
    canvas.paste(before_img, (0, 0))
    canvas.paste(after_img, (w + 8, 0))
    canvas.save(out_png)
    return out_png


def stretch_before(cube_rgb_dn: np.ndarray, bounds) -> np.ndarray:
    return apply_stretch(cube_rgb_dn, bounds)
