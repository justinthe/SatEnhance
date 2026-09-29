"""GeoTIFF output: windowed writer, 8-bit RGB derivative, COG conversion (PRD section 9.6)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import geometry_mask
from rasterio.windows import Window
from rasterio.windows import transform as win_transform
from shapely.geometry.base import BaseGeometry

DISCLAIMER = (
    "Super-resolved pixels are model-predicted detail, not measured data. Suitable for "
    "visualisation; do not use as ground truth for measurement or legal boundaries."
)
REFLECTANCE_SCALE = 10000


class EnhancedWriter:
    """Writes uint16 reflectance*10000 blocks into a tiled GeoTIFF as they are produced."""

    def __init__(
        self, path: Path, *, count: int, height: int, width: int, transform: Affine, crs,
        band_names: list[str], mask_geom: BaseGeometry | None = None, tags: dict | None = None,
    ):
        self.path = Path(path)
        self._part = self.path.with_name(self.path.name + ".part")
        self.transform = transform
        self.mask_geom = mask_geom
        self._ds = rasterio.open(
            self._part, "w", driver="GTiff", count=count, height=height, width=width,
            dtype="uint16", crs=crs, transform=transform, nodata=0, tiled=True,
            blockxsize=256, blockysize=256, compress="deflate", predictor=2, BIGTIFF="IF_SAFER",
        )
        for i, name in enumerate(band_names, 1):
            self._ds.set_band_description(i, name)
        self._ds.update_tags(
            SCALE_FACTOR=str(1 / REFLECTANCE_SCALE), UNITS="surface reflectance (x10000)",
            DISCLAIMER=DISCLAIMER, **(tags or {}),
        )

    def write_block(self, refl: np.ndarray, valid: np.ndarray, window: Window) -> None:
        """refl: float (C,h,w) reflectance; valid: bool (h,w) for the same window."""
        dn = np.clip(np.rint(refl * REFLECTANCE_SCALE), 1, 65535).astype("uint16")
        keep = valid.copy()
        if self.mask_geom is not None:
            keep &= ~geometry_mask(
                [self.mask_geom], out_shape=keep.shape,
                transform=win_transform(window, self.transform), all_touched=True,
            )
        dn[:, ~keep] = 0
        self._ds.write(dn, window=window)

    def close(self) -> Path:
        self._ds.close()
        os.replace(self._part, self.path)
        return self.path


def _decimated(ds: rasterio.io.DatasetReader, bands: tuple[int, ...], max_side: int) -> np.ndarray:
    step = max(1, int(np.ceil(max(ds.width, ds.height) / max_side)))
    out_shape = (len(bands), max(1, ds.height // step), max(1, ds.width // step))
    return ds.read([b + 1 for b in bands], out_shape=out_shape,
                   resampling=rasterio.enums.Resampling.nearest)


def compute_stretch(path: Path, rgb_idx: tuple[int, int, int]) -> list[tuple[float, float]]:
    """2-98 percentile per RGB band over valid pixels (decimated read)."""
    with rasterio.open(path) as ds:
        arr = _decimated(ds, rgb_idx, 2048).astype("float32")
    bounds = []
    for band in arr:
        vals = band[band > 0]
        if vals.size == 0:
            bounds.append((0.0, 1.0))
            continue
        lo, hi = np.percentile(vals, [2, 98])
        bounds.append((float(lo), float(max(hi, lo + 1))))
    return bounds


def apply_stretch(arr: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    """arr: (3,h,w) DN -> uint8 (3,h,w); 0 stays 0 (nodata)."""
    out = np.zeros(arr.shape, dtype="uint8")
    for i, (lo, hi) in enumerate(bounds):
        scaled = np.clip((arr[i].astype("float32") - lo) / (hi - lo), 0, 1)
        out[i] = np.where(arr[i] > 0, np.maximum(np.rint(scaled * 255), 1), 0).astype("uint8")
    return out


def write_rgb8(main_path: Path, rgb_path: Path, rgb_idx: tuple[int, int, int],
               bounds: list[tuple[float, float]], block: int = 1024) -> Path:
    part = rgb_path.with_name(rgb_path.name + ".part")
    with rasterio.open(main_path) as src:
        profile = src.profile.copy()
        profile.update(count=3, dtype="uint8", nodata=0, predictor=1, BIGTIFF="IF_SAFER")
        with rasterio.open(part, "w", **profile) as dst:
            dst.set_band_description(1, "red")
            dst.set_band_description(2, "green")
            dst.set_band_description(3, "blue")
            for r in range(0, src.height, block):
                for c in range(0, src.width, block):
                    win = Window(c, r, min(block, src.width - c), min(block, src.height - r))
                    data = src.read([i + 1 for i in rgb_idx], window=win)
                    dst.write(apply_stretch(data, bounds), window=win)
    os.replace(part, rgb_path)
    return rgb_path


def to_cog(path: Path) -> Path:
    import rasterio.shutil

    tmp = path.with_name(path.stem + ".cog.tmp.tif")
    rasterio.shutil.copy(path, tmp, driver="COG", compress="DEFLATE", BIGTIFF="IF_SAFER")
    os.replace(tmp, path)
    return path
