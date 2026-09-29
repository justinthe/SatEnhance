"""Assemble the model input cube from band GeoTIFFs (PRD section 9.3 step 2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .inputs import REF_BAND, EnhanceInput
from .variants import Variant

REFLECTANCE_MODES = ("offset-corrected", "raw-div10000")


@dataclass
class Cube:
    data: np.ndarray  # float32 (C, H, W) reflectance
    valid: np.ndarray  # bool (H, W): at least one band has data
    transform: Affine
    crs: object
    band_names: list[str]


def build_cube(inp: EnhanceInput, variant: Variant, mode: str = "offset-corrected") -> Cube:
    if mode not in REFLECTANCE_MODES:
        raise ValueError(f"reflectance mode must be one of {REFLECTANCE_MODES}")
    ref_name = REF_BAND if REF_BAND in variant.bands else variant.bands[0]
    with rasterio.open(inp.band_path(ref_name)) as ref:
        transform, crs, h, w = ref.transform, ref.crs, ref.height, ref.width

    out = np.zeros((len(variant.bands), h, w), dtype="float32")
    valid = np.zeros((h, w), dtype=bool)
    for i, band in enumerate(variant.bands):
        info = inp.manifest.scene.bands[band]
        with rasterio.open(inp.band_path(band)) as src:
            if (src.width, src.height) == (w, h):
                dn = src.read(1).astype("float32")
            else:  # 20 m -> 10 m grid, bilinear
                dn = np.zeros((h, w), dtype="float32")
                reproject(
                    source=rasterio.band(src, 1), destination=dn,
                    dst_transform=transform, dst_crs=crs, resampling=Resampling.bilinear,
                )
        has_data = dn > 0
        valid |= has_data
        if mode == "offset-corrected":
            refl = dn * info.scale + info.offset
        else:
            refl = dn / 10000.0
        out[i] = np.where(has_data, np.clip(refl, 0.0, None), 0.0)
    return Cube(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), valid, transform, crs,
                list(variant.bands))
