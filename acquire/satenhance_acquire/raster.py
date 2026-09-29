"""Small rasterio helpers shared by selection and download."""

from __future__ import annotations

import math

import rasterio
from rasterio.warp import transform_geom
from rasterio.windows import Window
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry


def aoi_in_crs(aoi: BaseGeometry, crs) -> BaseGeometry:
    return shape(transform_geom("EPSG:4326", crs, mapping(aoi)))


def snapped_window(
    src: rasterio.io.DatasetReader, bounds: tuple[float, float, float, float], even: bool = False
) -> Window:
    """Pixel window covering `bounds` (in src CRS), grid-aligned and clipped to the raster.

    With even=True, offsets are floored and sizes ceiled to multiples of 2 so that a
    10 m window maps exactly onto the 20 m grid.
    """
    win = rasterio.windows.from_bounds(*bounds, transform=src.transform)
    # Round away float noise so exact pixel edges don't spill into a neighbouring pixel.
    col0, row0 = math.floor(round(win.col_off, 6)), math.floor(round(win.row_off, 6))
    col1 = math.ceil(round(win.col_off + win.width, 6))
    row1 = math.ceil(round(win.row_off + win.height, 6))
    if even:
        col0, row0 = col0 - col0 % 2, row0 - row0 % 2
        col1, row1 = col1 + col1 % 2, row1 + row1 % 2
    col0, row0 = max(col0, 0), max(row0, 0)
    col1, row1 = min(col1, src.width), min(row1, src.height)
    return Window(col0, row0, max(col1 - col0, 0), max(row1 - row0, 0))
