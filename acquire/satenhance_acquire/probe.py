"""`probe`: one quick end-to-end check of catalogue + S3 access (FIX_PLAN C3).

The Copernicus details this project relies on (collection id, asset names, S3 access) could not
be verified offline. This searches once, shows what the first scene really looks like, and tries
to read real pixels, so a mismatch turns into one readable line instead of a failed run.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import rasterio
from rasterio.windows import Window
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from shapely.geometry.base import BaseGeometry

from .mosaic import _open, group_candidates
from .providers.base import Provider

REQUIRED = ["B02", "B03", "B04", "B08", "SCL"]


def probe(provider: Provider, aoi: BaseGeometry, start: date, end: date,
          out: Callable[[str], None] = print) -> None:
    """Prints a report; raises SatEnhanceError on the first real problem."""
    out(f"provider: {provider.name}")
    provider.check_auth()
    out("credentials: present")

    cands = provider.search(aoi, start, end, 100.0, limit=50)
    out(f"catalogue: {len(cands)} scene(s) intersect the AOI between {start} and {end}")
    if not cands:
        raise SatEnhanceError(
            ExitCode.NO_DATA,
            "The catalogue returned no scenes. Widen the dates or check the AOI; if this AOI "
            "should have data, the collection id or query may be wrong for this catalogue.",
        )
    groups = group_candidates(cands, aoi)
    multi = sum(1 for g in groups if len(g.tiles) > 1)
    out(f"passes: {len(groups)} ({multi} with more than one tile)")

    first = min(cands, key=lambda c: (c.tile_cloud if c.tile_cloud is not None else 100.0))
    out("first scene (lowest tile cloud):")
    out(f"  id:        {first.id}")
    out(f"  datetime:  {first.datetime}")
    out(f"  cloud:     {first.tile_cloud}")
    out(f"  baseline:  {first.processing_baseline}")
    out(f"  platform:  {first.platform}  orbit: {first.relative_orbit}  tile: {first.tile_id}")
    out(f"  assets found: {sorted(first.assets)}")
    missing = []
    for band in REQUIRED:
        try:
            out(f"  {band:<4} -> {provider.href(first, band)}")
        except SatEnhanceError:
            missing.append(band)
            out(f"  {band:<4} -> MISSING")
    if missing:
        raise SatEnhanceError(
            ExitCode.NETWORK_FAILURE,
            f"The scene has no asset for {missing}. Asset names differ from what this tool "
            f"expects; send the 'assets found' line above so the mapping can be fixed.",
        )

    out("reading real pixels:")
    with rasterio.Env(**provider.gdal_env()):
        for band in ("SCL", "B04"):
            with _open(provider, first, band) as src:
                out(f"  {band}: crs={src.crs} size={src.width}x{src.height} "
                    f"res={abs(src.transform.a):g} m dtype={src.dtypes[0]}")
                w = Window(src.width // 2, src.height // 2, min(64, src.width), min(64, src.height))
                try:
                    data = src.read(1, window=w)
                except rasterio.errors.RasterioIOError as e:
                    raise SatEnhanceError(
                        ExitCode.NETWORK_FAILURE, f"Opened {band} but could not read pixels: {e}"
                    ) from e
                out(f"  {band}: read a {data.shape[1]}x{data.shape[0]} window, "
                    f"non-zero pixels: {int((data > 0).sum())}")
    out("PROBE OK")
