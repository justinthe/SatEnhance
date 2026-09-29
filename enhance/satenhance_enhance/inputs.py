"""Locate and validate System 1 output (PRD section 9.3 step 1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rasterio
from satenhance_common import manifest as mf
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from satenhance_common.paths import read_latest

from .variants import VARIANTS, default_variant

REF_BAND = "B04"


@dataclass
class EnhanceInput:
    run_dir: Path
    manifest: mf.Manifest

    def band_path(self, band: str) -> Path:
        return self.run_dir / self.manifest.scene.bands[band].file


def _bad(msg: str) -> SatEnhanceError:
    return SatEnhanceError(ExitCode.ENHANCE_INPUT_INVALID, msg)


def resolve_run_dir(run_id: str | None, in_dir: Path | None, rawdata_root: Path) -> Path:
    if in_dir:
        return Path(in_dir)
    run_id = run_id or read_latest(rawdata_root)
    if not run_id:
        raise _bad(f"No --run-id given and no LATEST pointer in {rawdata_root}")
    return Path(rawdata_root) / run_id


def load_input(run_dir: Path, variant: str | None = None) -> EnhanceInput:
    """Validate manifest, files and grid consistency for the chosen (or default) variant."""
    man = mf.load(run_dir)
    variant = variant or default_variant(man.query.sensor)
    needed = VARIANTS[variant].bands
    inp = EnhanceInput(Path(run_dir), man)
    missing = [b for b in needed if b not in man.scene.bands]
    if missing:
        raise _bad(
            f"Variant '{variant}' needs bands {needed} but the manifest only has "
            f"{sorted(man.scene.bands)} (missing {missing}). Re-run System 1 with a matching "
            f"--sensor."
        )
    absent = [str(inp.band_path(b)) for b in needed if not inp.band_path(b).exists()]
    if absent:
        raise _bad(f"Band file(s) not found: {absent}")

    with rasterio.open(inp.band_path(REF_BAND if REF_BAND in needed else needed[0])) as ref:
        ref_crs, ref_bounds = ref.crs, ref.bounds
        ref_grid = (ref.transform, ref.width, ref.height)
    for b in needed:
        info = man.scene.bands[b]
        with rasterio.open(inp.band_path(b)) as src:
            if src.crs != ref_crs:
                raise _bad(f"Band {b} CRS {src.crs} differs from reference {ref_crs}")
            tol = 0.5
            if any(abs(a - c) > tol for a, c in zip(src.bounds, ref_bounds, strict=True)):
                raise _bad(f"Band {b} extent {tuple(src.bounds)} differs from {tuple(ref_bounds)}")
            if info.res_m == 10 and (src.transform, src.width, src.height) != ref_grid:
                raise _bad(f"10 m band {b} is not on the same pixel grid as the reference band")
    return inp
