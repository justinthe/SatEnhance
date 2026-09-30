"""System 2 orchestration."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import rasterio
from rasterio.warp import transform_geom
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from shapely.geometry import shape

from . import __version__
from .cube import REFLECTANCE_MODES, build_cube
from .georef import scaled_transform
from .infer import run_inference
from .inputs import load_input, resolve_run_dir
from .mem import choose_block
from .models import load_model, resolve_device
from .preview import stretch_before, write_preview
from .report import write_report
from .variants import VARIANTS, default_variant
from .write import EnhancedWriter, compute_stretch, to_cog, write_rgb8

log = logging.getLogger(__name__)


@dataclass
class EnhanceParams:
    run_id: str | None = None
    in_dir: Path | None = None
    rawdata: Path = Path("/data/rawdata")
    out_dir: Path = Path("/data/output")
    cache_dir: Path = Path("/data/cache")
    family: str = "lite"
    variant: str = "auto"
    overlap: int = 32
    block: int = 512
    device: str = "auto"
    cog: bool = False
    clip_to_polygon: bool = False
    reflectance: str = "offset-corrected"


def _res_label(res_m: float) -> str:
    return f"{res_m:g}".replace(".", "p") + "m"


def _mask_geometry(run_dir: Path, crs):
    aoi_path = run_dir / "aoi.geojson"
    if not aoi_path.exists():
        raise SatEnhanceError(ExitCode.ENHANCE_INPUT_INVALID,
                              "--clip-to-polygon needs aoi.geojson in the rawdata run")
    gj = json.loads(aoi_path.read_text())["features"][0]["geometry"]
    return shape(transform_geom("EPSG:4326", crs, gj))


def enhance(p: EnhanceParams) -> Path:
    t0 = time.time()
    if p.reflectance not in REFLECTANCE_MODES:
        raise SatEnhanceError(ExitCode.INVALID_INPUT, f"--reflectance must be one of {REFLECTANCE_MODES}")
    if p.overlap < 0 or p.overlap % 2 or p.block < 128:
        raise SatEnhanceError(ExitCode.INVALID_INPUT, "--overlap must be even and >= 0; --block >= 128")

    resolve_device(p.device, p.family)  # fail fast (exit 21) before touching any input

    run_dir = resolve_run_dir(p.run_id, p.in_dir, p.rawdata)
    from satenhance_common import manifest as mf

    man = mf.load(run_dir)
    variant_name = default_variant(man.query.sensor) if p.variant == "auto" else p.variant
    if variant_name not in VARIANTS:
        raise SatEnhanceError(ExitCode.INVALID_INPUT,
                              f"Unknown variant '{p.variant}'. Choose auto or {sorted(VARIANTS)}")
    variant = VARIANTS[variant_name]
    inp = load_input(run_dir, variant_name)

    model = load_model(variant_name, p.family, p.device, p.cache_dir)

    cube = build_cube(inp, variant, p.reflectance)
    _, h, w = cube.data.shape
    s = model.scale
    res_out = abs(cube.transform.a) / s
    out_dir = Path(p.out_dir) / man.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"{man.scene.id}_{_res_label(res_out)}"
    main_path = out_dir / f"enhanced_{label}.tif"
    rgb_path = out_dir / f"enhanced_{label}_rgb8.tif"

    mask = _mask_geometry(run_dir, cube.crs) if p.clip_to_polygon else None
    writer = EnhancedWriter(
        main_path, count=len(variant.bands), height=h * s, width=w * s,
        transform=scaled_transform(cube.transform, s), crs=cube.crs,
        band_names=variant.bands, mask_geom=mask,
        tags={"SCENE_ID": man.scene.id, "MODEL": model.source, "VARIANT": variant_name},
    )
    block, mem_warning = choose_block(p.block, len(variant.bands), p.overlap, s)
    if mem_warning:
        log.warning("%s", mem_warning)
    try:
        stats = run_inference(model, cube, writer, block=block, overlap=p.overlap)
    finally:
        writer.close()

    bounds = compute_stretch(main_path, variant.rgb_idx)
    write_rgb8(main_path, rgb_path, variant.rgb_idx, bounds)
    before = stretch_before(
        (cube.data[list(variant.rgb_idx)] * 10000).astype("float32"), bounds
    )
    before[:, ~cube.valid] = 0
    preview_path = write_preview(out_dir / "preview_before_after.png", before, rgb_path)
    if p.cog:
        to_cog(main_path)
        to_cog(rgb_path)

    import torch

    warnings = list(stats.warnings)
    if mem_warning:
        warnings.append(mem_warning)
    if model.stub:
        warnings.append("STUB MODEL: bicubic interpolation, not super-resolution (test mode)")
    try:
        import sen2sr

        sen2sr_version = sen2sr.__version__
    except Exception:  # noqa: BLE001
        sen2sr_version = "unknown"
    with rasterio.open(main_path) as ds:
        out_shape, out_res = (ds.height, ds.width), ds.res
    write_report(
        out_dir / "enhance_report.json",
        tool_version=__version__, run_id=man.run_id, scene=man.scene.id,
        sensor=man.query.sensor, variant=variant_name, model_family=p.family,
        model_source=model.source, stub_model=model.stub, device=model.device,
        sen2sr_version=sen2sr_version, torch_version=torch.__version__,
        input_shape=[h, w], output_shape=list(out_shape), output_pixel_size_m=out_res[0],
        reflectance_mode=p.reflectance, block_requested=p.block, block_used=block,
        expected_output_bands=list(variant.bands), blocks=stats.blocks, oom_splits=stats.oom_splits,
        clip_to_polygon=p.clip_to_polygon, cog=p.cog,
        outputs={"main": main_path.name, "rgb8": rgb_path.name, "preview": preview_path.name},
        runtime_s=round(time.time() - t0, 2), warnings=warnings,
    )
    return out_dir
