import json

import numpy as np
import pytest
import rasterio
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from satenhance_enhance import inputs, models
from satenhance_enhance.cli import app
from satenhance_enhance.cube import build_cube
from satenhance_enhance.georef import scaled_transform
from satenhance_enhance.infer import _predict_square
from satenhance_enhance.runner import EnhanceParams, enhance
from satenhance_enhance.synthetic import make_rawdata
from satenhance_enhance.variants import VARIANTS
from typer.testing import CliRunner


def params(tmp_path, **kw):
    base = dict(rawdata=tmp_path / "raw", out_dir=tmp_path / "out", cache_dir=tmp_path / "c",
                block=128)
    base.update(kw)
    return EnhanceParams(**base)


@pytest.fixture
def raw(tmp_path):
    return make_rawdata(tmp_path / "raw", size=160)


# ---- inputs ------------------------------------------------------------------------
def test_missing_manifest_exit_20(tmp_path):
    (tmp_path / "raw" / "x").mkdir(parents=True)
    with pytest.raises(SatEnhanceError) as e:
        enhance(params(tmp_path, run_id="x"))
    assert e.value.code == ExitCode.ENHANCE_INPUT_INVALID


def test_no_latest_exit_20(tmp_path):
    (tmp_path / "raw").mkdir()
    with pytest.raises(SatEnhanceError) as e:
        enhance(params(tmp_path))
    assert e.value.code == ExitCode.ENHANCE_INPUT_INVALID


def test_missing_band_file_exit_20(raw, tmp_path):
    (raw / "S2_SYNTH" / "B08.tif").unlink()
    with pytest.raises(SatEnhanceError) as e:
        enhance(params(tmp_path))
    assert e.value.code == ExitCode.ENHANCE_INPUT_INVALID and "B08" in e.value.message


def test_misaligned_grid_exit_20(raw):
    p = raw / "S2_SYNTH" / "B02.tif"
    with rasterio.open(p) as s:
        prof, data = s.profile, s.read()
    prof.update(transform=prof["transform"] * rasterio.Affine.translation(3, 0))
    with rasterio.open(p, "w", **prof) as d:
        d.write(data)
    with pytest.raises(SatEnhanceError) as e:
        inputs.load_input(raw)
    assert e.value.code == ExitCode.ENHANCE_INPUT_INVALID


def test_variant_needs_bands_exit_20(raw, tmp_path):
    with pytest.raises(SatEnhanceError) as e:
        enhance(params(tmp_path, variant="multispectral_x4"))
    assert e.value.code == ExitCode.ENHANCE_INPUT_INVALID


# ---- cube --------------------------------------------------------------------------
def test_cube_scaling_and_nodata(tmp_path):
    run = make_rawdata(tmp_path / "raw", size=32, nodata_corner=True)
    inp = inputs.load_input(run)
    c = build_cube(inp, VARIANTS["rgbn_x4"], "offset-corrected")
    assert c.data.shape == (4, 32, 32) and c.data.dtype == np.float32
    assert (c.data[:, :3, :3] == 0).all() and not c.valid[0, 0] and c.valid[-1, -1]
    with rasterio.open(inp.band_path("B04")) as s:
        dn = s.read(1).astype("float32")
    b04 = c.data[0]  # RGBN order: B04 first
    assert b04[-1, -1] == pytest.approx(max(dn[-1, -1] * 1e-4 - 0.1, 0), rel=1e-5)
    raw_mode = build_cube(inp, VARIANTS["rgbn_x4"], "raw-div10000")
    assert raw_mode.data[0, -1, -1] == pytest.approx(dn[-1, -1] / 1e4, rel=1e-5)


def test_cube_multispectral_resamples_20m(tmp_path):
    run = make_rawdata(tmp_path / "raw", sensor="multispectral", size=32)
    c = build_cube(inputs.load_input(run), VARIANTS["multispectral_x4"])
    assert c.data.shape == (10, 32, 32) and np.isfinite(c.data).all()
    assert c.band_names[3] == "B05"


# ---- end to end (stub model) -------------------------------------------------------
def test_end_to_end_geometry_and_outputs(raw, tmp_path):
    out = enhance(params(tmp_path))
    files = {p.name for p in out.iterdir()}
    assert {"enhanced_SYNTH_2p5m.tif", "enhanced_SYNTH_2p5m_rgb8.tif",
            "preview_before_after.png", "enhance_report.json"} <= files
    assert not list(out.glob("*.part"))
    with rasterio.open(raw / "S2_SYNTH" / "B04.tif") as src, \
            rasterio.open(out / "enhanced_SYNTH_2p5m.tif") as dst, \
            rasterio.open(out / "enhanced_SYNTH_2p5m_rgb8.tif") as rgb:
        assert dst.res == (2.5, 2.5) and dst.crs == src.crs
        assert dst.bounds == src.bounds
        assert (dst.height, dst.width) == (src.height * 4, src.width * 4)
        assert dst.count == 4 and dst.dtypes[0] == "uint16"
        assert dst.descriptions == ("B04", "B03", "B02", "B08")
        assert dst.tags()["DISCLAIMER"].startswith("Super-resolved")
        data = dst.read()
        assert data.min() >= 1  # all valid; nothing left unwritten
        assert rgb.count == 3 and rgb.dtypes[0] == "uint8" and rgb.read().max() > 100
    rep = json.loads((out / "enhance_report.json").read_text())
    assert rep["stub_model"] and rep["output_pixel_size_m"] == 2.5 and rep["blocks"] == 4
    assert any("STUB" in w for w in rep["warnings"])


def test_latest_is_used_and_run_id_selects(raw, tmp_path):
    other = make_rawdata(tmp_path / "raw", size=32, run_id="20260202T000000_other")
    assert enhance(params(tmp_path)).name == other.name  # LATEST
    assert enhance(params(tmp_path, run_id=raw.name)).name == raw.name


def test_nodata_stays_nodata_and_clip_polygon(tmp_path):
    run = make_rawdata(tmp_path / "raw", size=160, nodata_corner=True)
    out = enhance(params(tmp_path, clip_to_polygon=True))
    with rasterio.open(out / "enhanced_SYNTH_2p5m.tif") as dst:
        d = dst.read(1)
    assert (d[:60, :60] == 0).all()  # input nodata corner (16 px * 4) stays nodata
    assert d[-1, -1] > 0
    assert run.exists()


def test_clip_polygon_masks_outside(tmp_path):
    from shapely.geometry import box, mapping
    run = make_rawdata(tmp_path / "raw", size=160)
    aoi = json.loads((run / "aoi.geojson").read_text())
    from shapely.geometry import shape
    g = shape(aoi["features"][0]["geometry"])
    minx, miny, maxx, maxy = g.bounds
    half = box(minx, miny, (minx + maxx) / 2, maxy)  # left half only
    aoi["features"][0]["geometry"] = mapping(half)
    (run / "aoi.geojson").write_text(json.dumps(aoi))
    out = enhance(params(tmp_path, clip_to_polygon=True))
    with rasterio.open(out / "enhanced_SYNTH_2p5m.tif") as dst:
        d = dst.read(1)
    w = d.shape[1]
    assert (d[:, : w // 2 - 8] > 0).all()
    assert (d[:, w // 2 + 8 :] == 0).all()


def test_cog_output(raw, tmp_path):
    out = enhance(params(tmp_path, cog=True))
    with rasterio.open(out / "enhanced_SYNTH_2p5m.tif") as dst:
        assert dst.profile["driver"] == "GTiff" and dst.tags(ns="IMAGE_STRUCTURE").get(
            "LAYOUT") == "COG"


def test_multispectral_end_to_end(tmp_path):
    make_rawdata(tmp_path / "raw", sensor="multispectral", size=32)
    out = enhance(params(tmp_path))
    with rasterio.open(next(out.glob("enhanced_*_2p5m.tif"))) as dst:
        assert dst.count == 10 and dst.shape == (128, 128)


# ---- model / device rules ----------------------------------------------------------
def test_full_model_on_cpu_exit_21(raw, tmp_path):
    with pytest.raises(SatEnhanceError) as e:
        enhance(params(tmp_path, family="full"))
    assert e.value.code == ExitCode.MODEL_UNSUPPORTED
    assert not (tmp_path / "out").exists()  # failed before doing any work


def test_full_model_on_cpu_exit_21_even_without_rawdata(tmp_path):
    (tmp_path / "raw").mkdir()
    with pytest.raises(SatEnhanceError) as e:
        enhance(params(tmp_path, family="full"))
    assert e.value.code == ExitCode.MODEL_UNSUPPORTED  # hardware check precedes input check


def test_cuda_requested_without_gpu_exit_21(raw, tmp_path):
    import torch
    if torch.cuda.is_available():
        pytest.skip("GPU present")
    with pytest.raises(SatEnhanceError) as e:
        enhance(params(tmp_path, device="cuda"))
    assert e.value.code == ExitCode.MODEL_UNSUPPORTED


def test_bad_params_exit_2(raw, tmp_path):
    for kw in (dict(reflectance="x"), dict(overlap=31), dict(block=64), dict(variant="nope")):
        with pytest.raises(SatEnhanceError) as e:
            enhance(params(tmp_path, **kw))
        assert e.value.code == ExitCode.INVALID_INPUT


# ---- inference tiling --------------------------------------------------------------
@pytest.mark.parametrize("shape", [(4, 128, 128), (4, 100, 100), (4, 130, 300), (4, 400, 257)])
def test_predict_square_shapes_and_no_holes(shape):
    """predict_large leaves holes for exactly-128 inputs and assumes squares; wrapper fixes it."""
    yy, xx = np.mgrid[0 : shape[1], 0 : shape[2]].astype("float32")
    x = np.broadcast_to(0.1 + 0.5 * (xx + yy) / (shape[1] + shape[2]), shape).copy()
    m = models.load_model("rgbn_x4", "lite", "cpu", "/tmp/x")
    y = _predict_square(m, x, 32)
    assert y.shape == (4, shape[1] * 4, shape[2] * 4)
    assert (y > 0).all()  # no unwritten zero margins


def test_blocked_inference_has_no_seams(tmp_path):
    """Smooth gradient through small blocks must equal a single-shot upsample (no block seams)."""
    size = 300
    make_rawdata(tmp_path / "raw", size=size)
    out_blocked = enhance(params(tmp_path, block=128, out_dir=tmp_path / "o1"))
    out_big = enhance(params(tmp_path, block=1024, out_dir=tmp_path / "o2"))
    with rasterio.open(out_blocked / "enhanced_SYNTH_2p5m.tif") as a, \
            rasterio.open(out_big / "enhanced_SYNTH_2p5m.tif") as b:
        da, db = a.read().astype("float32"), b.read().astype("float32")
    assert da.shape == (4, size * 4, size * 4)
    # The bicubic stub is local, so blocking must not change the interior at all: any seam at a
    # block border would show up as a difference here. Only the outermost 2 input px (8 output
    # px) differ, because bottom/right blocks are reflect-padded to a square before inference.
    assert np.abs(da - db)[:, 8:-8, 8:-8].max() <= 1
    assert np.abs(da - db).max() < 100
    assert np.abs(np.diff(da[0], axis=1)).max() < 60  # no jumps across block borders


def test_scaled_transform():
    from affine import Affine
    t = scaled_transform(Affine(10, 0, 100, 0, -10, 500), 4)
    assert (t.a, t.e, t.c, t.f) == (2.5, -2.5, 100, 500)


# ---- CLI ---------------------------------------------------------------------------
def test_cli_shorthand_and_exit_codes(raw, tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["satenhance-enhance", "--run-id", raw.name])
    r = CliRunner().invoke(app, ["run", "--rawdata", str(tmp_path / "raw"),
                                 "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "c")])
    assert r.exit_code == 0, r.output
    assert any(ln.startswith("SATENHANCE_OUTPUT=") for ln in r.stdout.splitlines())
    r = CliRunner().invoke(app, ["run", "--rawdata", str(tmp_path / "raw"), "--run-id", "nope",
                                 "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "c")])
    assert r.exit_code == 20
    r = CliRunner().invoke(app, ["run", "--rawdata", str(tmp_path / "raw"), "--model", "full",
                                 "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "c")])
    assert r.exit_code == 21


def test_report_records_mosaic_tiles_and_reads_manifest_1_1(raw, tmp_path):
    from satenhance_common import manifest as mfm
    man = mfm.load(raw)
    man.scene.mosaic = True
    man.scene.tiles = [mfm.TileInfo(id="T1", crs="EPSG:32750", coverage_pct=60.0),
                       mfm.TileInfo(id="T2", crs="EPSG:32750", coverage_pct=40.0)]
    mfm.save(man, raw)
    out = enhance(params(tmp_path))
    rep = json.loads((out / "enhance_report.json").read_text())
    assert rep["mosaic"] is True and rep["source_tiles"] == ["T1", "T2"]


@pytest.mark.parametrize("flavor,needle", [
    ("gpu", "no GPU is visible inside this GPU image"),
    ("cpu", "this is the CPU image"),
    ("", "no CUDA GPU is available"),
])
def test_no_gpu_message_says_why_for_each_image(monkeypatch, flavor, needle):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("SATENHANCE_IMAGE_FLAVOR", flavor)
    for kwargs in (dict(requested="cuda", family="lite"), dict(requested="auto", family="full")):
        with pytest.raises(SatEnhanceError) as e:
            models.resolve_device(**kwargs)
        assert e.value.code == ExitCode.MODEL_UNSUPPORTED and needle in e.value.message
