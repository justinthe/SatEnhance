import json
from datetime import date

import pytest
import rasterio
from satenhance_acquire import nodata, pipeline
from satenhance_acquire.aoi import from_geometry, write_aoi
from satenhance_acquire.download import reflectance_offset
from satenhance_acquire.providers.fixture import FixtureProvider, build_fixture, fixture_aoi
from satenhance_acquire.select import select_best
from satenhance_common import manifest as mf
from satenhance_common.exit_codes import ExitCode, SatEnhanceError

SCENES = [
    {"id": "S2_A", "datetime": "2026-01-05T02:00:00Z", "tile_cloud": 5.0},
    {"id": "S2_B", "datetime": "2026-01-15T02:00:00Z", "tile_cloud": 5.0},  # same cloud, later
    {"id": "S2_C", "datetime": "2026-01-20T02:00:00Z", "tile_cloud": 3.0, "scl_cloud_frac": 0.9},
    {"id": "S2_D", "datetime": "2026-03-10T02:00:00Z", "tile_cloud": 8.0},  # outside Jan range
]


@pytest.fixture
def provider(tmp_path):
    return FixtureProvider(build_fixture(tmp_path / "fx", SCENES))


@pytest.fixture
def aoi_file(tmp_path):
    p = tmp_path / "site.geojson"
    write_aoi(from_geometry(fixture_aoi(), label="t"), p)
    return p


def params(tmp_path, aoi_file_, **kw):
    base = dict(
        start=date(2026, 1, 1), end=date(2026, 1, 31), max_cloud=10, sensor="rgb",
        out_dir=tmp_path / "rawdata", cache_dir=tmp_path / "cache", aoi_file=aoi_file_,
        interactive=False,
    )
    base.update(kw)
    return pipeline.AcquireParams(**base)


def test_select_prefers_aoi_local_cloud_then_recency(provider):
    aoi = fixture_aoi()
    cands = provider.search(aoi, date(2026, 1, 1), date(2026, 1, 31), 100)
    best, assessments, passing = select_best(
        provider, aoi, cands, max_cloud=10, min_coverage=95
    )
    assert best.id == "S2_B"  # A and B tie at 0% cloud; B is more recent
    by = {a.id: a for a in assessments}
    assert by["S2_C"].status == "rejected" and "cloud" in by["S2_C"].reason
    assert by["S2_A"].aoi_coverage == pytest.approx(100.0)


def test_full_acquire_writes_manifest_and_bands(tmp_path, provider, aoi_file):
    run_dir = pipeline.acquire(params(tmp_path, aoi_file), provider)
    man = mf.load(run_dir)
    assert man.scene.id == "S2_B" and man.query.sensor == "rgb"
    assert set(man.scene.bands) == {"B02", "B03", "B04", "B08", "SCL"}
    assert man.scene.bands["B04"].offset == pytest.approx(-0.1)
    assert man.scene.bands["SCL"].offset == 0
    assert (run_dir / "aoi.geojson").exists() and (run_dir / "search_results.json").exists()
    assert (tmp_path / "rawdata" / "LATEST").read_text().strip() == run_dir.name
    assert not list(run_dir.rglob("*.part"))
    with rasterio.open(run_dir / man.scene.bands["B04"].file) as b4, \
            rasterio.open(run_dir / man.scene.bands["SCL"].file) as scl:
        assert b4.res == (10.0, 10.0) and scl.res == (20.0, 20.0)
        assert b4.bounds == scl.bounds  # windows aligned across resolutions
        assert b4.width % 2 == 0 and b4.height % 2 == 0
        assert 150 <= b4.width <= 250  # roughly the central half of the 400 px scene


def test_multispectral_bands_and_resume(tmp_path, provider, aoi_file):
    run_dir = pipeline.acquire(params(tmp_path, aoi_file, sensor="multispectral"), provider)
    man = mf.load(run_dir)
    assert len(man.scene.bands) == 11
    scene_dir = run_dir / f"S2_{man.scene.id}"
    f = run_dir / man.scene.bands["B02"].file
    mtime = f.stat().st_mtime_ns
    from satenhance_acquire.download import download_scene
    cand = next(c for c in provider.search(fixture_aoi(), date(2026, 1, 1), date(2026, 1, 31), 100)
                if c.id == man.scene.id)
    from satenhance_acquire.mosaic import group_candidates
    download_scene(provider, group_candidates([cand])[0], fixture_aoi(), "multispectral", scene_dir)
    assert f.stat().st_mtime_ns == mtime  # complete files are skipped


@pytest.mark.parametrize("sensor", ["sar", "lidar", "hyperspectral", "banana"])
def test_unsupported_sensors_exit_2(sensor, tmp_path, provider, aoi_file):
    with pytest.raises(SatEnhanceError) as e:
        pipeline.acquire(params(tmp_path, aoi_file, sensor=sensor), provider)
    assert e.value.code == ExitCode.INVALID_INPUT


@pytest.mark.parametrize(
    "kw",
    [dict(start=date(2026, 2, 1)), dict(end=date(2999, 1, 1)), dict(max_cloud=101),
     dict(aoi_text="x")],
)
def test_validation(kw, tmp_path, provider, aoi_file):
    with pytest.raises(SatEnhanceError) as e:
        pipeline.acquire(params(tmp_path, aoi_file, **kw), provider)
    assert e.value.code == ExitCode.INVALID_INPUT


def test_area_cap(tmp_path, provider, aoi_file):
    with pytest.raises(SatEnhanceError) as e:
        pipeline.acquire(params(tmp_path, aoi_file, max_area_km2=0.5), provider)
    assert e.value.code == ExitCode.AOI_TOO_LARGE


def test_no_credentials_exit_4(tmp_path, aoi_file, monkeypatch):
    from satenhance_acquire.providers.cdse import CdseProvider
    monkeypatch.delenv("CDSE_S3_ACCESS_KEY", raising=False)
    with pytest.raises(SatEnhanceError) as e:
        pipeline.acquire(params(tmp_path, aoi_file), CdseProvider(client=object(), env={}))
    assert e.value.code == ExitCode.AUTH_FAILURE


def test_no_data_non_interactive_exit_10_with_report(tmp_path, provider, aoi_file):
    # 3% cloud allowed: A/B are 0% AOI cloud so use a window with only the cloudy scene
    p = params(tmp_path, aoi_file, start=date(2026, 1, 18), end=date(2026, 1, 25), max_cloud=10)
    with pytest.raises(SatEnhanceError) as e:
        pipeline.acquire(p, provider)
    assert e.value.code == ExitCode.NO_DATA
    report = json.loads(next((tmp_path / "rawdata").glob("*/no_data_report.json")).read_text())
    assert report["suggested_max_cloud"] is not None and report["suggested_max_cloud"] > 10
    assert report["nearest"] and report["query"]["max_cloud"] == 10


def test_no_data_empty_window_suggests_dates(tmp_path, provider, aoi_file):
    p = params(tmp_path, aoi_file, start=date(2026, 2, 1), end=date(2026, 2, 10))
    with pytest.raises(SatEnhanceError) as e:
        pipeline.acquire(p, provider)
    assert e.value.code == ExitCode.NO_DATA
    report = json.loads(next((tmp_path / "rawdata").glob("*/no_data_report.json")).read_text())
    assert report["suggested_start"] and report["suggested_end"]
    assert report["suggested_start"] <= "2026-02-01" and report["suggested_end"] >= "2026-02-10"


def test_no_data_interactive_retry_succeeds(tmp_path, provider, aoi_file):
    p = params(tmp_path, aoi_file, start=date(2026, 1, 18), end=date(2026, 1, 25),
               interactive=True)
    answers = iter(["9", "1"])  # invalid then "raise cloud"
    run_dir = pipeline.acquire(p, provider, input_fn=lambda prompt: next(answers))
    assert mf.load(run_dir).scene.id == "S2_C"
    assert mf.load(run_dir).query.max_cloud > 10


def test_no_data_interactive_quit(tmp_path, provider, aoi_file):
    p = params(tmp_path, aoi_file, start=date(2026, 2, 1), end=date(2026, 2, 10), interactive=True)
    with pytest.raises(SatEnhanceError) as e:
        pipeline.acquire(p, provider, input_fn=lambda prompt: "5")
    assert e.value.code == ExitCode.NO_DATA


def test_retry_menu_manual_entry():
    diag = nodata.Diagnosis(query={"start": "2026-01-01", "end": "2026-01-31", "max_cloud": 5},
                            searched_candidates=0)
    answers = iter(["4", "abc", "4", "30", "2026-01-02", "2026-01-20", "80"])
    c, s, e, mc = nodata.retry_menu(diag, date(2026, 1, 1), date(2026, 1, 31), 5, 95, "x",
                                    input_fn=lambda p: next(answers), today=date(2026, 6, 1))
    assert (c, s, e, mc) == (30.0, date(2026, 1, 2), date(2026, 1, 20), 80.0)


@pytest.mark.parametrize(
    "baseline,when,expected",
    [("05.11", "2026-01-01T00:00:00Z", -0.1), ("03.01", "2021-01-01T00:00:00Z", 0.0),
     (None, "2023-05-01T00:00:00Z", -0.1), (None, "2021-05-01T00:00:00Z", 0.0)],
)
def test_reflectance_offset(baseline, when, expected):
    assert reflectance_offset(baseline, when) == expected
