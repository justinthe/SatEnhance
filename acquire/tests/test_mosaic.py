"""Multi-tile AOIs: grouping, grid, per-pixel choice, reprojection, offset harmonisation."""

import json
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from satenhance_acquire import pipeline
from satenhance_acquire.aoi import from_geometry, write_aoi
from satenhance_acquire.mosaic import Acquisition, group_candidates, plan_grid
from satenhance_acquire.providers.base import Candidate
from satenhance_acquire.providers.fixture import FixtureProvider, build_fixture
from satenhance_common import manifest as mf
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from shapely.geometry import box, mapping
from shapely.ops import transform

WHEN = "2026-01-10T02:31:00Z"
ORIGIN_A = (400000.0, 6470000.0)
CRS = "EPSG:32750"


def tile(id_, origin, **kw):
    sc = {"id": id_, "datetime": WHEN, "tile_cloud": 5.0, "platform": "sentinel-2a",
          "relative_orbit": 74, "noise": 0, "origin": origin, "size_10m": 300}
    sc.update(kw)
    return sc


def aoi_utm(x0, y0, x1, y1, crs=CRS):
    """WGS84 polygon for a UTM box."""
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    return transform(tr, box(x0, y0, x1, y1))


def run(tmp_path, scenes, aoi, name="run", **kw):
    fx = build_fixture(tmp_path / f"fx_{name}", scenes)
    aoi_file = tmp_path / f"{name}.geojson"
    write_aoi(from_geometry(aoi, label="t"), aoi_file)
    input_fn = kw.pop("input_fn", None)
    p = pipeline.AcquireParams(
        start=date(2026, 1, 1), end=date(2026, 1, 31), max_cloud=kw.pop("max_cloud", 10),
        sensor=kw.pop("sensor", "rgb"), out_dir=tmp_path / f"raw_{name}",
        cache_dir=tmp_path / "cache", aoi_file=aoi_file, interactive=kw.pop("interactive", False),
        **kw,
    )
    return pipeline.acquire(p, FixtureProvider(fx), input_fn=input_fn)


def band(run_dir, man, name):
    with rasterio.open(run_dir / man.scene.bands[name].file) as d:
        return d.read(1), d.transform, d.crs


# AOI: 2 km wide box straddling the seam between tile A (x 400000-403000) and B (402500-405500)
STRADDLE = (401500, 6468500, 403500, 6469500)


def test_same_zone_mosaic_is_identical_to_one_big_tile(tmp_path):
    two = [tile("A", ORIGIN_A), tile("B", (402500.0, 6470000.0))]
    big = [tile("BIG", ORIGIN_A, size_10m=550)]
    aoi = aoi_utm(*STRADDLE)
    run_m = run(tmp_path, two, aoi, "mosaic")
    run_b = run(tmp_path, big, aoi, "big")
    man_m, man_b = mf.load(run_m), mf.load(run_b)

    assert man_m.scene.mosaic is True and man_b.scene.mosaic is False
    assert man_m.scene.id.startswith("MOSAIC_S2A_20260110T023100_R74_2T")
    assert man_m.schema_version == "1.1"
    assert sorted(t.id for t in man_m.scene.tiles) == ["A", "B"]
    assert all(not t.resampled for t in man_m.scene.tiles)  # same lattice: no resampling
    assert sum(t.coverage_pct for t in man_m.scene.tiles) == pytest.approx(100, abs=0.5)
    assert man_m.scene.aoi_coverage == pytest.approx(100.0)

    for b in ("B02", "B04", "B08", "SCL"):
        arr_m, t_m, crs_m = band(run_m, man_m, b)
        arr_b, t_b, crs_b = band(run_b, man_b, b)
        assert crs_m == crs_b and t_m == t_b
        assert np.array_equal(arr_m, arr_b), f"{b} differs from the single big tile"
        assert (arr_m > 0).all()  # no holes at the seam


def test_alignment_of_10m_and_20m_grids(tmp_path):
    aoi = aoi_utm(*STRADDLE)
    run_dir = run(tmp_path, [tile("A", ORIGIN_A), tile("B", (402500.0, 6470000.0))], aoi)
    man = mf.load(run_dir)
    b4, _, _ = band(run_dir, man, "B04")
    with rasterio.open(run_dir / man.scene.bands["B04"].file) as r10, \
            rasterio.open(run_dir / man.scene.bands["SCL"].file) as r20:
        assert r10.bounds == r20.bounds
        assert r10.width % 2 == 0 and r10.height % 2 == 0
        assert (r20.width, r20.height) == (r10.width // 2, r10.height // 2)
    assert b4.shape[1] >= 200  # ~2 km at 10 m


def test_no_group_when_passes_differ_gives_coverage_rejection(tmp_path):
    """Different relative orbits are different passes: never mosaicked together."""
    scenes = [tile("A", ORIGIN_A, relative_orbit=74), tile("B", (402500.0, 6470000.0), relative_orbit=31)]
    with pytest.raises(SatEnhanceError) as e:
        run(tmp_path, scenes, aoi_utm(*STRADDLE))
    assert e.value.code == ExitCode.NO_DATA
    report = json.loads(next((tmp_path / "raw_run").glob("*/no_data_report.json")).read_text())
    assert report["rejected_for_coverage"] == 2 and report["rejected_for_cloud"] == 0
    assert report["suggested_min_coverage"] == 75.0  # A alone covers 75% of the AOI
    assert "The problem is coverage" in report["message"]


def test_interactive_menu_can_accept_partial_coverage(tmp_path):
    scenes = [tile("A", ORIGIN_A, relative_orbit=74), tile("B", (402500.0, 6470000.0), relative_orbit=31)]
    answers = iter(["6"])
    run_dir = run(tmp_path, scenes, aoi_utm(*STRADDLE), interactive=True,
                  input_fn=lambda prompt: next(answers))
    man = mf.load(run_dir)
    assert man.scene.id == "A" and man.scene.aoi_coverage == pytest.approx(75.0, abs=1.0)


def test_overlap_prefers_the_clear_tile(tmp_path):
    # A is cloudy over its east 30% (x 402100-403000), which overlaps B's clear west edge
    scenes = [tile("A", ORIGIN_A, scl_cloud_frac=0.3, scl_cloud_side="right"),
              tile("B", (402500.0, 6470000.0))]
    run_dir = run(tmp_path, scenes, aoi_utm(*STRADDLE), max_cloud=50)
    man = mf.load(run_dir)
    scl, t, _ = band(run_dir, man, "SCL")
    x = t.c + (np.arange(scl.shape[1]) + 0.5) * 20  # pixel-centre easting
    overlap = (x >= 402500) & (x < 403000)
    assert (scl[:, overlap] == 4).all()  # clear pixels from B, not A's cloud
    only_a_cloudy = (x >= 402100) & (x < 402500)
    assert (scl[:, only_a_cloudy] == 9).all()  # only A can supply these, cloud stays
    assert 0 < man.scene.aoi_cloud_fraction < 30


def test_single_contributing_tile_is_an_ordinary_scene(tmp_path):
    aoi = aoi_utm(401000, 6468500, 402000, 6469500)  # entirely inside A (and not in B)
    run_dir = run(tmp_path, [tile("A", ORIGIN_A), tile("B", (402500.0, 6470000.0))], aoi)
    man = mf.load(run_dir)
    assert man.scene.id == "A" and man.scene.mosaic is False
    assert [t.id for t in man.scene.tiles] == ["A"]
    assert (run_dir / "S2_A").is_dir()


def test_zone_boundary_tiles_are_reprojected(tmp_path):
    t50 = Transformer.from_crs("EPSG:4326", "EPSG:32750", always_xy=True).transform
    t51 = Transformer.from_crs("EPSG:4326", "EPSG:32751", always_xy=True).transform
    a = tile("T_A50", t50(119.955, -31.90), crs="EPSG:32750", size_10m=400)
    b = tile("T_B51", t51(119.990, -31.90), crs="EPSG:32751", size_10m=400)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)  # noqa: F841
    from shapely.geometry import box as _box
    aoi = _box(119.975, -31.925, 120.015, -31.915)  # spans both tiles and the zone boundary
    run_dir = run(tmp_path, [a, b], aoi, max_cloud=50)
    man = mf.load(run_dir)

    assert man.scene.mosaic is True
    assert {t.crs for t in man.scene.tiles} == {"EPSG:32750", "EPSG:32751"}
    assert {t.id: t.resampled for t in man.scene.tiles} == {"T_A50": False, "T_B51": True}
    scl, t, crs = band(run_dir, man, "SCL")
    assert str(crs) == "EPSG:32750"  # the tie is broken by tile id: zone 50 lattice is the target
    b04, _, _ = band(run_dir, man, "B04")
    # inside the AOI everything must be filled
    aoi_utm50 = transform(Transformer.from_crs("EPSG:4326", "EPSG:32750", always_xy=True).transform, aoi)
    from rasterio.features import geometry_mask
    with rasterio.open(run_dir / man.scene.bands["B04"].file) as r:
        inside = ~geometry_mask([mapping(aoi_utm50)], out_shape=r.shape, transform=r.transform)
    assert (b04[inside] > 0).mean() > 0.999
    assert set(np.unique(scl)) <= {0, 4}  # nearest-neighbour: class codes stay valid classes
    assert man.scene.aoi_coverage > 99


def test_processing_baseline_offsets_are_harmonised(tmp_path):
    a = tile("A", ORIGIN_A, baseline="05.11")                          # DN offset convention
    b = tile("B", (402500.0, 6470000.0), baseline="03.01")            # older: no offset
    run_dir = run(tmp_path, [a, b], aoi_utm(*STRADDLE))
    man = mf.load(run_dir)
    assert man.scene.bands["B04"].offset == pytest.approx(-0.1)
    arr, t, _ = band(run_dir, man, "B04")
    fx = tmp_path / "fx_run"
    with rasterio.open(fx / "B_B04.tif") as rb:
        raw_b, tb = rb.read(1), rb.transform
    # columns east of A's edge (x >= 403000) can only come from B
    xs = t.c + (np.arange(arr.shape[1]) + 0.5) * 10
    cols_m = np.where(xs >= 403000)[0]
    cols_b = ((xs[cols_m] - tb.c) / 10).astype(int)
    rows_m = np.arange(arr.shape[0])
    rows_b = ((tb.f - (t.f - (rows_m + 0.5) * 10)) / 10).astype(int)
    assert np.array_equal(arr[np.ix_(rows_m, cols_m)], raw_b[np.ix_(rows_b, cols_b)] + 1000)
    # and A's own pixels are untouched
    with rasterio.open(fx / "A_B04.tif") as ra:
        raw_a, ta = ra.read(1), ra.transform
    cols_m = np.where(xs < 402500)[0]
    cols_a = ((xs[cols_m] - ta.c) / 10).astype(int)
    rows_a = ((ta.f - (t.f - (rows_m + 0.5) * 10)) / 10).astype(int)
    assert np.array_equal(arr[np.ix_(rows_m, cols_m)], raw_a[np.ix_(rows_a, cols_a)])


# ---- grouping unit tests ------------------------------------------------------------------
def cand(id_, when=WHEN, platform="sentinel-2a", orbit="74", footprint=None):
    return Candidate(id=id_, datetime=when, tile_cloud=1.0, processing_baseline="05.11",
                     assets={}, platform=platform, relative_orbit=orbit, footprint=footprint)


def test_grouping_rules():
    later = (datetime(2026, 1, 10, 2, 31, tzinfo=UTC) + timedelta(minutes=30)).isoformat()
    groups = group_candidates([
        cand("a"), cand("b", when="2026-01-10T02:31:07Z"),             # same pass
        cand("c", when=later),                                          # 30 min later: new group
        cand("d", platform="sentinel-2b"),                              # other satellite
        cand("e", orbit="31"),                                          # other orbit
    ])
    ids = sorted(sorted(t.id for t in g.tiles) for g in groups)
    assert ids == [["a", "b"], ["c"], ["d"], ["e"]]
    single = next(g for g in groups if [t.id for t in g.tiles] == ["c"])
    assert single.id == "c"  # a single tile keeps its own id
    assert next(g for g in groups if len(g.tiles) == 2).id.startswith("MOSAIC_S2A_")


def test_group_capped_at_four_keeping_tiles_that_cover_the_aoi():
    aoi = box(0, 0, 1, 1)
    near = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    far = {"type": "Polygon", "coordinates": [[[5, 5], [6, 5], [6, 6], [5, 6], [5, 5]]]}
    cands = [cand(f"far{i}", footprint=far) for i in range(4)] + [cand("near", footprint=near)]
    (g,) = group_candidates(cands, aoi)
    assert len(g.tiles) == 4 and "near" in [t.id for t in g.tiles]


def test_plan_grid_is_even_and_covers_the_aoi(tmp_path):
    fx = build_fixture(tmp_path / "fx", [tile("A", ORIGIN_A)])
    provider = FixtureProvider(fx)
    c = provider.search(None, date(2026, 1, 1), date(2026, 1, 31), 100)[0]
    aoi = aoi_utm(401234, 6468765, 402345, 6469876)
    grid = plan_grid(provider, aoi, [c])
    assert grid.width % 2 == 0 and grid.height % 2 == 0 and grid.res == 10
    minx, miny, maxx, maxy = transform(
        Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform, aoi).bounds
    left, top = grid.transform.c, grid.transform.f
    assert left <= minx and left + grid.width * 10 >= maxx
    assert top >= maxy and top - grid.height * 10 <= miny
    assert (left - ORIGIN_A[0]) % 20 == 0  # even offsets: the 20 m grid lines up
    g20 = grid.at(20)
    assert (g20.width, g20.height) == (grid.width // 2, grid.height // 2)
    assert Acquisition("x", [c]).is_mosaic is False
    assert mf.SCHEMA_VERSION == "1.1"
