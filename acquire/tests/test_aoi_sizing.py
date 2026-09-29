import pytest

from satenhance_acquire import aoi, sizing
from satenhance_common.exit_codes import ExitCode, SatEnhanceError


@pytest.mark.parametrize(
    "fixture",
    ["geojson_file", "kml_file", "kmz_file", "shp_zip", "gpkg_file", "utm_geojson"],
)
def test_formats_load_and_match(fixture, request, perth):
    res = aoi.load_aoi(request.getfixturevalue(fixture))
    assert res.geometry.geom_type == "Polygon"
    assert res.geometry.symmetric_difference(perth).area < 1e-6
    assert res.feature_count == 1


def test_area_about_right(geojson_file):
    res = aoi.load_aoi(geojson_file)
    assert 4.0 < sizing.geodesic_area_km2(res.geometry) < 4.4  # ~1.89 x 2.22 km


@pytest.mark.parametrize("fixture", ["no_crs_shp", "points_file", "lines_file", "garbage_file"])
def test_bad_inputs_exit_2(fixture, request):
    with pytest.raises(SatEnhanceError) as e:
        aoi.load_aoi(request.getfixturevalue(fixture))
    assert e.value.code == ExitCode.INVALID_INPUT


def test_missing_file():
    with pytest.raises(SatEnhanceError) as e:
        aoi.load_aoi("/nope/x.geojson")
    assert e.value.code == ExitCode.INVALID_INPUT


def test_bowtie_repaired(bowtie_file):
    res = aoi.load_aoi(bowtie_file)
    assert res.geometry.is_valid and not res.geometry.is_empty


def test_mixed_drops_points(mixed_file):
    res = aoi.load_aoi(mixed_file)
    assert res.feature_count == 1


def test_multiple_features_dissolved(two_polys_file):
    res = aoi.load_aoi(two_polys_file)
    assert res.feature_count == 2
    assert res.geometry.geom_type == "MultiPolygon"


def test_write_aoi_roundtrip(geojson_file, tmp_path):
    res = aoi.load_aoi(geojson_file)
    out = tmp_path / "out.geojson"
    aoi.write_aoi(res, out)
    again = aoi.load_aoi(out)
    assert again.geometry.symmetric_difference(res.geometry).area < 1e-9


def test_size_cap(geojson_file):
    geom = aoi.load_aoi(geojson_file).geometry
    est = sizing.check_size(geom, "rgb", 100)
    assert est.output_mb == pytest.approx(est.rawdata_mb * 16)
    with pytest.raises(SatEnhanceError) as e:
        sizing.check_size(geom, "rgb", 1)
    assert e.value.code == ExitCode.AOI_TOO_LARGE
    assert "km2" in e.value.message
