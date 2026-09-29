import zipfile

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

# ~2 km x 2 km square near Perth, WA
PERTH = Polygon([(115.85, -31.96), (115.87, -31.96), (115.87, -31.94), (115.85, -31.94)])


def _gdf(geom=PERTH, crs="EPSG:4326"):
    return gpd.GeoDataFrame({"name": ["a"]}, geometry=[geom], crs=crs)


@pytest.fixture
def perth():
    return PERTH


@pytest.fixture
def make_gdf():
    return _gdf


@pytest.fixture
def geojson_file(tmp_path):
    p = tmp_path / "aoi.geojson"
    _gdf().to_file(p, driver="GeoJSON")
    return p


@pytest.fixture
def kml_file(tmp_path):
    p = tmp_path / "aoi.kml"
    _gdf().to_file(p, driver="KML")
    return p


@pytest.fixture
def kmz_file(tmp_path, kml_file):
    p = tmp_path / "aoi.kmz"
    with zipfile.ZipFile(p, "w") as z:
        z.write(kml_file, "doc.kml")
    return p


@pytest.fixture
def shp_zip(tmp_path):
    d = tmp_path / "shp"
    d.mkdir()
    _gdf().to_file(d / "aoi.shp")
    p = tmp_path / "aoi_shp.zip"
    with zipfile.ZipFile(p, "w") as z:
        for f in d.iterdir():
            z.write(f, f.name)
    return p


@pytest.fixture
def gpkg_file(tmp_path):
    p = tmp_path / "aoi.gpkg"
    _gdf().to_file(p, driver="GPKG")
    return p


@pytest.fixture
def utm_geojson(tmp_path):
    p = tmp_path / "utm.gpkg"
    _gdf().to_crs("EPSG:32750").to_file(p, driver="GPKG")
    return p


@pytest.fixture
def no_crs_shp(tmp_path):
    d = tmp_path / "nocrs"
    d.mkdir()
    g = gpd.GeoDataFrame({"n": [1]}, geometry=[PERTH])
    g.to_file(d / "x.shp")
    (d / "x.prj").unlink(missing_ok=True)
    return d / "x.shp"


@pytest.fixture
def points_file(tmp_path):
    p = tmp_path / "pts.geojson"
    gpd.GeoDataFrame(geometry=[Point(115.86, -31.95)], crs="EPSG:4326").to_file(p, driver="GeoJSON")
    return p


@pytest.fixture
def lines_file(tmp_path):
    p = tmp_path / "lines.geojson"
    gpd.GeoDataFrame(
        geometry=[LineString([(115.85, -31.95), (115.87, -31.95)])], crs="EPSG:4326"
    ).to_file(p, driver="GeoJSON")
    return p


@pytest.fixture
def bowtie_file(tmp_path):
    p = tmp_path / "bow.geojson"
    bow = Polygon([(115.85, -31.96), (115.87, -31.94), (115.87, -31.96), (115.85, -31.94)])
    gpd.GeoDataFrame(geometry=[bow], crs="EPSG:4326").to_file(p, driver="GeoJSON")
    return p


@pytest.fixture
def mixed_file(tmp_path):
    p = tmp_path / "mixed.geojson"
    gpd.GeoDataFrame(geometry=[PERTH, Point(115.0, -31.0)], crs="EPSG:4326").to_file(
        p, driver="GeoJSON"
    )
    return p


@pytest.fixture
def two_polys_file(tmp_path):
    p = tmp_path / "two.geojson"
    a = PERTH
    b = Polygon([(115.90, -31.96), (115.92, -31.96), (115.92, -31.94), (115.90, -31.94)])
    gpd.GeoDataFrame(geometry=[a, b], crs="EPSG:4326").to_file(p, driver="GeoJSON")
    return p


@pytest.fixture
def garbage_file(tmp_path):
    p = tmp_path / "junk.shp"
    p.write_text("not a shapefile")
    return p
