import json
from datetime import date

import pystac
import pytest
from pystac_client.exceptions import APIError
from satenhance_acquire.aoi import from_geometry, write_aoi
from satenhance_acquire.cli import app
from satenhance_acquire.providers.cdse import CdseProvider, item_to_candidate
from satenhance_acquire.providers.fixture import build_fixture, fixture_aoi
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from typer.testing import CliRunner


def make_item(cloud=5.0, baseline="05.11"):
    return pystac.Item.from_dict({
        "type": "Feature", "stac_version": "1.0.0", "id": "S2X_TEST",
        "geometry": {"type": "Polygon", "coordinates": [[[115, -32], [116, -32], [116, -31], [115, -31], [115, -32]]]},
        "bbox": [115, -32, 116, -31], "links": [],
        "properties": {"datetime": "2026-01-05T02:00:00Z", "eo:cloud_cover": cloud,
                       "processing:version": baseline, "platform": "sentinel-2a"},
        "assets": {
            "B04_10m": {"href": "s3://eodata/x/B04_10m.jp2"},
            "B05_20m": {"href": "s3://eodata/x/B05_20m.jp2"},
            "SCL_20m": {"href": "s3://eodata/x/SCL_20m.jp2"},
        },
    })


class FakeSearch:
    def __init__(self, items): self._items = items
    def items(self): return iter(self._items)


class FakeClient:
    def __init__(self, items=(), fail_query=False, always_fail=False):
        self.items, self.fail_query, self.always_fail, self.calls = list(items), fail_query, always_fail, []

    def search(self, **kw):
        self.calls.append(kw)
        if self.always_fail or (self.fail_query and "query" in kw):
            err = APIError("boom")
            err.status_code = 400 if self.fail_query else 503
            raise err
        return FakeSearch(self.items)


ENV = {"CDSE_S3_ACCESS_KEY": "ak", "CDSE_S3_SECRET_KEY": "sk"}


def test_item_to_candidate_maps_assets():
    c = item_to_candidate(make_item())
    assert c.tile_cloud == 5.0 and c.processing_baseline == "05.11"
    assert set(c.assets) == {"B04", "B05", "SCL"}
    p = CdseProvider(client=FakeClient(), env=ENV)
    assert p.href(c, "B04") == "/vsis3/eodata/x/B04_10m.jp2"
    with pytest.raises(SatEnhanceError):
        p.href(c, "B02")


def test_gdal_env_and_auth():
    p = CdseProvider(client=FakeClient(), env=ENV)
    p.check_auth()
    env = p.gdal_env()
    assert env["AWS_S3_ENDPOINT"] == "eodata.dataspace.copernicus.eu"
    assert env["AWS_VIRTUAL_HOSTING"] == "FALSE" and env["AWS_ACCESS_KEY_ID"] == "ak"
    with pytest.raises(SatEnhanceError) as e:
        CdseProvider(client=FakeClient(), env={}).check_auth()
    assert e.value.code == ExitCode.AUTH_FAILURE


def test_search_query_and_client_side_filter():
    client = FakeClient([make_item(5), make_item(60)])
    p = CdseProvider(client=client, env=ENV)
    out = p.search(fixture_aoi(), date(2026, 1, 1), date(2026, 1, 31), 20)
    assert len(out) == 1 and out[0].tile_cloud == 5
    kw = client.calls[0]
    assert kw["collections"] == ["sentinel-2-l2a"] and kw["query"] == {"eo:cloud_cover": {"lte": 20}}
    assert kw["datetime"].startswith("2026-01-01") and kw["intersects"]["type"] == "Polygon"


def test_search_falls_back_when_query_unsupported():
    client = FakeClient([make_item(5), make_item(60)], fail_query=True)
    out = CdseProvider(client=client, env=ENV).search(fixture_aoi(), date(2026, 1, 1), date(2026, 1, 31), 20)
    assert [c.tile_cloud for c in out] == [5]
    assert "query" not in client.calls[-1]


def test_search_failure_is_network_error(monkeypatch):
    import tenacity
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda s: None)
    p = CdseProvider(client=FakeClient(always_fail=True), env=ENV)
    with pytest.raises(SatEnhanceError) as e:
        p.search(fixture_aoi(), date(2026, 1, 1), date(2026, 1, 31), 20)
    assert e.value.code == ExitCode.NETWORK_FAILURE


# ---- CLI ---------------------------------------------------------------------------
@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    fx = build_fixture(tmp_path / "fx", [
        {"id": "S2_A", "datetime": "2026-01-05T02:00:00Z", "tile_cloud": 5.0}])
    aoi = tmp_path / "site.geojson"
    write_aoi(from_geometry(fixture_aoi(), label="t"), aoi)
    monkeypatch.setenv("SATENHANCE_PROVIDER", "fixture")
    monkeypatch.setenv("SATENHANCE_FIXTURE_DIR", str(fx))
    return tmp_path, aoi


def invoke(tmp_path, aoi, *extra):
    return CliRunner().invoke(app, [
        "--aoi-file", str(aoi), "--out", str(tmp_path / "raw"), "--cache", str(tmp_path / "c"),
        "--non-interactive", *extra])


def test_cli_success_prints_run_id(cli_env):
    tmp_path, aoi = cli_env
    r = invoke(tmp_path, aoi, "--start", "2026-01-01", "--end", "2026-01-31")
    assert r.exit_code == 0, r.output
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("SATENHANCE_RUN_ID=")]
    assert line and (tmp_path / "raw" / line[-1].split("=", 1)[1] / "manifest.json").exists()


@pytest.mark.parametrize("extra,code", [
    (["--start", "2026-01-01", "--end", "2026-01-31", "--sensor", "lidar"], 2),
    (["--start", "nope", "--end", "2026-01-31"], 2),
    (["--start", "2026-02-01", "--end", "2026-02-05"], 10),
    (["--start", "2026-01-01", "--end", "2026-01-31", "--max-area-km2", "0.1"], 3),
])
def test_cli_exit_codes(cli_env, extra, code):
    tmp_path, aoi = cli_env
    assert invoke(tmp_path, aoi, *extra).exit_code == code


def test_cli_geocode_needs_confirmation(cli_env, monkeypatch):
    tmp_path, _ = cli_env
    from satenhance_acquire import geocode
    cache = tmp_path / "c" / "geocode"
    cache.mkdir(parents=True)
    (cache / (__import__("hashlib").sha1(b"perth").hexdigest() + ".json")).write_text(json.dumps([{
        "display_name": "Perth", "boundingbox": ["-32.0", "-31.9", "115.8", "115.9"]}]))
    assert geocode  # cache pre-seeded so no network is needed
    r = CliRunner().invoke(app, ["--aoi-text", "perth", "--start", "2026-01-01", "--end", "2026-01-31",
                                 "--out", str(tmp_path / "raw"), "--cache", str(tmp_path / "c"),
                                 "--non-interactive"])
    assert r.exit_code == 11


# ---- search paging / sorting (C4) and GDAL options (C2) -----------------------------------
class PagingClient(FakeClient):
    """Server that honours max_items but ignores sortby/query (date-ordered, like many STAC APIs)."""

    def search(self, **kw):
        self.calls.append(kw)
        return FakeSearch(self.items[: kw.get("max_items")])


def _items(n, best_index):
    out = []
    for i in range(n):
        it = make_item(cloud=3.0 if i == best_index else 40.0)
        it.id = f"S2_{i:03d}"
        out.append(it)
    return out


def test_search_reaches_best_scene_beyond_first_page():
    client = PagingClient(_items(120, best_index=110))
    out = CdseProvider(client=client, env=ENV).search(
        fixture_aoi(), date(2025, 1, 1), date(2026, 1, 1), 100
    )
    assert len(out) == 120 and min(c.tile_cloud for c in out) == 3.0
    kw = client.calls[0]
    assert kw["max_items"] == 500 and kw["limit"] == 100  # paged, not the old 50-item cut-off
    assert kw["sortby"] == [{"field": "properties.eo:cloud_cover", "direction": "asc"}]


def test_search_warns_when_cap_reached(caplog):
    import logging
    client = PagingClient(_items(600, best_index=0))
    with caplog.at_level(logging.WARNING):
        out = CdseProvider(client=client, env=ENV).search(
            fixture_aoi(), date(2025, 1, 1), date(2026, 1, 1), 100
        )
    assert len(out) == 500 and "cap" in caplog.text


def test_search_falls_back_step_by_step():
    """sortby unsupported -> retry with query only -> retry with a plain search."""
    class Picky(FakeClient):
        def search(self, **kw):
            self.calls.append(kw)
            if "sortby" in kw or "query" in kw:
                err = APIError("bad request")
                err.status_code = 400
                raise err
            return FakeSearch(self.items)

    client = Picky([make_item(5)])
    out = CdseProvider(client=client, env=ENV).search(
        fixture_aoi(), date(2026, 1, 1), date(2026, 1, 31), 20)
    assert len(out) == 1
    assert ["sortby" in c for c in client.calls] == [True, False, False]
    assert ["query" in c for c in client.calls] == [True, True, False]


def test_gdal_env_disables_directory_listing_and_sets_timeouts():
    env = CdseProvider(client=FakeClient(), env=ENV).gdal_env()
    assert env["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
    assert int(env["GDAL_HTTP_TIMEOUT"]) >= 60 and int(env["GDAL_HTTP_CONNECTTIMEOUT"]) > 0
