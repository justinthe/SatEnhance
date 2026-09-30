"""The real S3 read path (`/vsis3/` through an AWSSession) against a local S3 server.

The first real run failed with `EnvError: GDAL's AWS config options can not be directly set`
because credentials were passed to rasterio as config options. No test opened a raster over S3
then; these do.
"""

import json
import socket
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import boto3
import numpy as np
import pystac
import pytest
import rasterio
from moto.server import ThreadedMotoServer
from satenhance_acquire import pipeline
from satenhance_acquire.aoi import from_geometry, write_aoi
from satenhance_acquire.cli import app
from satenhance_acquire.providers.cdse import BAND_RES, CdseProvider
from satenhance_acquire.providers.fixture import FixtureProvider, build_fixture, fixture_aoi
from satenhance_common import manifest as mf
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from typer.testing import CliRunner

WHEN = "2026-01-10T02:31:00Z"
KEYS = {"CDSE_S3_ACCESS_KEY": "test-access", "CDSE_S3_SECRET_KEY": "test-secret"}
SCENE = {"id": "S2_S3TEST", "datetime": WHEN, "tile_cloud": 4.0, "platform": "sentinel-2a",
         "relative_orbit": 74, "noise": 0}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class FakeClient:
    def __init__(self, items):
        self.items = items

    def search(self, **kw):
        return FakeSearch(self.items)


def stac_item(scene_id: str, prefix: str = "Sentinel-2/MSI/L2A") -> pystac.Item:
    assets = {}
    for band, res in BAND_RES.items():
        key = f"{band}_{res}m"
        assets[key] = {"href": f"s3://eodata/{prefix}/{scene_id}/{key}.tif"}
    return pystac.Item.from_dict({
        "type": "Feature", "stac_version": "1.0.0", "id": scene_id, "geometry": None,
        "links": [],
        "properties": {"datetime": WHEN, "eo:cloud_cover": 4.0, "processing:version": "05.11",
                       "platform": "sentinel-2a", "sat:relative_orbit": 74},
        "assets": assets,
    })


@pytest.fixture
def s3(tmp_path):
    """A local S3 server holding one scene under s3://eodata/, plus the same scene on disk."""
    port = _free_port()
    server = ThreadedMotoServer(port=port, verbose=False)
    server.start()
    endpoint = f"http://127.0.0.1:{port}"
    client = boto3.client("s3", endpoint_url=endpoint, aws_access_key_id="x",
                          aws_secret_access_key="y", region_name="us-east-1")
    client.create_bucket(Bucket="eodata")
    fx = build_fixture(tmp_path / "fx", [SCENE])
    for band, res in BAND_RES.items():
        client.upload_file(str(fx / f"{SCENE['id']}_{band}.tif"), "eodata",
                           f"Sentinel-2/MSI/L2A/{SCENE['id']}/{band}_{res}m.tif")
    env = {**KEYS, "CDSE_S3_ENDPOINT": endpoint}
    yield {"env": env, "endpoint": endpoint, "fixture": fx, "client": client, "port": port}
    server.stop()


def provider(s3, **overrides):
    return CdseProvider(client=FakeClient([stac_item(SCENE["id"])]), env={**s3["env"], **overrides})


def params(tmp_path, aoi_file, name):
    return pipeline.AcquireParams(
        start=date(2026, 1, 1), end=date(2026, 1, 31), max_cloud=10, sensor="rgb",
        out_dir=tmp_path / name, cache_dir=tmp_path / "cache", aoi_file=aoi_file,
        interactive=False)


@pytest.fixture
def aoi_file(tmp_path):
    p = tmp_path / "site.geojson"
    write_aoi(from_geometry(fixture_aoi(), label="t"), p)
    return p


def read(run_dir, man, band):
    with rasterio.open(run_dir / man.scene.bands[band].file) as d:
        return d.read(1), d.transform, d.crs


def test_download_through_vsis3_matches_the_local_scene(tmp_path, s3, aoi_file):
    run_s3 = pipeline.acquire(params(tmp_path, aoi_file, "raw_s3"), provider(s3))
    run_local = pipeline.acquire(params(tmp_path, aoi_file, "raw_local"),
                                 FixtureProvider(s3["fixture"]))
    man_s3, man_local = mf.load(run_s3), mf.load(run_local)
    assert man_s3.scene.id == SCENE["id"] and set(man_s3.scene.bands) == set(man_local.scene.bands)
    for band in man_local.scene.bands:
        a, ta, ca = read(run_s3, man_s3, band)
        b, tb, cb = read(run_local, man_local, band)
        assert ca == cb and ta == tb and np.array_equal(a, b), f"{band} differs over S3"
        assert (a > 0).all()


def test_credentials_never_reach_rasterio_as_config_options(tmp_path, s3, aoi_file, monkeypatch):
    seen = []
    real_env = rasterio.Env

    def spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real_env(*args, **kwargs)
    monkeypatch.setattr(rasterio, "Env", spy)
    pipeline.acquire(params(tmp_path, aoi_file, "raw"), provider(s3))
    assert seen, "provider never created a rasterio.Env"
    for kwargs in seen:
        assert "session" in kwargs  # credentials travel in an AWSSession
        flat = {k.upper() for k in kwargs}
        assert not flat & {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}
        assert kwargs["AWS_VIRTUAL_HOSTING"] == "FALSE"


def test_secret_never_appears_in_logs_or_reports(tmp_path, s3, aoi_file, caplog):
    import logging
    with caplog.at_level(logging.DEBUG):
        run_dir = pipeline.acquire(params(tmp_path, aoi_file, "raw"), provider(s3))
    blob = caplog.text + "".join(p.read_text() for p in run_dir.rglob("*.json"))
    assert "test-secret" not in blob and "test-access" not in blob


def test_endpoint_and_region_are_configurable(s3):
    p = provider(s3)
    assert p._endpoint() == (s3["endpoint"].replace("http://", ""), False)
    assert p.gdal_options()["AWS_HTTPS"] == "NO"
    default = CdseProvider(client=FakeClient([]), env=KEYS)
    assert default._endpoint() == ("eodata.dataspace.copernicus.eu", True)
    assert default.gdal_options()["AWS_HTTPS"] == "YES"


class _Forbidden(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _deny(self):
        body = (b'<?xml version="1.0"?><Error><Code>SignatureDoesNotMatch</Code>'
                b"<Message>The request signature we calculated does not match</Message></Error>")
        self.send_response(403)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_HEAD = _deny


def test_wrong_keys_give_exit_4_with_a_hint(tmp_path, aoi_file):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Forbidden)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        env = {**KEYS, "CDSE_S3_ENDPOINT": f"http://127.0.0.1:{httpd.server_address[1]}"}
        prov = CdseProvider(client=FakeClient([stac_item(SCENE["id"])]), env=env)
        with pytest.raises(SatEnhanceError) as e:
            pipeline.acquire(params(tmp_path, aoi_file, "raw"), prov)
        assert e.value.code == ExitCode.AUTH_FAILURE
        assert ".env" in e.value.message and "probe" in e.value.message
    finally:
        httpd.shutdown()


def test_missing_object_is_exit_5_and_names_the_path(tmp_path, s3, aoi_file):
    item = stac_item(SCENE["id"], prefix="Sentinel-2/MSI/L2A/WRONG-PREFIX")
    prov = CdseProvider(client=FakeClient([item]), env=s3["env"])
    with pytest.raises(SatEnhanceError) as e:
        pipeline.acquire(params(tmp_path, aoi_file, "raw"), prov)
    assert e.value.code == ExitCode.NETWORK_FAILURE
    assert "WRONG-PREFIX" in e.value.message and "not in the bucket" in e.value.message


def test_unreachable_endpoint_is_exit_5_not_unexpected(tmp_path, aoi_file):
    env = {**KEYS, "CDSE_S3_ENDPOINT": f"http://127.0.0.1:{_free_port()}"}  # nothing listens
    prov = CdseProvider(client=FakeClient([stac_item(SCENE["id"])]), env=env)
    with pytest.raises(SatEnhanceError) as e:
        pipeline.acquire(params(tmp_path, aoi_file, "raw"), prov)
    assert e.value.code == ExitCode.NETWORK_FAILURE


def test_probe_reads_pixels_over_s3(tmp_path, s3, aoi_file, monkeypatch):
    from satenhance_acquire import cli
    monkeypatch.setattr(cli, "get_provider", lambda: provider(s3))
    r = CliRunner().invoke(app, ["probe", "--aoi-file", str(aoi_file), "--start", "2026-01-01",
                                 "--end", "2026-01-31", "--cache", str(tmp_path / "c")])
    assert r.exit_code == 0, r.output
    assert "reading real pixels" in r.output and "PROBE OK" in r.output
    assert "s3://eodata" in r.output or "/vsis3/eodata" in r.output


def test_run_over_s3_via_cli_writes_report_in_the_run_folder_on_failure(tmp_path, aoi_file, monkeypatch):
    """A failure deep in the S3 path must be reported next to the run, not as 'unexpected'."""
    from satenhance_acquire import cli
    env = {**KEYS, "CDSE_S3_ENDPOINT": f"http://127.0.0.1:{_free_port()}"}
    monkeypatch.setattr(cli, "get_provider",
                        lambda: CdseProvider(client=FakeClient([stac_item(SCENE["id"])]), env=env))
    out = tmp_path / "raw"
    r = CliRunner().invoke(app, ["run", "--aoi-file", str(aoi_file), "--start", "2026-01-01",
                                 "--end", "2026-01-31", "--out", str(out),
                                 "--cache", str(tmp_path / "c"), "--non-interactive"])
    assert r.exit_code == 5, r.output
    run_dir = next(out.glob("*_site"))
    data = json.loads((run_dir / "error_report.json").read_text())
    assert data["exit_name"] == "NETWORK_FAILURE"


# ---- reports for unexpected failures, sizes, container paths -----------------------------------
def test_unexpected_exception_still_reports_in_the_run_folder_with_traceback(tmp_path, aoi_file,
                                                                             monkeypatch):
    from satenhance_acquire import cli
    from satenhance_acquire import pipeline as pl
    monkeypatch.setenv("SATENHANCE_PROVIDER", "fixture")
    fx = build_fixture(tmp_path / "fx", [SCENE])
    monkeypatch.setenv("SATENHANCE_FIXTURE_DIR", str(fx))

    def boom(*a, **k):
        raise ValueError("kaboom")
    monkeypatch.setattr(pl, "download_scene", boom)
    out = tmp_path / "raw"
    r = CliRunner().invoke(app, ["run", "--aoi-file", str(aoi_file), "--start", "2026-01-01",
                                 "--end", "2026-01-31", "--out", str(out),
                                 "--cache", str(tmp_path / "c"), "--non-interactive"])
    assert r.exit_code == 1 and cli  # unexpected -> 1
    run_dir = next(out.glob("*_site"))
    data = json.loads((run_dir / "error_report.json").read_text())  # in the run folder, not raw/
    assert "ValueError: kaboom" in data["message"] and "kaboom" in data["traceback"]
    assert not (out / "error_report.json").exists()


def test_host_path_translates_container_paths():
    from satenhance_common.report import host_path
    assert host_path("/data/rawdata/x/error_report.json").startswith("./rawdata/x/error_report.json")
    assert host_path("/elsewhere/file") == "/elsewhere/file"


@pytest.mark.parametrize("mb,text", [(0.0, "0 KB"), (0.25, "256 KB"), (5.4, "5 MB"),
                                     (1234, "1.2 GB")])
def test_format_size(mb, text):
    from satenhance_acquire.sizing import format_size
    assert format_size(mb) == text
