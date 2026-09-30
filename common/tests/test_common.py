import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest
from satenhance_common import manifest as mf
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from satenhance_common.logging import RedactFilter
from satenhance_common.paths import make_run_id, read_latest, slugify, write_latest


def _manifest() -> mf.Manifest:
    return mf.Manifest(
        run_id="20260101T000000_x",
        created_utc="2026-01-01T00:00:00Z",
        query=mf.Query(aoi_source="file:x.geojson", start="2026-01-01", end="2026-01-31",
                       max_cloud=10, sensor="rgb"),
        aoi=mf.Aoi(bbox=[1, 2, 3, 4], area_km2=1.5),
        scene=mf.Scene(id="S2A_X", datetime="2026-01-05T00:00:00Z", aoi_cloud_fraction=1.0,
                       aoi_coverage=100.0,
                       bands={"B04": mf.BandInfo(file="S2_X/B04.tif", res_m=10)}),
    )


def test_schema_file_in_sync():
    on_disk = json.loads(
        (Path(mf.__file__).parent / "manifest.schema.json").read_text())
    assert on_disk == mf.json_schema()


def test_roundtrip(tmp_path):
    m = _manifest()
    mf.save(m, tmp_path)
    assert mf.load(tmp_path) == m


def test_load_missing_and_invalid(tmp_path):
    with pytest.raises(SatEnhanceError) as e:
        mf.load(tmp_path)
    assert e.value.code == ExitCode.ENHANCE_INPUT_INVALID
    (tmp_path / "manifest.json").write_text("{not json")
    with pytest.raises(SatEnhanceError):
        mf.load(tmp_path)
    (tmp_path / "manifest.json").write_text('{"run_id": "x"}')
    with pytest.raises(SatEnhanceError):
        mf.load(tmp_path)


def test_run_id_and_latest(tmp_path):
    rid = make_run_id("Perth City, WA", datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))
    assert rid == "20260102T030405_perth-city-wa"
    assert slugify("!!!") == "aoi"
    assert read_latest(tmp_path) is None
    write_latest(tmp_path, rid)
    assert read_latest(tmp_path) == rid


def test_redaction(monkeypatch):
    monkeypatch.setenv("CDSE_S3_SECRET_KEY", "supersecretvalue")
    rec = logging.LogRecord("x", logging.INFO, "", 0, "token=supersecretvalue", (), None)
    RedactFilter().filter(rec)
    assert "supersecretvalue" not in rec.getMessage()


def test_schema_1_0_manifest_still_loads(tmp_path):
    m = _manifest()
    data = json.loads(m.model_dump_json())
    data["schema_version"] = "1.0"
    del data["scene"]["tiles"], data["scene"]["mosaic"]  # a 1.0 manifest has neither
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    loaded = mf.load(tmp_path)
    assert loaded.scene.tiles == [] and loaded.scene.mosaic is False


def test_unknown_schema_version_rejected(tmp_path):
    data = json.loads(_manifest().model_dump_json())
    data["schema_version"] = "2.0"
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(SatEnhanceError) as e:
        mf.load(tmp_path)
    assert "2.0" in e.value.message


def test_tiles_roundtrip(tmp_path):
    m = _manifest()
    m.scene.tiles = [mf.TileInfo(id="T1", crs="EPSG:32750", resampled=False, coverage_pct=60.0),
                     mf.TileInfo(id="T2", crs="EPSG:32751", resampled=True, coverage_pct=40.0)]
    m.scene.mosaic = True
    mf.save(m, tmp_path)
    assert mf.load(tmp_path) == m


def test_error_report_written_and_redacted(tmp_path):
    from satenhance_common.report import FILENAME, write_error_report

    err = SatEnhanceError(ExitCode.NO_DATA, "nothing matched")
    err.context["run_dir"] = "/data/rawdata/x"
    path = write_error_report(tmp_path / "out", tool="acquire", exc=err, stage="search",
                              argv=["--start", "2026-01-01", "--secret-key=abc"])
    data = json.loads(path.read_text())
    assert path.name == FILENAME and data["exit_code"] == 10 and data["exit_name"] == "NO_DATA"
    assert data["stage"] == "search" and data["context"]["run_dir"] == "/data/rawdata/x"
    assert "abc" not in path.read_text() and "***" in data["argv"]
    assert "python" in data["versions"]


def test_error_report_for_unexpected_exception_and_unwritable_dir(tmp_path):
    from satenhance_common.report import write_error_report

    p = write_error_report(tmp_path, tool="enhance", exc=ValueError("boom"))
    assert json.loads(p.read_text())["exit_code"] == 1
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert write_error_report(blocker / "sub", tool="enhance", exc=ValueError("x")) is None
