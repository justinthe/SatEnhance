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
