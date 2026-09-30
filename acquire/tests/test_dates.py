from datetime import UTC, date, datetime, timedelta

import pytest
from satenhance_acquire.aoi import from_geometry, write_aoi
from satenhance_acquire.cli import app
from satenhance_acquire.dates import DEFAULT_DAYS, parse_date, resolve_window
from satenhance_acquire.providers.fixture import build_fixture, fixture_aoi
from satenhance_common import manifest as mf
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from typer.testing import CliRunner

TODAY = date(2026, 9, 30)


def test_neither_given_is_last_30_days():
    assert resolve_window(None, None, today=TODAY) == (date(2026, 8, 31), TODAY, True)


def test_only_end():
    assert resolve_window(None, date(2026, 3, 31), today=TODAY) == (
        date(2026, 3, 1), date(2026, 3, 31), True)


def test_only_start_capped_at_today():
    assert resolve_window(date(2026, 1, 1), None, today=TODAY) == (
        date(2026, 1, 1), date(2026, 1, 31), True)
    assert resolve_window(date(2026, 9, 20), None, today=TODAY) == (
        date(2026, 9, 20), TODAY, True)


def test_both_given_unchanged_and_custom_days():
    assert resolve_window(date(2026, 1, 1), date(2026, 2, 1), today=TODAY) == (
        date(2026, 1, 1), date(2026, 2, 1), False)
    assert resolve_window(None, None, days=60, today=TODAY)[0] == date(2026, 8, 1)


@pytest.mark.parametrize("kw", [dict(days=0), dict(days=99999), dict(start=date(2027, 1, 1))])
def test_invalid(kw):
    kw.setdefault("start", None)
    with pytest.raises(SatEnhanceError) as e:
        resolve_window(kw.pop("start"), None, today=TODAY, **kw)
    assert e.value.code == ExitCode.INVALID_INPUT


def test_parse_date():
    assert parse_date("2026-01-05") == date(2026, 1, 5)
    with pytest.raises(SatEnhanceError):
        parse_date("05/01/2026")


def test_cli_without_dates_uses_last_30_days(tmp_path, monkeypatch):
    now = datetime.now(UTC)
    scenes = [
        {"id": "RECENT", "datetime": (now - timedelta(days=5)).strftime("%Y-%m-%dT02:00:00Z"),
         "tile_cloud": 3.0},
        {"id": "OLD", "datetime": (now - timedelta(days=90)).strftime("%Y-%m-%dT02:00:00Z"),
         "tile_cloud": 1.0},  # clearer, but outside the default window
    ]
    fx = build_fixture(tmp_path / "fx", scenes)
    aoi = tmp_path / "site.geojson"
    write_aoi(from_geometry(fixture_aoi(), label="t"), aoi)
    monkeypatch.setenv("SATENHANCE_PROVIDER", "fixture")
    monkeypatch.setenv("SATENHANCE_FIXTURE_DIR", str(fx))
    args = ["--aoi-file", str(aoi), "--out", str(tmp_path / "raw"), "--cache", str(tmp_path / "c"),
            "--non-interactive", "--max-cloud", "10"]
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    run_dir = next((tmp_path / "raw").glob("*_site"))
    man = mf.load(run_dir)
    assert man.scene.id == "RECENT"
    assert (date.fromisoformat(man.query.end) - date.fromisoformat(man.query.start)).days == DEFAULT_DAYS
    # a longer default window brings the old scene into consideration (it loses the tie on recency)
    r = CliRunner().invoke(app, [*args, "--days", "120"])
    assert r.exit_code == 0, r.output
    newest = mf.load(sorted((tmp_path / "raw").glob("*_site"))[-1])
    assert (date.fromisoformat(newest.query.end) - date.fromisoformat(newest.query.start)).days == 120
    assert "OLD" in [a.id for a in newest.alternates]
