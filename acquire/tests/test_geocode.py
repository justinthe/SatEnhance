
import pytest
import responses
from satenhance_acquire import geocode
from satenhance_common.exit_codes import ExitCode, SatEnhanceError

PERTH_RESPONSE = [
    {
        "display_name": "Perth, Western Australia, Australia",
        "category": "boundary", "type": "administrative",
        "boundingbox": ["-32.02", "-31.93", "115.84", "115.90"],
        "geojson": {"type": "Polygon", "coordinates": [[[115.84, -32.02], [115.90, -32.02],
                                                          [115.90, -31.93], [115.84, -31.93],
                                                          [115.84, -32.02]]]},
    },
    {
        "display_name": "Perth, Scotland, United Kingdom",
        "category": "place", "type": "city",
        "boundingbox": ["56.35", "56.45", "-3.50", "-3.35"],
        "geojson": {"type": "Point", "coordinates": [-3.43, 56.4]},
    },
]


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch):
    monkeypatch.setattr(geocode.time, "sleep", lambda s: None)


@responses.activate
def test_search_parses_and_caches(tmp_path):
    responses.add(responses.GET, geocode.NOMINATIM_URL, json=PERTH_RESPONSE)
    c = geocode.search("Perth City, Western Australia", cache_dir=tmp_path, user_agent="t/1")
    assert len(c) == 2 and c[0].has_boundary and not c[1].has_boundary
    assert c[1].geometry.geom_type == "Polygon"  # point falls back to bbox
    assert responses.calls[0].request.headers["User-Agent"] == "t/1"
    assert "polygon_geojson=1" in responses.calls[0].request.url
    geocode.search("perth city, western australia", cache_dir=tmp_path, user_agent="t/1")
    assert len(responses.calls) == 1  # served from cache


@responses.activate
def test_no_results_and_network_error(tmp_path):
    responses.add(responses.GET, geocode.NOMINATIM_URL, json=[])
    with pytest.raises(SatEnhanceError) as e:
        geocode.search("zzzz", cache_dir=tmp_path)
    assert e.value.code == ExitCode.INVALID_INPUT
    responses.replace(responses.GET, geocode.NOMINATIM_URL, status=500)
    with pytest.raises(SatEnhanceError) as e:
        geocode.search("qqqq", cache_dir=tmp_path)
    assert e.value.code == ExitCode.NETWORK_FAILURE


def _cands(tmp_path):
    (tmp_path / "geocode").mkdir()
    with responses.RequestsMock() as r:
        r.add(responses.GET, geocode.NOMINATIM_URL, json=PERTH_RESPONSE)
        return geocode.search("perth", cache_dir=tmp_path)


def test_choose_yes_and_noninteractive(tmp_path):
    c = _cands(tmp_path)
    assert geocode.choose(c, yes=True, interactive=False) is c[0]
    with pytest.raises(SatEnhanceError) as e:
        geocode.choose(c, yes=False, interactive=False)
    assert e.value.code == ExitCode.GEOCODE_NEEDS_CONFIRMATION


def test_choose_interactive(tmp_path):
    c = _cands(tmp_path)
    assert geocode.choose(c, yes=False, interactive=True, input_fn=lambda p: "") is c[0]
    answers = iter(["next", "y"])
    assert geocode.choose(c, yes=False, interactive=True, input_fn=lambda p: next(answers)) is c[1]
    with pytest.raises(SatEnhanceError):
        geocode.choose(c, yes=False, interactive=True, input_fn=lambda p: "n")
    with pytest.raises(SatEnhanceError):  # ran out of candidates
        geocode.choose(c, yes=False, interactive=True, input_fn=lambda p: "next")
