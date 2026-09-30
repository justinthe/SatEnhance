import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from satenhance_enhance import models
from satenhance_enhance.download import (
    COMPLETE_MARKER,
    download_file,
    fetch_model,
    is_complete,
)
from satenhance_enhance.variants import VARIANTS

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


@pytest.fixture(autouse=True)
def _real_load_path(monkeypatch):
    """These tests exercise the real (non-stub) load path; conftest defaults to the stub."""
    monkeypatch.setenv("SATENHANCE_STUB_MODEL", "0")


class Server:
    """Tiny HTTP server with Range support and a knob to cut connections mid-file."""

    def __init__(self, files: dict[str, bytes], cut_first: dict[str, int] | None = None,
                 ignore_range: bool = False):
        self.files, self.cut_first, self.ignore_range = files, dict(cut_first or {}), ignore_range
        self.requests: list[tuple[str, str | None]] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_GET(self):
                data = outer.files.get(self.path)
                rng = self.headers.get("Range")
                outer.requests.append((self.path, rng))
                if data is None:
                    self.send_error(404)
                    return
                start = 0
                if rng and not outer.ignore_range:
                    start = int(rng.split("=")[1].rstrip("-"))
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(data)}")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                cut = outer.cut_first.get(self.path, 0)
                if cut and start == 0 or (cut and outer.cut_first.get(self.path)):
                    outer.cut_first[self.path] = 0
                    self.wfile.write(body[: max(cut, 1)])
                    self.wfile.flush()
                    self.connection.close()
                    return
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.httpd.shutdown()


def stac(base: str, names: list[str]) -> bytes:
    item = {
        "type": "Feature", "stac_version": "1.0.0", "id": "m", "geometry": None, "links": [],
        "properties": {"datetime": "2026-01-01T00:00:00Z"},
        "assets": {n: {"href": f"{base}/m/{n}"} for n in names},
    }
    return json.dumps(item).encode()


@pytest.fixture
def server():
    servers = []

    def make(**kw):
        files = {"/m/weights.bin": PAYLOAD, "/m/load.py": b"print('hi')\n"}
        s = Server(files, **kw)
        s.files["/m/mlm.json"] = stac(s.base, ["weights.bin", "load.py"])
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


nosleep = lambda s: None  # noqa: E731


def test_download_file_resumes_after_cut(server, tmp_path):
    s = server(cut_first={"/m/weights.bin": 300_000})
    import requests
    dest = tmp_path / "w.bin"
    download_file(f"{s.base}/m/weights.bin", dest, session=requests.Session(), sleep=nosleep)
    assert dest.read_bytes() == PAYLOAD
    ranges = [r for p, r in s.requests if p == "/m/weights.bin"]
    assert ranges[0] is None and ranges[1] and ranges[1].startswith("bytes=")  # resumed
    assert not (tmp_path / "w.bin.part").exists()


def test_download_file_server_ignores_range(server, tmp_path):
    s = server(cut_first={"/m/weights.bin": 200_000}, ignore_range=True)
    import requests
    dest = tmp_path / "w.bin"
    download_file(f"{s.base}/m/weights.bin", dest, session=requests.Session(), sleep=nosleep)
    assert dest.read_bytes() == PAYLOAD  # restarted cleanly, not appended


def test_download_gives_up_when_no_progress(tmp_path):
    import requests
    s = Server({})  # everything 404s
    try:
        with pytest.raises(SatEnhanceError) as e:
            download_file(f"{s.base}/nope", tmp_path / "x", session=requests.Session(),
                          attempts=2, sleep=nosleep)
        assert e.value.code == ExitCode.NETWORK_FAILURE
    finally:
        s.close()


def test_fetch_model_publishes_atomically_with_marker(server, tmp_path):
    s = server(cut_first={"/m/weights.bin": 100_000})
    target = tmp_path / "models" / "lite_rgbn"
    fetch_model(f"{s.base}/m/mlm.json", target, sleep=nosleep)
    assert is_complete(target) and (target / COMPLETE_MARKER).exists()
    assert (target / "weights.bin").read_bytes() == PAYLOAD
    assert (target / "mlm.json").exists() and (target / "load.py").exists()
    assert not (tmp_path / "models" / ".lite_rgbn.partial").exists()


def test_incomplete_cache_is_not_trusted_and_is_replaced(server, tmp_path):
    """The old bug: mlm.json present + truncated weights was treated as a finished download."""
    s = server()
    target = tmp_path / "models" / "lite_rgbn"
    target.mkdir(parents=True)
    (target / "mlm.json").write_text("{}")
    (target / "weights.bin").write_bytes(b"truncated")
    assert not is_complete(target)
    fetch_model(f"{s.base}/m/mlm.json", target, sleep=nosleep)
    assert (target / "weights.bin").read_bytes() == PAYLOAD


def test_failed_fetch_leaves_no_complete_cache_and_resumes_next_time(server, tmp_path):
    s = server()
    target = tmp_path / "models" / "m"
    s.files.pop("/m/load.py")  # second asset 404s
    with pytest.raises(SatEnhanceError):
        fetch_model(f"{s.base}/m/mlm.json", target, attempts=2, sleep=nosleep)
    assert not target.exists()
    partial = tmp_path / "models" / ".m.partial"
    assert (partial / "weights.bin").exists()  # finished file kept
    s.files["/m/load.py"] = b"ok"
    before = len([r for r in s.requests if r[0] == "/m/weights.bin"])
    fetch_model(f"{s.base}/m/mlm.json", target, sleep=nosleep)
    assert is_complete(target)
    assert len([r for r in s.requests if r[0] == "/m/weights.bin"]) == before  # not re-fetched


# ---- loading logic ---------------------------------------------------------------------
def _patch_fetch(monkeypatch, calls):
    def fake(url, target, **kw):
        calls.append(url)
        target.mkdir(parents=True, exist_ok=True)
        (target / COMPLETE_MARKER).write_text("{}")
        return target
    monkeypatch.setattr(models, "fetch_model", fake)


def test_missing_python_package_is_exit_21_and_named(tmp_path, monkeypatch):
    calls = []
    _patch_fetch(monkeypatch, calls)

    def boom(target, device):
        raise ModuleNotFoundError("No module named 'mamba_ssm'", name="mamba_ssm")
    monkeypatch.setattr(models, "_load_compiled", boom)
    with pytest.raises(SatEnhanceError) as e:
        models.load_model("rgbn_x4", "lite", "cpu", tmp_path)
    assert e.value.code == ExitCode.MODEL_UNSUPPORTED and "mamba_ssm" in e.value.message


def test_corrupt_cache_heals_once(tmp_path, monkeypatch):
    calls = []
    _patch_fetch(monkeypatch, calls)
    target = models.cache_dir_for(tmp_path, VARIANTS["rgbn_x4"], "lite")
    target.mkdir(parents=True)
    (target / COMPLETE_MARKER).write_text("{}")  # looks complete but is corrupt
    attempts = []

    def flaky(t, device):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("PytorchStreamReader failed reading zip archive")
        return object.__new__(type("M", (), {"eval": lambda self: None, "to": lambda self, d: self}))
    monkeypatch.setattr(models, "_load_compiled", flaky)
    m = models.load_model("rgbn_x4", "lite", "cpu", tmp_path)
    assert len(attempts) == 2 and len(calls) == 1 and m.family == "lite"


def test_fresh_download_that_will_not_load_is_exit_21_with_torch_version(tmp_path, monkeypatch):
    _patch_fetch(monkeypatch, [])

    def bad(t, d):
        raise RuntimeError("unsupported serialization version")
    monkeypatch.setattr(models, "_load_compiled", bad)
    with pytest.raises(SatEnhanceError) as e:
        models.load_model("rgbn_x4", "lite", "cpu", tmp_path)
    assert e.value.code == ExitCode.MODEL_UNSUPPORTED
    assert "torch" in e.value.message and "unsupported serialization" in e.value.message


def test_download_failure_is_exit_5(tmp_path, monkeypatch):
    def fail(url, target, **kw):
        raise SatEnhanceError(ExitCode.NETWORK_FAILURE, "no route")
    monkeypatch.setattr(models, "fetch_model", fail)
    with pytest.raises(SatEnhanceError) as e:
        models.load_model("rgbn_x4", "lite", "cpu", tmp_path)
    assert e.value.code == ExitCode.NETWORK_FAILURE


# ---- band-count validation -------------------------------------------------------------
def test_wrong_band_count_from_model_is_exit_22():
    import torch
    from satenhance_enhance.infer import _predict_square

    class Wrong(torch.nn.Module):
        def forward(self, x):  # returns 3 bands instead of 4
            return torch.nn.functional.interpolate(x[:, :3], scale_factor=4)

    m = models.LoadedModel(Wrong(), "cpu", "lite", VARIANTS["rgbn_x4"], "test")
    with pytest.raises(SatEnhanceError) as e:
        _predict_square(m, np.ones((4, 130, 130), dtype="float32"), 32)
    assert e.value.code == ExitCode.INFERENCE_FAILURE and "returned 3 bands" in e.value.message
