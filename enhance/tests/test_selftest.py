import json
import sys

from PIL import Image
from satenhance_common.exit_codes import ExitCode
from satenhance_enhance import models, selftest
from satenhance_enhance.cli import app
from satenhance_enhance.synthetic import make_rawdata
from typer.testing import CliRunner


def test_selftest_passes_with_stub_and_prints_a_table(tmp_path):
    lines = []
    code = selftest.selftest("lite", "cpu", tmp_path, out=lines.append)
    text = "\n".join(lines)
    assert code == 0 and "SELFTEST OK" in text
    assert "PASS  rgbn_x4" in text and "PASS  multispectral_x4" in text and "STUB" in text


def test_selftest_reports_missing_package_with_exit_21(tmp_path, monkeypatch):
    monkeypatch.setenv("SATENHANCE_STUB_MODEL", "0")

    def fake_fetch(url, target, **kw):
        target.mkdir(parents=True, exist_ok=True)
        (target / ".complete").write_text("{}")
    monkeypatch.setattr(models, "fetch_model", fake_fetch)

    def boom(t, d):
        raise ModuleNotFoundError("No module named 'mamba_ssm'", name="mamba_ssm")
    monkeypatch.setattr(models, "_load_compiled", boom)
    lines = []
    code = selftest.selftest("lite", "cpu", tmp_path, ["rgbn_x4"], out=lines.append)
    assert code == int(ExitCode.MODEL_UNSUPPORTED)
    assert "FAIL" in "\n".join(lines) and "mamba_ssm" in "\n".join(lines)


def test_selftest_catches_a_model_with_wrong_output_shape(tmp_path, monkeypatch):
    import torch

    class Wrong(torch.nn.Module):
        def forward(self, x):  # x2 instead of x4, right bands
            return torch.nn.functional.interpolate(x, scale_factor=2)

    real = models.load_model

    def load(variant, family, device, cache):
        m = real(variant, family, device, cache)
        m.module = Wrong()
        return m
    monkeypatch.setattr(selftest, "load_model", load)
    r = selftest.selftest_variant("rgbn_x4", "lite", "cpu", tmp_path)
    assert not r.ok and r.code == int(ExitCode.INFERENCE_FAILURE)


def test_cli_selftest_exit_codes(tmp_path):
    ok = CliRunner().invoke(app, ["selftest", "--cache", str(tmp_path), "--device", "cpu"])
    assert ok.exit_code == 0 and "SELFTEST OK" in ok.output
    bad = CliRunner().invoke(app, ["selftest", "--cache", str(tmp_path), "--variant", "nope"])
    assert bad.exit_code == 2
    full = CliRunner().invoke(app, ["selftest", "--cache", str(tmp_path), "--model", "full",
                                    "--device", "cpu"])
    assert full.exit_code == 21  # full model needs a GPU


def test_prefetch_runs_the_selftest(tmp_path):
    r = CliRunner().invoke(app, ["prefetch", "--cache", str(tmp_path)])
    assert r.exit_code == 0 and "SELFTEST OK" in r.output
    r = CliRunner().invoke(app, ["prefetch", "--cache", str(tmp_path), "--no-selftest"])
    assert r.exit_code == 0 and "SELFTEST" not in r.output


def test_compare_reflectance_writes_side_by_side_png(tmp_path):
    raw = tmp_path / "raw"
    run = make_rawdata(raw, size=160)
    r = CliRunner().invoke(app, ["selftest", "--compare-reflectance", run.name, "--rawdata",
                                 str(raw), "--out", str(tmp_path / "out"), "--cache",
                                 str(tmp_path / "c"), "--device", "cpu"])
    assert r.exit_code == 0, r.output
    png = tmp_path / "out" / run.name / "reflectance_compare.png"
    assert png.exists()
    img = Image.open(png)
    assert img.width > 3 * 600 and img.height > 600  # 3 panels of 160 px x4 (+ gaps)
    assert "offset-corrected" in r.output and "raw-div10000" in r.output


def test_failed_run_writes_error_report(tmp_path):
    r = CliRunner().invoke(app, ["run", "--rawdata", str(tmp_path / "raw"), "--run-id", "nope",
                                 "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "c")])
    assert r.exit_code == 20
    data = json.loads((tmp_path / "out" / "nope" / "error_report.json").read_text())
    assert data["exit_code"] == 20 and data["tool"] == "satenhance-enhance"
    assert "Manifest not found" in data["message"]


def test_shorthand_dispatch(monkeypatch):
    from satenhance_enhance import cli
    monkeypatch.setattr(cli, "app", lambda: None)
    for first, expect in (("--run-id", "run"), ("selftest", "selftest"), ("prefetch", "prefetch")):
        monkeypatch.setattr(sys, "argv", ["satenhance-enhance", first])
        cli.main()
        assert sys.argv[1] == expect


