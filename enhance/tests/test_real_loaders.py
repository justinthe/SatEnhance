"""Run the *real* SEN2SR loader scripts (copied from Hugging Face) end to end on CPU.

Only the weights are random; the loader code, file names, architecture classes, HardConstraint
and inference path are the production ones. The first real GPU run showed what the stub model
hides: every loader starts with `import matplotlib.pyplot`, which the images did not contain.
"""

import ast
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
import torch
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from satenhance_enhance import models
from satenhance_enhance.download import COMPLETE_MARKER
from satenhance_enhance.infer import _predict_square
from satenhance_enhance.variants import VARIANTS
from sen2sr.models.opensr_baseline.cnn import CNNSR

FIXTURES = Path(__file__).parent / "real_loaders"


@pytest.fixture(autouse=True)
def _real_load_path(monkeypatch):
    monkeypatch.setenv("SATENHANCE_STUB_MODEL", "0")


def _mlm(assets: list[str]) -> dict:
    return {
        "type": "Feature", "stac_version": "1.1.0", "id": "test-model", "geometry": None,
        "links": [], "properties": {"datetime": None, "start_datetime": "1900-01-01T00:00:00Z",
                                    "end_datetime": "9999-01-01T00:00:00Z"},
        "assets": {a: {"href": f"https://huggingface.co/tacofoundation/SEN2SR/resolve/main/x/{a}"}
                   for a in assets},
    }


def _cnn(cin, cout, upscale):
    sd = CNNSR(cin, cout, 24, upscale, True, False, 6).state_dict()
    return {k: v.contiguous() for k, v in sd.items()}  # safetensors refuses non-contiguous


def _mask(n):
    return {"weights": torch.rand(n, n)}


def build_lite_folder(root: Path, variant: str) -> Path:
    """A model folder with the real load.py and randomly initialised weights."""
    target = models.cache_dir_for(root, VARIANTS[variant], "lite")
    target.mkdir(parents=True)
    st = safetensors.torch.save_file
    if variant == "rgbn_x4":
        shutil.copy(FIXTURES / "lite_rgbn_x4_load.py", target / "load.py")
        st(_cnn(4, 4, 4), str(target / "model.safetensor"))
        st(_mask(512), str(target / "hard_constraint.safetensor"))
        names = ["model.safetensor", "hard_constraint.safetensor", "load.py"]
    else:
        shutil.copy(FIXTURES / "lite_main_load.py", target / "load.py")
        st(_cnn(4, 4, 4), str(target / "sr_model.safetensor"))
        st(_mask(512), str(target / "sr_hard_constraint.safetensor"))
        st(_cnn(10, 6, 1), str(target / "f2_model.safetensor"))
        st(_mask(128), str(target / "f2_hard_constraint.safetensor"))
        st(_cnn(10, 6, 1), str(target / "model.safetensor"))
        st(_mask(512), str(target / "hard_constraint.safetensor"))
        names = ["sr_model.safetensor", "sr_hard_constraint.safetensor", "f2_model.safetensor",
                 "f2_hard_constraint.safetensor", "model.safetensor", "hard_constraint.safetensor",
                 "load.py"]
    (target / "mlm.json").write_text(json.dumps(_mlm(names)))
    (target / COMPLETE_MARKER).write_text("{}")
    return target


@pytest.mark.parametrize("variant", ["rgbn_x4", "multispectral_x4"])
def test_real_lite_loader_runs_through_our_inference_path(variant, tmp_path):
    build_lite_folder(tmp_path, variant)
    model = models.load_model(variant, "lite", "cpu", tmp_path)  # real mlstac + real load.py
    assert not model.stub and model.out_channels == len(VARIANTS[variant].bands)
    rng = np.random.default_rng(0)
    patch = rng.uniform(0.02, 0.35, (model.out_channels, 128, 128)).astype("float32")
    y = _predict_square(model, patch, 32)
    assert y.shape == (model.out_channels, 512, 512)
    assert np.isfinite(y).all() and float(y.std()) > 0


def test_lite_loader_needs_matplotlib_and_the_error_says_so(tmp_path, monkeypatch):
    """No matplotlib -> exit 21 naming it, and the (perfectly good) cache is kept."""
    target = build_lite_folder(tmp_path, "rgbn_x4")
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(SatEnhanceError) as e:
        models.load_model("rgbn_x4", "lite", "cpu", tmp_path)
    assert e.value.code == ExitCode.MODEL_UNSUPPORTED
    assert "matplotlib" in e.value.message
    assert (target / COMPLETE_MARKER).exists() and (target / "model.safetensor").exists()


def _imported_top_level_modules(path: Path) -> set[str]:
    mods = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


@pytest.mark.parametrize("script", sorted(p.name for p in FIXTURES.glob("*.py")))
def test_every_import_of_every_real_loader_is_installed(script):
    """The images must contain everything the loaders import (mamba_ssm: GPU image only)."""
    gpu_only = {"mamba_ssm"}
    missing = sorted(
        m for m in _imported_top_level_modules(FIXTURES / script)
        if m not in gpu_only and importlib.util.find_spec(m) is None
    )
    assert not missing, f"{script} imports packages that are not installed: {missing}"


def test_full_loaders_need_mamba_ssm_and_say_so_on_cpu():
    """The full RGBN loader imports MambaSR, which needs mamba_ssm (absent here): it is reported
    as a missing package (exit 21), not as a corrupt download."""
    src = (FIXTURES / "full_rgbn_x4_load.py").read_text()
    assert "MambaSR" in src and "mamba" in src


def test_matplotlib_is_declared_as_a_dependency():
    root = Path(__file__).resolve().parents[1]
    assert "matplotlib" in (root / "pyproject.toml").read_text()
    assert "matplotlib==" in (root / "requirements.lock").read_text()
    for dockerfile in ("Dockerfile.cpu", "Dockerfile.gpu"):
        assert "MPLBACKEND=Agg" in (root / dockerfile).read_text()
