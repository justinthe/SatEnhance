"""Guard: the GPU image's runtime stage must carry what the builder stage had (FIX_PLAN_GPU_RUNTIME)."""
import re
from pathlib import Path

from satenhance_enhance import models

DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile.gpu"


def _stages():
    text = DOCKERFILE.read_text().replace("\\\n", " ")
    parts = re.split(r"(?m)^FROM ", text)[1:]
    return parts[0], parts[-1]


def test_runtime_stage_installs_python_build_bits():
    _, runtime = _stages()
    for pkg in ("python3-distutils", "python3.11-dev", "gcc"):
        assert re.search(rf"\b{re.escape(pkg)}\b", runtime), f"runtime stage lacks {pkg}"
    assert "check_gpu_runtime.py" in runtime
    assert "TRITON_CACHE_DIR" in runtime


def test_builder_really_upgrades_setuptools_and_pins_it():
    builder, _ = _stages()
    assert re.search(r'--upgrade\s+"?setuptools', builder)
    assert "setuptools" in re.search(r"grep -iE '([^']+)'", builder).group(1)


def test_distutils_error_gets_image_hint():
    exc = RuntimeError("Failed to load: No module named 'distutils.core'")
    assert models.missing_module(exc) == "distutils"
    assert models.missing_module(exc, full=True) == "distutils.core"
    hint = models.module_hint("distutils.core")
    assert "python3-distutils" in hint and "rebuild" in hint
