"""Model registry, device selection and loading (PRD sections 9.2 and 9.4)."""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from satenhance_common.exit_codes import ExitCode, SatEnhanceError

from .download import fetch_model, is_complete
from .variants import VARIANTS, Variant

log = logging.getLogger(__name__)

STUB_ENV = "SATENHANCE_STUB_MODEL"


@dataclass
class LoadedModel:
    module: object  # torch.nn.Module
    device: str
    family: str  # lite | full
    variant: Variant
    source: str
    stub: bool = False
    dtype: str = "float32"  # dtype of the model's parameters (inputs are cast to match)

    @property
    def scale(self) -> int:
        return self.variant.scale

    @property
    def out_channels(self) -> int:
        return len(self.variant.bands)


def stub_enabled() -> bool:
    return os.environ.get(STUB_ENV) == "1"


def _no_gpu_message(what: str) -> str:
    """Say *why* there is no GPU, using which image we are in."""
    flavor = os.environ.get("SATENHANCE_IMAGE_FLAVOR", "")
    if flavor == "gpu":
        return (
            f"{what}, but no GPU is visible inside this GPU image. Check that the NVIDIA "
            "Container Toolkit is installed and Docker was restarted, and run ./scripts/doctor.sh "
            "(it tests the GPU inside the image)."
        )
    if flavor == "cpu":
        return (
            f"{what}, but this is the CPU image (CPU-only PyTorch). Build and use the GPU image: "
            "./scripts/build.sh --gpu-only, then ./scripts/run_system2.sh --gpu ..."
        )
    return f"{what}, but no CUDA GPU is available"


def resolve_device(requested: str, family: str) -> str:
    """auto -> cuda if available. Never silently downgrade a requested model (PRD 9.4)."""
    import torch

    has_cuda = torch.cuda.is_available()
    if requested == "cuda" and not has_cuda:
        raise SatEnhanceError(ExitCode.MODEL_UNSUPPORTED, _no_gpu_message("--device cuda requested"))
    device = requested if requested != "auto" else ("cuda" if has_cuda else "cpu")
    if family == "full" and device != "cuda":
        raise SatEnhanceError(
            ExitCode.MODEL_UNSUPPORTED,
            _no_gpu_message("The full SEN2SR (Mamba) model needs an NVIDIA GPU")
            + " Or use --model lite, which runs on CPU.",
        )
    return device


def model_url(variant: Variant, family: str) -> str:
    return variant.url(family)


def _stub_module(scale: int):
    import torch

    class BicubicStub(torch.nn.Module):
        """Test-only stand-in: bicubic upsampling. NOT super-resolution."""

        def forward(self, x):
            return torch.nn.functional.interpolate(
                x, scale_factor=scale, mode="bicubic", align_corners=False
            ).clamp_min(0)

    return BicubicStub()


def cache_dir_for(cache_root: Path, variant: Variant, family: str) -> Path:
    return Path(cache_root) / "models" / f"{family}_{variant.name}"


def load_model(variant_name: str, family: str, device_req: str, cache_root: Path) -> LoadedModel:
    variant = VARIANTS[variant_name]
    device = resolve_device(device_req, family)  # fail before any download
    url = model_url(variant, family)
    if stub_enabled():
        log.warning("%s=1: using bicubic STUB model, output is NOT super-resolved", STUB_ENV)
        module = _stub_module(variant.scale).to(device)
        return LoadedModel(module, device, family, variant, "stub:bicubic", stub=True)

    target = cache_dir_for(cache_root, variant, family)
    module = _load_from_cache(url, target, device, variant, family)
    module.eval()
    return LoadedModel(module, device, family, variant, url, dtype=_param_dtype(module))


_MISSING_RE = re.compile(r"No module named '([\w.]+)'|import of ([\w.]+) halted")


def missing_module(exc: BaseException | None, full: bool = False) -> str | None:
    """Name of the missing Python package behind an exception, if there is one.

    mlstac runs the model's load.py and re-raises whatever it hits as RuntimeError("Failed to
    load Python module ...: No module named 'x'"), so the ModuleNotFoundError is only in the
    exception chain (and in the text). Look in both.
    """
    seen = 0
    while exc is not None and seen < 10:
        if isinstance(exc, ModuleNotFoundError) and exc.name:
            return exc.name if full else exc.name.split(".")[0]
        m = _MISSING_RE.search(str(exc))
        if m:
            name = m.group(1) or m.group(2)
            return name if full else name.split(".")[0]
        exc = exc.__cause__ or exc.__context__
        seen += 1
    return None


def module_hint(mod: str) -> str:
    pkg = mod.split(".")[0]
    if pkg in ("distutils", "setuptools"):
        return ("The image's Python is incomplete (python3-distutils / setuptools); rebuild the "
                "GPU image: ./scripts/build.sh --gpu-only")
    if pkg == "mamba_ssm":
        return "Use the GPU image (./scripts/build.sh --gpu-only): mamba_ssm is installed there."
    return "Rebuild the image with that package added (see docs/HOWTO.md, Troubleshooting)."


def _traceback_tail(exc: BaseException, n: int = 6) -> str:
    import traceback

    lines = [ln for ln in "".join(traceback.format_exception(exc)).splitlines() if ln.strip()]
    return "\nImport trace (last lines):\n  " + "\n  ".join(lines[-n:])


def _param_dtype(module) -> str:
    try:
        return str(next(module.parameters()).dtype).replace("torch.", "")
    except (StopIteration, AttributeError):
        return "float32"


def _torch_version() -> str:
    import torch

    return torch.__version__


def _load_compiled(target: Path, device: str):
    import mlstac

    return mlstac.load(str(target)).compiled_model(device=device).to(device)


def _load_from_cache(url: str, target: Path, device: str, variant: Variant, family: str):
    """Download if the cache is missing/incomplete, load, and self-heal once if loading fails."""
    downloaded_now = False
    for attempt in (1, 2):
        if not is_complete(target):
            log.info("Downloading model %s -> %s", url, target)
            fetch_model(url, target)  # raises SatEnhanceError(NETWORK_FAILURE)
            downloaded_now = True
        try:
            return _load_compiled(target, device)
        except Exception as e:  # noqa: BLE001  (torch/mlstac raise many types)
            mod = missing_module(e, full=True)
            if mod:
                # Not a corrupt download: keep the cache, name the module.
                err = SatEnhanceError(
                    ExitCode.MODEL_UNSUPPORTED,
                    f"The {family} model '{variant.name}' needs the Python module '{mod}', which "
                    f"is not available in this image. {module_hint(mod)}{_traceback_tail(e)}",
                )
                err.context["module"] = mod
                raise err from e
            if attempt == 1 and not downloaded_now:
                log.warning("Cached model failed to load (%s); re-downloading once", e)
                shutil.rmtree(target, ignore_errors=True)
                continue
            extra = ""
            if isinstance(e.__cause__, ImportError) or "cannot import name" in str(e):
                extra = (" The installed sen2sr package may not match this model; the image pins "
                         "sen2sr in enhance/requirements.lock.")
            raise SatEnhanceError(
                ExitCode.MODEL_UNSUPPORTED,
                f"The {family} model '{variant.name}' downloaded fine but could not be loaded "
                f"with torch {_torch_version()} on {device}: {e}. The model files may need a "
                f"different torch version (see docs/HOWTO.md, Troubleshooting).{extra}",
            ) from e
    raise AssertionError("unreachable")


def prefetch(variants: list[str], family: str, cache_root: Path) -> list[str]:
    done = []
    for v in variants:
        load_model(v, family, "cpu" if family == "lite" else "auto", cache_root)
        done.append(v)
    return done
