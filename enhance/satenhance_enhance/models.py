"""Model registry, device selection and loading (PRD sections 9.2 and 9.4)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from satenhance_common.exit_codes import ExitCode, SatEnhanceError

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

    @property
    def scale(self) -> int:
        return self.variant.scale


def stub_enabled() -> bool:
    return os.environ.get(STUB_ENV) == "1"


def resolve_device(requested: str, family: str) -> str:
    """auto -> cuda if available. Never silently downgrade a requested model (PRD 9.4)."""
    import torch

    has_cuda = torch.cuda.is_available()
    if requested == "cuda" and not has_cuda:
        raise SatEnhanceError(
            ExitCode.MODEL_UNSUPPORTED, "--device cuda requested but no CUDA GPU is available"
        )
    device = requested if requested != "auto" else ("cuda" if has_cuda else "cpu")
    if family == "full" and device != "cuda":
        raise SatEnhanceError(
            ExitCode.MODEL_UNSUPPORTED,
            "The full SEN2SR (Mamba) model needs an NVIDIA GPU; none is available. "
            "Use --model lite, or run the GPU image on a GPU host.",
        )
    return device


def model_url(variant: Variant, family: str) -> str:
    if family == "full":
        if not variant.full_url:
            raise SatEnhanceError(
                ExitCode.MODEL_UNSUPPORTED,
                f"No full-model weights are defined for variant '{variant.name}'; use --model lite",
            )
        return variant.full_url
    return variant.lite_url


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

    import mlstac

    target = cache_dir_for(cache_root, variant, family)
    try:
        if not (target / "mlm.json").exists():
            log.info("Downloading model %s -> %s", url, target)
            target.mkdir(parents=True, exist_ok=True)
            mlstac.download(file=url, output_dir=str(target))
        module = mlstac.load(str(target)).compiled_model(device=device).to(device)
    except Exception as e:  # network / HF / mlstac errors surface in many types
        raise SatEnhanceError(
            ExitCode.NETWORK_FAILURE,
            f"Could not fetch or load model from {url}: {e}. "
            "If offline, run 'prefetch' first with network access.",
        ) from e
    module.eval()
    return LoadedModel(module, device, family, variant, url)


def prefetch(variants: list[str], family: str, cache_root: Path) -> list[str]:
    done = []
    for v in variants:
        load_model(v, family, "cpu" if family == "lite" else "auto", cache_root)
        done.append(v)
    return done
