"""Model variants and band orders (from the SEN2SR README examples)."""

from __future__ import annotations

from dataclasses import dataclass

RGBN = ["B04", "B03", "B02", "B08"]
MS10 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

HF = "https://huggingface.co/tacofoundation/sen2sr/resolve/main"


@dataclass(frozen=True)
class Variant:
    name: str
    bands: list[str]
    scale: int
    lite_url: str
    # Path of the full (Mamba) model is not shown in the SEN2SR README. [VERIFY]
    full_url: str | None
    # Indices of R,G,B within the output for the 8-bit preview
    rgb_idx: tuple[int, int, int]


VARIANTS: dict[str, Variant] = {
    "rgbn_x4": Variant(
        "rgbn_x4", RGBN, 4, f"{HF}/SEN2SRLite/NonReference_RGBN_x4/mlm.json", None, (0, 1, 2)
    ),
    "multispectral_x4": Variant(
        "multispectral_x4", MS10, 4, f"{HF}/SEN2SRLite/main/mlm.json",
        f"{HF}/SEN2SR/main/mlm.json", (2, 1, 0),
    ),
}


def default_variant(sensor: str) -> str:
    return {"rgb": "rgbn_x4", "multispectral": "multispectral_x4"}[sensor]
