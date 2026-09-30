"""Model variants and band orders.

The model files live in the Hugging Face repo ``tacofoundation/sen2sr`` (verified against the
repo listing, see tests/test_variants.py): two families, each with three variant folders.

    SEN2SRLite/  NonReference_RGBN_x4   4 bands  10 m -> 2.5 m   (small CNN, CPU or GPU)
                 main                   10 bands 10 m -> 2.5 m
                 Reference_RSWIR_x2     20 m bands -> 10 m       (not used yet)
    SEN2SR/      NonReference_RGBN_x4   4 bands  10 m -> 2.5 m   (Mamba, needs a GPU)
                 main                   10 bands 10 m -> 2.5 m   (Mamba + Swin2SR, needs a GPU)
                 Reference_RSWIR_x2     20 m bands -> 10 m       (not used yet)
"""

from __future__ import annotations

from dataclasses import dataclass

RGBN = ["B04", "B03", "B02", "B08"]
MS10 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

HF = "https://huggingface.co/tacofoundation/sen2sr/resolve/main"
FAMILY_DIR = {"lite": "SEN2SRLite", "full": "SEN2SR"}


@dataclass(frozen=True)
class Variant:
    name: str
    bands: list[str]
    scale: int
    folder: str  # variant folder in the repo, same for both families
    # Indices of R,G,B within the output for the 8-bit preview
    rgb_idx: tuple[int, int, int]

    def url(self, family: str) -> str:
        return f"{HF}/{FAMILY_DIR[family]}/{self.folder}/mlm.json"

    @property
    def lite_url(self) -> str:
        return self.url("lite")

    @property
    def full_url(self) -> str:
        return self.url("full")


VARIANTS: dict[str, Variant] = {
    "rgbn_x4": Variant("rgbn_x4", RGBN, 4, "NonReference_RGBN_x4", (0, 1, 2)),
    "multispectral_x4": Variant("multispectral_x4", MS10, 4, "main", (2, 1, 0)),
}


def default_variant(sensor: str) -> str:
    return {"rgb": "rgbn_x4", "multispectral": "multispectral_x4"}[sensor]
