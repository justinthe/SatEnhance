"""The variant table must match the real model repository (tacofoundation/sen2sr)."""

import os
import re

import pytest
from satenhance_enhance import models
from satenhance_enhance.variants import FAMILY_DIR, HF, VARIANTS

# Listing of hf://models/tacofoundation/sen2sr taken on 2026-09-30 (files named mlm.json).
REPO_MLM = {
    "SEN2SR/NonReference_RGBN_x4/mlm.json",
    "SEN2SR/Reference_RSWIR_x2/mlm.json",
    "SEN2SR/main/mlm.json",
    "SEN2SRLite/NonReference_RGBN_x4/mlm.json",
    "SEN2SRLite/Reference_RSWIR_x2/mlm.json",
    "SEN2SRLite/main/mlm.json",
}
# The first real GPU run failed with "No full-model weights are defined for variant 'rgbn_x4'":
# the table had no entry although SEN2SR/NonReference_RGBN_x4 exists.


@pytest.mark.parametrize("family", ["lite", "full"])
@pytest.mark.parametrize("name", list(VARIANTS))
def test_every_variant_exists_in_both_families(name, family):
    url = VARIANTS[name].url(family)
    assert url.startswith(HF + "/") and url.endswith("/mlm.json")
    assert url[len(HF) + 1:] in REPO_MLM, f"{name}/{family} is not in the model repository"
    assert models.model_url(VARIANTS[name], family) == url


def test_family_directories():
    assert FAMILY_DIR == {"lite": "SEN2SRLite", "full": "SEN2SR"}


def test_the_case_from_the_failed_gpu_run():
    """--model full on RGB data (variant rgbn_x4) has weights."""
    assert VARIANTS["rgbn_x4"].full_url.endswith("/SEN2SR/NonReference_RGBN_x4/mlm.json")
    assert VARIANTS["multispectral_x4"].full_url.endswith("/SEN2SR/main/mlm.json")


def test_band_counts_match_the_model_descriptions():
    """mlm.json: RGBN models take B04,B03,B02,B08; 'main' takes the 10 bands B02..B12."""
    assert VARIANTS["rgbn_x4"].bands == ["B04", "B03", "B02", "B08"]
    assert VARIANTS["multispectral_x4"].bands == [
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    assert all(v.scale == 4 for v in VARIANTS.values())


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("SATENHANCE_LIVE"),
                    reason="opt-in: needs network access to Hugging Face (SATENHANCE_LIVE=1 pytest -m live)")
def test_table_matches_the_live_repository():
    """Opt-in: compare with the actual Hugging Face repo."""
    import requests

    found = set()
    for family in FAMILY_DIR.values():
        r = requests.get(
            f"https://huggingface.co/api/models/tacofoundation/sen2sr/tree/main/{family}",
            params={"recursive": "true"}, timeout=30)
        r.raise_for_status()
        found |= {e["path"] for e in r.json() if re.search(r"/mlm\.json$", e["path"])}
    assert found == REPO_MLM
    for name in VARIANTS:
        for family in FAMILY_DIR:
            assert VARIANTS[name].url(family)[len(HF) + 1:] in found
