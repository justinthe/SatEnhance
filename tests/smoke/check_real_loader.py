"""Runs INSIDE the enhance image with the real loader scripts mounted at /loaders.

Builds a Lite model folder from the REAL load.py plus random weights, then loads it through the
production path (mlstac -> load.py -> our inference). Fails if the image lacks a package the
loaders import (the first GPU run showed: matplotlib).
"""

import ast
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["SATENHANCE_STUB_MODEL"] = "0"
import numpy as np  # noqa: E402
import safetensors.torch  # noqa: E402
import torch  # noqa: E402
from satenhance_enhance import models  # noqa: E402
from satenhance_enhance.download import COMPLETE_MARKER  # noqa: E402
from satenhance_enhance.infer import _predict_square  # noqa: E402
from satenhance_enhance.variants import VARIANTS  # noqa: E402
from sen2sr.models.opensr_baseline.cnn import CNNSR  # noqa: E402

LOADERS = Path("/loaders")
flavor = os.environ.get("SATENHANCE_IMAGE_FLAVOR", "")

# 1) every import of every real loader resolves (mamba_ssm only in the GPU image)
missing = []
for f in sorted(LOADERS.glob("*.py")):
    for node in ast.walk(ast.parse(f.read_text())):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module.split(".")[0]]
        for n in names:
            if n == "mamba_ssm" and flavor != "gpu":
                continue
            if importlib.util.find_spec(n) is None:
                missing.append(f"{f.name}: {n}")
assert not missing, f"packages the loaders import are missing from this image: {missing}"

# 2) the real Lite RGBN loader, random weights, production load path
cache = Path("/tmp/loader_check")
shutil.rmtree(cache, ignore_errors=True)
target = models.cache_dir_for(cache, VARIANTS["rgbn_x4"], "lite")
target.mkdir(parents=True)
shutil.copy(LOADERS / "lite_rgbn_x4_load.py", target / "load.py")
sd = {k: v.contiguous() for k, v in CNNSR(4, 4, 24, 4, True, False, 6).state_dict().items()}
safetensors.torch.save_file(sd, str(target / "model.safetensor"))
safetensors.torch.save_file({"weights": torch.rand(512, 512)}, str(target / "hard_constraint.safetensor"))
assets = {n: {"href": f"https://example.invalid/{n}"}
          for n in ("model.safetensor", "hard_constraint.safetensor", "load.py")}
(target / "mlm.json").write_text(json.dumps({
    "type": "Feature", "stac_version": "1.1.0", "id": "check", "geometry": None, "links": [],
    "properties": {"datetime": None, "start_datetime": "1900-01-01T00:00:00Z",
                   "end_datetime": "9999-01-01T00:00:00Z"}, "assets": assets}))
(target / COMPLETE_MARKER).write_text("{}")

m = models.load_model("rgbn_x4", "lite", "cpu", cache)
y = _predict_square(m, np.full((4, 128, 128), 0.2, dtype="float32"), 32)
assert y.shape == (4, 512, 512) and np.isfinite(y).all(), y.shape
print(f"OK real Lite loader ran: {y.shape}, torch {torch.__version__}, flavor={flavor or 'none'}")
sys.exit(0)
