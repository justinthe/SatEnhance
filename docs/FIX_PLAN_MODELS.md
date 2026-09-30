# Fix Plan: SEN2SR model table and model loading (`--model full` on RGB data)

| | |
|---|---|
| **Status** | Plan only; nothing in this document is implemented yet |
| **Date** | 2026-09-30 |
| **Trigger** | `./scripts/run_system2.sh --gpu --model full` → `ERROR No full-model weights are defined for variant 'rgbn_x4'; use --model lite` (exit 21) |

## 1. What the error tells us

The check that failed runs **after** the GPU check (`resolve_device`, `runner.py`). So this run proves three things that were untested until now:
- the GPU image **built** on your machine (including the `mamba-ssm` compile), and
- the container **can see the GPU** (`torch.cuda.is_available()` was true), and
- System 2 found and validated your rawdata run.

The run stopped because of a wrong entry in my model table. It never reached downloading or running a model.

## 2. Root cause (verified against the real model repository)

I listed the SEN2SR model repository on Hugging Face (`tacofoundation/sen2sr`, read through the Hugging Face connector; the sandbox can't reach huggingface.co directly). It contains **six** models:

| Family | Variant folder | Input → output | In my table? |
|---|---|---|---|
| SEN2SRLite | `NonReference_RGBN_x4` | B04,B03,B02,B08 at 10 m → same 4 bands at 2.5 m | yes |
| SEN2SRLite | `main` | 10 bands → 10 bands at 2.5 m | yes |
| SEN2SRLite | `Reference_RSWIR_x2` | 20 m bands → 10 m | no (deferred earlier) |
| **SEN2SR (full)** | **`NonReference_RGBN_x4`** | **4 bands → 2.5 m (Mamba, 55 MB)** | **no: this is the bug** |
| SEN2SR (full) | `main` | 10 bands → 2.5 m (Mamba + Swin2SR, ~380 MB total) | yes; my guessed path `SEN2SR/main/mlm.json` is correct |
| SEN2SR (full) | `Reference_RSWIR_x2` | 20 m → 10 m | no |

`enhance/satenhance_enhance/variants.py` sets `full_url=None` for `rgbn_x4`. The SEN2SR README only showed the full model for the 10-band case, and I wrongly concluded that no full RGBN model existed. Your latest rawdata run is `--sensor rgb` (the default), so `auto` picked `rgbn_x4`, and `--model full` had nothing to load.

**My own test encoded the mistake:** `test_full_model_has_no_rgbn_weights` asserts that the full RGBN model does *not* exist. `selftest --model full` also silently skipped variants without a full URL, so nothing flagged the gap.

## 3. The next failures this would hit (found by reading the real model files)

I read each model's `mlm.json` and its `load.py`, the Python file that `mlstac` downloads and runs to build the model. Fixing only the table would lead straight into these:

### 3.1 `matplotlib` is missing from both images (affects **every** real model, CPU included)
All four `load.py` files I read (Lite RGBN, Lite main, full RGBN, full main) start with `import matplotlib.pyplot as plt`. Neither image installs `matplotlib`. So **no real model can load at all**, including `--model lite` on CPU. The earlier tests used the stub model and never ran the real loader code, which is how this slipped through.

### 3.2 The error for 3.1 would be misleading, and would trigger a needless re-download
`mlstac` catches the import error and re-raises it as `RuntimeError("Failed to load Python module ...: No module named 'matplotlib'")`. My loader only recognises a *direct* `ModuleNotFoundError`. So it would:
1. treat the cached model as corrupt and re-download it (up to ~380 MB for full `main`), then
2. report "could not be loaded with torch …", which points you at PyTorch instead of the one missing package.

### 3.3 Data type (checked: not a problem, but guard it)
The model descriptions say `float16`, but the weights are float32. The full RGBN `model.safetensor` is 55,189,944 bytes, and 13,759,444 parameters × 4 bytes ≈ 55.0 MB. So our float32 input matches. I'm still adding a cheap guard in case a model ever ships half-precision weights.

### 3.4 Asset URLs use a different repo spelling
Inside `mlm.json` the weight links point to `huggingface.co/tacofoundation/SEN2SR/...` (capitalised). Our table uses `.../tacofoundation/sen2sr/...`. I believe Hugging Face treats repo names case-insensitively and redirects **[VERIFY: huggingface.co is blocked in this sandbox]**, but a 404 here would stop every download.

## 4. The fix

### 4.1 Correct the model table (the reported bug)
- `variants.py`: set `full_url` for `rgbn_x4` to `…/SEN2SR/NonReference_RGBN_x4/mlm.json`.
- Store the source repo path per family, so both URLs come from one place.
- Replace the wrong test with one that checks the whole table against a recorded snapshot of the repo listing above (6 `mlm.json` paths).
- Add an **opt-in live test** (`pytest -m live`) that lists the Hugging Face repo and fails if the table and the repo disagree.

### 4.2 Install what the model loaders import
- Add `matplotlib` (headless) to `enhance/requirements.lock`. Both CPU and GPU images use that file.
- Set `MPLBACKEND=Agg` and `MPLCONFIGDIR=/tmp/matplotlib` in both Dockerfiles, so importing `pyplot` never needs a display or a writable home directory.
- Check that everything else the loaders import is installed: `safetensors`, `sen2sr`, `timm`, `einops`, and `mamba_ssm` for full. From the `mlm.json` `dependencies` lists, these are present: `safetensors` comes with `mlstac`, `mamba_ssm` is in the GPU image.

### 4.3 Test with the *real* loader code
- Save the four real `load.py` files as test fixtures, with their source URL recorded. They're 3–9 KB, and SEN2SR's code licence is permissive (MIT/CC0; the discrepancy is noted in the PRD).
- **Lite, runnable here on CPU:** build a model folder from the real `mlm.json` and `load.py` plus randomly initialised weights of the correct architecture (`CNNSR` from the installed `sen2sr`). Then run the **real** `mlstac.load(...).compiled_model(...)` path and our inference on a patch. That proves the imports (including matplotlib), file names and output shape with genuine loader code. It would have caught 3.1.
- **Full (Mamba):** needs `mamba_ssm` and a GPU, so it can't run here. It gets a GPU-only test, skipped without CUDA, that you can run with `selftest --model full`.
- An **isolation test** that imports each real `load.py` inside the enhance environment and reports any missing module by name.

### 4.4 Correct error messages and no needless re-downloads
- When a load fails, `models._load_from_cache` walks the exception chain (`__cause__`/`__context__`) and also checks for mlstac's `"No module named 'X'"` text. A missing package → exit 21 naming the package, **without** deleting or re-downloading the cache.
- Only a genuinely unreadable or corrupt weight file triggers the one self-healing re-download.

### 4.5 Guards
- **Data type:** after loading, read the model's parameter dtype and cast the input to match (a no-op for today's float32 weights).
- **Repo spelling:** in `download.fetch_model`, rewrite `huggingface.co/<org>/<repo>/` in asset links to the exact spelling of the `mlm.json` URL we requested. That's harmless if Hugging Face already redirects, and fixes it if it doesn't.

### 4.6 `selftest` and `prefetch` must not skip silently
- `selftest --model full` and `prefetch --model full` list every variant, and a variant without full weights shows as **FAIL/UNAVAILABLE**, not missing from the table.
- `prefetch --model full` downloads both full variants, about 435 MB in total. Before starting it prints the expected size, taken from the `file:size` fields.

### 4.7 Where the error report goes
Your report went to `./output/error_report.json` because no `--run-id` was given. System 2 will resolve `rawdata/LATEST` before anything else, so the report lands in `./output/<run_id>/error_report.json` next to the outputs.

### 4.8 Docs
- HOWTO System 2 table: `--model full` works for both `rgb` (Mamba RGBN) and `multispectral` data.
- Prefetch sizes for each model.
- A troubleshooting row for "needs the Python package 'X'".

## 5. Optional, and not needed for this bug: the 20 m → 10 m models
Both families also ship `Reference_RSWIR_x2`: 10 bands in, 6 bands out at 10 m. Supporting it needs the variant table to hold separate **output** bands and output resolution, because right now output bands = input bands and scale = 4. I'd leave this out unless you want 10 m output of the 20 m bands.

## 6. Verification
**Here:**
- unit tests;
- the new real-loader test for Lite on CPU (genuine `load.py`, random weights);
- the container smoke test, which runs that same real-loader check inside the CPU image to prove `matplotlib` is installed there;
- CI.

**Only on your machine:**
```bash
git pull && ./scripts/build.sh --gpu      # rebuild: adds matplotlib to both images
./scripts/run_system2.sh --gpu prefetch --model full     # downloads ~435 MB, then runs selftest
./scripts/run_system2.sh --gpu --model full              # enhance your latest rawdata run
```
If `prefetch` or `selftest` fails, send me the output. It names the variant and the exact missing package or error.

## 7. Order of work
1. 4.1 + 4.2 + 4.3 (table, matplotlib, real-loader tests) → unit tests + smoke test → push.
2. 4.4 + 4.5 + 4.6 + 4.7 (messages, guards, selftest/prefetch, report location).
3. 4.8 docs.

One commit per step, pushed to `claude/practical-ptolemy-w1xnmq`. `main` only when you ask.
