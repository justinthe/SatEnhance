# Fix Plan: GPU image runtime (`needs the Python package 'distutils'`)

| | |
|---|---|
| **Status** | Plan only; nothing in this document is implemented yet |
| **Date** | 2026-09-30 |
| **Trigger** | `./scripts/run_system2.sh --gpu prefetch --model full` → `ERROR The full model 'rgbn_x4' needs the Python package 'distutils', which is not installed in this image.` |

## 1. What the run proved

These all worked on your machine for the first time:
- The full RGB+NIR model was found in the corrected table.
- All four model files downloaded from Hugging Face. The `SEN2SR`/`sen2sr` spelling was a non-issue.
- `matplotlib` loaded (`generated new fontManager`).
- The model's `load.py` ran.

It failed while `load.py` imported the Mamba model.

## 2. Root cause (reproduced here, not guessed)

**The import chain:** `load.py` imports `MambaSR`, which imports `mamba_ssm`, which loads `triton` kernels. `triton/runtime/build.py` (triton 3.2.0, the version torch 2.6 pins) starts with `import setuptools`, and `setuptools` then needs `distutils.core`, which fails.

**Why only the GPU image:** `enhance/Dockerfile.gpu` has two stages, and they install different Python packages.

| Stage | Ubuntu packages | `distutils` |
|---|---|---|
| builder | `python3.11 python3.11-venv python3.11-dev …` | complete: `python3.11-venv` depends on `python3-distutils` |
| runtime | `python3.11` only | **stub only**: Ubuntu puts `distutils/__init__.py` in the base package and `distutils.core`/`distutils.sysconfig` in `python3-distutils` |

**Why `setuptools` didn't cover for it:** the venv still has **Ubuntu's `setuptools` 59.6**. The builder's `pip install setuptools` didn't upgrade it, because it was already "satisfied". `setuptools` 59.6 relies on the system `distutils`. Version 60 and later bundle their own copy, which would have avoided the problem.

**Reproduction:** this used the same two-stage layout, on `ubuntu:22.04` with Ubuntu's own packages:
```
BUILDER ok 3.11.0rc1 setuptools 59.6.0 /usr/lib/python3.11/distutils/__init__.py
RUNTIME import distutils OK            <- the stub
RUNTIME ModuleNotFoundError: No module named 'distutils.core'
RUNTIME ModuleNotFoundError: No module named 'distutils.sysconfig'
```

**Both fixes verified the same way:**
```
A+B OK: setuptools 84.0.0                           (python3-distutils in runtime + upgraded setuptools)
B only OK: setuptools 84.0.0 .../setuptools/_distutils   (upgraded setuptools alone)
```

**Also found:** Ubuntu 22.04's `python3.11` package is **3.11.0 release candidate 1** (`3.11.0~rc1-1~22.04.1`), not a final release. It works (torch loaded and saw your GPU), but see §4.5.

## 3. The next failure, fixed in the same change

Reading `triton/runtime/build.py` shows what happens once the import succeeds. On the **first kernel launch**, `triton` compiles a small C launcher at run time:
- it needs a C compiler: `shutil.which("gcc")` or `clang`, otherwise `RuntimeError: Failed to find C compiler`;
- it needs Python's C headers (`Python.h`, from `python3.11-dev`) via `sysconfig.get_paths()["include"]`.

The runtime stage has **neither**. The builder has them, which again is why nothing complained during the build. So after the `distutils` fix, the first real inference would fail here.

## 4. The fix

### 4.1 Same Python in both stages (the root cause)
In the runtime stage of `enhance/Dockerfile.gpu`, install everything the Python stack needs at run time:
```
python3.11 python3.11-dev python3-distutils gcc libc6-dev libexpat1
```
- `python3-distutils` provides the complete `distutils`. It's the package `python3.11-venv` pulls into the builder.
- `python3.11-dev`, `gcc` and `libc6-dev` are what `triton` compiles its launcher with (§3). That's roughly 60–80 MB more **[VERIFY exact size after build]**.

### 4.2 A real `setuptools` upgrade (second line of defence)
- In the builder, use `pip install --upgrade "setuptools>=70"`. A plain `pip install setuptools` is a no-op when Ubuntu's 59.6 is present.
- Pin the resulting version into the torch constraints file, so later installs can't move it.
- Verified: with this alone, `distutils.core` works through `setuptools`' bundled copy.

### 4.3 Check the runtime stage at build time (why this slipped through)
The CUDA check I added earlier runs in the **builder**, which has everything. Add a check at the end of the **runtime** stage that fails the build with a clear message if any of these fail:
- `import setuptools, distutils.core, distutils.sysconfig`;
- `import triton, triton.runtime.build`, which is exactly the failing import;
- a C compiler exists and `Python.h` is at `sysconfig.get_paths()["include"]`;
- `import mamba_ssm` and `from sen2sr.models.opensr_baseline.mamba import MambaSR`.

`docker build` has no GPU. These are import checks only, so they should not need the NVIDIA driver **[VERIFY: if `mamba_ssm`'s compiled extension refuses to load without `libcuda`, only a missing-module error fails the build; driver errors are reported as warnings]**.

### 4.4 Persist `triton`'s compiled kernels
- Set `TRITON_CACHE_DIR=/data/cache/triton`, so the launcher and kernels compiled on the first run are reused on later runs instead of being rebuilt every time.
- `./cache` is already mounted, so this adds no new volume.

### 4.5 Optional: move off the Python release candidate
Ubuntu 22.04 only offers 3.11.0rc1. The cleanest way off it is a standalone CPython 3.11 build (`uv python install 3.11`, copied into both stages), which is a full final release and includes a complete `distutils`. **I recommend 4.1–4.4 now and this as a separate follow-up.** It changes the base of the GPU image, and it needs a GPU rebuild on your side to verify. It isn't required to fix this error.

### 4.6 Clearer error message
The message said the "package" was `distutils`. The real module was `distutils.core`, and `distutils` isn't installable with pip.
- Report the full module name.
- Map the known ones to a real hint: `distutils.*` → "the image's Python is incomplete (python3-distutils / setuptools); rebuild the GPU image".
- Also include the last lines of the import traceback in the message, not only in `error_report.json`.

### 4.7 Tests
- **Dockerfile guard test** (runs here and in CI): parse `enhance/Dockerfile.gpu` and fail if the runtime stage doesn't install `python3-distutils`, `python3.11-dev` and `gcc`, or if the builder's `setuptools` install lacks `--upgrade`. This keeps the stages from drifting apart again.
- **Stage-split reproduction test** (Docker, runs in the smoke test here): the two-stage `ubuntu:22.04` build from §2, with the new runtime packages. It asserts that `import setuptools, distutils.core` works, that `gcc` exists, and that `Python.h` is present. This checks the exact Ubuntu packaging behaviour without needing the multi-GB CUDA images.
- **Message test:** `missing_module()` on `"No module named 'distutils.core'"` gives the new hint, not "install package distutils".

## 5. Verification

**Here:** the unit tests, the Dockerfile guard test, and the stage-split reproduction with the fixed package list (§4.7), plus CI. I can't build the real GPU image here: it pulls about 4 GB of CUDA base images, the sandbox disk is small, and Docker Hub rate-limits the sandbox.

**On your machine** (the GPU rebuild is quick for the pip layers, since they're cached; the apt layer re-runs):
```bash
git pull && ./scripts/build.sh --gpu-only        # the new runtime check fails the build if anything is still missing
./scripts/run_system2.sh --gpu prefetch --model full   # the model is already cached; this just loads and self-tests it
./scripts/run_system2.sh --gpu --model full            # first real full-model run
```
The first full-model inference compiles `triton` kernels, which may take a minute. Later runs reuse the cache (§4.4).

## 6. Order of work
1. 4.1 + 4.2 + 4.3 + 4.4 (Dockerfile) with the tests from 4.7, then run the tests and the reproduction, then push.
2. 4.6 (message).
3. Docs: HOWTO troubleshooting row, and a note that the first full-model run compiles kernels.

One commit per step, pushed to `claude/practical-ptolemy-w1xnmq`. `main` only when you ask. 4.5 waits for your go-ahead.
