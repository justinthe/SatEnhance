# SatEnhance — Fix Plan (errors seen so far + likely next errors)

| | |
|---|---|
| **Status** | Plan only, nothing in this document is implemented yet |
| **Date** | 2026-09-30 |
| **Branch** | `claude/practical-ptolemy-w1xnmq` (at `5ddc95b`); `main` is at `959f091` |

This plan covers four groups:

- **A:** errors that have actually happened (in your runs, in CI, or here).
- **B:** System 2 failures that are likely on the first real run.
- **C:** System 1 failures that are likely on the first real run.
- **D:** tooling so the next failure is quick to diagnose.

Every item gives the evidence, the root cause, the fix, and how it will be verified.

> Honesty note: anything marked **[VERIFY]** is my best understanding of a third-party service or library that I could not test from this sandbox (no GPU, no access to Copernicus or Hugging Face). Items in B and C are *predicted* failures, based on reading the code of `mlstac`, `sen2sr`, GDAL behaviour, and the logs you sent. They are not observed failures.

---

## A. Errors already seen

### A1. CI is red on every push: `ModuleNotFoundError: No module named 'pyproj'` (System 2)
- **Evidence:** GitHub Actions run 36668701854, job `test-enhance`: 6 failed, 10 errors. The other four jobs pass, including the container `smoke` job.
- **Root cause:** `enhance/satenhance_enhance/synthetic.py` imports `pyproj`, but `pyproj` is not a dependency of the enhance package or image. My local venv had it via the acquire package, which hid the problem.
- **Impact beyond CI:** `docs/HOWTO.md` §10 tells you to run `python -m satenhance_enhance.synthetic` inside the System 2 container. That command will crash the same way.
- **Fix:** reproject the synthetic AOI with `rasterio.warp.transform_geom` (already a dependency) and drop the `pyproj` import. Then add a CI and unit guard that imports every `satenhance_enhance` module inside the *enhance-only* environment.
- **Verify:** `test-enhance` goes green in CI. Running `docker compose run … enhance-cpu -m satenhance_enhance.synthetic` works.

### A2. Image build dies on a single slow download (`ReadTimeoutError` from files.pythonhosted.org)
- **Evidence:** both of your build logs. The second failure happened *after* `PIP_RETRIES=10` was added. The log shows no retry lines, just one timeout and then the build died.
- **Root cause:** pip's retries cover connecting and starting a response. They do not cover a read timeout part-way through a large file. On a link doing ~100 KB/s to 3 MB/s, a 170 MB wheel can stall long enough to hit that.
- **Fix:**
  1. Wrap every `pip install` in the Dockerfiles in a small shell retry loop (up to 5 attempts, with back-off). The BuildKit pip cache mount keeps every wheel that finished downloading, so each attempt resumes from where the last one stopped instead of starting over. A partly downloaded wheel is still fetched again.
  2. Raise `PIP_DEFAULT_TIMEOUT` from 120 to 300 s.
- **Verify:** here, simulate a flaky index with a proxy that drops the connection mid-transfer. The build completes on attempt 2 or 3, and the log shows the resumed wheels being served from cache.

### A3. All images build in parallel and compete for your bandwidth
- **Evidence:** your log shows `acquire`, `enhance-cpu` and `enhance-gpu` all building at once. The CUDA base layers (1.37 GB + 2.65 GB + 2 × 670 MB) and the pip downloads were fetched at the same time, which makes the timeouts in A2 more likely.
- **Fix:** `build.sh` builds one service at a time, in the order acquire, then cpu, then gpu. It also prints roughly how much will be downloaded before starting.
- **Verify:** the build log shows sequential builds.

### A4. GPU image: pip replaced the CUDA 12.4 PyTorch with a CUDA 13 build
- **Evidence:** your log shows `mamba-ssm 2.3.2.post1`, which requires `triton>=3.5.0`. That made pip download `torch-2.14.0` plus `nvidia-*-cu13` libraries on top of the base image's `torch 2.6.0+cu124`.
- **Status:** a fix is committed (`5ddc95b`) but **untested**:
  - torch is frozen with a constraints file,
  - the default is `MAMBA_SPEC="mamba-ssm<2.3"`,
  - the build fails if the CUDA version doesn't match.
- **Remaining work:**
  - Also pin `triton` to the version that ships with the installed torch, so `mamba-ssm` 2.2.x cannot pull a newer one.
  - Set `MAX_JOBS=4` so the `mamba-ssm` source compile doesn't exhaust RAM.
  - Let `setup.py` use `mamba-ssm`'s prebuilt wheel when one exists for torch 2.6 + cu12 **[VERIFY]**. That avoids a long compile.
  - Also install `causal-conv1d`, if the SEN2SR full model needs it (check in B3).
- **Verify:** only you can do this: `./scripts/build.sh --gpu-only` on your GPU machine. The build's own CUDA check will stop with a clear message if torch is still swapped.

### A5. CPU image: torch/torchvision mismatch caused a second, larger torch download
- **Evidence:** in your log the CPU dependency step took 540 s, against 52 s for the torch step.
- **Status:** a fix is committed (`5ddc95b`): torch and torchvision are installed from the same index, then frozen. **Verified here:** in the rebuild, the requirements step installed nothing torch-related (`torch==2.14.0`, `torchvision==0.29.0` stayed as installed, with no "Attempting uninstall: torch").
- **Remaining work:** none beyond A2 and A3.

### A6. "pull access denied for satenhance-acquire" during build
- **Evidence:** your first log. It's harmless (Compose tries to pull our locally built image before building it), but it looks like an error.
- **Fix:** add `pull_policy: build` to our three services in `docker-compose.yml` so Compose never tries to pull them.
- **Verify:** the message is gone from the build output.

### A7. Errors found and already fixed during development (no action; listed for completeness)
| Error | Fix (commit) |
|---|---|
| `libexpat.so.1` missing in `python:3.11-slim` (rasterio import failed) | install `libexpat1` if absent (`65f1047`) |
| First `run_system1.sh` silently started a 15-min build | `require_image` check (`959f091`) |
| `--model full` on CPU exited 20 instead of 21 when there was no rawdata | device check moved first (`742480d`) |
| CDSE catalogue search retried permanent 4xx errors | retry only 5xx / 429 / network (`f5a6a3a`) |
| `sen2sr.predict_large` assumes square input and leaves unwritten pixels for exactly-128 px input | pad to a square ≥129 px, then crop (`ed5029e`) |

---

## B. System 2 (enhance): likely failures on the first real run

These all sit on code paths the stub model bypasses, so no test has exercised them yet.

### B1. A half-finished model download is cached forever
- **Evidence (from `mlstac/main.py`):** `mlstac.download()` writes `mlm.json` **first**, then streams the weight files with a 30 s timeout and **no retry**. Our cache check (`models.py`) only asks whether `mlm.json` exists.
- **Failure:** on your connection a weight download that times out leaves `mlm.json` plus a truncated file. Every later run skips the download and then fails to load the model (exit 5, confusing message) until you delete `cache/models/` by hand.
- **Fix:**
  - Download into `cache/models/.<name>.partial/`, retrying up to 5 times with back-off.
  - Check that every asset listed in `mlm.json` exists and is not empty.
  - Write a `.complete` marker, then rename the folder into place. The cache check becomes "is the `.complete` marker there?".
  - A corrupt cache is detected on load and re-downloaded once automatically.
- **Verify:** unit test with a fake HTTP server that cuts the connection mid-file: the first run fails cleanly, the second resumes and succeeds, and a truncated cache is never used.

### B2. Model weights and the installed PyTorch may be incompatible
- **Evidence:** SEN2SR weights are `.pt2` files (`torch.export` archives) loaded by Python code that `mlstac` downloads from Hugging Face and runs with `exec`. `.pt2` archives can be tied to the torch version that created them **[VERIFY]**. The CPU image has torch 2.14 and the GPU image torch 2.6, so they may differ in what they can load.
- **Fix:**
  - Add a `satenhance-enhance selftest [--model lite|full]` command, exposed as `./scripts/run_system2.sh selftest`. It loads each model variant, runs one random 128×128 patch, and checks that the output shape is ×4, has the right band count and contains no NaNs.
  - Run it automatically at the end of `prefetch`, and record the torch version each variant was tested with.
  - If a variant fails to load, the error names the torch version and suggests the build arg to change.
- **Verify:** you run `./scripts/run_system2.sh prefetch` once. I can't run it here because Hugging Face is blocked.

### B3. The model loader may need packages we don't install
- **Evidence:** the downloaded loader code may import extra modules: `mamba_ssm` and possibly `causal_conv1d` for the full model, maybe `safetensors` or `einops` for lite. That code isn't visible until download.
- **Fix:**
  - `selftest` catches `ImportError` and reports exactly which module is missing, as a clear exit 21 message.
  - After prefetching once (on your machine or anywhere Hugging Face is reachable), scan the loader for imports and add the missing ones to `enhance/requirements.lock` or the GPU Dockerfile.
- **Verify:** part of the B2 `selftest` run.

### B4. The model returns a different band count than we expect
- **Failure:** if a variant outputs, say, 3 or 10 bands when we expect 4, the GeoTIFF writer fails with a cryptic rasterio error.
- **Fix:** `_predict_square` validates the channel count as well as the ×4 size. On a mismatch it exits 22 naming the variant and the band counts. The expected output bands for each variant also get written into the report.
- **Verify:** unit test with a stub that returns the wrong band count.

### B5. The GPU is present but not visible inside the container
- **Evidence:** the GPU image relies on Compose `deploy.resources.reservations.devices`. `has_gpu()` detects the NVIDIA runtime by grepping `docker info` for "nvidia", which may miss setups that use the newer CDI mode **[VERIFY]**.
- **Failures:**
  - A GPU machine silently runs on CPU.
  - `--gpu` runs, but `torch.cuda.is_available()` is False inside the container, giving exit 21 with a misleading "no GPU" message.
- **Fix:**
  - `has_gpu()` tries a real `docker run --rm --gpus all <gpu image> nvidia-smi -L` once and caches the answer for the session.
  - Inside the container, if the GPU image is running but CUDA isn't visible, the message says "GPU image without GPU access: check the NVIDIA Container Toolkit / `--gpus`" instead of just "no GPU".
  - The compose file also gets `gpus: all` where supported **[VERIFY Compose version]**.
- **Verify:** you run `./scripts/doctor.sh` (D1) on the GPU machine.

### B6. Reflectance convention is uncertain
- **Evidence:** SEN2SR's examples divide by 10,000. Our default also subtracts the baseline ≥04.00 offset of 1000. If SEN2SR was trained on data without that offset removed, colours will be subtly wrong.
- **Fix:** `selftest --compare-reflectance <run_id>` enhances one 256×256 crop in both modes and writes a side-by-side PNG, so the choice is made by looking once. After you've seen it we set the default in one line. No code guesses.
- **Verify:** you look at the PNG.

### B7. Out of memory on large inputs
- **Evidence:** peak memory isn't measured yet. The CPU image has no memory guard.
- **Fix:** before inference, estimate peak memory for the chosen `--block`. If it exceeds 70% of the container's available memory, lower the block size automatically and log it. OOM is already retried once at half size on CUDA; do the same on CPU (`MemoryError`).
- **Verify:** unit test with a small fake memory limit.

---

## C. System 1 (acquire): likely failures on the first real run

### C1. `--aoi-file` with a relative path breaks when run from another directory
- **Evidence (code):** `scripts/_common.sh` changes into the repo root *before* `run_system1.sh` checks `[[ -f "$f" ]]`, so a relative path is resolved against the repo root, not the directory you ran it from. Your run worked only because you started in the repo root.
- **Fix:** save the caller's directory before changing into the repo root, and resolve `--aoi-file` against it. Also stop two AOI files with the same name from overwriting each other in `aoi/` by prefixing a short hash.
- **Verify:** smoke test step that runs the script from `/tmp` with a relative path.

### C2. GDAL tries to list the Copernicus bucket and fails or crawls
- **Evidence:** when opening a `/vsis3/` path, GDAL lists the parent directory unless told not to **[VERIFY CDSE behaviour]**. Listing `eodata` is slow and may be refused, which would show up as a confusing 403 (exit 4) or exit 5.
- **Fix:**
  - Add `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`, `AWS_REGION=default` and `GDAL_HTTP_TIMEOUT` / `GDAL_HTTP_MAX_RETRY` to `CdseProvider.gdal_env()`.
  - Decide `AWS_REGION` with the probe in C3.
- **Verify:** unit test that the options are passed. For real, the C3 probe.

### C3. CDSE catalogue and asset details differ from my assumptions
- **Evidence:** these are all unverified **[VERIFY]**: the collection id `sentinel-2-l2a`, asset keys like `B04_10m` / `SCL_20m`, `s3://` hrefs, and the processing-baseline property name.
- **Fix:** add `satenhance-acquire probe --aoi-file … --start … --end …` (and `./scripts/run_system1.sh probe …`). It:
  - searches once and prints how many items were found,
  - prints the first item's asset keys, hrefs and properties,
  - opens that item's SCL band through the same GDAL settings, reporting OK or the exact GDAL error.

  It takes a minute and turns any mismatch into a one-line fix. Asset-key matching also becomes more tolerant (case-insensitive; also accept `B04` with a `gsd` of 10).
- **Verify:** you run the probe once with real keys and send me the output if it fails.

### C4. Wide date ranges miss the best scene
- **Evidence (code):** the search asks for at most 50 items, and STAC servers usually return them in date order, so over a long range the clearest scene may never be looked at.
- **Fix:**
  - Request `sortby=eo:cloud_cover` when the server supports it.
  - Otherwise page through up to 500 items and sort by cloud on our side.
  - Log a warning if the cap was reached.
- **Verify:** unit test with a fake 120-item catalogue where the best scene is on page 3.

### C5. AOI crossing two Sentinel-2 tiles is reported as "no data"
- **Evidence (code):** each scene is judged on its own. An AOI split across two tiles gets <95% coverage from each one, so every scene is rejected and the user is told to raise cloud cover, which won't help.
- **Fix:**
  - In the no-data report and the interactive menu, separate "rejected for cloud" from "rejected for coverage". When coverage is the reason, say the AOI spans tile edges and offer `--min-coverage` as a menu option.
  - Mosaicking two tiles stays out of scope for v1, unless you want it.
- **Verify:** unit test with a fixture AOI straddling two fake tiles.

### C6. Missing `--start` / `--end` gives an unfriendly message
- **Evidence:** your first run (`./scripts/run_system1.sh --aoi-file data/map.kml`) had no dates. It never got that far because of the build, but it would have stopped with Typer's generic "Missing option '--start'" (exit 2).
- **Fix:** keep dates required (see open question Q1), but make the error message show an example command.
- **Verify:** CLI test.

---

## D. Diagnosis tooling (so the next failure takes minutes, not a round trip)

### D1. `./scripts/doctor.sh`
One command that prints PASS / FAIL with a fix hint for each check:
- Docker daemon and Compose version (≥2.24, needed for `env_file: required`).
- Free disk space (warn below 30 GB; the GPU build needs more).
- Which images exist.
- `.env` present, and whether the CDSE keys are set (it never prints the values).
- GPU: `nvidia-smi` on the host, plus a real `--gpus all` test.
- Network reachability of: pypi.org, files.pythonhosted.org, download.pytorch.org, registry-1.docker.io, stac.dataspace.copernicus.eu, eodata.dataspace.copernicus.eu, nominatim.openstreetmap.org, huggingface.co.
- Model cache status (which variants have the `.complete` marker).

### D2. Better run reports
- Every failure writes a short `error_report.json` next to the outputs, with the exit code, message, stage and tool versions.
- The scripts print where that file is.

### D3. CI hardening
- The `test-enhance` job installs torch and torchvision from the CPU index together.
- Add a job that imports every module inside each image (catches A1-type gaps).
- Move to current `actions/checkout` and `actions/setup-python` major versions (the logs warn that Node 20 is deprecated) **[VERIFY latest major versions]**.

---

## E. Order of work and commits

| Step | Items | Why this order |
|---|---|---|
| 1 | A1, D3 | CI green first, so every later push is checked |
| 2 | A2, A3, A6 | Unblocks your builds on a slow connection |
| 3 | B1, B4, B7 | System 2 robustness that can be fully tested here |
| 4 | C1, C2, C4, C5, C6 | System 1 robustness that can be fully tested here |
| 5 | D1, D2, C3 (`probe`), B2/B3/B6 (`selftest`) | Tools for the checks only your machine can do |
| 6 | A4 remainder, B5 | GPU; needs your machine to confirm |

One commit per step, each pushed to `claude/practical-ptolemy-w1xnmq` only after unit tests, lint and the container smoke test pass here. `main` is updated only when you ask.

## F. What you will need to run (things I cannot test here)

1. `git pull`, then `./scripts/doctor.sh`. Send me any FAIL lines.
2. `./scripts/build.sh`, which now resumes after timeouts. Add `--gpu` only when you want the GPU image.
3. Fill in `.env` with your Copernicus S3 keys, then `./scripts/run_system1.sh probe --aoi-file data/map.kml --start 2026-01-01 --end 2026-03-31`.
4. `./scripts/run_system2.sh prefetch`, which runs `selftest` automatically.
5. A real run: `./scripts/run_pipeline.sh --aoi-file data/map.kml --start … --end …`.
6. Optional: `selftest --compare-reflectance <run_id>`, then tell me which PNG looks right.

## G. Open questions for you

- **Q1.** Should `--start` / `--end` stay required, or default to "last 30 days" when omitted?
- **Q2.** Should an AOI crossing two Sentinel-2 tiles stay out of scope for v1 (clear message plus `--min-coverage`), or do you want the two tiles mosaicked?
- **Q3.** Is the GPU image actually needed now? If your GPU machine isn't the one you're testing on, I'd do the GPU work (step 6) last or leave it for later.
