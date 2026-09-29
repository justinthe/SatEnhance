# SatEnhance v1 — Implementation Plan

> On approval, the first step copies this file into the repo as `docs/IMPLEMENTATION_PLAN.md`, so it is versioned with the code.

## Context

`docs/PRD.md` (branch `claude/practical-ptolemy-w1xnmq`) specifies a two-stage containerised pipeline:
**System 1 (acquire)** resolves an AOI (vector file or place name), finds the best Sentinel-2 scene on Copernicus Data Space (CDSE), and writes it into `rawdata/`. **System 2 (enhance)** runs SEN2SR super-resolution (10 m → 2.5 m) and writes GeoTIFFs to `output/`. Three scripts run each system alone or chained. The repo currently holds only `README.md` and `docs/PRD.md`, so all code is new.

### Decisions (from both interviews)
| Topic | Decision |
|---|---|
| Sensor | `rgb`, `multispectral` (Sentinel-2 L2A). `sar`, `lidar`, `hyperspectral` are **rejected** in v1 (Sentinel-1 dropped) |
| Interface | CLI + shell scripts |
| Source | CDSE, behind a swappable `Provider` interface |
| Compute | CPU image (lite model) + GPU image (lite + full) |
| Text AOI | Nominatim, confirm the match (`--yes` to auto-accept) |
| Size | Hard cap, default 100 km²; tiled inference |
| Scene | Single best (lowest AOI-local cloud, then most recent) |
| No data | Interactive prompt; `--non-interactive` → exit 10 + `no_data_report.json` |
| Outputs | GeoTIFF (multi-band + 8-bit RGB) + before/after PNG; `--cog` optional |
| CRS / clip | Native UTM, AOI bbox; `--clip-to-polygon` optional mask |
| Testing | Unit + mocked HTTP + CPU container smoke test; live CDSE test opt-in |

### Defaults I'm assuming for the remaining PRD open questions (say if wrong)
- `rawdata/` is **kept** after enhancement (no auto-clean).
- Default `--max-cloud 20`, `--max-area-km2 100`.
- Scheduling: no cron setup; non-interactive flags are enough.
- The repo README is rewritten to remove the "sub-meter" and "real time" claims.
- The PRD is updated to match these decisions (S1 dropped, exit-code table adjusted).

### Environment constraints found while planning
The sandbox network policy **blocks** `stac.dataspace.copernicus.eu`, `huggingface.co`, `nominatim.openstreetmap.org` and `download.pytorch.org`. PyPI, files.pythonhosted.org and Docker Hub are reachable, and Docker runs (4 CPU, 15 GB RAM, no GPU). Consequences:
- CDSE and Nominatim code is tested only against recorded responses here.
- Real SEN2SR weights can't be downloaded here, so the container smoke test uses a **stub model** (bicubic ×4) through the same code path. Real-model runs must be verified on your machine.
- The GPU image is written but can't be built and run here. It is only syntax-checked (`docker build --check`, or hadolint if available).
- The CPU image takes the torch index as a build arg (`TORCH_INDEX_URL`, default `https://download.pytorch.org/whl/cpu`). Here I'll build with the PyPI index instead.

---

## Repository layout (target)

```
SatEnhance/
├── README.md                     # rewritten: quickstart, honest claims, disclaimer
├── docs/PRD.md  docs/IMPLEMENTATION_PLAN.md
├── docker-compose.yml
├── .env.example  .gitignore  .dockerignore
├── Makefile                      # build, test, lint, smoke, prefetch-models
├── scripts/
│   ├── build.sh  run_system1.sh  run_system2.sh  run_pipeline.sh
│   └── _common.sh                # env loading, TTY detection, GPU detection, uid/gid
├── common/                       # package: satenhance_common (installed in both images)
│   ├── pyproject.toml
│   └── satenhance_common/  exit_codes.py  manifest.py  manifest.schema.json  logging.py  paths.py
├── acquire/
│   ├── Dockerfile  pyproject.toml
│   ├── satenhance_acquire/
│   │   ├── cli.py  aoi.py  geocode.py  sizing.py  select.py  download.py  nodata.py  prompts.py
│   │   └── providers/  base.py  cdse.py
│   └── tests/  (fixtures/ with recorded STAC + Nominatim JSON)
├── enhance/
│   ├── Dockerfile.cpu  Dockerfile.gpu  pyproject.toml
│   ├── satenhance_enhance/
│   │   ├── cli.py  inputs.py  cube.py  models.py  infer.py  georef.py  write.py  preview.py  report.py
│   └── tests/
├── tests/smoke/  make_synthetic_rawdata.py  test_pipeline_smoke.sh
└── .github/workflows/ci.yml
```

Three installable packages are used so the acquire image carries no torch, and the enhance image carries no geocoding / STAC code. The manifest schema lives in `common` so both sides validate against the same file.

---

## Implementation steps

### Step 0 — Scaffolding
- Copy this plan to `docs/IMPLEMENTATION_PLAN.md`.
- Add `.gitignore` (`rawdata/ output/ cache/ .env secrets/`), `.dockerignore`, `.env.example` (`CDSE_S3_ACCESS_KEY`, `CDSE_S3_SECRET_KEY`, `CDSE_USERNAME`, `CDSE_PASSWORD`, `NOMINATIM_USER_AGENT`, `SATENHANCE_MAX_AREA_KM2`), a `Makefile`, and a ruff config.
- Tooling: Python 3.11, `pyproject.toml` per package (hatchling), `pytest`, `ruff`. Pin with `uv pip compile` into a `requirements.lock` per image.

### Step 1 — `satenhance_common`
- `exit_codes.py`: an `IntEnum` matching PRD §12.1 (0,1,2,3,4,5,10,11,20,21,22), plus a `SatEnhanceError(code, message)` base exception. Each CLI maps it to `sys.exit(code)`.
- `manifest.py`: pydantic v2 models (`Manifest`, `Query`, `Aoi`, `Scene`, `BandInfo`, `Alternate`). `load(path)` / `save(path)` validate against `manifest.schema.json` (generated from the models and committed, with a test ensuring they stay in sync).
- `logging.py`: human-readable by default; `--log-json` switches to JSON lines. A redaction filter masks any env value whose name contains `KEY`, `SECRET` or `PASSWORD`.
- `paths.py`: `run_id` builder `YYYYMMDDTHHMMSS_<slug>` and a `LATEST` pointer-file helper.

### Step 2 — AOI resolver (`acquire/aoi.py`)
- `load_aoi(path) -> AoiResult(geometry_wgs84, source, feature_count)`.
  - `.kmz` → unzip to a temp dir → first `.kml`. `.zip` → find `.shp` inside (read via `pyogrio` with `/vsizip/`).
  - Everything else goes through `geopandas.read_file` (pyogrio engine), which covers GeoJSON, JSON, SHP, KML, GPKG, GML and FGB. On a driver error → `SatEnhanceError(INVALID_INPUT)`, naming the file and driver.
- Normalise per PRD §7.2: keep Polygon/MultiPolygon only (points/lines → error), require a CRS (GeoJSON defaults to 4326), `to_crs(4326)`, `shapely.make_valid`, drop empties, `union_all()`.
- `sizing.py`: geodesic area via `pyproj.Geod(ellps="WGS84").geometry_area_perimeter`. Also estimates rawdata and output size: pixels = area / 100 m² × bands × 2 bytes; output ×16. Over the cap → exit 3 with area, cap and estimates.
- Writes `aoi.geojson`.

### Step 3 — Geocoding (`acquire/geocode.py`, `prompts.py`)
- `requests` GET `https://nominatim.openstreetmap.org/search?q=…&format=jsonv2&polygon_geojson=1&limit=5`, with a `User-Agent` from env (required, with a default containing the project name). Throttle to ≥1 s between calls. On-disk JSON cache at `/data/cache/geocode/<sha1(query)>.json`.
- Use the returned polygon if there is one, else the bbox polygon. Show display name, type, bbox and area. Prompt `Use this? [Y/n/next]`, cycling through up to 5 candidates.
- `--yes` takes the top result. With `--non-interactive` and no `--yes` → exit 11. Zero results → exit 2.
- The resulting geometry passes through the same normalisation and size cap as files do.

### Step 4 — Provider layer (`providers/base.py`, `providers/cdse.py`)
- `class Provider(Protocol)`: `search(aoi, start, end, max_cloud, limit) -> list[Candidate]`, `open_band(candidate, band) -> rasterio dataset/URL`, `check_auth()`.
- `CdseProvider`:
  - Search: `pystac_client.Client.open("https://stac.dataspace.copernicus.eu/v1")`, collection `sentinel-2-l2a`, `intersects=aoi`, `datetime=start/end`, `query={"eo:cloud_cover": {"lte": max_cloud}}`. **[VERIFY endpoint, collection id, and whether it supports the `query` extension or needs `filter` (CQL2). The code will try CQL2 `filter` and fall back to client-side filtering.]**
  - Data access: the asset hrefs are expected to be `s3://eodata/...` JP2 files. Read them with rasterio through GDAL `/vsis3/`, with `AWS_S3_ENDPOINT=eodata.dataspace.copernicus.eu`, `AWS_VIRTUAL_HOSTING=FALSE`, `AWS_HTTPS=YES`, and the keys from `CDSE_S3_ACCESS_KEY/SECRET`. This gives windowed reads, so only the AOI window is downloaded. **[VERIFY: S3 key issuance and quotas at CDSE. If windowed S3 access turns out not to be viable, fallback is OData full-product download with an OAuth token (username/password), then local extraction — kept behind the same interface but implemented only if needed.]**
  - `check_auth()` does a cheap HEAD/`gdalinfo`-style open of a known object → exit 4 on 401/403.
  - Retries: `urllib3.Retry` / `tenacity`, 4 attempts, exponential back-off, then exit 5.
- Candidates are normalised to `Candidate(id, datetime, tile_cloud, processing_baseline, assets{band→href}, footprint)`.

### Step 5 — Scene selection (`select.py`)
- Shortlist the top N (default 5) by tile cloud. For each, read the **SCL** band over the AOI window (20 m, cheap) and compute:
  - `aoi_coverage` = share of AOI pixels with SCL ≠ 0 (no-data).
  - `aoi_cloud_fraction` = share of covered pixels in SCL classes {3 cloud shadow, 8 medium cloud, 9 high cloud, 10 cirrus}.
- Reject if coverage < `--min-coverage` (default 95). Warn below 100.
- Reject if `aoi_cloud_fraction > max_cloud`. Rank by (cloud asc, datetime desc).
- If several S2 tiles cover the AOI on the same date, v1 picks the single tile with the best coverage. Cross-tile mosaicking is out of scope, which the size cap makes rare.
- `search_results.json` records every candidate with its metrics and the reason it was rejected.

### Step 6 — Download (`download.py`)
- Band sets: `rgb` → B02, B03, B04, B08, SCL. `multispectral` → B02–B08, B8A, B11, B12, SCL.
- For each band, compute the AOI bbox window in the band's native CRS/resolution (snapped to the pixel grid). Read the window, write a tiled deflate GeoTIFF `S2_<id>/<band>.tif` via `*.part` + atomic rename. Skip if it already exists with the expected shape.
- Scale/offset from scene metadata: `processing_baseline >= 04.00` → offset −0.1 (DN offset 1000), else 0; scale 1e-4. Recorded per band in the manifest. **[VERIFY against product metadata `BOA_ADD_OFFSET`. I'll read it from STAC properties if exposed, else derive it from the baseline.]**
- Write `manifest.json` (status `complete`), update `rawdata/LATEST`, and print `SATENHANCE_RUN_ID=<id>` as the last stdout line.

### Step 7 — No-data flow (`nodata.py`)
- When zero scenes pass, run a **diagnostic search**: same AOI, date window widened by ±max(30 days, range length), cloud ≤ 100, limit 20. Compute suggestions:
  - `suggested_max_cloud` = ceil of the lowest cloud among in-range candidates (+2 margin), capped at 100.
  - `suggested_dates` = the smallest widening that includes the nearest candidate under the current cloud limit.
- Interactive (stdin is a TTY and not `--non-interactive`): the menu from PRD §8.4, re-search, loop.
- Non-interactive: write `no_data_report.json` (query, counts, nearest candidates, suggestions), exit 10.

### Step 8 — System 1 CLI (`cli.py`, Typer)
`satenhance-acquire --aoi-file|--aoi-text --start --end [--max-cloud 20] [--sensor rgb] [--out /data/rawdata] [--max-area-km2 100] [--min-coverage 95] [--non-interactive] [--yes] [--log-json]`
- Validation: exactly one AOI option; `start ≤ end ≤ today`. `lidar`, `hyperspectral`, `sar` → exit 2 with an explanation (sar says "planned, not in v1").

### Step 9 — Enhance: inputs and cube (`inputs.py`, `cube.py`)
- Load and validate the manifest (bad → exit 20). Check the band files exist and the 10 m bands share CRS, transform and shape.
- Build a `float32 (C,H,W)` cube: `DN*scale + offset`, clip at 0, `nan_to_num`. For multispectral, resample 20 m bands onto the 10 m grid with bilinear interpolation (`rasterio.warp.reproject`).
- The band order matches SEN2SR's examples: RGBN = [B04,B03,B02,B08], 10-band = [B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12].
- Note: I'm not certain whether SEN2SR expects the −0.1 offset already applied. Its examples use `cubo` data ÷ 10,000. The reflectance mode goes behind a `--reflectance {offset-corrected,raw-div10000}` flag, default `offset-corrected`, flagged in the README to validate visually.

### Step 10 — Models (`models.py`)
- Registry `{(family, variant): mlm.json URL}` from the SEN2SR README:
  - lite main: `…/SEN2SRLite/main/mlm.json`
  - lite RGBN: `…/SEN2SRLite/NonReference_RGBN_x4/mlm.json`
  - lite RSWIR ×2: `…/SEN2SRLite/Reference_RSWIR_x2/mlm.json`
  - full: `…/SEN2SR/main/mlm.json` **[VERIFY path. It isn't shown in the README.]**
- `load_model(family, variant, device, cache=/data/cache/models)`: `mlstac.download` if the model is absent, then `mlstac.load(dir).compiled_model(device)`. Download failure → exit 5; missing CUDA for `full` → exit 21 before any download.
- `auto` variant: `rgb` → RGBN_x4; `multispectral` → lite main (or full with `--model full`).
- **Stub model** for tests: selected only when `SATENHANCE_STUB_MODEL=1`. It is a torch module doing bicubic ×4 interpolation, so the full pipeline and container run offline. It is clearly labelled in `enhance_report.json`.
- `make prefetch-models` / `enhance prefetch` subcommand fills the cache volume ahead of time.

### Step 11 — Inference with bounded memory (`infer.py`, `georef.py`, `write.py`)
- Outer blocking: iterate over 512×512 input blocks with a 32 px halo (reusing SEN2SR's `define_iteration` idea). For each block, call `sen2sr.predict_large(model, X_block, overlap=32)` under `torch.inference_mode()`, crop the halo ×4, and write that window straight to the open output GeoTIFF. Peak memory then scales with block size, not AOI size. A tqdm progress bar is shown only on a TTY.
- CUDA OOM → retry the block at half size once, then exit 22.
- `georef.py`: output transform = input transform with pixel size ÷ scale factor (4, or 2 for RSWIR), same CRS, same extent.
- `write.py`:
  - `enhanced_<scene>_2p5m.tif`: uint16 reflectance×10000, tiled 512, deflate+predictor 2, nodata 0, band descriptions set.
  - `enhanced_<scene>_rgb_2p5m.tif`: 8-bit RGB, 2–98 percentile stretch computed on a decimated read.
  - `--cog`: convert with the GDAL COG driver.
  - `--clip-to-polygon`: `rasterio.features.geometry_mask` from `aoi.geojson` (reprojected to UTM), set outside pixels to nodata.
- `preview.py`: before/after PNG (Pillow), with the input upsampled nearest-neighbour ×4 next to the output, downscaled to ≤2000 px wide.
- `report.py`: `enhance_report.json` with the model URL/family/variant, sen2sr and torch versions, device, runtime, block/tile counts, warnings, stub flag, and the "model-predicted detail" disclaimer.

### Step 12 — System 2 CLI
`satenhance-enhance [--run-id ID|--in DIR] [--out /data/output] [--model lite|full] [--variant auto|rgbn_x4|multispectral_x4|rswir_x2] [--overlap 32] [--block 512] [--device auto|cpu|cuda] [--cog] [--clip-to-polygon] [--reflectance …] [--log-json]` plus `satenhance-enhance prefetch`.
With no `--run-id`, it reads `rawdata/LATEST`.

### Step 13 — Containers
- `acquire/Dockerfile`: `python:3.11-slim`; pip wheels for rasterio, pyogrio, pyproj and shapely bundle GDAL/PROJ, so no system GDAL is needed. Non-root user `app` (uid 1000). `ENTRYPOINT ["satenhance-acquire"]`.
- `enhance/Dockerfile.cpu`: `python:3.11-slim` + `torch` from `ARG TORCH_INDEX_URL` + `sen2sr mlstac rasterio pillow`. `cubo` is not needed because System 1 replaces it. **[VERIFY that `mlstac` / `sen2sr` don't import `cubo` at runtime.]**
- `enhance/Dockerfile.gpu`: multi-stage. The builder is `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`: install python3.11, torch cu124, `mamba-ssm --no-build-isolation` (pinned), sen2sr, mlstac. The runtime stage is `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` with a copied venv. **[VERIFY a known-good torch / mamba-ssm / CUDA combo. The SEN2SR README warns that this build is fragile.]**
- `docker-compose.yml`: services `acquire`, `enhance-cpu`, `enhance-gpu` (profile `gpu`, `deploy.resources.reservations.devices: [{driver: nvidia, count: all, capabilities: [gpu]}]`). Volumes `./rawdata:/data/rawdata`, `./output:/data/output`, `./cache:/data/cache`. `env_file: .env`. `user: "${UID:-1000}:${GID:-1000}"` so host files are owned by the user. `enhance-*` services get `network_mode: none` when `SATENHANCE_OFFLINE=1`.

### Step 14 — Scripts (`scripts/`)
- `_common.sh`: `set -euo pipefail`, repo-root detection, load `.env`, export UID/GID, create the `rawdata/output/cache` dirs. `tty_flags` picks `-T` when there is no TTY, and `has_gpu` checks `nvidia-smi` plus `docker info | grep -i nvidia`.
- `build.sh [--cpu-only|--gpu]`: `docker compose build` for the selected services.
- `run_system1.sh …args`: `docker compose run --rm $(tty_flags) acquire "$@"`. With no TTY it appends `--non-interactive` unless already present. Echoes the run id from `rawdata/LATEST`. Propagates the exit code.
- `run_system2.sh [--cpu|--gpu] …args`: GPU auto-detect, then `docker compose run --rm enhance-{cpu,gpu} "$@"`.
- `run_pipeline.sh [--cpu|--gpu] <system1 args> [-- <system2 args>]`: runs System 1, then reads the run id it printed (`SATENHANCE_RUN_ID=` line, captured via `tee`; `LATEST` only as fallback). On a non-zero exit it stops and exits with the same code. Otherwise it runs System 2 with `--run-id` and prints both output paths.

### Step 15 — Tests
- **acquire unit** (pytest): fixtures generated in `conftest.py` with geopandas (GeoJSON, KML, KMZ, SHP-zip, GPKG, no-CRS SHP, a point file, an invalid bow-tie polygon), so there are no binary fixtures. Plus area calc, the size cap, the run-id slug, sensor rejection and date validation.
- **geocode**: `responses`-mocked Nominatim JSON (a Perth multi-match), the confirmation loop with scripted stdin, `--yes`, and exit 11.
- **provider/select**: recorded STAC JSON fixtures (hand-authored to match the STAC item shape) served via `responses`. SCL windows use small synthetic rasters exercising coverage, cloud, ranking and tie-break.
- **no-data**: the diagnostic search, suggestion maths, interactive loop via `monkeypatch` input, and the non-interactive exit-10 report.
- **download**: `open_band` is monkeypatched to local synthetic JP2/GTiff files; checks windowing, `.part` resume and the manifest schema.
- **enhance unit**: manifest validation errors (exit 20), cube scaling/offset, 20 m→10 m resample, georef (pixel size 2.5, same bounds), stub-model shape ×4 with no NaNs, a **seam test** (a synthetic smooth gradient through the blocked path, so no discontinuity above tolerance at block borders), polygon mask, COG flag, and `--model full` on CPU → exit 21.
- **smoke** (`tests/smoke/test_pipeline_smoke.sh`, `make smoke`): build the acquire and enhance-cpu images, generate a synthetic `rawdata/<run>` with `make_synthetic_rawdata.py` (a valid manifest and 128×128 bands), run `run_system2.sh --cpu` with `SATENHANCE_STUB_MODEL=1`, and assert the outputs exist with the correct size/CRS (via a rasterio one-liner in the container). Also run `run_system1.sh --sensor lidar …` → expect exit 2, and a missing-credentials run → expect exit 4.
- **opt-in live test** (`pytest -m live`, needs `.env` CDSE creds): a tiny AOI search + download. Not run in CI or here.

### Step 16 — CI (`.github/workflows/ci.yml`)
Jobs: `lint` (ruff), `test-acquire`, `test-enhance` (CPU torch), `smoke` (build CPU images + smoke script), `gpu-dockerfile-check` (build `--check` only).

### Step 17 — Docs
- Rewrite `README.md`: what it does, the honest 2.5 m claim, prerequisites (Docker, CDSE account + S3 keys, NVIDIA toolkit for GPU), quickstart for all three scripts, parameters, exit codes, the disclaimer, data licences/attribution (Copernicus, OSM ODbL, SEN2SR) with **[VERIFY]** notes.
- Update `docs/PRD.md`: S1 dropped from v1, decisions above, open questions resolved.

---

## Critical files (new)
- `common/satenhance_common/manifest.py`: the System 1 ↔ System 2 contract
- `acquire/satenhance_acquire/providers/cdse.py`: highest external-API risk
- `acquire/satenhance_acquire/select.py`: SCL-based cloud/coverage ranking
- `enhance/satenhance_enhance/infer.py`: blocked inference and memory bounds
- `scripts/run_pipeline.sh`: run-id handoff and exit-code propagation
- `enhance/Dockerfile.gpu`: fragile mamba-ssm build

## Reused external code
- `sen2sr.predict_large(model, X, overlap)` for per-block tiling; the `define_iteration` / `fix_lastchunk` pattern (`sen2sr/utils.py`) for the outer block grid
- `mlstac.download` / `mlstac.load(...).compiled_model(device)` for weights
- `pystac-client`, `rasterio` (windowed reads, `/vsis3/`, `/vsizip/`), `geopandas`/`pyogrio`, `pyproj.Geod`, `shapely.make_valid`

## Commit strategy
One commit per step group (0–1, 2–3, 4–8, 9–12, 13–14, 15–16, 17), pushed to `claude/practical-ptolemy-w1xnmq` after tests pass at each point. No PR unless you ask.

---

## Verification
1. `make lint` and `make test` all pass locally (acquire, enhance, common).
2. `./scripts/build.sh --cpu-only` builds `acquire` and `enhance-cpu`.
3. `make smoke`: System 2 runs in a container on synthetic rawdata with the stub model. Checks: output pixel size = 2.5 m, bounds equal input bounds, CRS unchanged, PNG + report written, exit 0. The lidar → exit 2 and no-creds → exit 4 paths are also checked through the scripts.
4. `run_pipeline.sh` is exercised with the acquire stage mocked (`SATENHANCE_PROVIDER=fixture`, a test-only provider that serves the synthetic scene). This proves the run-id handoff and exit-code propagation, including the no-data → exit 10 path without a TTY.
5. **What I can't verify here, which you should run on your machine:** live CDSE search/download, Nominatim geocoding of "Perth City, Western Australia", real SEN2SR weights (visual check of the reflectance mode), and the GPU image build + `--model full`. I'll give you the exact commands in the README "First run" section.
