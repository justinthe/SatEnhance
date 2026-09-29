# SatEnhance — Product Requirements Document

| | |
|---|---|
| **Status** | v0.2 — decisions applied; v1 implemented (see README) |
| **Date** | 2026-09-29 |
| **Owner** | Justin |
| **Scope** | v1 (CLI + containers + automation scripts) |

> **Verification notes.** Items marked **[VERIFY]** are things I believe to be true but could not confirm with certainty (external APIs, licences, exact model behaviour). Check them against primary sources before building. Facts about SEN2SR below come from reading its README and source in the cloned repository at `github.com/ESAOpenSR/SEN2SR`.

---

## 1. Summary

SatEnhance is a containerised, two-stage pipeline:

- **System 1 — Acquire:** takes an Area of Interest (AOI), date range, cloud-cover limit and product type, then downloads Sentinel-1 / Sentinel-2 data into `rawdata/`.
- **System 2 — Enhance:** takes the Sentinel-2 data from `rawdata/`, applies the SEN2SR super-resolution models, and writes georeferenced enhanced rasters into `output/`.

Each system runs standalone, or chained by a single orchestrator script. Everything runs in Docker containers with a CPU variant and a GPU variant.

## 2. Problem and goals

Sentinel-2 is free and frequent, but its 10 m pixels are too coarse for many GIS, urban-planning and environmental tasks. Getting from "I want imagery of Perth" to a sharpened GeoTIFF currently means hand-writing STAC queries, handling cloud filtering, clipping, tiling and model inference. SatEnhance automates that end to end.

### Goals (v1)
1. Accept an AOI as a vector file **or** a plain-text place name.
2. Download the best-matching Sentinel-2 scene for the AOI and time window.
3. Recover gracefully when no data matches (interactive retry or a machine-readable failure).
4. Super-resolve Sentinel-2 imagery using SEN2SR, with tiling for large AOIs.
5. Ship as containers with three run scripts: System 1 only, System 2 only, both chained.

### Non-goals (v1)
- LiDAR and hyperspectral data. Sentinel satellites do not provide LiDAR. Sentinel-2 is multispectral (13 bands), not hyperspectral. See §6.
- **Sentinel-1 (SAR) altogether** — dropped from v1 by decision. SEN2SR is built for Sentinel-2 (see §9.1), so SAR could only ever be download-only. `--sensor sar` is rejected with a clear message.
- Web UI or REST API (interface decision: CLI + shell scripts only).
- Cloud-free compositing / mosaicking across dates (decision: single best scene).
- Real-time or streaming acquisition. Despite the repo README's "real time" wording, v1 is batch.
- Sub-meter output. SEN2SR's documented output is **2.5 m** (see §8.1). The current repo README claims "sub-meter", which should be corrected.

## 3. Users and use cases

**Primary user:** a GIS analyst / developer comfortable with a terminal and Docker.

| # | Use case |
|---|---|
| UC1 | "Download the clearest Sentinel-2 image of Perth City, WA for Jan–Mar 2026, max 10% cloud." |
| UC2 | "Here's my `site.geojson` — get me an enhanced image for last month." |
| UC3 | "I already have raw Sentinel-2 files in `rawdata/`; just enhance them." |
| UC4 | "Run the whole pipeline nightly, unattended, and tell me if it failed." |
| UC5 | ~~"Get Sentinel-1 SAR for this AOI"~~ — deferred (not in v1). |

## 4. Decisions already made (from the interview)

| Topic | Decision |
|---|---|
| Sensor param | Maps to Sentinel-2 products (`rgb`, `multispectral`); `sar` (Sentinel-1) dropped from v1; LiDAR/hyperspectral rejected |
| Interface | CLI + shell scripts only |
| Data source | Copernicus Data Space Ecosystem (CDSE) |
| Compute | Both GPU and CPU image variants |
| Text AOI geocoding | Nominatim (OpenStreetMap), user confirms the match |
| Large AOIs | Configurable max area + tiled inference with overlap |
| Scene selection | Single best scene (lowest cloud cover, most recent as tiebreaker) |
| No-data handling | Interactive prompt, plus `--non-interactive` mode with distinct exit code and JSON report |

## 5. System overview

```
                    ┌─────────────────────────── SatEnhance ───────────────────────────┐
 AOI file / text ─▶ │  SYSTEM 1: acquire                SYSTEM 2: enhance              │
 dates, cloud,      │  ┌──────────────────┐  rawdata/  ┌──────────────────┐  output/    │
 product type       │  │ AOI resolver     │  ───────▶  │ Input validator  │  ───────▶   │
                    │  │ CDSE STAC search │  + manifest│ SEN2SR inference │  GeoTIFFs   │
                    │  │ Download + clip  │  .json     │ Tiling + blend   │  + report   │
                    │  └──────────────────┘            └──────────────────┘             │
                    └───────────────────────────────────────────────────────────────────┘
```

**Contract between systems:** the only coupling is the filesystem. System 1 writes scene data plus a `manifest.json` into `rawdata/<run_id>/`. System 2 reads that folder and writes to `output/<run_id>/`. This lets either system be run, tested or replaced independently.

## 6. Parameters

### 6.1 System 1 parameters

| Parameter | Flag | Type | Required | Default | Notes |
|---|---|---|---|---|---|
| AOI (file) | `--aoi-file PATH` | path | one of file/text | — | See §7 |
| AOI (text) | `--aoi-text "Perth City, Western Australia"` | string | one of file/text | — | Geocoded via Nominatim |
| Start date | `--start YYYY-MM-DD` | date | yes | — | Inclusive |
| End date | `--end YYYY-MM-DD` | date | yes | — | Inclusive; must be ≥ start and not in the future |
| Max cloud cover | `--max-cloud N` | 0–100 (%) | no | 20 | Applies to Sentinel-2 only |
| Sensor / product | `--sensor {rgb,multispectral,sar}` | enum | no | `rgb` | See §6.2 |
| Output dir | `--out DIR` | path | no | `/data/rawdata` | Container path |
| Non-interactive | `--non-interactive` | flag | no | off | Never prompt; fail with exit code 10 on no data |
| Max AOI area | `--max-area-km2 N` | number | no | 100 | Hard cap on the AOI **bounding box**; see §7.4 |
| Min coverage | `--min-coverage N` | 0–100 (%) | no | 95 | Minimum share of the AOI the scene must cover |
| Cache dir | `--cache DIR` | path | no | `/data/cache` | Geocode cache |
| Confirm geocode | `--yes` | flag | no | off | Auto-accept the top geocode match (for automation) |

### 6.2 The `sensor` parameter

Your original list was RGB / LiDAR / Hyperspectral. Sentinel-1 and Sentinel-2 cannot supply LiDAR, and Sentinel-2 is multispectral rather than hyperspectral. The parameter is therefore redefined as a **product type**:

| Value | Mission | Bands / product | Enhanceable by System 2? |
|---|---|---|---|
| `rgb` | Sentinel-2 L2A | B02, B03, B04 (true colour). B08 (NIR) is also fetched because the RGBN model variant needs it. | Yes (10 m → 2.5 m) |
| `multispectral` | Sentinel-2 L2A | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 (the 10 bands used in SEN2SR's examples) | Yes (see §8.2) |
| `sar` | Sentinel-1 GRD | — | **Not supported in v1** (rejected with a message; planned) |
| `lidar`, `hyperspectral` | — | Not available from Sentinel | Rejected with a clear error message |

Rejecting `lidar` / `hyperspectral` explicitly (rather than silently substituting) prevents users assuming they got something they did not.

### 6.3 System 2 parameters

| Parameter | Flag | Default | Notes |
|---|---|---|---|
| Input dir | `--in DIR` | `/data/rawdata` | Reads `manifest.json` |
| Output dir | `--out DIR` | `/data/output` | |
| Model | `--model {lite,full}` | `lite` | `full` needs the GPU image (§8.3) |
| Variant | `--variant {auto,rgbn_x4,multispectral_x4}` | `auto` | `auto` picks from the manifest's product type. The 20 m → 10 m `rswir_x2` model is **deferred**: its input/output layout could not be verified without the weights |
| Tile overlap | `--overlap N` | 32 | Pixels; matches SEN2SR's `predict_large` example |
| Device | `--device {auto,cpu,cuda}` | `auto` | |
| COG | `--cog` | off | Write Cloud-Optimised GeoTIFFs |
| Polygon mask | `--clip-to-polygon` | off | Mask pixels outside the AOI polygon (default: whole bounding box) |
| Reflectance mode | `--reflectance {offset-corrected,raw-div10000}` | `offset-corrected` | How DNs are converted before the model; see §8.3 [VERIFY visually] |
| Block size | `--block N` | 512 | Input pixels per processing block (bounds memory) |

## 7. AOI handling (System 1)

### 7.1 Accepted inputs

| Input | Extensions / form | Library |
|---|---|---|
| GeoJSON / JSON | `.geojson`, `.json` | GeoPandas / Fiona |
| Shapefile | `.shp` (+ `.dbf`, `.shx`, `.prj`), or a `.zip` containing them | GeoPandas |
| KML | `.kml` | GDAL/OGR KML or LIBKML driver |
| KMZ | `.kmz` (zipped KML) | Unzip, then as KML |
| Other OGR vectors | `.gpkg`, `.gml`, `.fgb`, etc. | GDAL/OGR auto-detect |
| Text | `--aoi-text "…"` | Nominatim |

The resolver tries by extension first, then falls back to OGR auto-detection. Anything not readable fails with exit code 2 and a message naming the file and the driver error.

### 7.2 Normalisation rules
1. Read every feature; keep only Polygon / MultiPolygon. Points and lines are rejected with a message (v1 does not buffer them).
2. Reproject to EPSG:4326. If the file has no CRS, fail (exit code 2) rather than guess. Exception: GeoJSON, which is defined as EPSG:4326 by spec.
3. Repair invalid geometries (`make_valid`), drop empty ones.
4. Dissolve multiple features into one AOI. Log the count.
5. Compute the bounding box and the geodesic area (km²), and write the normalised AOI to `rawdata/<run_id>/aoi.geojson`.

### 7.3 Text geocoding (Nominatim)
- Query with `polygon_geojson=1` to obtain the boundary; fall back to the bounding box if none is returned.
- Interactive mode: print the top match (display name, type, bbox, area in km²) and ask `Use this? [Y/n/next]`. `next` shows the next candidate (up to 5).
- `--yes` accepts the top result. In `--non-interactive` mode without `--yes`, fail with exit code 11 rather than guessing.
- Must send a descriptive `User-Agent`, and stay within Nominatim's usage policy (about one request per second, no bulk use) **[VERIFY current policy]**. Cache results on disk.
- Ambiguity note: "Perth" exists in several countries, and "Perth City" is administratively a small area within the greater Perth metro. The confirmation step is important for this reason.

### 7.4 Size limits
- Default hard cap: **100 km²**, applied to the AOI's **bounding box** because that is what is downloaded and enhanced (a thin diagonal polygon can have a much bigger bbox). The error reports both figures.
- Configurable via via `--max-area-km2` or env `SATENHANCE_MAX_AREA_KM2`). Larger AOIs exit with code 3 and a message showing the AOI area, the cap, and a size estimate.
- Before downloading, print an estimate: number of Sentinel-2 tiles intersected, approximate download size, and (from System 2) approximate enhanced output size.
- Reason for the cap: a 10 m → 2.5 m upscale multiplies pixel count by 16. 100 km² at 10 m is about 1 million pixels per band; at 2.5 m it is about 16 million per band. This is manageable, but memory grows quickly with more bands and larger areas. The cap should be tuned after benchmarking.

## 8. System 1 — Acquire: functional requirements

### 8.1 Data source: Copernicus Data Space Ecosystem (CDSE)
- **Catalogue search:** CDSE's STAC API **[VERIFY exact endpoint and collection IDs, e.g. `sentinel-2-l2a`, `sentinel-1-grd`]**.
- **Authentication:** free CDSE account. Credentials are supplied via environment variables / Docker secrets (never baked into images or committed). Data access uses either OAuth tokens or S3-compatible access keys **[VERIFY which method is current and what quotas apply]**.
- **Provider abstraction:** the search and download code sits behind a `Provider` interface (`search()`, `download()`), so another source (AWS Earth Search, Planetary Computer) can be added later without touching the rest.

### 8.2 Search and selection
1. Search the STAC catalogue with `intersects=<AOI>`, `datetime=<start>/<end>`, and a tile-level `eo:cloud_cover` prefilter. The prefilter is deliberately looser than `--max-cloud` (`max-cloud + 30`, capped at 100) because tile cloud is only a proxy; the exact test uses AOI-local cloud (step 2).
2. **Cloud cover caveat:** the catalogue's `eo:cloud_cover` is a *whole-tile* figure. The AOI might be cloud-free while the tile reads 40%, or the reverse. Requirement: after shortlisting candidates, compute **AOI-local cloud fraction** from the Scene Classification Layer (SCL) band (classes for cloud shadow, medium/high-probability cloud, cirrus) and use that for the final ranking and the threshold check.
3. Also require the AOI to be fully covered by the scene (or report the covered percentage). Partial coverage below 100% is a warning, and below a configurable minimum (default 95%) a rejection.
4. Rank: lowest AOI-local cloud fraction first; ties broken by most recent acquisition date.
5. Select **one** scene. Record the runners-up in the manifest.

### 8.3 Download and preparation
- Download only the required bands (not the full SAFE archive) and only the window covering the AOI where the source supports range reads (COGs / JP2 windowed reads).
- Clip to AOI bounding box (keep the bbox, not the exact polygon, so the model has pixels to work with; the polygon mask is applied at the end of System 2 if `--clip-to-polygon` is set).
- Save each band as GeoTIFF in native resolution, preserving CRS and transform. Do not resample in System 1.
- Store reflectance as delivered (integer DN with the L2A scale factor and offset recorded in the manifest). **[VERIFY:** processing baseline 04.00 and later adds a +1000 offset to L2A DNs; the SEN2SR examples divide by 10,000. The offset handling must be checked and made explicit in the manifest so System 2 applies the right scaling.**]**
- Resumable: partial downloads are written to `*.part` and renamed on completion; re-running skips complete files (checksum or size check).
- Retries with exponential back-off on network / 5xx errors (default 4 attempts).

### 8.4 No-data handling

When zero scenes pass the filters:

**Interactive mode (default in a TTY).** Print what was searched and what was found, e.g.:
```
No Sentinel-2 scene found for Perth City, WA between 2026-01-01 and 2026-01-31
with cloud cover ≤ 10%.
Closest candidates: 2026-01-14 (14% cloud), 2026-01-24 (22% cloud).
What would you like to do?
  [1] Raise max cloud cover (suggest 15%)
  [2] Widen date range (suggest 2025-12-01 → 2026-02-28)
  [3] Both
  [4] Enter values manually
  [5] Quit
```
The suggestions are computed from the nearest candidates in the catalogue, so they are data-driven and not arbitrary. After the user chooses, the search reruns. Loop until success or quit.

**Non-interactive mode (`--non-interactive`).** No prompt. Exit with code **10** and write `rawdata/<run_id>/no_data_report.json` containing the query, the counts, the nearest candidates and suggested relaxed parameters. The orchestrator script and any scheduler can read this.

### 8.5 System 1 outputs

```
rawdata/<run_id>/
├── manifest.json          # see §10
├── aoi.geojson            # normalised AOI
├── search_results.json    # candidates considered + ranking
├── S2_<scene_id>/
│   ├── B02.tif  B03.tif  B04.tif  B08.tif  ...
│   └── SCL.tif
└── S1_<scene_id>/         # only when --sensor sar
    └── VV.tif  VH.tif
```

`run_id` = `YYYYMMDDTHHMMSS_<aoi-slug>`.

## 9. System 2 — Enhance: functional requirements

### 9.1 About SEN2SR (from the cloned repository)
- Python package `sen2sr` (v0.8.5 in `pyproject.toml`), models fetched from Hugging Face (`tacofoundation/sen2sr`) via `mlstac`.
- **Sentinel-2 only.** All documented use cases take Sentinel-2 L2A bands. No SAR support.
- Documented use cases:
  1. 10 m + 20 m bands (10 bands) → 2.5 m
  2. 10 m bands (B04, B03, B02, B08) → 2.5 m (`NonReference_RGBN_x4`)
  3. 20 m bands (B05, B06, B07, B8A, B11, B12) → 10 m (`Reference_RSWIR_x2`)
- **Two model families:** `SEN2SRLite` (runs on CPU or GPU) and the full `SEN2SR` (Mamba architecture) which the README says requires a GPU and CUDA > 12.
- Input convention in the examples: reflectance as float32 = DN / 10,000, NaNs replaced with 0.
- Patch size is 128 × 128; `sen2sr.predict_large(model, X, overlap=32)` tiles larger inputs with overlap and reassembles them.
- **Licence discrepancy:** the README badge says MIT, but the `LICENSE` file in the repo is CC0 1.0. **[VERIFY]** Also check the licence terms of the Hugging Face model weights separately. This matters if SatEnhance is ever distributed commercially.
- **Correction to earlier expectations:** output is 2.5 m, not sub-meter.

### 9.2 Variant selection (`--variant auto`)

| Manifest product | Variant used | Output |
|---|---|---|
| `rgb` | `NonReference_RGBN_x4` | 4-band (R, G, B, NIR) at 2.5 m; also a 3-band RGB preview |
| `multispectral` | `SEN2SRLite` (10 bands → 2.5 m) or full `SEN2SR` if `--model full` | 10 bands at 2.5 m |
| `sar` | — | Not applicable in v1 (System 1 rejects `sar`) |

### 9.3 Processing steps
1. **Validate input:** read `manifest.json`; check the required bands exist, share CRS and grid alignment, and have expected sizes. Fail with exit code 20 and an actionable message otherwise.
2. **Assemble the cube:** stack bands (resampling 20 m bands to the 10 m grid only where a model variant requires it), scale to reflectance using the manifest's scale/offset, replace NaN / inf with 0.
3. **Load the model:** download on first use into a persistent model-cache volume, so later runs work offline and don't re-download.
4. **Inference:** `predict_large` with the configured overlap, batching by available memory. Log tile counts and progress.
5. **Georeference the output:** new transform with pixel size 2.5 m (10 m / 4) over the same extent, same CRS as input.
6. **Write GeoTIFF(s)**, plus a side-by-side before/after PNG preview.
7. **Write `enhance_report.json`.**

### 9.4 Device selection
`--device auto`: use CUDA if `torch.cuda.is_available()`, else CPU. If `--model full` is requested and no GPU is present, fail early with exit code 21 and a message suggesting `--model lite`. No silent downgrade — the user asked for a specific model.

### 9.5 Memory and tile management
- Process in tiles so memory use does not scale with AOI size; write output windows incrementally rather than holding the full result in RAM.
- Configurable `--max-memory-gb` guard (advisory) to avoid OOM-killing the container.
- Overlap blending is delegated to `predict_large`. Requirement: a test that checks for visible seams on a synthetic gradient image.

### 9.6 System 2 outputs

```
output/<run_id>/
├── enhanced_<scene_id>_2p5m.tif       # multi-band GeoTIFF, 2.5 m
├── enhanced_<scene_id>_rgb_2p5m.tif   # 3-band RGB, 8-bit stretched (for viewing)
├── preview_before_after.png
└── enhance_report.json                 # model, version, device, runtime, tile count, warnings
```

Output GeoTIFFs are tiled, compressed (deflate), and optionally converted to Cloud-Optimised GeoTIFF with `--cog`.

### 9.7 Caveat that must appear in the report and README
Super-resolution output is **model-predicted detail**, not measured data. It is suitable for visualisation and some analysis, but should not be used as ground truth for measurement or legal boundaries. The report will include a short disclaimer.

## 10. Manifest (`manifest.json`) — System 1 → System 2 contract

```json
{
  "schema_version": "1.0",
  "run_id": "20260929T134600_perth-city",
  "created_utc": "2026-09-29T13:46:00Z",
  "query": {
    "aoi_source": "text:Perth City, Western Australia",
    "aoi_file": "aoi.geojson",
    "start": "2026-01-01", "end": "2026-03-31",
    "max_cloud": 10, "sensor": "rgb"
  },
  "aoi": { "crs": "EPSG:4326", "bbox": [115.85, -31.97, 115.87, -31.94], "area_km2": 12.3 },
  "scene": {
    "mission": "sentinel-2", "product": "L2A", "id": "…",
    "datetime": "2026-02-14T02:31:11Z",
    "tile_cloud_cover": 6.1, "aoi_cloud_fraction": 1.2, "aoi_coverage": 100.0,
    "processing_baseline": "…",
    "bands": { "B04": {"file": "S2_…/B04.tif", "res_m": 10, "scale": 0.0001, "offset": -0.1} }
  },
  "alternates": [ ],
  "status": "complete",
  "tool_version": "0.1.0"
}
```

(Exact scale/offset values must come from the scene metadata, not hard-coded. See the §8.3 [VERIFY] note.)

## 11. Containers

### 11.1 Images

| Image | Purpose | Base | Notes |
|---|---|---|---|
| `satenhance-acquire` | System 1 | `python:3.11-slim` + GDAL | No ML dependencies; small |
| `satenhance-enhance:cpu` | System 2 (CPU) | `python:3.11-slim` + PyTorch CPU wheels | Runs `lite` model only |
| `satenhance-enhance:gpu` | System 2 (GPU) | NVIDIA CUDA 12.x runtime base | Adds `mamba-ssm` for `--model full`. Build is slow and fragile per the SEN2SR README; pin versions |

Requirements:
- Non-root user in every image.
- Pinned dependency versions (lockfile), reproducible builds.
- Model weights are **not** baked into images by default; they're pulled into a named volume on first run. Optionally, a `make prefetch-models` step bakes them for offline use.
- Health/`--version` entry points on each image.

### 11.2 Volumes and configuration

| Mount | Container path | Purpose |
|---|---|---|
| `./rawdata` | `/data/rawdata` | System 1 output, System 2 input |
| `./output` | `/data/output` | System 2 output |
| `./cache` | `/data/cache` | Geocode cache, model cache |
| `./secrets` or env | — | CDSE credentials (never committed; `.env` in `.gitignore`, `.env.example` provided) |

### 11.3 Compose
A `docker-compose.yml` defines the services `acquire`, `enhance-cpu`, `enhance-gpu` (using Compose's GPU device reservation, under a profile so CPU-only machines are unaffected).

### 11.4 Network
System 1 needs outbound HTTPS to CDSE and Nominatim. System 2 needs outbound HTTPS only for the first model download; afterwards it can run with networking disabled.

## 12. Automation scripts

All scripts are POSIX-friendly bash in `scripts/`, use `set -euo pipefail`, load `.env`, and forward extra args.

| Script | Behaviour |
|---|---|
| `scripts/run_system1.sh` | Runs `acquire` with the given args. Prints the resulting `run_id` and rawdata path on success. |
| `scripts/run_system2.sh` | Runs `enhance` against an existing rawdata folder (`--run-id ID`, default: latest). Auto-selects GPU image if `nvidia-smi` and the Docker NVIDIA runtime are present, else CPU; override with `--cpu` / `--gpu`. |
| `scripts/run_pipeline.sh` | Runs System 1, then System 2 on that run's output. Stops if System 1 fails. Propagates exit codes. |
| `scripts/build.sh` | Builds the images. |

Example:
```bash
./scripts/run_pipeline.sh \
  --aoi-text "Perth City, Western Australia" --yes \
  --start 2026-01-01 --end 2026-03-31 --max-cloud 10 --sensor rgb
```

**Pipeline behaviour on no data:** when run in a terminal, System 1's interactive prompt works as normal inside the pipeline (the script allocates a TTY). When run without a TTY (cron/CI), the script adds `--non-interactive`, and exit code 10 propagates to the caller.

### 12.1 Exit codes

| Code | Meaning |
|---|---|
| 0 | Success (or System 2 skipped a SAR run) |
| 1 | Unexpected error |
| 2 | Invalid input (unreadable AOI, missing CRS, bad dates, unsupported sensor) |
| 3 | AOI exceeds max area |
| 4 | Authentication / credentials failure |
| 5 | Network / provider failure after retries |
| 10 | No data matched the query (non-interactive) |
| 11 | Geocode needs confirmation (non-interactive, no `--yes`) |
| 20 | System 2 input validation failed |
| 21 | Requested model not supported on this hardware |
| 22 | Inference failure (e.g. out of memory) |

## 13. Suggested repository layout

```
SatEnhance/
├── README.md
├── docs/PRD.md
├── docker-compose.yml
├── .env.example
├── scripts/                 # run_system1.sh, run_system2.sh, run_pipeline.sh, build.sh
├── acquire/                 # System 1
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── satenhance_acquire/  # aoi.py, geocode.py, providers/cdse.py, select.py, download.py, cli.py
├── enhance/                 # System 2
│   ├── Dockerfile.cpu
│   ├── Dockerfile.gpu
│   ├── pyproject.toml
│   └── satenhance_enhance/  # manifest.py, cube.py, model.py, infer.py, write.py, cli.py
├── shared/                  # manifest schema (JSON Schema), exit codes
├── rawdata/  output/  cache/    # gitignored, mounted volumes
└── tests/
```

## 14. Non-functional requirements

| Area | Requirement |
|---|---|
| Reproducibility | Same inputs + same model version → same scene selection; the manifest records everything needed to rerun |
| Observability | Structured logs (JSON with `--log-json`), human-readable by default; progress bars only when a TTY is attached |
| Security | No credentials in images, logs or manifests; secrets redacted in logs; non-root containers; downloaded data treated as untrusted (validate file types and sizes) |
| Performance targets | To be set after benchmarking — I'm not going to invent numbers. Suggested measurement: time and peak memory for a 25 km² and a 100 km² AOI on CPU and GPU |
| Portability | Linux with Docker is primary. macOS / Windows via Docker Desktop for CPU. GPU support needs Linux + NVIDIA Container Toolkit |
| Data volume | Docs must warn that downloads and outputs can be many GB, and that `rawdata/` and `output/` are not auto-cleaned |
| Licensing | Sentinel data is free and open under Copernicus terms **[VERIFY attribution requirements]**; Nominatim/OSM data requires ODbL attribution if outputs redistribute it; check SEN2SR code and weights licences (§9.1) |

## 15. Testing and acceptance criteria

### 15.1 Test plan
- **Unit:** each AOI format (fixture files for KML, KMZ, GeoJSON, SHP-zip, GPKG); reprojection; invalid geometry repair; area calculation; cloud-ranking logic; manifest schema validation.
- **Provider tests:** recorded HTTP responses (no live network in CI); one opt-in live smoke test using real credentials.
- **Integration:** tiny AOI end to end in CPU containers, using a pre-recorded small scene.
- **Model tests:** output shape = 4× input, transform correct, no NaNs, seam check on a synthetic image.
- **Failure paths:** no data (interactive with scripted stdin, and non-interactive), bad credentials, oversize AOI, SAR passed to System 2, `--model full` on CPU.

### 15.2 Acceptance criteria (v1 is done when…)
1. `run_system1.sh` with a GeoJSON AOI produces a valid `rawdata/<run_id>/` with a manifest that validates against the schema.
2. The same works with KML, KMZ, zipped Shapefile, and GeoPackage.
3. `--aoi-text "Perth City, Western Australia"` shows a match for confirmation and, once confirmed, downloads data.
4. An impossible query (e.g. `--max-cloud 0` in a cloudy window) triggers the retry prompt interactively, and exit code 10 plus `no_data_report.json` in non-interactive mode.
5. `run_system2.sh` on an existing rawdata folder produces a GeoTIFF whose pixel size is 2.5 m and whose extent matches the input.
6. `run_pipeline.sh` completes both stages unattended.
7. `--sensor lidar` and `--sensor hyperspectral` are rejected with a clear message; `--sensor sar` downloads and System 2 reports "skipped".
8. Both CPU and GPU System 2 images build and run; the CPU image completes `--model lite` end to end.
9. No credentials appear in images, logs, or the repo.

## 16. Milestones (suggested)

| # | Milestone | Contents |
|---|---|---|
| M0 | Scaffolding | Repo layout, compose file, CI skeleton, `.env.example` |
| M1 | AOI resolver | All vector formats + normalisation + tests |
| M2 | System 1 core | CDSE auth, STAC search, SCL cloud check, download, manifest |
| M3 | System 1 UX | Geocoding, no-data retry flow, size guard, `run_system1.sh` |
| M4 | System 2 CPU | Lite model, variant selection, tiling, GeoTIFF write, `run_system2.sh` |
| M5 | System 2 GPU | CUDA image, full model, device logic |
| M6 | Pipeline | `run_pipeline.sh`, exit-code propagation, docs |
| M7 | Hardening | Benchmarks, tune area cap, security review, README |

## 17. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| CDSE API/auth changes or quota limits **[VERIFY]** | System 1 breaks | Provider abstraction; pin and test against recorded responses |
| `mamba-ssm` build failures in the GPU image | Full model unusable | Pin CUDA/PyTorch versions; treat `lite` as the default path; document known-good combination |
| Whole-tile cloud metadata misleads selection | Poor scenes chosen | SCL-based AOI-local cloud check (§8.2) |
| L2A reflectance offset mis-handled | Wrong colours from the model | Record scale/offset in manifest; unit test on a known scene |
| Ambiguous place names geocode wrongly | Wrong area downloaded | Confirmation step; show bbox and area |
| Users expect sub-meter or "real" detail | Misuse of outputs | Clear 2.5 m statement and the model-prediction disclaimer (§9.7); fix repo README |
| SEN2SR licence ambiguity (MIT badge vs CC0 file, weights unknown) | Legal uncertainty | Verify before distribution |
| Very large outputs fill disk | Failed runs | Size estimate before download; cap; docs |

## 18. Open questions — resolved

| # | Question | Decision |
|---|---|---|
| 1 | Output formats | GeoTIFF (full bands + 8-bit RGB) and a before/after PNG; `--cog` optional |
| 2 | Sentinel-1 | Dropped from v1 |
| 3 | Output CRS | Native Sentinel-2 UTM zone |
| 4 | Polygon clipping | AOI bounding box by default; `--clip-to-polygon` masks outside the polygon |
| 5 | Retention | `rawdata/` is kept; nothing is auto-deleted |
| 6 | Defaults | `--max-cloud 20`, `--max-area-km2 100` (area cap applies to the bounding box) |
| 7 | Scheduling | No cron integration; `--non-interactive` + exit codes are enough for external schedulers |
| 8 | README claims | Corrected ("sub-meter" / "real time" removed) |

Still open / unverified (each marked **[VERIFY]** where it appears): CDSE STAC/S3 details, the full-model weight path, the Sentinel-2 reflectance offset convention expected by SEN2SR, the GPU image build, the SEN2SR licence and weights licence.

## 19. Assumptions (correct me if wrong)
- You have, or will create, a free CDSE account.
- Docker (and, for GPU, the NVIDIA Container Toolkit) is installed on the machine that runs this.
- v1 operates on one AOI and one scene per run.
- Python is the implementation language.
