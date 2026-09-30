# SatEnhance

A containerised two-stage pipeline:

1. **System 1 – acquire.** Give it an area of interest (a vector file *or* a place name), a date range and a cloud-cover limit. It finds the clearest [Sentinel-2](https://sentinels.copernicus.eu/copernicus/sentinel-2) L2A scene on the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) and downloads just the AOI window into `rawdata/`.
2. **System 2 – enhance.** Takes that Sentinel-2 data and super-resolves it from 10 m to **2.5 m** pixels with [SEN2SR](https://github.com/ESAOpenSR/SEN2SR), writing GeoTIFFs and a before/after preview into `output/`.

Both run in Docker, alone or chained by one script.

> **What you get, honestly.** The output pixel size is **2.5 m** (4× finer than Sentinel-2's 10 m bands), not sub-meter. The extra detail is *predicted by a neural network*, not measured. It is good for visualisation and some analysis, but don't use it as ground truth for measurement or legal boundaries.

## Quick start

Prerequisites: Docker with the Compose plugin. For the GPU image, an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/). A free [Copernicus Data Space](https://dataspace.copernicus.eu) account with S3 access keys.

```bash
cp .env.example .env            # then fill in CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY
./scripts/build.sh              # builds acquire + enhance-cpu   (add --gpu for the GPU image)
./scripts/run_system2.sh prefetch   # optional: download model weights into ./cache

# Everything in one go:
./scripts/run_pipeline.sh \
  --aoi-text "Perth City, Western Australia" \
  --max-cloud 10 --sensor rgb      # dates optional: defaults to the last 30 days
```

Results: `rawdata/<run_id>/` (inputs + `manifest.json`) and `output/<run_id>/` (enhanced imagery).

### The three scripts

| Script | What it does |
|---|---|
| `scripts/run_system1.sh <args>` | System 1 only → `rawdata/` |
| `scripts/run_system2.sh [--cpu\|--gpu] [--run-id ID] <args>` | System 2 only, on the latest (or a chosen) run. Uses the GPU image only if a GPU runtime is present and the GPU image has been built; otherwise CPU. Force one with --cpu/--gpuent |
| `scripts/run_pipeline.sh [--cpu\|--gpu] <system 1 args> [-- <system 2 args>]` | System 1 then System 2. Stops at the first failure and returns its exit code |
| `scripts/build.sh [--gpu\|--gpu-only]` | Build the images (default: acquire + CPU; the GPU image is opt-in) |

Without a terminal (cron, CI) `--non-interactive` is added automatically.

## System 1 parameters

| Flag | Default | Notes |
|---|---|---|
| `--aoi-file PATH` *or* `--aoi-text "…"` | – | Exactly one. Files: `.geojson/.json`, `.shp` (or a `.zip` of one), `.kml`, `.kmz`, `.gpkg`, and other GDAL/OGR vectors. Only polygons are used; multiple features are merged |
| `--start`, `--end` `YYYY-MM-DD` | last 30 days | Both optional. Neither given: the 30 days up to today (UTC). Only `--end`: the 30 days before it. Only `--start`: the 30 days after it (never past today). `end` cannot be in the future |
| `--days N` | 30 | Length of that default window |
| `--max-cloud N` | 20 | Percent cloud **over your AOI** (from the scene classification layer), not the whole tile |
| `--sensor` | `rgb` | `rgb` (B02/B03/B04 + B08) or `multispectral` (10 bands). `lidar`, `hyperspectral` and `sar` are rejected (see below) |
| `--max-area-km2 N` | 100 | Cap on the AOI's bounding box |
| `--min-coverage N` | 95 | Minimum % of the AOI the scene must cover |
| `--yes` | off | Accept the top geocoding match without asking |
| `--non-interactive` | off | Never prompt |

**Text AOIs** are geocoded with OpenStreetMap's Nominatim, and you are shown the match (name, box, area) to confirm. Names can be ambiguous ("Perth" exists in several countries), so check it.

**Why only `rgb` / `multispectral`?** Sentinel-2 is multispectral, not hyperspectral, and no Sentinel satellite produces LiDAR. Sentinel-1 is radar (SAR) and SEN2SR only works on Sentinel-2, so Sentinel-1 is not in v1.

**When no scene matches** (e.g. too cloudy) an interactive run shows the nearest candidates and offers to raise the cloud limit and/or widen the dates, with suggestions taken from the catalogue. A non-interactive run exits with code **10** and writes `rawdata/<run_id>/no_data_report.json` containing the same suggestions.

## System 2 parameters

| Flag | Default | Notes |
|---|---|---|
| `--run-id ID` / `--in DIR` | latest | Which rawdata run |
| `--model lite\|full` | `lite` | `full` (Mamba) needs an NVIDIA GPU and the GPU image; it fails with exit 21 otherwise, it never silently falls back |
| `--variant` | `auto` | `rgbn_x4` (RGB+NIR → 2.5 m) or `multispectral_x4` (10 bands → 2.5 m) |
| `--device auto\|cpu\|cuda` | `auto` | |
| `--cog` | off | Cloud-Optimised GeoTIFF output |
| `--clip-to-polygon` | off | Mask pixels outside the AOI polygon |
| `--overlap`, `--block` | 32, 512 | Tiling; `--block` bounds memory use |
| `--reflectance` | `offset-corrected` | See "Things to check on first real run" |

Outputs in `output/<run_id>/`: `enhanced_<scene>_2p5m.tif` (uint16, reflectance×10000, native UTM CRS), `…_rgb8.tif` (8-bit preview-stretched RGB), `preview_before_after.png`, `enhance_report.json`.

## Exit codes

| Code | Meaning | Code | Meaning |
|---|---|---|---|
| 0 | Success | 10 | No data matched (non-interactive) |
| 1 | Unexpected error | 11 | Geocode needs confirmation (use `--yes`) |
| 2 | Invalid input | 20 | System 2 input invalid |
| 3 | AOI over the area cap | 21 | Model unsupported on this hardware |
| 4 | Credentials missing/denied | 22 | Inference failed (e.g. out of memory) |
| 5 | Network / provider failure | | |

## Development

```bash
make venv && make lint && make test     # unit tests (no network, no GPU)
make smoke                              # builds CPU images and runs the container smoke test
```

The smoke test uses an offline fixture provider and a **bicubic stub in place of the SEN2SR model** (`SATENHANCE_STUB_MODEL=1`, clearly flagged in the report). It proves the containers, scripts, volumes, exit codes and GeoTIFF geometry, not real Copernicus access or real model quality.

**Step-by-step guide (API keys, scripts, every parameter, output locations): [`docs/HOWTO.md`](docs/HOWTO.md).**

Layout: `common/` (manifest + exit codes shared by both systems), `acquire/` (System 1), `enhance/` (System 2), `scripts/`, `docs/PRD.md`, `docs/IMPLEMENTATION_PLAN.md`.

## Things to check on your first real run

These could not be verified in the environment this was written in (no access to Copernicus, Hugging Face or a GPU). Each is marked **[VERIFY]** in the code or PRD.

1. **Copernicus access.** The STAC endpoint, collection id, asset names (`B04_10m`…) and the S3 access-key flow in `acquire/satenhance_acquire/providers/cdse.py`. Try a small AOI first.
2. **Model download and quality.** Run `./scripts/run_system2.sh prefetch`, then enhance a real scene and look at `preview_before_after.png`. The full-model weight path (`variants.py`) is a guess.
3. **Reflectance convention.** Sentinel-2 data processed with baseline ≥ 04.00 has a +1000 DN offset. The default subtracts it (`offset-corrected`); SEN2SR's own examples just divide by 10,000. If colours look off, try `--reflectance raw-div10000` and compare.
4. **GPU image.** `enhance/Dockerfile.gpu` compiles `mamba-ssm` and is untested; the SEN2SR README warns this build is fragile. Pin torch/CUDA versions via build args if it fails.
5. **Licences.** SEN2SR's README badge says MIT but its `LICENSE` file is CC0 1.0, and the model weights have their own terms. Sentinel data is free under Copernicus terms; Nominatim results are © OpenStreetMap contributors (ODbL). Check attribution requirements before redistributing.

## Networks with a TLS-inspecting proxy or blocked Debian mirrors

`CA_BUNDLE=/path/ca.pem ./scripts/build.sh` passes a CA bundle to `pip` during builds. `BASE_IMAGE=python:3.11 ./scripts/build.sh` uses the full Python image (which already has the system libraries needed) if `apt` can't reach Debian mirrors.
