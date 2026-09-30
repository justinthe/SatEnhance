# SatEnhance — How-To

Practical guide: what to sign up for, which script to run, what every parameter does, and where the results end up. For the design rationale see `PRD.md`; for a short overview see the top-level `README.md`.

> Items marked **[VERIFY]** come from my memory of third-party services (Copernicus, Hugging Face, Nominatim). I could not check them from the environment this was written in, so confirm them on the provider's site if a step doesn't match what you see.

---

## 1. What you need

| Need | For | Cost | Required? |
|---|---|---|---|
| Docker + Docker Compose plugin | Running everything | Free | **Yes** |
| Copernicus Data Space account + **S3 access keys** | Downloading Sentinel-2 (System 1) | Free | **Yes**, for real downloads |
| NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) | The GPU image / `--model full` | – | Optional (CPU works with `--model lite`) |
| Internet on first run | Downloading model weights | – | Yes, once (then cached) |

**Only one API credential is needed: the Copernicus S3 key pair.** The other services need no key:

| Service | Used for | Key needed? |
|---|---|---|
| Copernicus Data Space (STAC catalogue search) | Finding scenes | No key for searching **[VERIFY]** |
| Copernicus Data Space (S3 `eodata`) | Downloading the pixels | **Yes: S3 access key + secret key** |
| OpenStreetMap Nominatim | Turning "Perth City, Western Australia" into a polygon | No key. It needs a descriptive `User-Agent` (set in `.env`) and asks for at most ~1 request/second **[VERIFY current policy]** |
| Hugging Face (`tacofoundation/sen2sr`) | SEN2SR model weights | No token expected for public models **[VERIFY]**. Only add one if the download is refused |

---

## 2. Getting the Copernicus S3 keys

1. **Create a free account** at <https://dataspace.copernicus.eu> (Register) and confirm your email.
2. **Generate S3 credentials.** Copernicus documents an "S3 access" page where you create an access key + secret key for the `eodata` bucket. As far as I remember the key manager is at `https://eodata-s3keysmanager.dataspace.copernicus.eu` **[VERIFY: search their documentation for "S3 access" / "access keys" if that URL has moved]**.
3. **Copy both values immediately.** The secret is typically shown only once, and keys may expire, so note the expiry date **[VERIFY]**.
4. Put them in `.env` (next section). Never commit `.env` (it is in `.gitignore`).

Quotas and rate limits apply to free accounts **[VERIFY current limits]**. Start with a small AOI.

---

## 3. One-time setup

```bash
git clone <this repo> && cd SatEnhance      # or use your existing checkout

cp .env.example .env
$EDITOR .env
```

`.env` settings:

| Variable | What it is | Example |
|---|---|---|
| `CDSE_S3_ACCESS_KEY` | Copernicus S3 access key | `AKIA…`-style string from step 2 |
| `CDSE_S3_SECRET_KEY` | Copernicus S3 secret key | (keep private) |
| `NOMINATIM_USER_AGENT` | Identifies your app to Nominatim. Include a contact | `SatEnhance/0.1 (you@example.com)` |
| `SATENHANCE_MAX_AREA_KM2` | Default cap on AOI bounding-box area | `100` |

Then build the images and (optionally) fetch the model weights:

```bash
./scripts/build.sh                 # acquire + enhance-cpu (+ enhance-gpu if a GPU runtime is detected)
./scripts/run_system2.sh prefetch  # downloads SEN2SR weights into ./cache/models (needs internet)
```

Build options:

| Command | Builds |
|---|---|
| `./scripts/build.sh` | `acquire`, `enhance-cpu`, plus `enhance-gpu` if `nvidia-smi` and the Docker NVIDIA runtime are found |
| `./scripts/build.sh --cpu-only` | `acquire`, `enhance-cpu` |
| `./scripts/build.sh --gpu` | also `enhance-gpu` (slow: compiles `mamba-ssm`; **untested**, see Troubleshooting) |

Build-time environment variables (rarely needed):

| Variable | Use |
|---|---|
| `CA_BUNDLE=/path/ca.pem` | CA certificate for `pip` when your network inspects TLS |
| `BASE_IMAGE=python:3.11` | Use the full Python image if `apt` cannot reach Debian mirrors |
| `TORCH_INDEX_URL=…` | Where the CPU image gets PyTorch (default: PyTorch's CPU wheel index) |

---

## 4. Which script to run

All scripts live in `scripts/`, run from anywhere, and read `.env` automatically.

| I want to… | Run |
|---|---|
| Download imagery only | `./scripts/run_system1.sh <system 1 options>` |
| Enhance imagery I already downloaded | `./scripts/run_system2.sh [options]` |
| Do both, unattended | `./scripts/run_pipeline.sh <system 1 options> [-- <system 2 options>]` |
| Build the images | `./scripts/build.sh` |
| Pre-download model weights | `./scripts/run_system2.sh prefetch` |

### 4.1 System 1 only — `run_system1.sh`

```bash
# by place name (you'll be shown the match and asked to confirm)
./scripts/run_system1.sh --aoi-text "Perth City, Western Australia" \
    --start 2026-01-01 --end 2026-03-31 --max-cloud 10 --sensor rgb

# by file (geojson, shp/zip, kml, kmz, gpkg, ...)
./scripts/run_system1.sh --aoi-file ~/sites/my_site.kmz \
    --start 2026-01-01 --end 2026-03-31

# skip the geocode confirmation (for automation)
./scripts/run_system1.sh --aoi-text "Fremantle, Western Australia" --yes \
    --start 2026-02-01 --end 2026-02-28
```

Notes:
- An `--aoi-file` is copied into `./aoi/` and mounted read-only into the container. For a loose `.shp`, the `.dbf/.shx/.prj/.cpg` files next to it are copied too; a `.zip` of a shapefile is simpler.
- Without a terminal (cron, CI) `--non-interactive` is added automatically.
- On success the last line printed is `SATENHANCE_RUN_ID=<run_id>`.

### 4.2 System 2 only — `run_system2.sh`

```bash
./scripts/run_system2.sh                          # newest run in ./rawdata
./scripts/run_system2.sh --run-id 20260115T031200_perth-city
./scripts/run_system2.sh --cog --clip-to-polygon  # options are passed through
./scripts/run_system2.sh --gpu --model full       # GPU image, full model
./scripts/run_system2.sh --cpu                    # force the CPU image
```

`--cpu` / `--gpu` choose the image; with neither, the GPU image is used when a GPU runtime is detected, otherwise CPU. Everything else goes to the enhancer (section 6).

### 4.3 Both in one go — `run_pipeline.sh`

```bash
./scripts/run_pipeline.sh \
    --aoi-text "Perth City, Western Australia" --yes \
    --start 2026-01-01 --end 2026-03-31 --max-cloud 10 --sensor rgb \
    -- --cog
#   ^^^^^^^^^ System 1 options ^^^^^^^^^   ^^ System 2 options (after "--")
```

- Options before `--` go to System 1, options after go to System 2. `--cpu` / `--gpu` can go anywhere before the `--`.
- It reads the run id System 1 printed and passes it to System 2 automatically.
- **It stops at the first failure and returns that stage's exit code** (section 8). If System 1 finds no data, System 2 never runs.
- In a terminal, System 1's interactive prompts (geocode confirmation, "raise cloud cover?") work inside the pipeline. Without a terminal it runs non-interactively.

### 4.4 Running unattended (cron / CI)

```bash
./scripts/run_pipeline.sh --aoi-file /path/site.geojson --start 2026-01-01 --end 2026-01-31 \
    --max-cloud 15 < /dev/null
echo "exit code: $?"      # 10 = nothing matched; see rawdata/<run>/no_data_report.json
```

For place names, add `--yes` (otherwise the run stops with exit code 11 rather than guess).

---

## 5. System 1 parameters (download)

| Parameter | Default | What it does |
|---|---|---|
| `--aoi-file PATH` | – | Area of interest from a vector file: `.geojson`/`.json`, `.shp` or a `.zip` containing one, `.kml`, `.kmz`, `.gpkg`, or another format GDAL can read. Only polygons are used; several polygons are merged into one area. Points/lines alone are rejected. The file needs a coordinate system (GeoJSON is assumed WGS84) |
| `--aoi-text "…"` | – | Area of interest by name, geocoded with OpenStreetMap Nominatim. Use *exactly one* of `--aoi-file` / `--aoi-text` |
| `--start YYYY-MM-DD` | required | First day to search (inclusive) |
| `--end YYYY-MM-DD` | required | Last day to search (inclusive). Cannot be in the future |
| `--max-cloud N` | `20` | Highest acceptable cloud cover, in percent, **measured over your AOI** from the scene classification layer (not the whole ~100 km tile) |
| `--sensor` | `rgb` | `rgb` = Sentinel-2 bands B02, B03, B04 (true colour) plus B08 (near-infrared, needed by the enhancement model). `multispectral` = 10 bands (B02–B08, B8A, B11, B12). Anything else is rejected — see below |
| `--max-area-km2 N` | `100` (or `SATENHANCE_MAX_AREA_KM2`) | Refuses AOIs whose **bounding box** is bigger than this. The enhanced output is 16× more pixels than the input, so this protects your disk and memory |
| `--min-coverage N` | `95` | The scene must cover at least this % of your AOI (e.g. AOI sits on the edge of the satellite swath) |
| `--yes` | off | Accept the top geocoding match without asking |
| `--non-interactive` | off | Never prompt. On no data, exit 10 and write a report instead |
| `--out DIR` | `/data/rawdata` | Output folder inside the container (mapped to `./rawdata`; leave alone) |
| `--cache DIR` | `/data/cache` | Geocoding cache (mapped to `./cache`; leave alone) |
| `--log-json` | off | Machine-readable logs on stderr |

**Rejected sensors:** `lidar` (no Sentinel satellite provides it), `hyperspectral` (Sentinel-2 is multispectral), `sar` / Sentinel-1 (not in v1). You get exit code 2 and an explanation.

**How the scene is chosen:** the catalogue is searched for scenes intersecting your AOI in the date range, then the cloud fraction and coverage *over your AOI* are measured for the best candidates. The scene with the lowest AOI cloud wins; ties go to the most recent. One scene per run.

**If nothing matches** (interactive): you're shown the closest candidates and a menu — raise the cloud limit, widen the dates, both, enter values manually, or quit. The suggested values come from what actually exists in the catalogue. **Non-interactive:** exit code 10 and `rawdata/<run_id>/no_data_report.json` with the same suggestions.

**Text AOI tips:** names can be ambiguous ("Perth" exists in Australia and Scotland) and "Perth City" is a small administrative area compared with metropolitan Perth. Check the shown bounding box and area. The result must also fit under `--max-area-km2`.

---

## 6. System 2 parameters (enhance)

| Parameter | Default | What it does |
|---|---|---|
| `--run-id ID` | newest | Which folder in `./rawdata` to enhance. Default: whatever `rawdata/LATEST` names |
| `--in DIR` | – | Explicit path to a rawdata run folder (inside the container) instead of `--run-id` |
| `--model lite\|full` | `lite` | `lite` = SEN2SR-Lite, runs on CPU or GPU. `full` = the Mamba model; **needs an NVIDIA GPU** and the GPU image. Asking for `full` without one fails with exit 21; it never silently downgrades |
| `--variant` | `auto` | `auto` picks from how System 1 was run: `rgb` → `rgbn_x4` (R,G,B,NIR at 10 m → 2.5 m); `multispectral` → `multispectral_x4` (10 bands → 2.5 m). You can force one, but the needed bands must be in the download |
| `--device auto\|cpu\|cuda` | `auto` | Where to run inference. `cuda` without a GPU fails with exit 21 |
| `--cog` | off | Write Cloud-Optimised GeoTIFFs (better for web/GIS streaming) |
| `--clip-to-polygon` | off | Set pixels outside your AOI polygon to "no data". Default keeps the whole bounding rectangle |
| `--overlap N` | `32` | Pixels of overlap between model tiles (even number). Higher = fewer visible seams, slower |
| `--block N` | `512` | Input pixels per processing block (≥128). Lower it if you run out of memory; higher is slightly faster |
| `--reflectance` | `offset-corrected` | How raw pixel values become reflectance before the model. `offset-corrected` subtracts the +1000 offset used by newer Sentinel-2 products; `raw-div10000` just divides by 10,000 like SEN2SR's own examples. If colours look wrong, try the other **[VERIFY visually]** |
| `--rawdata DIR`, `--out DIR`, `--cache DIR` | `/data/...` | Container paths (mapped to `./rawdata`, `./output`, `./cache`; leave alone) |
| `--log-json` | off | Machine-readable logs |

`./scripts/run_system2.sh prefetch [--model lite]` downloads the model weights into `./cache/models` so later runs can work offline.

---

## 7. Where everything goes

All paths are relative to the repo folder on your machine.

```
rawdata/
├── LATEST                                   # name of the most recent run
└── <run_id>/                                # e.g. 20260929T134600_perth-city
    ├── manifest.json                        # what was downloaded + how to read it (System 2 input)
    ├── aoi.geojson                          # your AOI, cleaned up (WGS84, one geometry)
    ├── search_results.json                  # every scene considered, its AOI cloud/coverage, why rejected
    ├── no_data_report.json                  # only when nothing matched (exit 10)
    └── S2_<scene_id>/
        ├── B02.tif  B03.tif  B04.tif  B08.tif    # (+ B05 B06 B07 B8A B11 B12 for multispectral)
        └── SCL.tif                               # scene classification (cloud mask), 20 m

output/
└── <run_id>/                                # same run id as the rawdata run
    ├── enhanced_<scene>_2p5m.tif            # main result: all bands, 2.5 m, uint16
    ├── enhanced_<scene>_2p5m_rgb8.tif       # 8-bit RGB, contrast-stretched, for quick viewing
    ├── preview_before_after.png             # original vs enhanced, side by side
    └── enhance_report.json                  # model, versions, device, runtime, warnings

cache/                                       # geocoding results + downloaded model weights (safe to keep)
aoi/                                         # copies of AOI files you passed with --aoi-file
```

File details:
- **Rawdata bands** are windowed to your AOI (bounding box), in Sentinel-2's own UTM projection, at native resolution (10 m or 20 m), as compressed GeoTIFFs. Nothing is resampled in System 1.
- **`enhanced_*_2p5m.tif`**: pixel size 2.5 m, same projection and extent as the input, band names set (`B04`, `B03`, …). Values are **surface reflectance × 10,000** (divide by 10,000 for 0–1); `0` means no data. The `SCALE_FACTOR` tag records this.
- **Nothing is deleted automatically.** `rawdata/` and `output/` can grow to many gigabytes; clean up yourself. Rough size guide (uncompressed; files are compressed so real sizes are smaller): a 100 km² bounding box is about 1 million 10 m pixels, i.e. ≈ 2 MB per band as 16-bit data, and the 2.5 m enhanced output is 16× that, ≈ 32 MB per band (≈ 130 MB for RGB+NIR, ≈ 320 MB for 10 bands).

**Reading the result:** open the `.tif` in QGIS or any GIS. Use `_rgb8.tif` when you only need a picture.

> **Remember:** the extra detail is *predicted by a neural network*, not measured. Good for looking at and for many analyses, not for legal boundaries or precise measurement. This is also written into the GeoTIFF metadata and `enhance_report.json`.

---

## 8. Exit codes

Use these in your own scripts (`echo $?`).

| Code | Meaning | What to do |
|---|---|---|
| 0 | Success | – |
| 1 | Unexpected error | Re-run with `--log-json`, check the message |
| 2 | Invalid input (bad file, no CRS, bad dates, unsupported sensor) | Fix the input; the message says what |
| 3 | AOI bigger than the area cap | Use a smaller area or raise `--max-area-km2` |
| 4 | Credentials missing or refused | Check `.env` (section 2) |
| 5 | Network / provider failure after retries | Retry later; check connectivity to Copernicus |
| 10 | No scene matched | Read `no_data_report.json`; raise `--max-cloud` and/or widen the dates |
| 11 | Geocode needs confirmation | Add `--yes`, or run interactively |
| 20 | System 2 couldn't use the rawdata (missing bands/files) | Re-run System 1 with a matching `--sensor` |
| 21 | Requested model/device not available (`full` or `cuda` without a GPU) | Use `--model lite`, or the GPU image on a GPU machine |
| 22 | Inference failed (usually out of memory) | Lower `--block`, use a smaller AOI |

---

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Exit 4 straight away | `.env` missing or the S3 keys are blank/expired. Regenerate them |
| Exit 5, "Could not read SCL" | Network blocked to Copernicus, or S3 access not enabled for your account **[VERIFY]** |
| Exit 3 | AOI bounding box too large; the message shows both figures |
| Exit 10 every time | Cloudy season or short window. Raise `--max-cloud`, widen dates. Sentinel-2 revisits every ~5 days, so a month usually has several scenes |
| Geocoder picked the wrong place | Answer `next` at the prompt to see other matches, or use an `--aoi-file` |
| Exit 21 with `--model full` | No usable NVIDIA GPU in the container. Build/run the GPU image (`--gpu`) on a GPU host with the NVIDIA Container Toolkit |
| Exit 22 / container killed | Lower `--block` (e.g. 256) or use a smaller AOI |
| Colours look washed out or too dark | Try `--reflectance raw-div10000` |
| GPU image fails to build | `mamba-ssm` builds are fragile and this Dockerfile is untested. Pin a torch/CUDA/mamba-ssm combination that works for your GPU with the build args in `enhance/Dockerfile.gpu` |
| Files owned by root in `rawdata/`/`output/` | Run the scripts (not raw `docker run`); they pass your user id |
| `image 'satenhance-…' not found` | The images aren't built yet. Run `./scripts/build.sh` first |
| Build fails with `ReadTimeoutError` from `files.pythonhosted.org` (or another network error while `pip` downloads) | Slow or flaky connection. Just **re-run `./scripts/build.sh`**: `pip` now retries each download and keeps a cache between builds, so a re-run resumes instead of starting over. The System 2 image is the biggest download (PyTorch, several GB) |
| `pull access denied for satenhance-acquire` during a build | Harmless. Compose tries to pull our locally built image first, then builds it |
| First System 2 run is slow to start | It's downloading model weights. Use `prefetch` once |

---

## 10. Trying it without Copernicus access

The test suite and the container smoke test run entirely offline with synthetic data and a **stand-in model** (plain bicubic ×4, clearly flagged in the report). They verify the plumbing, not real quality:

```bash
make venv && make test     # unit tests
make smoke                 # builds the CPU images and runs the full container test
```

To try System 2 alone on synthetic data with the stand-in model:

```bash
docker compose run --rm -T --entrypoint python enhance-cpu -m satenhance_enhance.synthetic --out /data/rawdata
SATENHANCE_STUB_MODEL=1 ./scripts/run_system2.sh --cpu
```
