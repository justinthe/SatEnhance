# Fix Plan: Copernicus S3 access (`EnvError` on first real run)

| | |
|---|---|
| **Status** | Plan only; nothing in this document is implemented yet |
| **Date** | 2026-09-30 |
| **Trigger** | `./scripts/run_system1.sh --aoi-file data/aoi.geojson` failed with `EnvError: GDAL's AWS config options can not be directly set. AWS credentials are handled exclusively by boto3.` |

## 1. What happened

```
INFO Searching 2026-08-31 -> 2026-09-30 (default window: 30 days)
INFO AOI aoi: bbox 0.4 km2 (polygon 0.3 km2); est. raw 0 MB, enhanced 0 MB
INFO 3 candidate scene(s) in catalogue
ERROR Unexpected error: EnvError: GDAL's AWS config options can not be directly set. ...
```

**What already works on your real setup:** the `.env` loading, the default 30-day window, AOI reading, the size check, and the **real Copernicus catalogue search**, which found 3 scenes. None of that had been tested against the live service before.

**What failed:** the first attempt to read pixels. It happens before any network request is made, so it says nothing yet about your keys, the asset URLs or S3 access.

## 2. Root cause (confirmed, not guessed)

`CdseProvider.gdal_env()` returns the S3 keys as GDAL config options (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`). Every pixel read does `rasterio.Env(**provider.gdal_env())`. rasterio deliberately refuses those two options. Its `Env.__init__` (`rasterio/env.py`, installed 1.4.4) raises this exact `EnvError` whenever either key is passed as an option. Credentials must go through a rasterio `AWSSession`, which needs `boto3`.

**Reproduced here:** the current call raises the same `EnvError`.

**Verified here:** the fix reads real pixels through GDAL's `/vsis3/` from a local S3 server. That test used a custom endpoint, path-style URLs and HTTP range requests, which is the same shape as Copernicus's `eodata` bucket:

```python
from rasterio.session import AWSSession
sess = AWSSession(aws_access_key_id=..., aws_secret_access_key=...,
                  endpoint_url="eodata.dataspace.copernicus.eu", region_name=...)
with rasterio.Env(session=sess, AWS_VIRTUAL_HOSTING="FALSE", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", ...):
    rasterio.open("/vsis3/eodata/...")        # windowed read worked (HTTP 206)
```

**Why the tests missed it:**
- The unit tests only checked that `gdal_env()` returned the right *dictionary*. They never opened a raster with it.
- The smoke test uses the offline fixture provider, whose settings are empty.
- So the one code path that talks to S3 had no test.

That gap is fixed in step 4 below, not only the bug.

## 3. Secondary issues seen in the same output

| # | Issue | Fix |
|---|---|---|
| S1 | The error is logged as "Unexpected error" (exit 1). The report went to `rawdata/error_report.json`, not the run's own folder, because only our own errors carry the run folder with them | In `pipeline.acquire`, attach `run_dir` to *any* exception on the way up (wrap non-SatEnhance errors as exit 1 with context), so the report lands in `rawdata/<run_id>/` |
| S2 | "Details: /data/rawdata/error_report.json" is a path inside the container; on your machine it is `./rawdata/error_report.json` | The scripts rewrite `/data/` to `./` in that one line (or the CLI prints both) |
| S3 | "est. raw 0 MB, enhanced 0 MB" for a small AOI looks broken | Print sizes in KB below 1 MB |
| S4 | A failed run leaves its folder in `rawdata/` | Intentional: it holds `aoi.geojson`, `search_results.json` and the error report for debugging. `rawdata/LATEST` only moves on success, so System 2 never picks up a failed run. No change; documented in HOWTO |

## 4. The fix

### 4.1 Credentials through an `AWSSession` (the actual bug)
- Add `boto3` to System 1's dependencies (`acquire/pyproject.toml`, `acquire/requirements.in`, `acquire/requirements.lock`). It is only a few MB. System 2 doesn't need it.
- Replace `Provider.gdal_env() -> dict` with `Provider.rasterio_env() -> rasterio.Env`, one context manager that every pixel read uses:
  - **CDSE:** `rasterio.Env(session=AWSSession(key, secret, endpoint_url=S3_ENDPOINT, region_name=S3_REGION), AWS_VIRTUAL_HOSTING="FALSE", AWS_HTTPS="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", timeouts...)`. No keys in the option dict.
  - **Fixture provider:** plain `rasterio.Env()`.
- Update the 4 call sites (`mosaic.py` ×2, `download.py`, `probe.py`) to `with provider.rasterio_env():`. The earlier `gdal_env()` dict stays only as the list of non-secret options, which makes it easy to test.
- Region: `S3_REGION` defaults to `"default"` and can be overridden with `CDSE_S3_REGION` in `.env`. I believe Copernicus's own S3 examples use the region `default` **[VERIFY with `probe`]**. GDAL signs requests with the region, so a wrong value would show up as a 403 in `probe`.
- Endpoint override: `CDSE_S3_ENDPOINT` in `.env` (default `eodata.dataspace.copernicus.eu`). The tests use it to point at a local S3 server, and you can use it if Copernicus changes the hostname.

### 4.2 Clear messages for S3 failures
- `_open()` already turns GDAL 401/403 errors into exit 4. Extend the check to the messages GDAL gives for a bad signature or wrong region (`SignatureDoesNotMatch`, `AuthorizationHeaderMalformed`, `InvalidAccessKeyId`). The hint says: check both keys in `.env`, check whether they have expired, and run `probe`.
- A 404 on an asset → exit 5, "the catalogue points at a file that is not in the bucket", naming the path.
- A raw `EnvError` or `CPLE_*` from rasterio in the S3 path is caught and reported as exit 5 with the GDAL text, never as "Unexpected error".

### 4.3 Tests that exercise real S3 reads (closes the gap)
- **New `acquire/tests/test_s3_access.py`**, using `moto`'s local S3 server (test-only dependency, runs offline, already tried here):
  - upload small GeoTIFF bands to a fake `eodata` bucket;
  - point `CdseProvider` at it with `CDSE_S3_ENDPOINT=127.0.0.1:<port>` and plain HTTP (a test-only switch);
  - run the **real** `plan_mosaic` → `download_scene` path through `/vsis3/`, and check that pixels and bounds are correct;
  - check that the options passed to rasterio never contain `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY` (regression guard for this exact bug).
- **Auth-failure test:** moto doesn't check signatures, so a small local HTTP server returning `403 SignatureDoesNotMatch` checks that we exit 4 with the right hint.
- **`probe` against the fake bucket:** it prints `PROBE OK` and reads pixels through `/vsis3/`.
- **Smoke test:** add one run in the container against the moto server, so the acquire *image* is proven to have `boto3` and working S3 access, not just the dev environment.

### 4.4 Docs
- HOWTO troubleshooting rows for exit 4 and 5 from S3, including the new `CDSE_S3_REGION` and `CDSE_S3_ENDPOINT` settings.
- `.env.example`: add both settings, commented out with their defaults.

## 5. What will still be unverified after this fix
These can only be checked with your keys against the real service:
1. That Copernicus accepts the region/endpoint/path-style combination. `probe` shows a clear OK or FAIL.
2. The real asset URLs in the catalogue (`s3://eodata/...` is expected). `probe` lists them.
3. That the keys in your `.env` are active.

**After the fix, please run:**
```bash
git pull && ./scripts/build.sh          # acquire image gets boto3 (small, fast rebuild)
./scripts/run_system1.sh probe --aoi-file data/aoi.geojson
./scripts/run_system1.sh --aoi-file data/aoi.geojson
```
If `probe` fails, send me its output. It shows the asset names and URLs and the exact S3 error.

## 6. Order of work
1. 4.1 and 4.3 together (fix + tests that would have caught it), then run the full unit suite and smoke test here, and push.
2. 4.2, S1, S2, S3 (messages and reports).
3. 4.4 docs.

One commit per step, pushed to `claude/practical-ptolemy-w1xnmq`. `main` only when you ask.
