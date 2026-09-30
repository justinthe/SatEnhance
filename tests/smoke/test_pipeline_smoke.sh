#!/usr/bin/env bash
# Container smoke test. Needs Docker; needs no network beyond building the images.
#   bash tests/smoke/test_pipeline_smoke.sh
# Env:
#   SMOKE_SKIP_BUILD=1   reuse existing images (satenhance-acquire:latest, satenhance-enhance:cpu)
#   CA_BUNDLE=/path.pem  CA bundle for `pip` during the build (TLS-inspecting proxies)
#   TORCH_INDEX_URL=...  torch index for the CPU image (default: PyTorch's CPU index)
#
# Uses an offline fixture provider and a bicubic STUB model, so it exercises the containers,
# scripts, volumes, exit codes and file formats - NOT real Copernicus access or real SEN2SR
# weights. Runs in a temporary directory so it never touches your ./rawdata or ./output.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE="$(mktemp -d)"
trap 'rm -rf "$SMOKE"' EXIT

if [[ "${SMOKE_SKIP_BUILD:-0}" != 1 ]]; then
  "$REPO/scripts/build.sh" --cpu-only
fi

cp -r "$REPO/scripts" "$REPO/docker-compose.yml" "$SMOKE/"
cd "$SMOKE"
mkdir -p rawdata output cache

export SATENHANCE_PROVIDER=fixture
export SATENHANCE_FIXTURE_DIR=/data/cache/fixture
export SATENHANCE_STUB_MODEL=1
export CDSE_S3_ACCESS_KEY="" CDSE_S3_SECRET_KEY=""

pass=0; fail=0
expect_rc() {  # expect_rc <expected> <description> <command...>
  local want="$1" desc="$2"; shift 2
  set +e; "$@" > "$SMOKE/last.log" 2>&1; local rc=$?; set -e
  if [[ $rc -eq $want ]]; then echo "PASS  $desc (exit $rc)"; pass=$((pass+1))
  else echo "FAIL  $desc: expected exit $want, got $rc"; sed 's/^/      | /' "$SMOKE/last.log" | tail -25
       fail=$((fail+1)); fi
}
compose_py() {  # run a python script from stdin inside a service
  local service="$1"; shift
  docker compose --profile gpu run --rm -T --entrypoint python "$service" "$@"
}

echo "--- fixture"
export UID GID="$(id -g)"
compose_py acquire - < "$REPO/tests/smoke/make_fixture.py"
AOI=cache/smoke/site.geojson
COMMON=(--aoi-file "$AOI" --sensor rgb)

echo "--- System 1 error paths"
expect_rc 2  "lidar sensor is rejected"    ./scripts/run_system1.sh "${COMMON[@]:0:2}" --start 2026-01-01 --end 2026-01-31 --sensor lidar
expect_rc 2  "end date in the future"      ./scripts/run_system1.sh "${COMMON[@]}" --start 2026-01-01 --end 2999-01-01
expect_rc 3  "AOI over the area cap"       ./scripts/run_system1.sh "${COMMON[@]}" --start 2026-01-01 --end 2026-01-31 --max-area-km2 0.1
expect_rc 10 "no data -> exit 10"          ./scripts/run_system1.sh "${COMMON[@]}" --start 2026-02-01 --end 2026-02-10
(cd rawdata && ls -d */no_data_report.json >/dev/null 2>&1) && { echo "PASS  no_data_report.json written"; pass=$((pass+1)); } \
  || { echo "FAIL  no_data_report.json missing"; fail=$((fail+1)); }
expect_rc 4  "missing CDSE credentials"    env SATENHANCE_PROVIDER=cdse ./scripts/run_system1.sh "${COMMON[@]}" --start 2026-01-01 --end 2026-01-31

echo "--- System 2 error paths"
expect_rc 20 "System 2 with no rawdata"    ./scripts/run_system2.sh --cpu --run-id does-not-exist
expect_rc 21 "full model on CPU"           ./scripts/run_system2.sh --cpu --model full

echo "--- System 1 alone, then System 2 alone"
rm -rf rawdata/* output/*
expect_rc 0  "run_system1.sh succeeds"     ./scripts/run_system1.sh "${COMMON[@]}" --start 2026-01-01 --end 2026-01-31 --max-cloud 10
expect_rc 0  "relative --aoi-file works from another directory" \
  bash -c 'cd cache/smoke && ../../scripts/run_system1.sh --aoi-file site.geojson --start 2026-01-01 --end 2026-01-31 --max-cloud 10'
expect_rc 0  "--aoi-file=PATH form is accepted" \
  ./scripts/run_system1.sh --aoi-file="$AOI" --start 2026-01-01 --end 2026-01-31 --max-cloud 10
RUN1="$(tr -d '\r\n' < rawdata/LATEST)"
expect_rc 0  "run_system2.sh succeeds"     ./scripts/run_system2.sh --cpu --run-id "$RUN1"
expect_rc 0  "output is geometrically sane" compose_py enhance-cpu - "$RUN1" < "$REPO/tests/smoke/check_output.py"

echo "--- full pipeline"
rm -rf rawdata/* output/*
expect_rc 0  "run_pipeline.sh succeeds"    ./scripts/run_pipeline.sh --cpu "${COMMON[@]}" --start 2026-01-01 --end 2026-01-31 --max-cloud 10 -- --cog
RUN2="$(tr -d '\r\n' < rawdata/LATEST)"
expect_rc 0  "pipeline output is sane"     compose_py enhance-cpu - "$RUN2" < "$REPO/tests/smoke/check_output.py"
rm -rf rawdata/* output/*
expect_rc 10 "pipeline propagates exit 10 (no data)" ./scripts/run_pipeline.sh --cpu "${COMMON[@]}" --start 2026-02-01 --end 2026-02-10
[[ -z "$(ls -A output)" ]] && { echo "PASS  System 2 not run after System 1 failure"; pass=$((pass+1)); } \
  || { echo "FAIL  output/ not empty after failed System 1"; fail=$((fail+1)); }

echo "--- mosaic: AOI straddling two tiles of one pass"
rm -rf rawdata/* output/*
expect_rc 0  "run_pipeline.sh mosaics two tiles and enhances the result" \
  env SATENHANCE_FIXTURE_DIR=/data/cache/fixture_mosaic ./scripts/run_pipeline.sh --cpu \
    --aoi-file cache/smoke/straddle.geojson --start 2026-01-01 --end 2026-01-31 --max-cloud 10
RUN3="$(tr -d '\r\n' < rawdata/LATEST)"
expect_rc 0  "manifest 1.1 lists both tiles, no holes at the seam" compose_py acquire - "$RUN3" < "$REPO/tests/smoke/check_mosaic.py"
expect_rc 0  "enhanced mosaic is geometrically sane" compose_py enhance-cpu - "$RUN3" MOSAIC_ < "$REPO/tests/smoke/check_output.py"

echo "--- diagnosis tools"
expect_rc 0  "doctor.sh runs (network and disk-size checks neutralised)" env DOCTOR_SKIP_NETWORK=1 DOCTOR_FREE_KB=104857600 ./scripts/doctor.sh
expect_rc 0  "probe reports catalogue + pixel access" \
  ./scripts/run_system1.sh probe --aoi-file "$AOI" --start 2026-01-01 --end 2026-01-31
expect_rc 0  "selftest loads each variant and runs a patch" ./scripts/run_system2.sh --cpu selftest --device cpu
expect_rc 0  "failed runs leave an error_report.json" bash -c \
  'rm -f rawdata/error_report.json; ! ./scripts/run_system1.sh --aoi-file cache/smoke/site.geojson --sensor lidar >/dev/null 2>&1; ls rawdata/error_report.json'

echo "--- real CDSE provider inside the acquire image (the EnvError seen on the first real run)"
expect_rc 0  "S3 session can be created (boto3 present, no EnvError)" compose_py acquire -c "
from satenhance_acquire.providers.cdse import CdseProvider
p = CdseProvider(env={'CDSE_S3_ACCESS_KEY': 'a', 'CDSE_S3_SECRET_KEY': 'b'})
with p.rasterio_env():
    import rasterio; print('ok', rasterio.__version__)
"

echo "--- real SEN2SR loader scripts inside the enhance image (matplotlib etc.)"
expect_rc 0  "real Lite loader loads and runs (random weights)" \
  docker compose --profile gpu run --rm -T -v "$REPO/enhance/tests/real_loaders:/loaders:ro" \
    --entrypoint python enhance-cpu - < "$REPO/tests/smoke/check_real_loader.py"

echo "--- System 2 helper commands inside the enhance image"
rm -rf rawdata/* output/*
expect_rc 0  "synthetic rawdata generator runs in the enhance image" compose_py enhance-cpu -m satenhance_enhance.synthetic --out /data/rawdata
expect_rc 0  "System 2 runs on the synthetic rawdata" ./scripts/run_system2.sh --cpu

echo "--- GPU image Python packaging (stage split, plain ubuntu:22.04)"
secret=(); [[ -n "${CA_BUNDLE:-}" ]] && secret=(--secret "id=cabundle,src=$CA_BUNDLE")
expect_rc 0  "old runtime package set lacks system distutils.core" \
  docker build -q -f tests/smoke/stage_split.Dockerfile --target runtime-old "${secret[@]}" .
expect_rc 0  "new runtime package set has distutils, gcc and Python.h" \
  docker build -q -f tests/smoke/stage_split.Dockerfile --target runtime-new "${secret[@]}" .

echo
echo "smoke: $pass passed, $fail failed"
[[ $fail -eq 0 ]]
