#!/usr/bin/env bash
# Environment check: prints PASS / WARN / FAIL with a fix hint for each item, and exits 1 if
# anything FAILs. Run it first when something does not work, and send me any FAIL lines.
#   ./scripts/doctor.sh
# Never prints secret values.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
set +e   # a failing check must not stop the script

MIN_DISK_GB="${DOCTOR_MIN_DISK_GB:-30}"
NPASS=0; NWARN=0; NFAIL=0
if [[ -t 1 ]]; then G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else G=; Y=; R=; N=; fi
pass() { NPASS=$((NPASS+1)); echo "${G}PASS${N}  $1"; }
warn() { NWARN=$((NWARN+1)); echo "${Y}WARN${N}  $1"; [[ -n "${2:-}" ]] && echo "      -> $2"; }
fail() { NFAIL=$((NFAIL+1)); echo "${R}FAIL${N}  $1"; [[ -n "${2:-}" ]] && echo "      -> $2"; }
info() { echo "      $1"; }
section() { echo; echo "== $1"; }

section "Docker"
if ! command -v docker >/dev/null 2>&1; then
  fail "docker is not installed" "Install Docker: https://docs.docker.com/get-docker/"
elif ! docker info >/dev/null 2>&1; then
  fail "the Docker daemon is not reachable" "Start Docker (e.g. 'sudo systemctl start docker') and make sure your user may use it"
else
  pass "Docker daemon is running"
  cv="$(docker compose version --short 2>/dev/null | sed 's/^v//')"
  if [[ -z "$cv" ]]; then
    fail "the Docker Compose plugin is missing" "Install it: https://docs.docker.com/compose/install/"
  elif [[ "$(printf '%s\n%s\n' 2.24 "$cv" | sort -V | head -n1)" != "2.24" ]]; then
    fail "Docker Compose $cv is too old (need 2.24 or newer)" "Upgrade the Compose plugin"
  else
    pass "Docker Compose $cv"
  fi
fi

section "Disk space"
avail_kb="${DOCTOR_FREE_KB:-$(df -Pk "$REPO_ROOT" 2>/dev/null | awk 'NR==2{print $4}')}"
if [[ -n "$avail_kb" ]]; then
  gb=$((avail_kb / 1024 / 1024))
  if [[ $gb -lt 10 ]]; then fail "only ${gb} GB free in $REPO_ROOT" "Builds and models need ~10 GB minimum; free some space or prune Docker: docker system prune"
  elif [[ $gb -lt $MIN_DISK_GB ]]; then warn "${gb} GB free in $REPO_ROOT" "The GPU image and model weights can need more than ${MIN_DISK_GB} GB"
  else pass "${gb} GB free"; fi
else
  warn "could not read free disk space"
fi

section "Images"
for spec in "satenhance-acquire:latest|System 1|./scripts/build.sh" "satenhance-enhance:cpu|System 2 (CPU)|./scripts/build.sh"; do
  IFS='|' read -r img label hint <<<"$spec"
  if docker image inspect "$img" >/dev/null 2>&1; then pass "$label image ($img) is built"
  else warn "$label image ($img) is not built" "$hint"; fi
done
if docker image inspect satenhance-enhance:gpu >/dev/null 2>&1; then pass "System 2 (GPU) image is built"
else info "System 2 (GPU) image not built (only needed for --model full): ./scripts/build.sh --gpu-only"; fi

section "Credentials (.env)"
if [[ ! -f "$REPO_ROOT/.env" ]]; then
  warn ".env is missing" "cp .env.example .env, then fill in the Copernicus S3 keys (docs/HOWTO.md section 2)"
else
  pass ".env exists"
  for var in CDSE_S3_ACCESS_KEY CDSE_S3_SECRET_KEY; do
    if [[ -n "${!var:-}" ]]; then pass "$var is set"
    else warn "$var is empty" "Needed to download imagery; see docs/HOWTO.md section 2"; fi
  done
  ua="${NOMINATIM_USER_AGENT:-}"
  if [[ -z "$ua" || "$ua" == *"you@example.com"* ]]; then
    warn "NOMINATIM_USER_AGENT is not set to your contact" "Put a contact in it (e.g. SatEnhance/0.1 (me@mydomain)); only needed for --aoi-text"
  else pass "NOMINATIM_USER_AGENT is set"; fi
fi

section "GPU (only needed for --model full)"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  pass "host GPU: $(nvidia-smi -L 2>/dev/null | head -n1)"
  if docker image inspect satenhance-enhance:gpu >/dev/null 2>&1; then
    if docker run --rm --gpus all --entrypoint nvidia-smi satenhance-enhance:gpu -L >/dev/null 2>&1; then
      pass "the GPU is visible inside the GPU image"
    else
      fail "the GPU is NOT visible inside the GPU image" "Install/configure the NVIDIA Container Toolkit and restart Docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
    fi
  elif docker info 2>/dev/null | grep -qiE 'nvidia|cdi'; then
    pass "Docker has an NVIDIA runtime"
  else
    warn "Docker shows no NVIDIA runtime" "Install the NVIDIA Container Toolkit, then build with ./scripts/build.sh --gpu-only"
  fi
else
  info "no NVIDIA GPU visible on the host; use --model lite (CPU works)"
fi

section "Network"
if [[ -n "${DOCTOR_SKIP_NETWORK:-}" ]]; then
  info "skipped (DOCTOR_SKIP_NETWORK set)"
else
  urls="${DOCTOR_URLS:-pypi.org|https://pypi.org/simple/ files.pythonhosted.org|https://files.pythonhosted.org/ download.pytorch.org|https://download.pytorch.org/whl/cpu/ registry-1.docker.io|https://registry-1.docker.io/v2/ stac.dataspace.copernicus.eu|https://stac.dataspace.copernicus.eu/v1/ eodata.dataspace.copernicus.eu|https://eodata.dataspace.copernicus.eu/ nominatim.openstreetmap.org|https://nominatim.openstreetmap.org/status.php huggingface.co|https://huggingface.co/}"
  for item in $urls; do
    host="${item%%|*}"; url="${item#*|}"
    code="$(curl -sS -m 10 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null)"
    if [[ -z "$code" || "$code" == "000" ]]; then
      fail "$host is unreachable" "Check your internet, proxy or firewall; builds need pypi/pytorch/docker, System 1 needs Copernicus, prefetch needs huggingface"
    elif [[ "$code" == "429" ]]; then
      warn "$host answered 429 (rate limited)" "Wait a few minutes and retry (Docker Hub limits anonymous pulls)"
    else
      pass "$host reachable (HTTP $code)"
    fi
  done
fi

section "Model cache"
found=0
for d in "$REPO_ROOT"/cache/models/*/; do
  [[ -d "$d" ]] || continue
  found=1
  if [[ -f "$d/.complete" ]]; then pass "model $(basename "$d") is complete"
  else warn "model $(basename "$d") is incomplete" "./scripts/run_system2.sh prefetch (it resumes)"; fi
done
[[ $found -eq 0 ]] && info "no models cached yet: ./scripts/run_system2.sh prefetch"

echo
echo "doctor: $NPASS passed, $NWARN warning(s), $NFAIL failed"
[[ $NFAIL -eq 0 ]]
