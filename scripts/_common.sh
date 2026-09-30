# Sourced by the run scripts. Not executable on its own.
set -euo pipefail

# Remember where the user ran the script from: relative paths in arguments are relative to it.
CALLER_DIR="$PWD"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load .env without `source`: values such as `SatEnhance/0.1 (me@example.com)` are not valid
# shell syntax, but they are valid in a docker-compose env file. Parse KEY=VALUE lines instead:
# comments and blank lines are skipped, one pair of surrounding quotes is removed, and a value
# is never evaluated.
load_env() {
  local file="$1" line key val
  [[ -f "$file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[1]}"; val="${BASH_REMATCH[2]}"
    if [[ "$val" =~ ^\"(.*)\"$ ]] || [[ "$val" =~ ^\'(.*)\'$ ]]; then val="${BASH_REMATCH[1]}"; fi
    export "$key=$val"
  done < "$file"
}
load_env "$REPO_ROOT/.env"

# bash's UID is readonly; exporting it makes it visible to docker compose (${UID}).
export UID
GID="${GID:-$(id -g)}"; export GID
mkdir -p rawdata output cache

# `docker compose run` needs -T when there is no TTY (cron, CI, pipes).
# Only stdin/stderr decide: the pipeline pipes stdout through tee but is still interactive.
tty_flag() {
  if [[ -t 0 ]]; then echo ""; else echo "-T"; fi
}

has_tty() { [[ -t 0 && -t 2 ]]; }

# A GPU counts only if the host sees one AND Docker can hand it to a container. When the GPU
# image exists this is tested for real (nvidia-smi inside it), which also catches a missing
# NVIDIA Container Toolkit; otherwise fall back to looking for an NVIDIA runtime in `docker info`.
has_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 || return 1
  if docker image inspect satenhance-enhance:gpu >/dev/null 2>&1; then
    docker run --rm --gpus all --entrypoint nvidia-smi satenhance-enhance:gpu -L >/dev/null 2>&1
    return
  fi
  docker info 2>/dev/null | grep -qiE 'nvidia|cdi'
}

has_arg() {
  local needle="$1"; shift
  local a
  for a in "$@"; do [[ "$a" == "$needle" ]] && return 0; done
  return 1
}

compose() { docker compose "$@"; }

die() { echo "error: $*" >&2; exit 2; }

# Fail with a clear message instead of silently starting a long implicit build.
require_image() {  # require_image <image> [build command hint]
  docker image inspect "$1" >/dev/null 2>&1 \
    || { echo "error: image '$1' not found. Build it first:  ${2:-./scripts/build.sh}" >&2; exit 1; }
}
