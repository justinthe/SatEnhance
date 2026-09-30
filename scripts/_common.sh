# Sourced by the run scripts. Not executable on its own.
set -euo pipefail

# Remember where the user ran the script from: relative paths in arguments are relative to it.
CALLER_DIR="$PWD"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load .env (if present) without exporting comments/blank lines.
if [[ -f .env ]]; then
  set -a; . ./.env; set +a
fi

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

has_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 \
    && docker info 2>/dev/null | grep -qi nvidia
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
require_image() {
  docker image inspect "$1" >/dev/null 2>&1 \
    || { echo "error: image '$1' not found. Build it first:  ./scripts/build.sh" >&2; exit 1; }
}
