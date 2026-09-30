#!/usr/bin/env bash
# System 1: download Sentinel-2 data into ./rawdata
#   ./scripts/run_system1.sh --aoi-text "Perth City, Western Australia" --yes \
#       --start 2026-01-01 --end 2026-03-31 --max-cloud 10 --sensor rgb
#   ./scripts/run_system1.sh --aoi-file site.geojson --start ... --end ...
# All arguments are passed to `satenhance-acquire` (see --help). Without a terminal (cron/CI),
# --non-interactive is added automatically. --aoi-file may be anywhere (relative paths are
# relative to the directory you run this from); it is copied into ./aoi for the container.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_image satenhance-acquire:latest
args=("$@")
mkdir -p aoi
if ! has_tty && ! has_arg --non-interactive "${args[@]}"; then
  args+=(--non-interactive)
fi

# Stage an AOI file into ./aoi/<hash>/ (mounted read-only in the container) and return the
# in-container path. The hash of the absolute path keeps same-named files from different
# folders apart; sidecar files (.dbf, .shx, .prj, ...) travel with a loose .shp.
stage_aoi() {
  local f="$1"
  [[ "$f" = /* ]] || f="$CALLER_DIR/$f"      # relative paths are relative to where you ran us
  [[ -f "$f" ]] || die "AOI file not found: $f"
  local dir base ext
  dir="$(cd "$(dirname "$f")" && pwd)"; f="$dir/$(basename "$f")"
  local key; key="$(printf '%s' "$f" | cksum | cut -d' ' -f1)"
  mkdir -p "aoi/$key"
  cp -f "$f" "aoi/$key/"
  base="${f%.*}"
  for ext in dbf shx prj cpg qix; do
    if [[ -f "$base.$ext" ]]; then cp -f "$base.$ext" "aoi/$key/"; fi
  done
  echo "/data/aoi/$key/$(basename "$f")"
}

for i in "${!args[@]}"; do
  case "${args[$i]}" in
    --aoi-file)   args[$((i+1))]="$(stage_aoi "${args[$((i+1))]}")" ;;
    --aoi-file=*) args[$i]="--aoi-file=$(stage_aoi "${args[$i]#--aoi-file=}")" ;;
  esac
done

# shellcheck disable=SC2046
compose run --rm $(tty_flag) -v "$REPO_ROOT/aoi:/data/aoi:ro" acquire "${args[@]}"
