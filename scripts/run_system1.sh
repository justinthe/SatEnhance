#!/usr/bin/env bash
# System 1: download Sentinel-2 data into ./rawdata
#   ./scripts/run_system1.sh --aoi-text "Perth City, Western Australia" --yes \
#       --start 2026-01-01 --end 2026-03-31 --max-cloud 10 --sensor rgb
#   ./scripts/run_system1.sh --aoi-file site.geojson --start ... --end ...
# All arguments are passed to `satenhance-acquire` (see --help). Without a terminal (cron/CI),
# --non-interactive is added automatically. AOI files must live inside this repo directory
# (they are mounted as /data/... via ./rawdata, ./cache or ./aoi).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

args=("$@")
mkdir -p aoi
if ! has_tty && ! has_arg --non-interactive "${args[@]}"; then
  args+=(--non-interactive)
fi

# A relative --aoi-file is mapped into the container via the ./aoi mount.
for i in "${!args[@]}"; do
  if [[ "${args[$i]}" == "--aoi-file" ]]; then
    f="${args[$((i+1))]}"
    [[ -f "$f" ]] || die "AOI file not found: $f"
    mkdir -p aoi
    cp -f "$f" "aoi/$(basename "$f")"
    # copy sidecar files for shapefiles
    base="${f%.*}"
    for ext in dbf shx prj cpg qix; do [[ -f "$base.$ext" ]] && cp -f "$base.$ext" "aoi/" || true; done
    args[$((i+1))]="/data/aoi/$(basename "$f")"
  fi
done

# shellcheck disable=SC2046
compose run --rm $(tty_flag) -v "$REPO_ROOT/aoi:/data/aoi:ro" acquire "${args[@]}"
