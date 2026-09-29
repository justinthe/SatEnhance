#!/usr/bin/env bash
# System 1 then System 2, automatically.
#   ./scripts/run_pipeline.sh [--cpu|--gpu] <system 1 args> [-- <system 2 args>]
#   ./scripts/run_pipeline.sh --aoi-text "Perth City, Western Australia" --yes \
#       --start 2026-01-01 --end 2026-03-31 --max-cloud 10 -- --cog
# Stops at the first failure and exits with that stage's exit code (10 = no data found,
# see README for the full table).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
HERE="$(dirname "${BASH_SOURCE[0]}")"

s1=(); s2=(); wrapper=(); seen_sep=0
for a in "$@"; do
  if [[ $seen_sep -eq 1 ]]; then s2+=("$a")
  elif [[ "$a" == "--" ]]; then seen_sep=1
  elif [[ "$a" == "--cpu" || "$a" == "--gpu" ]]; then wrapper+=("$a")
  else s1+=("$a"); fi
done

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

echo "=== System 1: acquire ===" >&2
set +e
# stdout carries SATENHANCE_RUN_ID=...; keep it visible and capture it.
"$HERE/run_system1.sh" "${s1[@]}" | tee "$log"
rc=${PIPESTATUS[0]}
set -e
if [[ $rc -ne 0 ]]; then
  echo "System 1 failed with exit code $rc" >&2
  exit "$rc"
fi

run_id="$(grep -h '^SATENHANCE_RUN_ID=' "$log" | tail -n1 | cut -d= -f2 | tr -d '\r' || true)"
if [[ -z "$run_id" && -f rawdata/LATEST ]]; then
  run_id="$(tr -d '\r\n' < rawdata/LATEST)"
fi
[[ -n "$run_id" ]] || { echo "Could not determine run id from System 1" >&2; exit 1; }

echo "=== System 2: enhance ($run_id) ===" >&2
set +e
"$HERE/run_system2.sh" "${wrapper[@]+"${wrapper[@]}"}" --run-id "$run_id" "${s2[@]+"${s2[@]}"}"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  echo "System 2 failed with exit code $rc" >&2
  exit "$rc"
fi
echo "Done. Raw data: rawdata/$run_id  Enhanced: output/$run_id" >&2
