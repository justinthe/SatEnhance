#!/usr/bin/env bash
# Build the container images.
#   ./scripts/build.sh              acquire + enhance-cpu (+ enhance-gpu if a GPU runtime is found)
#   ./scripts/build.sh --cpu-only   acquire + enhance-cpu
#   ./scripts/build.sh --gpu        also build enhance-gpu (slow: compiles mamba-ssm)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

mode=auto
for a in "$@"; do
  case "$a" in
    --cpu-only) mode=cpu ;;
    --gpu) mode=gpu ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) die "unknown option $a" ;;
  esac
done

services=(acquire enhance-cpu)
if [[ "$mode" == gpu ]] || { [[ "$mode" == auto ]] && has_gpu; }; then
  services+=(enhance-gpu)
fi
echo "Building: ${services[*]}"
compose --profile gpu build "${services[@]}"
