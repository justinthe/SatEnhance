#!/usr/bin/env bash
# Build the container images.
#   ./scripts/build.sh              acquire + enhance-cpu
#   ./scripts/build.sh --gpu        also build enhance-gpu (large download; compiles mamba-ssm)
#   ./scripts/build.sh --gpu-only   only build enhance-gpu
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

mode=cpu
for a in "$@"; do
  case "$a" in
    --cpu-only) mode=cpu ;;   # kept for compatibility: this is now the default
    --gpu) mode=gpu ;;
    --gpu-only) mode=gpu-only ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) die "unknown option $a" ;;
  esac
done

services=(acquire enhance-cpu)
[[ "$mode" == gpu ]] && services+=(enhance-gpu)
[[ "$mode" == gpu-only ]] && services=(enhance-gpu)
echo "Building: ${services[*]}"
compose --profile gpu build "${services[@]}"
