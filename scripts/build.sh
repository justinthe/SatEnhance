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
# One image at a time: building them in parallel makes them compete for bandwidth, which on a
# slow link is what causes pip read timeouts. Re-running resumes from the pip cache.
echo "Building (one at a time): ${services[*]}"
echo "Rough downloads: acquire ~0.2 GB, enhance-cpu ~1-6 GB (PyTorch), enhance-gpu ~6 GB + CUDA base images"
for svc in "${services[@]}"; do
  echo "=== building $svc ===" >&2
  compose --profile gpu build "$svc"
done
