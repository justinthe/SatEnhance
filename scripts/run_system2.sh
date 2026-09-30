#!/usr/bin/env bash
# System 2: enhance rawdata into ./output
#   ./scripts/run_system2.sh                     # latest run in ./rawdata
#   ./scripts/run_system2.sh --run-id 2026...    # a specific run
#   ./scripts/run_system2.sh --model full --gpu
#   ./scripts/run_system2.sh prefetch            # download model weights into ./cache
# Wrapper flags: --cpu / --gpu choose the image (default: GPU if available, else CPU).
# All other arguments go to `satenhance-enhance`.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

target=auto
args=()
for a in "$@"; do
  case "$a" in
    --cpu) target=cpu ;;
    --gpu) target=gpu ;;
    *) args+=("$a") ;;
  esac
done
if [[ "$target" == auto ]]; then
  # GPU only if a GPU runtime is present AND the GPU image has been built.
  if has_gpu && docker image inspect satenhance-enhance:gpu >/dev/null 2>&1; then
    target=gpu
  else
    target=cpu
  fi
fi
service="enhance-$target"
if [[ "$target" == gpu ]]; then
  require_image "satenhance-enhance:gpu" "./scripts/build.sh --gpu-only"
else
  require_image "satenhance-enhance:cpu"
fi
echo "Using service: $service" >&2

# shellcheck disable=SC2046
compose --profile gpu run --rm $(tty_flag) "$service" "${args[@]+"${args[@]}"}"
