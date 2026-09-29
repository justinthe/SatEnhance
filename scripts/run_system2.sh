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
  if has_gpu; then target=gpu; else target=cpu; fi
fi
service="enhance-$target"
echo "Using service: $service" >&2

# shellcheck disable=SC2046
compose --profile gpu run --rm $(tty_flag) "$service" "${args[@]+"${args[@]}"}"
