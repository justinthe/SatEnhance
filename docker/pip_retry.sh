#!/bin/sh
# pip install with retries, for slow or flaky networks.
#   pip-retry <arguments for `pip install`>
# pip's own --retries only covers connection setup; a read timeout in the middle of a large
# wheel aborts the whole install. Retrying the whole command is cheap when the BuildKit pip
# cache mount is in use: wheels that finished downloading are served from the cache.
attempts="${PIP_RETRY_ATTEMPTS:-5}"
i=1
while :; do
  if pip install "$@"; then
    exit 0
  fi
  if [ "$i" -ge "$attempts" ]; then
    echo "pip-retry: giving up after $attempts attempts" >&2
    exit 1
  fi
  wait=$((i * ${PIP_RETRY_BACKOFF:-10}))
  echo "pip-retry: attempt $i/$attempts failed; retrying in ${wait}s (finished wheels are cached)" >&2
  sleep "$wait"
  i=$((i + 1))
done
