# System 1: build from the repo root:  docker build -f acquire/Dockerfile .
# Override for networks that cannot reach the Debian mirrors, e.g. --build-arg BASE_IMAGE=python:3.11
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PIP_CACHE_DIR=/root/.cache/pip PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 HOME=/tmp

# rasterio wheels need libexpat at runtime (not present in python:*-slim)
RUN ldconfig -p | grep -q libexpat.so.1 || \
    (apt-get update && apt-get install -y --no-install-recommends libexpat1 \
     && rm -rf /var/lib/apt/lists/*)

COPY docker/pip_retry.sh /usr/local/bin/pip-retry

WORKDIR /src
COPY acquire/requirements.lock /tmp/requirements.lock
RUN --mount=type=secret,id=cabundle,target=/tmp/ca.pem,required=false \
    --mount=type=cache,target=/root/.cache/pip \
    if [ -s /tmp/ca.pem ]; then export PIP_CERT=/tmp/ca.pem; fi; \
    pip-retry -r /tmp/requirements.lock

COPY common ./common
COPY acquire ./acquire
RUN --mount=type=secret,id=cabundle,target=/tmp/ca.pem,required=false \
    --mount=type=cache,target=/root/.cache/pip \
    if [ -s /tmp/ca.pem ]; then export PIP_CERT=/tmp/ca.pem; fi; \
    pip-retry --no-deps ./common ./acquire

RUN useradd -m -u 1000 app && mkdir -p /data/rawdata /data/cache && chown -R app /data
USER app
WORKDIR /data
ENTRYPOINT ["satenhance-acquire"]
CMD ["--help"]
