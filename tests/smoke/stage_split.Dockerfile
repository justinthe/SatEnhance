# Reproduces the two-stage layout of enhance/Dockerfile.gpu on plain ubuntu:22.04 (no CUDA, no GPU)
# to prove the Ubuntu packaging behaviour behind FIX_PLAN_GPU_RUNTIME.md:
#   builder = python3.11-venv (pulls in python3-distutils); runtime must list it explicitly.
FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3.11-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3.11 -m venv /opt/venv
RUN --mount=type=secret,id=cabundle,target=/tmp/ca.pem,required=false \
    if [ -s /tmp/ca.pem ]; then export PIP_CERT=/tmp/ca.pem; fi; \
    /opt/venv/bin/pip install -q --upgrade "setuptools>=70"

# What the OLD runtime stage had: must FAIL to import distutils.core.
FROM ubuntu:22.04 AS runtime-old
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
# The system Python only has the distutils stub here (the bug). The venv would still work thanks
# to the upgraded setuptools, so check the system interpreter:
RUN ! /usr/bin/python3.11 -c "import distutils.core"

# What the runtime stage of Dockerfile.gpu installs now.
FROM ubuntu:22.04 AS runtime-new
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-dev python3-distutils gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
RUN /opt/venv/bin/python -c "\
import os, shutil, sysconfig, setuptools, distutils.core, distutils.sysconfig; \
assert shutil.which('gcc'); \
assert os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')); \
print('runtime-new OK, setuptools', setuptools.__version__)" \
 && /usr/bin/python3.11 -c "import distutils.core, distutils.sysconfig"
