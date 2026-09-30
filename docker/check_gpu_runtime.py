# ruff: noqa: I001, E402
"""Build-time check for the GPU image's RUNTIME stage (docker build has no GPU).

Missing modules/tools fail the build. Driver-related errors only warn.
"""
import os
import shutil
import sysconfig

import setuptools  # noqa: F401
import distutils.core  # noqa: F401
import distutils.sysconfig  # noqa: F401

assert shutil.which("gcc"), "gcc missing (triton compiles a C launcher at run time)"
inc = os.path.join(sysconfig.get_paths()["include"], "Python.h")
assert os.path.exists(inc), f"{inc} missing (python3.11-dev)"

import triton  # noqa: E402, F401
import triton.runtime.build  # noqa: E402, F401

try:
    import mamba_ssm  # noqa: F401
    from sen2sr.models.opensr_baseline.mamba import MambaSR  # noqa: F401
except ModuleNotFoundError:
    raise
except Exception as e:  # noqa: BLE001
    print("WARNING (no GPU/driver at build time?):", e)
print("runtime stage OK")
