"""System 2 must run with only its own dependencies (the CI failure `No module named 'pyproj'`)."""

import subprocess
import sys
import textwrap

# Packages that belong to System 1 only; the enhance image does not install them.
SYSTEM1_ONLY = ["pyproj", "geopandas", "pyogrio", "pystac_client", "tenacity", "pandas"]


def test_enhance_imports_and_synthetic_run_without_system1_packages(tmp_path):
    code = textwrap.dedent(
        f"""
        import sys, importlib, pkgutil
        for name in {SYSTEM1_ONLY!r}:
            sys.modules[name] = None  # any import of these raises ImportError
        import satenhance_enhance
        for m in pkgutil.iter_modules(satenhance_enhance.__path__):
            importlib.import_module("satenhance_enhance." + m.name)
        from satenhance_enhance.synthetic import make_rawdata
        make_rawdata(r"{tmp_path}", size=16)
        print("ok")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
