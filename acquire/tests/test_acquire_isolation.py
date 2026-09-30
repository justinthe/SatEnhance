"""System 1 must not need the ML stack (its image has no torch)."""

import subprocess
import sys
import textwrap

SYSTEM2_ONLY = ["torch", "torchvision", "sen2sr", "mlstac", "timm", "einops", "PIL"]


def test_acquire_imports_without_system2_packages():
    code = textwrap.dedent(
        f"""
        import sys, importlib, pkgutil
        for name in {SYSTEM2_ONLY!r}:
            sys.modules[name] = None
        import satenhance_acquire
        for m in pkgutil.walk_packages(satenhance_acquire.__path__, "satenhance_acquire."):
            importlib.import_module(m.name)
        print("ok")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
