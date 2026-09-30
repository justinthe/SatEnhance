"""docker/pip_retry.sh retries a failing pip and gives up after the configured attempts."""

import os
import stat
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "docker" / "pip_retry.sh"


def _fake_pip(tmp_path: Path, fail_times: int) -> dict:
    counter = tmp_path / "count"
    counter.write_text("0")
    pip = tmp_path / "pip"
    pip.write_text(
        "#!/bin/sh\n"
        f"n=$(cat {counter}); n=$((n+1)); echo $n > {counter}\n"
        f'echo "args: $@" >> {tmp_path}/args\n'
        f"[ $n -gt {fail_times} ]\n"
    )
    pip.chmod(pip.stat().st_mode | stat.S_IEXEC)
    return {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "PIP_RETRY_BACKOFF": "0"}


def test_retries_until_success(tmp_path):
    env = _fake_pip(tmp_path, fail_times=2)
    r = subprocess.run(["sh", str(SCRIPT), "-r", "req.txt"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "count").read_text().strip() == "3"
    assert "retrying" in r.stderr
    assert "-r req.txt" in (tmp_path / "args").read_text()


def test_gives_up_after_max_attempts(tmp_path):
    env = _fake_pip(tmp_path, fail_times=99)
    env["PIP_RETRY_ATTEMPTS"] = "3"
    r = subprocess.run(["sh", str(SCRIPT), "x"], env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "giving up after 3 attempts" in r.stderr
    assert (tmp_path / "count").read_text().strip() == "3"
