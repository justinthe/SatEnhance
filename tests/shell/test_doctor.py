"""scripts/doctor.sh with fake `docker`, `curl` and `nvidia-smi` so every branch can be exercised."""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _exe(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def sandbox(tmp_path):
    """A copy of scripts/ (they cd to their own repo root) plus a fake-tool bin dir."""
    root = tmp_path / "repo"
    (root / "cache" / "models").mkdir(parents=True)
    shutil.copytree(REPO / "scripts", root / "scripts")
    (root / ".env").write_text("CDSE_S3_ACCESS_KEY=abc\nCDSE_S3_SECRET_KEY=def\n"
                               "NOMINATIM_USER_AGENT=Test/1 (me@mydomain.org)\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    # fake docker: behaviour steered by env vars
    _exe(bin_ / "docker", r'''
case "$1" in
  info) [ "$FAKE_DAEMON" = down ] && exit 1; echo "Runtimes: runc"; exit 0 ;;
  compose) echo "${FAKE_COMPOSE:-2.29.1}"; exit 0 ;;
  image) case " $FAKE_IMAGES " in *" $3 "*) exit 0 ;; *) exit 1 ;; esac ;;
  run) [ "$FAKE_GPU_IN_CONTAINER" = yes ] && exit 0 || exit 1 ;;
esac
exit 0
''')
    _exe(bin_ / "curl", r'''
for a in "$@"; do url="$a"; done
case "$FAKE_NET" in
  down) printf 000; exit 6 ;;
  limited) case "$url" in *docker*) printf 429; exit 0 ;; esac ;;
esac
printf 200
''')
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "FAKE_DAEMON": "up",
           "FAKE_IMAGES": "satenhance-acquire:latest satenhance-enhance:cpu", "FAKE_NET": "up",
           "HOME": str(tmp_path)}
    for k in ("CDSE_S3_ACCESS_KEY", "CDSE_S3_SECRET_KEY", "NOMINATIM_USER_AGENT"):
        env.pop(k, None)
    return root, bin_, env


def doctor(sandbox, **extra_env):
    root, _, env = sandbox
    extra_env.setdefault("DOCTOR_FREE_KB", str(200 * 1024 * 1024))  # tests must not depend on real disk
    r = subprocess.run(["bash", str(root / "scripts" / "doctor.sh")], env={**env, **extra_env},
                       capture_output=True, text=True, cwd=root)
    return r.returncode, r.stdout


def test_healthy_machine_passes(sandbox):
    rc, out = doctor(sandbox)
    assert rc == 0, out
    assert "FAIL" not in out and "doctor:" in out and "0 failed" in out
    assert "PASS  System 1 image" in out and "CDSE_S3_ACCESS_KEY is set" in out
    assert "abc" not in out and "def" not in out.replace("default", "")  # secrets never printed


def test_disk_space_thresholds(sandbox):
    gb = 1024 * 1024
    rc, out = doctor(sandbox, DOCTOR_FREE_KB=str(5 * gb))
    assert rc == 1 and "FAIL  only 5 GB free" in out
    rc, out = doctor(sandbox, DOCTOR_FREE_KB=str(20 * gb))
    assert rc == 0 and "WARN  20 GB free" in out
    rc, out = doctor(sandbox, DOCTOR_FREE_KB=str(50 * gb))
    assert rc == 0 and "PASS  50 GB free" in out


def test_daemon_down_fails(sandbox):
    rc, out = doctor(sandbox, FAKE_DAEMON="down")
    assert rc == 1 and "FAIL  the Docker daemon is not reachable" in out


def test_old_compose_fails_with_hint(sandbox):
    rc, out = doctor(sandbox, FAKE_COMPOSE="2.20.3")
    assert rc == 1 and "too old" in out and "2.24" in out


def test_missing_images_warn_with_build_hint(sandbox):
    rc, out = doctor(sandbox, FAKE_IMAGES="")
    assert rc == 0 and "WARN  System 1 image" in out and "./scripts/build.sh" in out


def test_missing_env_and_keys_warn(sandbox):
    root, _, _ = sandbox
    (root / ".env").write_text("CDSE_S3_ACCESS_KEY=\n")
    rc, out = doctor(sandbox)
    assert rc == 0 and "WARN  CDSE_S3_ACCESS_KEY is empty" in out
    assert "WARN  CDSE_S3_SECRET_KEY is empty" in out and "NOMINATIM_USER_AGENT is not set" in out
    (root / ".env").unlink()
    rc, out = doctor(sandbox)
    assert "WARN  .env is missing" in out


def test_network_down_fails_each_host(sandbox):
    rc, out = doctor(sandbox, FAKE_NET="down")
    assert rc == 1 and out.count("is unreachable") == 8 and "FAIL  pypi.org is unreachable" in out


def test_rate_limit_is_a_warning_not_a_failure(sandbox):
    rc, out = doctor(sandbox, FAKE_NET="limited")
    assert rc == 0 and "WARN  registry-1.docker.io answered 429" in out


def test_network_can_be_skipped(sandbox):
    rc, out = doctor(sandbox, DOCTOR_SKIP_NETWORK="1", FAKE_NET="down")
    assert rc == 0 and "skipped" in out


def test_model_cache_states(sandbox):
    root, _, _ = sandbox
    (root / "cache" / "models" / "lite_ok").mkdir()
    (root / "cache" / "models" / "lite_ok" / ".complete").write_text("{}")
    (root / "cache" / "models" / "lite_half").mkdir()
    rc, out = doctor(sandbox)
    assert "PASS  model lite_ok is complete" in out and "WARN  model lite_half is incomplete" in out


def test_gpu_visible_on_host_but_not_in_container_fails(sandbox):
    root, bin_, env = sandbox
    _exe(bin_ / "nvidia-smi", 'echo "GPU 0: Test GPU (UUID: x)"\n')
    rc, out = doctor(sandbox, FAKE_IMAGES="satenhance-enhance:gpu", FAKE_GPU_IN_CONTAINER="no")
    assert rc == 1 and "NOT visible inside the GPU image" in out and "NVIDIA Container Toolkit" in out
    rc, out = doctor(sandbox, FAKE_IMAGES="satenhance-enhance:gpu", FAKE_GPU_IN_CONTAINER="yes")
    assert rc == 0 and "PASS  the GPU is visible inside the GPU image" in out


def test_has_gpu_helper_uses_a_real_container_test(sandbox):
    root, bin_, env = sandbox
    _exe(bin_ / "nvidia-smi", "exit 0\n")
    cmd = f"source {root}/scripts/_common.sh; has_gpu && echo GPU || echo NOGPU"
    for visible, expect in (("yes", "GPU"), ("no", "NOGPU")):
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, cwd=root,
                           env={**env, "FAKE_IMAGES": "satenhance-enhance:gpu",
                                "FAKE_GPU_IN_CONTAINER": visible})
        assert r.stdout.strip() == expect


# ---- .env loading (a real bug: `source .env` chokes on `Name/1 (contact)`) --------------------
def _load(root, env_text, var):
    (root / ".env").write_text(env_text)
    cmd = f'source {root}/scripts/_common.sh; printf "%s" "${{{var}-UNSET}}"'
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, cwd=root,
                       env={k: v for k, v in os.environ.items() if k != var})
    assert r.returncode == 0, r.stderr
    return r.stdout


@pytest.mark.parametrize("text,var,expected", [
    ('NOMINATIM_USER_AGENT=SatEnhance/0.1 (you@example.com)\n', "NOMINATIM_USER_AGENT",
     "SatEnhance/0.1 (you@example.com)"),                              # unquoted, spaces + parens
    ('NOMINATIM_USER_AGENT="SatEnhance/0.1 (you@example.com)"\n', "NOMINATIM_USER_AGENT",
     "SatEnhance/0.1 (you@example.com)"),                              # double-quoted
    ("K='single quoted'\n", "K", "single quoted"),
    ("# comment\n\nK=value\n", "K", "value"),
    ("K=value\r\n", "K", "value"),                                      # Windows line endings
    ("K=$(echo pwned)\n", "K", "$(echo pwned)"),                        # never evaluated
    ("K=a=b=c\n", "K", "a=b=c"),
    ("  # indented comment\nK=1\n", "K", "1"),
    ("K=\n", "K", ""),
    ("nothing here\n", "K", "UNSET"),
])
def test_env_file_is_parsed_not_sourced(sandbox, text, var, expected):
    root, _, _ = sandbox
    assert _load(root, text, var) == expected


def test_shipped_env_example_loads_cleanly(sandbox):
    root, _, _ = sandbox
    shutil.copy(REPO / ".env.example", root / ".env.example")
    out = _load(root, (root / ".env.example").read_text(), "NOMINATIM_USER_AGENT")
    assert out.startswith("SatEnhance/") and "(" in out


def test_require_image_hint(sandbox):
    root, _, env = sandbox
    cmd = f"source {root}/scripts/_common.sh; require_image satenhance-enhance:gpu 'build it: ./scripts/build.sh --gpu-only'"
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, cwd=root,
                       env={**env, "FAKE_IMAGES": ""})
    assert r.returncode == 1 and "--gpu-only" in r.stderr
