"""Resumable, verified model download into an atomically published cache directory.

`mlstac.download` writes mlm.json first, then streams each weight file with a 30 s timeout and
no retry or resume. On a slow link that leaves a truncated model behind that a naive
"does mlm.json exist?" cache check would then trust forever. This replaces it:

* files download to `<cache>/.<name>.partial/` and resume with HTTP Range across retries
* every file is size-checked, then a `.complete` marker is written
* only then is the directory renamed into place, so `<cache>/<name>` is either absent or whole
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import pystac
import requests
from satenhance_common.exit_codes import ExitCode, SatEnhanceError

log = logging.getLogger(__name__)

COMPLETE_MARKER = ".complete"
CHUNK = 64 * 1024  # small so partial progress reaches disk before a dropped connection
TIMEOUT = (15, 90)  # connect, read seconds


def is_complete(model_dir: Path) -> bool:
    return (Path(model_dir) / COMPLETE_MARKER).exists()


def _net_error(msg: str, e: BaseException | None = None) -> SatEnhanceError:
    err = SatEnhanceError(ExitCode.NETWORK_FAILURE, msg)
    if e is not None:
        err.__cause__ = e
    return err


def _get_text(url: str, session: requests.Session, attempts: int, backoff: float,
              sleep: Callable[[float], None]) -> str:
    for i in range(1, attempts + 1):
        try:
            r = session.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            if i == attempts:
                raise _net_error(f"Could not fetch {url}: {e}", e) from e
            log.warning("fetch %s failed (%s); retry %d/%d", url, e, i, attempts - 1)
            sleep(backoff * i)
    raise AssertionError("unreachable")


def download_file(url: str, dest: Path, *, session: requests.Session, attempts: int = 5,
                  backoff: float = 5.0, sleep: Callable[[float], None] = time.sleep) -> Path:
    """Download `url` to `dest`, resuming a partial `dest.part`. Retries only count when an
    attempt made no progress, so a slow-but-moving link keeps going."""
    part = dest.with_name(dest.name + ".part")
    stalls = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        made_progress = False
        try:
            with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
                if r.status_code == 416 and have:  # already have every byte
                    part.replace(dest)
                    return dest
                r.raise_for_status()
                resumed = r.status_code == 206
                if have and not resumed:  # server ignored Range: start over
                    have = 0
                total = None
                if "Content-Range" in r.headers:
                    total = int(r.headers["Content-Range"].rsplit("/", 1)[-1])
                elif "Content-Length" in r.headers:
                    total = have + int(r.headers["Content-Length"])
                with open(part, "ab" if resumed else "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if chunk:
                            f.write(chunk)
                            made_progress = True
                size = part.stat().st_size
                if total is not None and size != total:
                    raise requests.ConnectionError(f"incomplete: {size} of {total} bytes")
                if size == 0:
                    raise requests.ConnectionError("empty response")
            part.replace(dest)
            return dest
        except (requests.RequestException, OSError) as e:
            if made_progress:
                stalls = 0
            else:
                stalls += 1
            if stalls >= attempts:
                raise _net_error(f"Download failed for {url}: {e}", e) from e
            log.warning("download %s interrupted (%s); resuming (%d/%d)", dest.name, e,
                        stalls, attempts)
            sleep(backoff * max(stalls, 1))


def fetch_model(url: str, target: Path, *, session: requests.Session | None = None,
                attempts: int = 5, backoff: float = 5.0,
                sleep: Callable[[float], None] = time.sleep) -> Path:
    """Download the MLM-STAC model at `url` into `target` (atomically)."""
    session = session or requests.Session()
    target = Path(target)
    partial = target.with_name(f".{target.name}.partial")
    partial.mkdir(parents=True, exist_ok=True)

    text = _get_text(url, session, attempts, backoff, sleep)
    try:
        item = pystac.Item.from_dict(json.loads(text))
    except Exception as e:  # noqa: BLE001  (bad JSON / not a STAC item: pystac raises several types)
        raise _net_error(f"{url} is not a valid model description: {e}", e) from e
    (partial / "mlm.json").write_text(json.dumps(item.to_dict(), indent=2, ensure_ascii=False))

    sizes: dict[str, int] = {}
    for key, asset in item.assets.items():
        name = Path(urlparse(asset.href).path).name
        dest = partial / name
        if dest.exists() and dest.stat().st_size > 0:
            log.info("model file %s already downloaded", name)
        else:
            log.info("downloading model file %s (%s)", name, key)
            download_file(asset.href, dest, session=session, attempts=attempts,
                          backoff=backoff, sleep=sleep)
        sizes[name] = dest.stat().st_size

    empty = [n for n, sz in sizes.items() if sz == 0]
    if empty:
        raise _net_error(f"Downloaded model file(s) are empty: {empty}")
    (partial / COMPLETE_MARKER).write_text(
        json.dumps({"url": url, "files": sizes, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    )
    if target.exists():
        shutil.rmtree(target)  # stale/incomplete cache from an earlier run or older version
    partial.rename(target)
    return target
