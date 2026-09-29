"""Copernicus Data Space Ecosystem provider.

[VERIFY] against current CDSE documentation before relying on it: the STAC root URL,
the collection id, the asset key naming (B04_10m ...), and S3 access-key issuance.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any

from pystac_client import Client
from pystac_client.exceptions import APIError
from satenhance_common.exit_codes import ExitCode, SatEnhanceError
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .base import Candidate

log = logging.getLogger(__name__)

STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
COLLECTION = "sentinel-2-l2a"
S3_ENDPOINT = "eodata.dataspace.copernicus.eu"

# Native resolution of each band; used to build candidate asset keys.
BAND_RES = {
    "B02": 10, "B03": 10, "B04": 10, "B08": 10,
    "B05": 20, "B06": 20, "B07": 20, "B8A": 20, "B11": 20, "B12": 20,
    "SCL": 20,
}


def asset_keys(band: str) -> list[str]:
    res = BAND_RES[band]
    return [f"{band}_{res}m", band, f"{band}_20m", f"{band}_10m", f"{band}_60m"]


def _baseline(props: dict) -> str | None:
    for k in ("processing:version", "s2:processing_baseline", "processing_baseline"):
        if props.get(k):
            return str(props[k])
    return None


def _transient(exc: BaseException) -> bool:
    """Retry network errors and 5xx/429 responses; 4xx (e.g. unsupported query) is final."""
    if isinstance(exc, APIError):
        code = getattr(exc, "status_code", None)
        return code is None or code >= 500 or code == 429
    return isinstance(exc, OSError)


def item_to_candidate(item: Any) -> Candidate:
    props = item.properties
    assets: dict[str, str] = {}
    for band in BAND_RES:
        for key in asset_keys(band):
            if key in item.assets:
                assets[band] = item.assets[key].href
                break
    return Candidate(
        id=item.id,
        datetime=props.get("datetime") or props.get("start_datetime") or "",
        tile_cloud=props.get("eo:cloud_cover"),
        processing_baseline=_baseline(props),
        assets=assets,
        footprint=item.geometry,
        properties={k: props[k] for k in ("platform", "s2:tile_id", "grid:code") if k in props},
    )


class CdseProvider:
    name = "cdse"

    def __init__(self, client: Any | None = None, env: dict[str, str] | None = None):
        self._client = client
        self._env = env if env is not None else os.environ

    # -- auth -------------------------------------------------------------
    def _keys(self) -> tuple[str, str]:
        ak, sk = self._env.get("CDSE_S3_ACCESS_KEY", ""), self._env.get("CDSE_S3_SECRET_KEY", "")
        return ak, sk

    def check_auth(self) -> None:
        ak, sk = self._keys()
        if not ak or not sk:
            raise SatEnhanceError(
                ExitCode.AUTH_FAILURE,
                "CDSE S3 credentials missing: set CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY "
                "(see .env.example).",
            )

    def gdal_env(self) -> dict[str, str]:
        ak, sk = self._keys()
        return {
            "AWS_S3_ENDPOINT": S3_ENDPOINT,
            "AWS_ACCESS_KEY_ID": ak,
            "AWS_SECRET_ACCESS_KEY": sk,
            "AWS_VIRTUAL_HOSTING": "FALSE",
            "AWS_HTTPS": "YES",
            "GDAL_HTTP_MAX_RETRY": "4",
            "GDAL_HTTP_RETRY_DELAY": "2",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".jp2,.tif,.xml",
        }

    def href(self, candidate: Candidate, band: str) -> str:
        try:
            href = candidate.assets[band]
        except KeyError as e:
            raise SatEnhanceError(
                ExitCode.NETWORK_FAILURE,
                f"Scene {candidate.id} has no asset for band {band} "
                f"(available: {sorted(candidate.assets)})",
            ) from e
        if href.startswith("s3://"):
            return "/vsis3/" + href[len("s3://"):]
        return href

    # -- search -----------------------------------------------------------
    def _open(self) -> Any:
        if self._client is None:
            self._client = Client.open(STAC_URL)
        return self._client

    @retry(
        retry=retry_if_exception(_transient),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=16),
        reraise=True,
    )
    def _run(self, **kwargs: Any) -> list[Any]:
        return list(self._open().search(**kwargs).items())

    def search(
        self, aoi: BaseGeometry, start: date, end: date, max_tile_cloud: float, limit: int = 50
    ) -> list[Candidate]:
        kwargs: dict[str, Any] = dict(
            collections=[COLLECTION],
            intersects=mapping(aoi),
            datetime=f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
            max_items=limit,
        )
        try:
            try:
                items = self._run(query={"eo:cloud_cover": {"lte": max_tile_cloud}}, **kwargs)
            except APIError as e:
                # Server may not support the query extension: filter client-side instead.
                log.warning("Server-side cloud filter rejected (%s); filtering client-side", e)
                items = self._run(**kwargs)
        except (APIError, OSError) as e:
            raise SatEnhanceError(
                ExitCode.NETWORK_FAILURE, f"CDSE catalogue search failed: {e}"
            ) from e
        cands = [item_to_candidate(i) for i in items]
        return [
            c for c in cands if c.tile_cloud is None or c.tile_cloud <= max_tile_cloud
        ]
