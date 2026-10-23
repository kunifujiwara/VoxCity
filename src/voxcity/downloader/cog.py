"""
Generic reader for Cloud-Optimized GeoTIFFs over anonymous HTTP(S).

Everything here goes over plain anonymous HTTPS. There is no boto3, no AWS
CLI and no credential lookup, so it works the same on a laptop, in CI and in
Google Colab without signing in to anything.

``fetch_cog_crop`` reads only the window covering a rectangle from each
source through GDAL's ``/vsicurl/`` (HTTP range requests), mosaics the
pieces, optionally resamples, and writes a small local GeoTIFF. Dataset
adapters live in ``aws_opendata.py`` and ``fabdem.py``.
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds

from ..utils.logging import get_logger

__all__ = ["fetch_cog_crop"]

_logger = get_logger(__name__)


def _bbox_from_rectangle_vertices(vertices: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) from (lon, lat) vertices."""
    if not vertices:
        raise ValueError("rectangle_vertices must be a non-empty sequence of (lon, lat)")
    lons = [v[0] for v in vertices]
    lats = [v[1] for v in vertices]
    return (min(lons), min(lats), max(lons), max(lats))

def _remote_exists(url: str, timeout: int) -> bool:
    """
    True if an anonymous HEAD returns 200.

    404 is the normal "no tile here" answer and is silent. Any other status
    or a network error also returns False but is logged as a warning, since
    it silently shrinks coverage.
    """
    try:
        status = requests.head(url, timeout=timeout).status_code
    except requests.RequestException as exc:
        _logger.warning("Tile probe failed for %s: %s", url, exc)
        return False
    if status == 200:
        return True
    if status != 404:
        _logger.warning("Tile probe for %s returned HTTP %s", url, status)
    return False


# GDAL options for reading COGs over HTTP: skip directory listings (one HEAD
# instead of a bucket LIST), only treat .tif as rasters, retry transient errors.
_GDAL_HTTP_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
}

_RESAMPLING = {
    "nearest": Resampling.nearest,
    "average": Resampling.average,
    "max": Resampling.max,
}


def _gdal_path(url: str) -> str:
    """Prefix http(s) URLs for GDAL's virtual curl filesystem; leave local paths alone."""
    if url.startswith(("http://", "https://")):
        return f"/vsicurl/{url}"
    return url


def fetch_cog_crop(
    urls: Sequence[str],
    rectangle_vertices: Sequence[Tuple[float, float]],
    out_path: str,
    *,
    res: Optional[float] = None,
    resampling: str = "nearest",
    timeout: int = 60,
) -> Optional[str]:
    """
    Crop one or more Cloud-Optimized GeoTIFFs to a rectangle and write a local GeoTIFF.

    Only the window covering the rectangle is read from each source (HTTP
    range requests for remote COGs), then the pieces are mosaicked. The output
    is in the CRS of the first openable source. Area not covered by any source
    is filled with 0.

    Args:
        urls: http(s) URLs or local paths of GeoTIFFs. Sources that fail to
              open are logged and skipped.
        rectangle_vertices: (lon, lat) vertices of the area of interest (WGS84).
        out_path: Where to write the cropped GeoTIFF.
        res: If given, resample the crop to this pixel size (in the source CRS
             units, metres for EPSG:3857) using ``resampling``.
        resampling: 'nearest' (default), 'average' or 'max'. Only used with ``res``.
        timeout: Per-request HTTP timeout in seconds (GDAL_HTTP_TIMEOUT).

    Returns:
        ``out_path``, or None if no source could be opened or the read failed.
    """
    if resampling not in _RESAMPLING:
        raise ValueError(f"resampling must be one of {sorted(_RESAMPLING)}, got {resampling!r}")

    min_lon, min_lat, max_lon, max_lat = _bbox_from_rectangle_vertices(rectangle_vertices)
    env = dict(_GDAL_HTTP_ENV, GDAL_HTTP_TIMEOUT=str(timeout))

    with rasterio.Env(**env):
        sources = []
        try:
            for url in urls:
                try:
                    src = rasterio.open(_gdal_path(url))
                except RasterioIOError as exc:
                    # Remote URLs were probed as existing, so failing to open
                    # one is unexpected; local paths are the caller's business.
                    level = logging.WARNING if url.startswith(("http://", "https://")) else logging.INFO
                    _logger.log(level, "Skipping %s: %s", url, exc)
                    continue
                if sources and src.crs != sources[0].crs:
                    _logger.warning(
                        "Skipping %s: CRS %s differs from %s", url, src.crs, sources[0].crs
                    )
                    src.close()
                    continue
                sources.append(src)
            if not sources:
                return None

            crs = sources[0].crs
            px, py = sources[0].res
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", crs, min_lon, min_lat, max_lon, max_lat
            )
            # One source pixel of padding so cell-centre sampling at the edge
            # never falls just outside the crop.
            bounds = (left - px, bottom - py, right + px, top + py)
            try:
                data, transform = merge(sources, bounds=bounds)
            except Exception as exc:  # noqa: BLE001
                # Broad on purpose: GDAL read-time errors (e.g. HTTP 503 mid-range-
                # request) surface as CPLE_* classes that are not public
                # RasterioError subclasses, so they can't be caught by type.
                _logger.warning("Reading COG window failed (%s); no raster written", exc)
                return None
        finally:
            for src in sources:
                src.close()

    if res is not None:
        data, transform = _resample(data, transform, crs, res, _RESAMPLING[resampling])

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": 1,
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data[0], 1)
    return out_path


def _resample(data, transform, crs, res: float, resampling: Resampling):
    """Warp a (1, H, W) array to pixel size ``res`` in the same CRS."""
    src = data[0]
    width_m = src.shape[1] * transform.a
    height_m = src.shape[0] * (-transform.e)
    width = max(1, int(math.ceil(width_m / res)))
    height = max(1, int(math.ceil(height_m / res)))
    dst_transform = from_origin(transform.c, transform.f, res, res)
    dst = np.zeros((height, width), dtype=src.dtype)
    reproject(
        src, dst,
        src_transform=transform, src_crs=crs,
        dst_transform=dst_transform, dst_crs=crs,
        resampling=resampling,
    )
    return dst[np.newaxis, ...], dst_transform


