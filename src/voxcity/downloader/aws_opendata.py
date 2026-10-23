"""
Downloader for rasters published on the AWS Open Data Registry.

Everything here goes over plain anonymous HTTPS. There is no boto3, no AWS
CLI and no credential lookup, so it works the same on a laptop, in CI and in
Google Colab without signing in to anything.

Two layers:

- ``fetch_cog_crop``: generic. Given URLs (or local paths) of Cloud-Optimized
  GeoTIFFs, read only the window covering a rectangle through GDAL's
  ``/vsicurl/`` (HTTP range requests), mosaic the pieces, optionally resample
  to a coarser resolution, and write a small local GeoTIFF.
- CHMv2 adapter: Meta / WRI High Resolution Canopy Height Maps v2
  (https://registry.opendata.aws/dataforgood-fb-forestsv2/, CC BY 4.0).
  Tiles are zoom-10 Web Mercator tiles named by their 10-digit Bing quadkey,
  EPSG:3857, ~1.19 m/px, uint8 with 255 as the "no canopy" sentinel.

Usage:
    path = fetch_chmv2_canopy_height(rectangle_vertices, output_dir)
"""

from __future__ import annotations

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
from .gsi import latlon_to_tile

__all__ = ["fetch_cog_crop", "chmv2_tile_urls", "fetch_chmv2_canopy_height", "CHMV2_BASE_URL"]

_logger = get_logger(__name__)

CHMV2_BASE_URL = (
    "https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3"
)
CHMV2_ZOOM = 10


def _bbox_from_rectangle_vertices(vertices: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) from (lon, lat) vertices."""
    if not vertices:
        raise ValueError("rectangle_vertices must be a non-empty sequence of (lon, lat)")
    lons = [v[0] for v in vertices]
    lats = [v[1] for v in vertices]
    return (min(lons), min(lats), max(lons), max(lats))


def _quadkey(x: int, y: int, zoom: int) -> str:
    """Bing Maps quadkey for XYZ tile (x, y) at ``zoom``."""
    digits = []
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        digits.append(str(digit))
    return "".join(digits)


def _quadkeys_for_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float, zoom: int = CHMV2_ZOOM
) -> List[str]:
    """Sorted, de-duplicated quadkeys of every ``zoom`` tile touching the bbox."""
    x0, y0 = latlon_to_tile(max_lat, min_lon, zoom)  # top-left
    x1, y1 = latlon_to_tile(min_lat, max_lon, zoom)  # bottom-right
    keys = {
        _quadkey(x, y, zoom)
        for x in range(min(x0, x1), max(x0, x1) + 1)
        for y in range(min(y0, y1), max(y0, y1) + 1)
    }
    return sorted(keys)


def _remote_exists(url: str, timeout: int) -> bool:
    """True if an anonymous HEAD returns 200. Network errors count as missing."""
    try:
        return requests.head(url, timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def chmv2_tile_urls(
    rectangle_vertices: Sequence[Tuple[float, float]],
    base_url: str = CHMV2_BASE_URL,
    timeout: int = 30,
) -> List[str]:
    """
    URLs of the CHMv2 tiles that cover ``rectangle_vertices`` and exist.

    The dataset covers land only, so a missing tile key simply means no
    coverage; it is dropped rather than treated as an error.
    """
    min_lon, min_lat, max_lon, max_lat = _bbox_from_rectangle_vertices(rectangle_vertices)
    base = base_url.rstrip("/")
    urls = []
    for qk in _quadkeys_for_bbox(min_lon, min_lat, max_lon, max_lat):
        url = f"{base}/chm/{qk}.tif"
        if _remote_exists(url, timeout):
            urls.append(url)
        else:
            _logger.debug("CHMv2 tile %s not available", qk)
    return urls


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
        ``out_path``, or None if no source could be opened.
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
                    sources.append(rasterio.open(_gdal_path(url)))
                except RasterioIOError as exc:
                    _logger.info("Skipping %s: %s", url, exc)
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
            data, transform = merge(sources, bounds=bounds)
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
