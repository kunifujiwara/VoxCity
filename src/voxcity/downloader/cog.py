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

__all__ = ["fetch_cog_crop", "probe_existing", "tile_origins_for"]

_logger = get_logger(__name__)


def _bbox_from_rectangle_vertices(vertices: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) from (lon, lat) vertices."""
    if not vertices:
        raise ValueError("rectangle_vertices must be a non-empty sequence of (lon, lat)")
    lons = [v[0] for v in vertices]
    lats = [v[1] for v in vertices]
    return (min(lons), min(lats), max(lons), max(lats))


# Tiles are 1 or 3 degrees; more than this many means an area far beyond any
# city-scale model (or vertices straddling the antimeridian).
_MAX_TILES = 16


def _lat_tag(lat: int) -> str:
    """'N52' / 'S05' for an integer latitude."""
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"


def _lon_tag(lon: int) -> str:
    """'E013' / 'W074' for an integer longitude."""
    return f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"


def _tile_origins(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float, step: int
) -> List[Tuple[int, int]]:
    """Sorted (lat0, lon0) south-west corners of every ``step``-degree tile touching the bbox."""
    lat_lo = math.floor(min_lat / step) * step
    lat_hi = math.floor(max_lat / step) * step
    lon_lo = math.floor(min_lon / step) * step
    lon_hi = math.floor(max_lon / step) * step
    return sorted(
        (lat, lon)
        for lat in range(lat_lo, lat_hi + 1, step)
        for lon in range(lon_lo, lon_hi + 1, step)
    )


def tile_origins_for(
    rectangle_vertices: Sequence[Tuple[float, float]], step: int, label: str
) -> List[Tuple[int, int]]:
    """
    Tile origins for ``rectangle_vertices``, or [] (with a warning) when the
    area spans more than 180 degrees of longitude or needs more than
    ``_MAX_TILES`` tiles.
    """
    min_lon, min_lat, max_lon, max_lat = _bbox_from_rectangle_vertices(rectangle_vertices)
    origins = _tile_origins(min_lon, min_lat, max_lon, max_lat, step)
    if max_lon - min_lon > 180 or len(origins) > _MAX_TILES:
        _logger.warning(
            "%s: area is too large or crosses the antimeridian (%d tiles, longitude "
            "span %.1f deg; limit %d tiles); skipping this source",
            label, len(origins), max_lon - min_lon, _MAX_TILES,
        )
        return []
    return origins


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


def probe_existing(urls: Sequence[str], timeout: int = 10) -> List[str]:
    """The subset of ``urls`` that answer an anonymous HEAD with 200, in order."""
    return [url for url in urls if _remote_exists(url, timeout)]


# GDAL options for reading COGs over HTTP: skip directory listings (one HEAD
# instead of a bucket LIST), only treat .tif rasters and .zip archives (FABDEM) as
# files, retry transient errors.
_GDAL_HTTP_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.zip",
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
    pad_pixels: int = 1,
    fill_nodata: bool = False,
) -> Optional[str]:
    """
    Crop one or more Cloud-Optimized GeoTIFFs to a rectangle and write a local GeoTIFF.

    Only the window covering the rectangle is read from each source (HTTP
    range requests for remote COGs), then the pieces are mosaicked. The output
    is in the CRS of the first openable source. Area not covered by any source
    is filled with 0 (or, with ``fill_nodata=True``, from the nearest valid pixel).

    Args:
        urls: http(s) URLs or local paths of GeoTIFFs. Sources that fail to
              open are logged and skipped.
        rectangle_vertices: (lon, lat) vertices of the area of interest (WGS84).
        out_path: Where to write the cropped GeoTIFF.
        res: If given, resample the crop to this pixel size (in the source CRS
             units, metres for EPSG:3857) using ``resampling``.
        resampling: 'nearest' (default), 'average' or 'max'. Only used with ``res``.
        timeout: Per-request HTTP timeout in seconds (GDAL_HTTP_TIMEOUT).
        pad_pixels: Source pixels of padding added on every side of the
             rectangle (default 1, so cell-centre sampling at the edge never
             falls outside the crop; DEM adapters use more for interpolation).
        fill_nodata: Merge as float32 with NaN for source nodata and for area
             no source covers, then fill those cells from the nearest valid
             pixel. Returns None if no pixel is valid. Real zero values are
             kept.

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
            # Padding so cell-centre sampling (and DEM interpolation) at the
            # edge never falls just outside the crop.
            bounds = (
                left - pad_pixels * px, bottom - pad_pixels * py,
                right + pad_pixels * px, top + pad_pixels * py,
            )
            merge_kwargs = {"nodata": np.nan, "dtype": "float32"} if fill_nodata else {}
            try:
                data, transform = merge(sources, bounds=bounds, **merge_kwargs)
            except Exception as exc:  # noqa: BLE001
                # Broad on purpose: GDAL read-time errors (e.g. HTTP 503 mid-range-
                # request) surface as CPLE_* classes that are not public
                # RasterioError subclasses, so they can't be caught by type.
                _logger.warning("Reading COG window failed (%s); no raster written", exc)
                return None
        finally:
            for src in sources:
                src.close()

    if fill_nodata:
        filled = _fill_nearest(data[0])
        if filled is None:
            _logger.warning("No valid pixels in the requested window; no raster written")
            return None
        data = filled[np.newaxis, ...]

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


def _fill_nearest(a: np.ndarray) -> Optional[np.ndarray]:
    """Replace non-finite cells with the nearest finite cell; None if none are finite."""
    from scipy import ndimage

    invalid = ~np.isfinite(a)
    if invalid.all():
        return None
    if not invalid.any():
        return a
    idx = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return a[tuple(idx)]


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


