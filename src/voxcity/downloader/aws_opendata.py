"""
Downloader for rasters published on the AWS Open Data Registry.

Everything here goes over plain anonymous HTTPS. There is no boto3, no AWS
CLI and no credential lookup, so it works the same on a laptop, in CI and in
Google Colab without signing in to anything.

Two layers:

- The generic Cloud-Optimized-GeoTIFF reader (``fetch_cog_crop``) now lives
  in ``cog.py`` and is re-exported here.
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

from ..utils.logging import get_logger
from .cog import _bbox_from_rectangle_vertices, _remote_exists, fetch_cog_crop
from .gsi import latlon_to_tile

__all__ = ["fetch_cog_crop", "chmv2_tile_urls", "fetch_chmv2_canopy_height", "CHMV2_BASE_URL"]

_logger = get_logger(__name__)

CHMV2_BASE_URL = (
    "https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3"
)
CHMV2_ZOOM = 10
_MERCATOR_MAX_LAT = 85.05112878
# Zoom-10 tiles are ~39 km wide; more than this many means an area far beyond
# any city-scale model (or antimeridian-crossing vertices), so skip the probes.
_MAX_CHMV2_TILES = 64


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
    # Web Mercator is undefined beyond ~85.05 degrees; clamp so polar or
    # global boxes neither crash nor produce out-of-range tile indices.
    max_lat = max(-_MERCATOR_MAX_LAT, min(_MERCATOR_MAX_LAT, max_lat))
    min_lat = max(-_MERCATOR_MAX_LAT, min(_MERCATOR_MAX_LAT, min_lat))
    last = 2**zoom - 1

    def _clamp(v: int) -> int:
        return max(0, min(last, v))

    x0, y0 = latlon_to_tile(max_lat, min_lon, zoom)  # top-left
    x1, y1 = latlon_to_tile(min_lat, max_lon, zoom)  # bottom-right
    x0, y0, x1, y1 = _clamp(x0), _clamp(y0), _clamp(x1), _clamp(y1)
    keys = {
        _quadkey(x, y, zoom)
        for x in range(min(x0, x1), max(x0, x1) + 1)
        for y in range(min(y0, y1), max(y0, y1) + 1)
    }
    return sorted(keys)


def chmv2_tile_urls(
    rectangle_vertices: Sequence[Tuple[float, float]],
    base_url: str = CHMV2_BASE_URL,
    timeout: int = 10,
) -> List[str]:
    """
    URLs of the CHMv2 tiles that cover ``rectangle_vertices`` and exist.

    The dataset covers land only, so a missing tile key simply means no
    coverage; it is dropped rather than treated as an error.

    Returns [] without any network probe when the area spans more than 180
    degrees of longitude (likely vertices straddling the antimeridian) or
    needs more than ``_MAX_CHMV2_TILES`` tiles.
    """
    min_lon, min_lat, max_lon, max_lat = _bbox_from_rectangle_vertices(rectangle_vertices)
    quadkeys = _quadkeys_for_bbox(min_lon, min_lat, max_lon, max_lat)
    if max_lon - min_lon > 180 or len(quadkeys) > _MAX_CHMV2_TILES:
        _logger.warning(
            "CHMv2: area is too large or crosses the antimeridian (%d tiles, "
            "longitude span %.1f deg; limit %d tiles); skipping the CHMv2 source",
            len(quadkeys), max_lon - min_lon, _MAX_CHMV2_TILES,
        )
        return []
    base = base_url.rstrip("/")
    urls = []
    for qk in quadkeys:
        url = f"{base}/chm/{qk}.tif"
        if _remote_exists(url, timeout):
            urls.append(url)
        else:
            _logger.debug("CHMv2 tile %s not available", qk)
    return urls


def fetch_chmv2_canopy_height(
    rectangle_vertices: Sequence[Tuple[float, float]],
    output_dir: str,
    *,
    res: Optional[float] = None,
    resampling: str = "nearest",
    base_url: str = CHMV2_BASE_URL,
) -> Optional[str]:
    """
    Write ``output_dir/canopy_height.tif`` cropped from Meta/WRI CHMv2 tiles.

    Anonymous HTTPS only; nothing to sign in to. Returns the GeoTIFF path, or
    None when no tile covers the area or none could be read (offline).

    ``res`` is in ground metres. It is converted to Web Mercator units at the
    latitude of the area's centre (EPSG:3857 metres are 1/cos(lat) ground
    metres). Aggregation windows ('max'/'average') are aligned to the crop,
    not to the voxel grid, so results are approximate per cell.
    """
    if res is not None:
        _, min_lat, _, max_lat = _bbox_from_rectangle_vertices(rectangle_vertices)
        res = res / math.cos(math.radians((min_lat + max_lat) / 2))
    urls = chmv2_tile_urls(rectangle_vertices, base_url=base_url)
    if not urls:
        _logger.info("CHMv2: no tiles cover this area (or the bucket is unreachable)")
        return None
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "canopy_height.tif")
    return fetch_cog_crop(urls, rectangle_vertices, out_path, res=res, resampling=resampling)
