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
