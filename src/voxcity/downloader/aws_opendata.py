"""
Downloader for rasters published on the AWS Open Data Registry.

Everything here goes over plain anonymous HTTPS. There is no boto3, no AWS
CLI and no credential lookup, so it works the same on a laptop, in CI and in
Google Colab without signing in to anything.

Two layers:

- The generic Cloud-Optimized-GeoTIFF reader (``fetch_cog_crop``) now lives
  in ``cog.py`` and is re-exported here.
- Copernicus DEM GLO-30 and USGS 3DEP 1/3 arc-second DEM adapters (DEM sources).
- ESA WorldCover 10 m adapter (land cover, written in VoxCity's class colours).
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

from ..utils.logging import get_logger
from .cog import (
    _bbox_from_rectangle_vertices,
    _lat_tag,
    _lon_tag,
    fetch_cog_crop,
    probe_existing,
    tile_origins_for,
)
from .gsi import latlon_to_tile

__all__ = [
    "fetch_cog_crop", "chmv2_tile_urls", "fetch_chmv2_canopy_height", "CHMV2_BASE_URL",
    "worldcover_tile_urls", "fetch_esa_worldcover",
    "copernicus_tile_urls", "fetch_copernicus_dem", "usgs_3dep_10m_tile_urls", "fetch_usgs_3dep_10m",
]

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
    return probe_existing([f"{base}/chm/{qk}.tif" for qk in quadkeys], timeout)


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


# --- Copernicus DEM GLO-30 (surface model; buildings and trees included) ---

COPERNICUS_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"


def _copernicus_tile_name(lat0: int, lon0: int) -> str:
    """Tile (and folder) name for the 1-degree tile with south-west corner (lat0, lon0)."""
    return f"Copernicus_DSM_COG_10_{_lat_tag(lat0)}_00_{_lon_tag(lon0)}_00_DEM"


def copernicus_tile_urls(
    rectangle_vertices: Sequence[Tuple[float, float]],
    base_url: str = COPERNICUS_BASE_URL,
    timeout: int = 10,
) -> List[str]:
    """URLs of the Copernicus GLO-30 tiles that cover the area and exist (open ocean has none)."""
    base = base_url.rstrip("/")
    names = [_copernicus_tile_name(lat, lon) for lat, lon in tile_origins_for(rectangle_vertices, 1, "Copernicus DEM")]
    return probe_existing([f"{base}/{n}/{n}.tif" for n in names], timeout)


def fetch_copernicus_dem(
    rectangle_vertices: Sequence[Tuple[float, float]],
    output_dir: str,
    *,
    base_url: str = COPERNICUS_BASE_URL,
) -> Optional[str]:
    """
    Write ``output_dir/dem.tif`` from Copernicus DEM GLO-30 on AWS (no login).

    Missing tiles (open ocean) are zero-filled by the mosaic, i.e. sea level.
    Returns None when no tile exists or none could be read.
    """
    urls = copernicus_tile_urls(rectangle_vertices, base_url=base_url)
    if not urls:
        _logger.info("Copernicus DEM: no tiles cover this area (or the bucket is unreachable)")
        return None
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "dem.tif")
    # ~120 m of padding at 1 arc-second, matching the 100 m buffer of the
    # Earth Engine DEM path, so interpolation at the edge has neighbours.
    return fetch_cog_crop(urls, rectangle_vertices, out_path, pad_pixels=4)


# --- USGS 3DEP 1/3 arc-second seamless DEM (bare earth, US) ---

USGS_3DEP_10M_BASE_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current"


def _usgs_3dep_tile_tag(lat0: int, lon0: int) -> str:
    """USGS tiles are named by their NORTH-west corner, e.g. 'n41w074' for lat 40..41, lon -74..-73."""
    return f"{_lat_tag(lat0 + 1)}{_lon_tag(lon0)}".lower()


def usgs_3dep_10m_tile_urls(
    rectangle_vertices: Sequence[Tuple[float, float]],
    base_url: str = USGS_3DEP_10M_BASE_URL,
    timeout: int = 10,
) -> List[str]:
    """URLs of the USGS 3DEP 1/3 arc-second tiles that cover the area and exist."""
    base = base_url.rstrip("/")
    tags = [_usgs_3dep_tile_tag(lat, lon) for lat, lon in tile_origins_for(rectangle_vertices, 1, "USGS 3DEP 10m")]
    return probe_existing([f"{base}/{t}/USGS_13_{t}.tif" for t in tags], timeout)


def fetch_usgs_3dep_10m(
    rectangle_vertices: Sequence[Tuple[float, float]],
    output_dir: str,
    *,
    base_url: str = USGS_3DEP_10M_BASE_URL,
) -> Optional[str]:
    """
    Write ``output_dir/dem.tif`` from the USGS 3DEP 1/3 arc-second DEM on AWS (no login).

    Cells with the dataset's nodata (-999999) are filled from the nearest valid
    pixel. Returns None when no tile exists or nothing valid could be read.
    """
    urls = usgs_3dep_10m_tile_urls(rectangle_vertices, base_url=base_url)
    if not urls:
        _logger.info("USGS 3DEP 10m: no tiles cover this area (or the bucket is unreachable)")
        return None
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "dem.tif")
    # ~100 m of padding at 1/3 arc-second.
    return fetch_cog_crop(urls, rectangle_vertices, out_path, pad_pixels=10, fill_nodata=True)


# --- ESA WorldCover 10 m 2021 v200 ---

WORLDCOVER_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
# Class codes in the order of get_land_cover_classes('ESA WorldCover').
_WORLDCOVER_CODES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)


def _worldcover_tile_name(lat0: int, lon0: int) -> str:
    """File name of the 3-degree tile with south-west corner (lat0, lon0)."""
    return f"ESA_WorldCover_10m_2021_v200_{_lat_tag(lat0)}{_lon_tag(lon0)}_Map.tif"


def worldcover_tile_urls(
    rectangle_vertices: Sequence[Tuple[float, float]],
    base_url: str = WORLDCOVER_BASE_URL,
    timeout: int = 10,
) -> List[str]:
    """URLs of the WorldCover tiles that cover the area and exist."""
    base = base_url.rstrip("/")
    names = [_worldcover_tile_name(lat, lon) for lat, lon in tile_origins_for(rectangle_vertices, 3, "ESA WorldCover")]
    return probe_existing([f"{base}/{n}" for n in names], timeout)


def _worldcover_rgb_lut() -> np.ndarray:
    """(256, 3) uint8 lookup from class code to the RGB VoxCity's ESA class table expects.

    Uses the colours of get_land_cover_classes('ESA WorldCover') so the
    existing colour matching maps every pixel exactly. Code 0 (outside the
    tile set, i.e. ocean) and any unknown code map to Open water.
    """
    from ..utils.lc import get_land_cover_classes

    colors = list(get_land_cover_classes("ESA WorldCover").keys())
    lut = np.empty((256, 3), dtype=np.uint8)
    lut[:] = colors[_WORLDCOVER_CODES.index(80)]
    for code, rgb in zip(_WORLDCOVER_CODES, colors):
        lut[code] = rgb
    return lut


def fetch_esa_worldcover(
    rectangle_vertices: Sequence[Tuple[float, float]],
    output_dir: str,
    *,
    base_url: str = WORLDCOVER_BASE_URL,
) -> Optional[str]:
    """
    Write ``output_dir/land_cover.tif`` (3-band RGB) from ESA WorldCover on AWS (no login).

    The RGB matches the 'ESA WorldCover' class table, so the rest of the
    land-cover pipeline is unchanged. Returns None when no tile exists or
    none could be read.
    """
    urls = worldcover_tile_urls(rectangle_vertices, base_url=base_url)
    if not urls:
        _logger.info("ESA WorldCover: no tiles cover this area (or the bucket is unreachable)")
        return None
    os.makedirs(output_dir, exist_ok=True)
    codes_path = os.path.join(output_dir, "_esa_worldcover_codes.tif")
    if fetch_cog_crop(urls, rectangle_vertices, codes_path) is None:
        return None
    try:
        with rasterio.open(codes_path) as src:
            codes = src.read(1)
            crs, transform = src.crs, src.transform
    finally:
        if os.path.exists(codes_path):
            os.remove(codes_path)
    rgb = _worldcover_rgb_lut()[codes]
    out_path = os.path.join(output_dir, "land_cover.tif")
    with rasterio.open(
        out_path, "w", driver="GTiff", height=codes.shape[0], width=codes.shape[1], count=3,
        dtype="uint8", crs=crs, transform=transform, compress="deflate",
    ) as dst:
        dst.write(np.moveaxis(rgb, -1, 0))
    return out_path
