"""
Downloader for Global Building Atlas (GBA) LOD1 polygons.

This module downloads GeoParquet tiles from the Global Building Atlas (GBA)
hosted at data.source.coop, selects tiles intersecting a user-specified
rectangle, loads them into a GeoDataFrame, and filters features to the
rectangle extent.

Tile scheme:
- Global 5x5-degree tiles named like: e010_n50_e015_n45.parquet
  - longitudes: e/w with 3-digit zero padding (e.g., e010, w060)
  - latitudes: n/s with 2-digit zero padding (e.g., n50, s25)
  - filename order: west_lon, north_lat, east_lon, south_lat

Usage:
    gdf = load_gdf_from_gba(rectangle_vertices=[(lon1, lat1), (lon2, lat2), ...])

Notes:
- Output CRS is EPSG:4326.
- Requires pyarrow or fastparquet for parquet reading via GeoPandas.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Polygon


def _bbox_from_rectangle_vertices(vertices: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """
    Convert rectangle vertices in (lon, lat) into bbox as (min_lon, min_lat, max_lon, max_lat).
    """
    if not vertices:
        raise ValueError("rectangle_vertices must be a non-empty sequence of (lon, lat)")
    lons = [v[0] for v in vertices]
    lats = [v[1] for v in vertices]
    return (min(lons), min(lats), max(lons), max(lats))


def _pad_lon(deg: int) -> str:
    return f"{abs(deg):03d}"


def _pad_lat(deg: int) -> str:
    return f"{abs(deg):02d}"


def _lon_tag(deg: int) -> str:
    return ("e" if deg >= 0 else "w") + _pad_lon(deg)


def _lat_tag(deg: int) -> str:
    return ("n" if deg >= 0 else "s") + _pad_lat(deg)


def _snap_down(value: float, step: int) -> int:
    return int(math.floor(value / step) * step)


def _snap_up(value: float, step: int) -> int:
    return int(math.ceil(value / step) * step)


def _generate_tile_bounds_for_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float, tile_size_deg: int = 5
) -> Iterable[Tuple[int, int, int, int]]:
    """
    Generate 5-degree tile bounds (west, south, east, north) covering bbox.
    All values are integer degrees aligned to tile_size_deg.
    """
    west = _snap_down(min_lon, tile_size_deg)
    east = _snap_up(max_lon, tile_size_deg)
    south = _snap_down(min_lat, tile_size_deg)
    north = _snap_up(max_lat, tile_size_deg)

    for lon in range(west, east, tile_size_deg):
        for lat in range(south, north, tile_size_deg):
            yield (lon, lat, lon + tile_size_deg, lat + tile_size_deg)


def _tile_filename(west: int, south: int, east: int, north: int) -> str:
    """
    Construct GBA tile filename for given integer-degree bounds.
    Naming convention examples:
      e010_n50_e015_n45.parquet
      e140_s25_e145_s30.parquet
      w060_s30_w055_s35.parquet
    """
    return f"{_lon_tag(west)}_{_lat_tag(north)}_{_lon_tag(east)}_{_lat_tag(south)}.parquet"


def _tile_url(base_url: str, west: int, south: int, east: int, north: int) -> str:
    filename = _tile_filename(west, south, east, north)
    return f"{base_url.rstrip('/')}/{filename}"


def _download_parquet(url: str, download_dir: str, timeout: int = 60) -> Optional[str]:
    """
    Download a parquet file to download_dir. Returns local filepath or None if not found.
    """
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                return None
            filename = os.path.basename(url)
            local_path = os.path.join(download_dir, filename)
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return local_path
    except requests.RequestException:
        return None


def _filter_to_rectangle(gdf: gpd.GeoDataFrame, rectangle: Polygon, clip: bool) -> gpd.GeoDataFrame:
    gdf = gdf[gdf.geometry.notnull()].copy()
    # Ensure CRS is WGS84
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    intersects = gdf.intersects(rectangle)
    gdf = gdf[intersects].copy()
    if clip and not gdf.empty:
        # GeoPandas clip performs overlay to trim geometries to rectangle
        gdf = gpd.clip(gdf, gpd.GeoSeries([rectangle], crs="EPSG:4326").to_frame("geometry"))
    return gdf


def load_gdf_from_gba(
    rectangle_vertices: Sequence[Tuple[float, float]],
    base_url: str = "https://data.source.coop/tge-labs/globalbuildingatlas-lod1",
    download_dir: Optional[str] = None,
    clip_to_rectangle: bool = False,
) -> Optional[gpd.GeoDataFrame]:
    """
    Download GBA tiles intersecting a rectangle and return combined GeoDataFrame.

    Args:
        rectangle_vertices: Sequence of (lon, lat) defining the area of interest.
        base_url: Base URL hosting GBA parquet tiles.
        download_dir: Optional directory to store downloaded tiles. If None, a
                      temporary directory is used and cleaned up by the OS later.
        clip_to_rectangle: If True, geometries are clipped to rectangle extent.

    Returns:
        GeoDataFrame with EPSG:4326 geometry and an 'id' column, or None if no data.
    """
    min_lon, min_lat, max_lon, max_lat = _bbox_from_rectangle_vertices(rectangle_vertices)
    rectangle = Polygon([
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat),
    ])

    tmp_dir_created = False
    if download_dir is None:
        download_dir = tempfile.mkdtemp(prefix="gba_tiles_")
        tmp_dir_created = True
    else:
        os.makedirs(download_dir, exist_ok=True)

    local_files: List[str] = []
    for west, south, east, north in _generate_tile_bounds_for_bbox(min_lon, min_lat, max_lon, max_lat):
        url = _tile_url(base_url, west, south, east, north)
        local = _download_parquet(url, download_dir)
        if local is not None:
            local_files.append(local)

    if not local_files:
        return None

    gdfs: List[gpd.GeoDataFrame] = []
    for path in local_files:
        try:
            # GeoParquet read
            gdf = gpd.read_parquet(path)
            if gdf is not None and not gdf.empty:
                gdfs.append(gdf)
        except Exception:
            # Skip unreadable tiles
            continue

    if not gdfs:
        return None

    combined = pd.concat(gdfs, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry")
    combined = _filter_to_rectangle(combined, rectangle, clip=clip_to_rectangle)

    if combined.empty:
        return None

    # Ensure sequential ids
    combined["id"] = combined.index.astype(int)
    combined.set_crs(epsg=4326, inplace=True, allow_override=True)
    return combined


