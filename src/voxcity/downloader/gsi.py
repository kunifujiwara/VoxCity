"""
Module for downloading bare-earth DEM (Digital Terrain Model) tiles from the
Geospatial Information Authority of Japan (GSI / 国土地理院) and converting them
into a VoxCity-compatible georeferenced GeoTIFF.

GSI publishes elevation tiles on the standard XYZ Web-Mercator tile grid at
``https://cyberjapandata.gsi.go.jp/xyz/{type}/{z}/{x}/{y}.txt``. Each tile is a
256x256 grid of elevation values (meters) as CSV text, with ``e`` marking
no-data. Resolutions, finest first: dem5a (5 m laser, z15), dem5b (5 m photo,
z15), dem10b (10 m nationwide, z14).

The mosaic is written natively in EPSG:3857 (no resampling); the downstream
``create_dem_grid_from_geotiff_polygon`` reprojects from the file's own CRS.

Example:
    >>> verts = [(140.09, 36.21), (140.12, 36.21), (140.12, 36.24), (140.09, 36.24)]
    >>> save_gsi_dem_as_geotiff(verts, "dem.tif")
"""

import os
import math
import time

import numpy as np
import requests
from osgeo import gdal, osr

__all__ = ["save_gsi_dem_as_geotiff"]

# Half the EPSG:3857 (Web Mercator) world extent, in meters.
_MERC_MAX = 20037508.342789244

GSI_TILE_SIZE = 256
GSI_NODATA = -9999.0

# DEM product types in priority order (finest resolution first).
GSI_DEM_TYPES = [
    {"type": "dem5a", "zoom": 15},   # 5 m mesh, airborne laser survey
    {"type": "dem5b", "zoom": 15},   # 5 m mesh, photogrammetry
    {"type": "dem10b", "zoom": 14},  # 10 m mesh, nationwide
]
_ZOOM_BY_TYPE = {item["type"]: item["zoom"] for item in GSI_DEM_TYPES}


def latlon_to_tile(lat, lon, zoom):
    """Convert lat/lon to integer XYZ Web-Mercator tile indices (x, y)."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi)
        / 2.0
        * n
    )
    return xtile, ytile


def tile_bounds_mercator(x, y, zoom):
    """Return (minx, miny, maxx, maxy) of a tile in EPSG:3857 meters."""
    world = 2 * _MERC_MAX
    tile = world / (2.0 ** zoom)
    minx = -_MERC_MAX + x * tile
    maxx = minx + tile
    maxy = _MERC_MAX - y * tile
    miny = maxy - tile
    return (minx, miny, maxx, maxy)


def _bbox_from_rectangle_vertices(rectangle_vertices):
    """Return (min_lon, min_lat, max_lon, max_lat) from (lon, lat) vertices."""
    if not rectangle_vertices:
        raise ValueError("rectangle_vertices is empty")
    lons = [p[0] for p in rectangle_vertices]
    lats = [p[1] for p in rectangle_vertices]
    return (min(lons), min(lats), max(lons), max(lats))


def tile_range_for_bbox(bbox, zoom):
    """Return (x_min, y_min, x_max, y_max) tile indices covering bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    x0, y0 = latlon_to_tile(max_lat, min_lon, zoom)  # top-left
    x1, y1 = latlon_to_tile(min_lat, max_lon, zoom)  # bottom-right
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
