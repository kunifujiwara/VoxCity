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


def parse_dem_tile_text(text, nodata=GSI_NODATA, size=GSI_TILE_SIZE):
    """Parse a GSI DEM ``.txt`` tile (CSV of meters, ``e`` = no-data).

    Returns a ``(size, size)`` float32 array. Ragged/short rows and missing
    rows are filled with ``nodata``.
    """
    arr = np.full((size, size), nodata, dtype=np.float32)
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    for i, line in enumerate(lines[:size]):
        cells = line.split(",")
        for j, cell in enumerate(cells[:size]):
            c = cell.strip()
            if c == "" or c == "e":
                continue
            try:
                arr[i, j] = float(c)
            except ValueError:
                continue  # malformed token (e.g. truncated mid-stream) -> nodata
    return arr


_GSI_XYZ_URL = "https://cyberjapandata.gsi.go.jp/xyz/{dem_type}/{zoom}/{x}/{y}.txt"


def check_dem_availability(lat, lon, *, timeout_s=5, sleep=0.2):
    """Probe the center point and return (dem_type, zoom) for the finest
    available GSI DEM product. Falls back to ('dem10b', 14)."""
    for item in GSI_DEM_TYPES:
        x, y = latlon_to_tile(lat, lon, item["zoom"])
        url = _GSI_XYZ_URL.format(dem_type=item["type"], zoom=item["zoom"], x=x, y=y)
        try:
            if sleep:
                time.sleep(sleep)
            resp = requests.get(url, timeout=timeout_s)
            if resp.status_code == 200:
                return item["type"], item["zoom"]
        except requests.exceptions.RequestException:
            continue
    return "dem10b", 14


def download_dem_tiles(tile_range, dem_type, zoom, *, nodata=GSI_NODATA,
                       sleep=0.4, timeout_s=10):
    """Download every tile in ``tile_range`` (x_min, y_min, x_max, y_max).

    Returns ``{(x, y): (256, 256) float32}``. Missing/failed tiles become
    nodata blocks. Raises ValueError if no tile was retrieved.
    """
    x_min, y_min, x_max, y_max = tile_range
    tiles = {}
    any_ok = False
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            url = _GSI_XYZ_URL.format(dem_type=dem_type, zoom=zoom, x=x, y=y)
            block = np.full((GSI_TILE_SIZE, GSI_TILE_SIZE), nodata, dtype=np.float32)
            try:
                if sleep:
                    time.sleep(sleep)
                resp = requests.get(url, timeout=timeout_s)
                if resp.status_code == 200:
                    block = parse_dem_tile_text(resp.text, nodata=nodata)
                    any_ok = True
            except requests.exceptions.RequestException:
                pass
            tiles[(x, y)] = block
    if not any_ok:
        raise ValueError(
            "No GSI DEM tiles available for the requested area "
            "(is it outside Japan coverage?)."
        )
    return tiles


def compose_dem_array(tiles, tile_range, nodata=GSI_NODATA):
    """Assemble per-tile 256x256 blocks into one float32 mosaic."""
    x_min, y_min, x_max, y_max = tile_range
    cols = (x_max - x_min + 1) * GSI_TILE_SIZE
    rows = (y_max - y_min + 1) * GSI_TILE_SIZE
    mosaic = np.full((rows, cols), nodata, dtype=np.float32)
    for (x, y), block in tiles.items():
        r0 = (y - y_min) * GSI_TILE_SIZE
        c0 = (x - x_min) * GSI_TILE_SIZE
        mosaic[r0:r0 + GSI_TILE_SIZE, c0:c0 + GSI_TILE_SIZE] = block
    return mosaic


def save_dem_as_geotiff(array, tile_range, zoom, filepath, nodata=GSI_NODATA):
    """Write the mosaic as a single-band float32 GeoTIFF in EPSG:3857."""
    x_min, y_min, x_max, y_max = tile_range
    origin_minx, _, _, origin_maxy = tile_bounds_mercator(x_min, y_min, zoom)
    pixel = (2 * _MERC_MAX) / (2.0 ** zoom) / GSI_TILE_SIZE

    out_dir = os.path.dirname(filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    height, width = array.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filepath, width, height, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform((origin_minx, pixel, 0, origin_maxy, 0, -pixel))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset.SetProjection(srs.ExportToWkt())

    band = dataset.GetRasterBand(1)
    band.WriteArray(np.asarray(array, dtype=np.float32))
    band.SetNoDataValue(float(nodata))
    dataset = None
    return filepath


def save_gsi_dem_as_geotiff(rectangle_vertices, filepath, dem_type=None,
                            nodata=GSI_NODATA, sleep=0.4, timeout_s=10):
    """Download GSI bare-earth DEM for an ROI and save an EPSG:3857 GeoTIFF.

    Args:
        rectangle_vertices: list of (lon, lat) tuples defining the ROI.
        filepath: output GeoTIFF path.
        dem_type: None to auto-detect the finest available product, or one of
                  'dem5a' / 'dem5b' / 'dem10b' to force it.
        nodata: no-data fill value.
        sleep: seconds between requests (politeness; set 0 in tests).
        timeout_s: per-request timeout.

    Returns:
        The written filepath.

    Note:
        Succeeds even if some tiles in the ROI failed to download (e.g. near
        coverage edges); inspect the returned GeoTIFF for unexpected nodata
        regions if the ROI may span a coverage boundary.

    Raises:
        ValueError: if rectangle_vertices is empty, dem_type is invalid, or
            no tiles cover the area.
    """
    bbox = _bbox_from_rectangle_vertices(rectangle_vertices)
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = (min_lat + max_lat) / 2.0
    mid_lon = (min_lon + max_lon) / 2.0

    if dem_type is None:
        dem_type, zoom = check_dem_availability(
            mid_lat, mid_lon, timeout_s=timeout_s, sleep=sleep
        )
    else:
        if dem_type not in _ZOOM_BY_TYPE:
            raise ValueError(
                f"Unknown dem_type {dem_type!r}; expected one of "
                f"{sorted(_ZOOM_BY_TYPE)}"
            )
        zoom = _ZOOM_BY_TYPE[dem_type]

    tile_range = tile_range_for_bbox(bbox, zoom)
    tiles = download_dem_tiles(
        tile_range, dem_type, zoom, nodata=nodata, sleep=sleep, timeout_s=timeout_s
    )
    mosaic = compose_dem_array(tiles, tile_range, nodata=nodata)
    save_dem_as_geotiff(mosaic, tile_range, zoom, filepath, nodata=nodata)
    return filepath
