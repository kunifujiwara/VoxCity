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


def _fetch_tile(dem_type, zoom, x, y, *, nodata=GSI_NODATA, sleep=0.4,
                timeout_s=10):
    """Fetch and parse a single GSI DEM tile.

    Returns ``(block, ok)`` where ``block`` is a freshly-allocated
    ``(256, 256)`` float32 array (all-``nodata`` on miss) and ``ok`` is True
    only on a successful HTTP 200 response.
    """
    url = _GSI_XYZ_URL.format(dem_type=dem_type, zoom=zoom, x=x, y=y)
    try:
        if sleep:
            time.sleep(sleep)
        resp = requests.get(url, timeout=timeout_s)
        if resp.status_code == 200:
            return parse_dem_tile_text(resp.text, nodata=nodata), True
    except requests.exceptions.RequestException:
        pass
    return np.full((GSI_TILE_SIZE, GSI_TILE_SIZE), nodata, dtype=np.float32), False


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
            block, ok = _fetch_tile(dem_type, zoom, x, y, nodata=nodata,
                                    sleep=sleep, timeout_s=timeout_s)
            any_ok = any_ok or ok
            tiles[(x, y)] = block
    if not any_ok:
        raise ValueError(
            "No GSI DEM tiles available for the requested area "
            "(is it outside Japan coverage?)."
        )
    return tiles


def _download_tiles_safe(tile_range, dem_type, zoom, *, nodata=GSI_NODATA,
                         sleep=0.4, timeout_s=10):
    """Like :func:`download_dem_tiles` but never raises on all-missing.

    Used for fallback layers, where an absent product is expected and must not
    abort the merge. Missing/failed tiles are returned as nodata blocks.
    """
    x_min, y_min, x_max, y_max = tile_range
    tiles = {}
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            block, _ = _fetch_tile(dem_type, zoom, x, y, nodata=nodata,
                                   sleep=sleep, timeout_s=timeout_s)
            tiles[(x, y)] = block
    return tiles


def _download_fine_merged(tile_range, *, nodata=GSI_NODATA, sleep=0.4,
                          timeout_s=10):
    """Build a z15 mosaic source by overlaying dem5a on dem5b at pixel level.

    For each tile in ``tile_range`` dem5a (5 m laser) is fetched first; dem5b
    (5 m photogrammetry) is requested *only* when the dem5a tile has missing
    pixels, and is used to fill those pixels. A tile fully covered by dem5a
    therefore incurs no extra request.

    Returns ``(tiles, any_ok)`` where ``tiles`` is ``{(x, y): (256, 256)
    float32}`` and ``any_ok`` is True if any 5 m tile was retrieved.
    """
    x_min, y_min, x_max, y_max = tile_range
    tiles = {}
    any_ok = False
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            block, ok = _fetch_tile("dem5a", 15, x, y, nodata=nodata,
                                    sleep=sleep, timeout_s=timeout_s)
            any_ok = any_ok or ok
            holes = block == nodata
            if holes.any():
                block5b, ok5b = _fetch_tile("dem5b", 15, x, y, nodata=nodata,
                                            sleep=sleep, timeout_s=timeout_s)
                any_ok = any_ok or ok5b
                block[holes] = block5b[holes]
            tiles[(x, y)] = block
    return tiles, any_ok


def _backfill_from_coarser(mosaic, fine_range, coarse_mosaic, coarse_range,
                           fine_zoom, coarse_zoom, nodata=GSI_NODATA):
    """Fill remaining no-data pixels of a fine mosaic from a coarser mosaic.

    Both grids subdivide the same global EPSG:3857 extent from the same origin
    and their pixel sizes differ by an exact power of two, so a fine pixel's
    global index maps to its covering coarse pixel by integer division - no
    interpolation or resampling artifacts. ``mosaic`` is modified in place and
    returned.
    """
    holes = mosaic == nodata
    if not holes.any():
        return mosaic

    step = 2 ** (fine_zoom - coarse_zoom)  # fine pixels per coarse pixel (axis)
    g_col0_f = fine_range[0] * GSI_TILE_SIZE
    g_row0_f = fine_range[1] * GSI_TILE_SIZE
    g_col0_c = coarse_range[0] * GSI_TILE_SIZE
    g_row0_c = coarse_range[1] * GSI_TILE_SIZE
    rows_c, cols_c = coarse_mosaic.shape

    rr, cc = np.nonzero(holes)
    lc = (g_col0_f + cc) // step - g_col0_c
    lr = (g_row0_f + rr) // step - g_row0_c
    in_bounds = (lc >= 0) & (lc < cols_c) & (lr >= 0) & (lr < rows_c)
    rr, cc, lr, lc = rr[in_bounds], cc[in_bounds], lr[in_bounds], lc[in_bounds]
    vals = coarse_mosaic[lr, lc]
    good = vals != nodata
    mosaic[rr[good], cc[good]] = vals[good]
    return mosaic


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
                            nodata=GSI_NODATA, sleep=0.4, timeout_s=10,
                            include_dem10b_fallback=True):
    """Download GSI bare-earth DEM for an ROI and save an EPSG:3857 GeoTIFF.

    Args:
        rectangle_vertices: list of (lon, lat) tuples defining the ROI.
        filepath: output GeoTIFF path.
        dem_type: None (default) to build a seamless mosaic by overlaying the
                  finest products available per pixel (see below), or one of
                  'dem5a' / 'dem5b' / 'dem10b' to force a single product with
                  no merging.
        nodata: no-data fill value.
        sleep: seconds between requests (politeness; set 0 in tests).
        timeout_s: per-request timeout.
        include_dem10b_fallback: when auto-merging, backfill pixels covered by
                  neither 5 m product with dem10b (10 m). Set False to leave
                  such pixels as no-data.

    Returns:
        The written filepath.

    Note:
        With ``dem_type=None`` the ROI is composed at z15 from dem5a, with
        dem5b filling any dem5a no-data pixels, and (optionally) dem10b filling
        whatever remains. This produces a seamless terrain even when the ROI
        straddles a dem5a coverage boundary, instead of leaving no-data holes.
        dem5b is fetched only for tiles where dem5a is incomplete, and dem10b
        only when 5 m no-data pixels remain, so fully dem5a-covered ROIs incur
        no extra requests. dem10b (z14) is half the resolution of the 5 m
        products; because both grids share the global mercator origin and
        differ by an exact power of two, its pixels are mapped without
        resampling artifacts.

    Raises:
        ValueError: if rectangle_vertices is empty, dem_type is invalid, or
            no tiles cover the area.
    """
    bbox = _bbox_from_rectangle_vertices(rectangle_vertices)

    # Forced single product: download exactly that product, no merging.
    if dem_type is not None:
        if dem_type not in _ZOOM_BY_TYPE:
            raise ValueError(
                f"Unknown dem_type {dem_type!r}; expected one of "
                f"{sorted(_ZOOM_BY_TYPE)}"
            )
        zoom = _ZOOM_BY_TYPE[dem_type]
        tile_range = tile_range_for_bbox(bbox, zoom)
        tiles = download_dem_tiles(
            tile_range, dem_type, zoom, nodata=nodata, sleep=sleep,
            timeout_s=timeout_s
        )
        mosaic = compose_dem_array(tiles, tile_range, nodata=nodata)
        save_dem_as_geotiff(mosaic, tile_range, zoom, filepath, nodata=nodata)
        return filepath

    # Auto: pixel-level overlay of dem5a on dem5b at z15, with optional dem10b
    # (z14) backfill for pixels neither 5 m product covers.
    fine_range = tile_range_for_bbox(bbox, 15)
    tiles, any_ok = _download_fine_merged(
        fine_range, nodata=nodata, sleep=sleep, timeout_s=timeout_s
    )
    mosaic = compose_dem_array(tiles, fine_range, nodata=nodata)

    if include_dem10b_fallback and (mosaic == nodata).any():
        coarse_range = tile_range_for_bbox(bbox, 14)
        coarse_tiles = _download_tiles_safe(
            coarse_range, "dem10b", 14, nodata=nodata, sleep=sleep,
            timeout_s=timeout_s
        )
        coarse_ok = any(
            bool((block != nodata).any()) for block in coarse_tiles.values()
        )
        if coarse_ok:
            coarse_mosaic = compose_dem_array(
                coarse_tiles, coarse_range, nodata=nodata
            )
            mosaic = _backfill_from_coarser(
                mosaic, fine_range, coarse_mosaic, coarse_range, 15, 14,
                nodata=nodata
            )
            any_ok = True

    if not any_ok:
        raise ValueError(
            "No GSI DEM tiles available for the requested area "
            "(is it outside Japan coverage?)."
        )

    save_dem_as_geotiff(mosaic, fine_range, 15, filepath, nodata=nodata)
    return filepath
