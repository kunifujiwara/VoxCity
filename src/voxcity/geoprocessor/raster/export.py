import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pyproj import CRS
from ..utils import setup_transformer
from ...utils.orientation import ensure_orientation, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP


def grid_to_geodataframe(grid_ori, rectangle_vertices, meshsize):
    """
    Converts a 2D grid to a GeoDataFrame with cell polygons and values.
    Output CRS: EPSG:4326
    """
    # Grids arrive in uv_m (SOUTH_UP) after Phase 3. Convert to NORTH_UP so that
    # row 0 = north, matching the max_y-based coordinate math below.
    grid = ensure_orientation(grid_ori.copy(), ORIENTATION_SOUTH_UP, ORIENTATION_NORTH_UP)

    min_lon = min(v[0] for v in rectangle_vertices)
    max_lon = max(v[0] for v in rectangle_vertices)
    min_lat = min(v[1] for v in rectangle_vertices)
    max_lat = max(v[1] for v in rectangle_vertices)

    rows, cols = grid.shape

    transformer_to_mercator = setup_transformer("EPSG:4326", "EPSG:3857")
    transformer_to_wgs84 = setup_transformer("EPSG:3857", "EPSG:4326")

    min_x, min_y = transformer_to_mercator.transform(min_lon, min_lat)
    max_x, max_y = transformer_to_mercator.transform(max_lon, max_lat)

    cell_size_x = (max_x - min_x) / cols
    cell_size_y = (max_y - min_y) / rows

    # Vectorized: compute all cell edges at once
    j_idx = np.arange(cols)
    i_idx = np.arange(rows)
    cell_min_xs = min_x + j_idx * cell_size_x
    cell_max_xs = min_x + (j_idx + 1) * cell_size_x
    cell_max_ys = max_y - i_idx * cell_size_y
    cell_min_ys = max_y - (i_idx + 1) * cell_size_y

    # Meshgrid for all cells (row-major: i varies slowest)
    jj, ii = np.meshgrid(j_idx, i_idx)
    min_xs_flat = cell_min_xs[jj.ravel()]
    max_xs_flat = cell_max_xs[jj.ravel()]
    min_ys_flat = cell_min_ys[ii.ravel()]
    max_ys_flat = cell_max_ys[ii.ravel()]

    # Batch transform all corners to WGS84
    min_lons, min_lats = transformer_to_wgs84.transform(min_xs_flat, min_ys_flat)
    max_lons, max_lats = transformer_to_wgs84.transform(max_xs_flat, max_ys_flat)

    polygons = [box(lo1, la1, lo2, la2) for lo1, la1, lo2, la2 in zip(min_lons, min_lats, max_lons, max_lats)]
    values = grid.ravel().tolist()

    gdf = gpd.GeoDataFrame({'geometry': polygons, 'value': values}, crs=CRS.from_epsg(4326))
    return gdf


def grid_to_point_geodataframe(grid_ori, rectangle_vertices, meshsize):
    """
    Converts a 2D grid to a GeoDataFrame with point geometries at cell centers and values.
    Output CRS: EPSG:4326
    """
    from shapely.geometry import Point

    # Grids arrive in uv_m (SOUTH_UP) after Phase 3. Convert to NORTH_UP so that
    # row 0 = north, matching the max_y-based coordinate math below.
    grid = ensure_orientation(grid_ori.copy(), ORIENTATION_SOUTH_UP, ORIENTATION_NORTH_UP)

    min_lon = min(v[0] for v in rectangle_vertices)
    max_lon = max(v[0] for v in rectangle_vertices)
    min_lat = min(v[1] for v in rectangle_vertices)
    max_lat = max(v[1] for v in rectangle_vertices)

    rows, cols = grid.shape

    transformer_to_mercator = setup_transformer("EPSG:4326", "EPSG:3857")
    transformer_to_wgs84 = setup_transformer("EPSG:3857", "EPSG:4326")

    min_x, min_y = transformer_to_mercator.transform(min_lon, min_lat)
    max_x, max_y = transformer_to_mercator.transform(max_lon, max_lat)

    cell_size_x = (max_x - min_x) / cols
    cell_size_y = (max_y - min_y) / rows

    # Vectorized: compute all cell centers at once
    j_idx = np.arange(cols)
    i_idx = np.arange(rows)
    center_xs = min_x + (j_idx + 0.5) * cell_size_x
    center_ys = max_y - (i_idx + 0.5) * cell_size_y

    jj, ii = np.meshgrid(j_idx, i_idx)
    cx_flat = center_xs[jj.ravel()]
    cy_flat = center_ys[ii.ravel()]

    # Batch transform to WGS84
    lons, lats = transformer_to_wgs84.transform(cx_flat, cy_flat)

    points = [Point(lo, la) for lo, la in zip(lons, lats)]
    values = grid.ravel().tolist()

    gdf = gpd.GeoDataFrame({'geometry': points, 'value': values}, crs=CRS.from_epsg(4326))
    return gdf


