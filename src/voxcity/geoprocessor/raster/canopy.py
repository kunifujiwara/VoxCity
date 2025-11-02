import warnings
from typing import List, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
from pyproj import CRS, Transformer, Geod

from ..utils import initialize_geod, calculate_distance, normalize_to_one_meter
from .core import calculate_grid_size


def create_vegetation_height_grid_from_gdf_polygon(veg_gdf, mesh_size, polygon):
    """
    Create a vegetation height grid from a GeoDataFrame of vegetation polygons/objects.
    Cells with vegetation take the max height of intersecting features.
    Returns north-up grid (row 0 = north).
    """
    if veg_gdf.crs is None:
        warnings.warn("veg_gdf has no CRS. Assuming EPSG:4326. ")
        veg_gdf = veg_gdf.set_crs(epsg=4326)
    else:
        if veg_gdf.crs.to_epsg() != 4326:
            veg_gdf = veg_gdf.to_crs(epsg=4326)

    if 'height' not in veg_gdf.columns:
        raise ValueError("Vegetation GeoDataFrame must have a 'height' column.")

    if isinstance(polygon, list):
        poly = Polygon(polygon)
    elif isinstance(polygon, Polygon):
        poly = polygon
    else:
        raise ValueError("polygon must be a list of (lon, lat) or a shapely Polygon.")

    left, bottom, right, top = poly.bounds
    geod = Geod(ellps="WGS84")
    _, _, width_m = geod.inv(left, bottom, right, bottom)
    _, _, height_m = geod.inv(left, bottom, left, top)

    num_cells_x = int(width_m / mesh_size + 0.5)
    num_cells_y = int(height_m / mesh_size + 0.5)

    if num_cells_x < 1 or num_cells_y < 1:
        warnings.warn("Polygon bounding box is smaller than mesh_size; returning empty array.")
        return np.array([])

    xs = np.linspace(left, right, num_cells_x)
    ys = np.linspace(top, bottom, num_cells_y)
    X, Y = np.meshgrid(xs, ys)
    xs_flat = X.ravel()
    ys_flat = Y.ravel()

    grid_points = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in zip(xs_flat, ys_flat)],
        crs="EPSG:4326"
    )

    joined = gpd.sjoin(
        grid_points,
        veg_gdf[['height', 'geometry']],
        how='left',
        predicate='intersects'
    )

    joined_agg = (
        joined.groupby(joined.index).agg({'height': 'max'})
    )

    veg_grid = np.zeros((num_cells_y, num_cells_x), dtype=float)
    for i, row_data in joined_agg.iterrows():
        if not np.isnan(row_data['height']):
            row_idx = i // num_cells_x
            col_idx = i % num_cells_x
            veg_grid[row_idx, col_idx] = row_data['height']

    return np.flipud(veg_grid)


def create_dem_grid_from_gdf_polygon(terrain_gdf, mesh_size, polygon):
    """
    Create a height grid from a terrain GeoDataFrame using nearest-neighbor sampling.
    Returns north-up grid.
    """
    if terrain_gdf.crs is None:
        warnings.warn("terrain_gdf has no CRS. Assuming EPSG:4326. ")
        terrain_gdf = terrain_gdf.set_crs(epsg=4326)
    else:
        if terrain_gdf.crs.to_epsg() != 4326:
            terrain_gdf = terrain_gdf.to_crs(epsg=4326)

    if isinstance(polygon, list):
        poly = Polygon(polygon)
    elif isinstance(polygon, Polygon):
        poly = polygon
    else:
        raise ValueError("`polygon` must be a list of (lon, lat) or a shapely Polygon.")

    left, bottom, right, top = poly.bounds
    geod = Geod(ellps="WGS84")
    _, _, width_m = geod.inv(left, bottom, right, bottom)
    _, _, height_m = geod.inv(left, bottom, left, top)
    num_cells_x = int(width_m / mesh_size + 0.5)
    num_cells_y = int(height_m / mesh_size + 0.5)
    if num_cells_x < 1 or num_cells_y < 1:
        warnings.warn("Polygon bounding box is smaller than mesh_size; returning empty array.")
        return np.array([])

    xs = np.linspace(left, right, num_cells_x)
    ys = np.linspace(top, bottom, num_cells_y)
    X, Y = np.meshgrid(xs, ys)
    xs_flat = X.ravel()
    ys_flat = Y.ravel()

    grid_points = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in zip(xs_flat, ys_flat)],
        crs="EPSG:4326"
    )

    if 'elevation' not in terrain_gdf.columns:
        raise ValueError("terrain_gdf must have an 'elevation' column.")

    try:
        centroid = poly.centroid
        lon_c, lat_c = float(centroid.x), float(centroid.y)
        zone = int((lon_c + 180.0) // 6) + 1
        epsg_proj = 32600 + zone if lat_c >= 0 else 32700 + zone
        terrain_proj = terrain_gdf.to_crs(epsg=epsg_proj)
        grid_points_proj = grid_points.to_crs(epsg=epsg_proj)

        grid_points_elev = gpd.sjoin_nearest(
            grid_points_proj,
            terrain_proj[['elevation', 'geometry']],
            how="left",
            distance_col="dist_to_terrain"
        )
        grid_points_elev.index = grid_points.index
    except Exception:
        grid_points_elev = gpd.sjoin_nearest(
            grid_points,
            terrain_gdf[['elevation', 'geometry']],
            how="left",
            distance_col="dist_to_terrain"
        )

    dem_grid = np.full((num_cells_y, num_cells_x), np.nan, dtype=float)
    for i, elevation_val in zip(grid_points_elev.index, grid_points_elev['elevation']):
        row = i // num_cells_x
        col = i % num_cells_x
        dem_grid[row, col] = elevation_val
    return np.flipud(dem_grid)


def create_canopy_grids_from_tree_gdf(tree_gdf, meshsize, rectangle_vertices):
    """
    Create canopy top and bottom height grids from a tree GeoDataFrame.
    Returns (canopy_height_grid, canopy_bottom_height_grid).
    """
    if tree_gdf is None or len(tree_gdf) == 0:
        return np.array([]), np.array([])

    required_cols = ['top_height', 'bottom_height', 'crown_diameter', 'geometry']
    for col in required_cols:
        if col not in tree_gdf.columns:
            raise ValueError(f"tree_gdf must contain '{col}' column.")

    if tree_gdf.crs is None:
        warnings.warn("tree_gdf has no CRS. Assuming EPSG:4326.")
        tree_gdf = tree_gdf.set_crs(epsg=4326)
    elif tree_gdf.crs.to_epsg() != 4326:
        tree_gdf = tree_gdf.to_crs(epsg=4326)

    geod = initialize_geod()
    vertex_0, vertex_1, vertex_3 = rectangle_vertices[0], rectangle_vertices[1], rectangle_vertices[3]

    dist_side_1 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_1[0], vertex_1[1])
    dist_side_2 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_3[0], vertex_3[1])

    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)
    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)

    origin = np.array(rectangle_vertices[0])
    grid_size, adjusted_meshsize = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)
    nx, ny = grid_size[0], grid_size[1]

    i_centers_m = (np.arange(nx) + 0.5) * adjusted_meshsize[0]
    j_centers_m = (np.arange(ny) + 0.5) * adjusted_meshsize[1]

    canopy_top = np.zeros((nx, ny), dtype=float)
    canopy_bottom = np.zeros((nx, ny), dtype=float)

    transform_mat = np.column_stack((u_vec, v_vec))
    try:
        transform_inv = np.linalg.inv(transform_mat)
    except np.linalg.LinAlgError:
        transform_inv = np.linalg.pinv(transform_mat)

    for _, row in tree_gdf.iterrows():
        geom = row['geometry']
        if geom is None or not hasattr(geom, 'x'):
            continue
        top_h = float(row.get('top_height', 0.0) or 0.0)
        bot_h = float(row.get('bottom_height', 0.0) or 0.0)
        dia = float(row.get('crown_diameter', 0.0) or 0.0)
        if dia <= 0 or top_h <= 0:
            continue
        if bot_h < 0:
            bot_h = 0.0
        if bot_h > top_h:
            top_h, bot_h = bot_h, top_h
        R = dia / 2.0
        a = max((top_h - bot_h) / 2.0, 0.0)
        z0 = (top_h + bot_h) / 2.0
        tree_lon = float(geom.x)
        tree_lat = float(geom.y)
        delta = np.array([tree_lon, tree_lat]) - origin
        alpha_beta = transform_inv @ delta
        alpha_m = alpha_beta[0]
        beta_m = alpha_beta[1]
        du_cells = int(R / adjusted_meshsize[0] + 2)
        dv_cells = int(R / adjusted_meshsize[1] + 2)
        i_center_idx = int(alpha_m / adjusted_meshsize[0])
        j_center_idx = int(beta_m / adjusted_meshsize[1])
        i_min = max(0, i_center_idx - du_cells)
        i_max = min(nx - 1, i_center_idx + du_cells)
        j_min = max(0, j_center_idx - dv_cells)
        j_max = min(ny - 1, j_center_idx + dv_cells)
        if i_min > i_max or j_min > j_max:
            continue
        ic = i_centers_m[i_min:i_max + 1][:, None]
        jc = j_centers_m[j_min:j_max + 1][None, :]
        di = ic - alpha_m
        dj = jc - beta_m
        r = np.sqrt(di * di + dj * dj)
        within = r <= R
        if not np.any(within):
            continue
        ratio = np.clip(r / max(R, 1e-9), 0.0, 1.0)
        factor = np.sqrt(1.0 - ratio * ratio)
        local_top = z0 + a * factor
        local_bot = z0 - a * factor
        local_top_masked = np.where(within, local_top, 0.0)
        local_bot_masked = np.where(within, local_bot, 0.0)
        canopy_top[i_min:i_max + 1, j_min:j_max + 1] = np.maximum(
            canopy_top[i_min:i_max + 1, j_min:j_max + 1], local_top_masked
        )
        canopy_bottom[i_min:i_max + 1, j_min:j_max + 1] = np.maximum(
            canopy_bottom[i_min:i_max + 1, j_min:j_max + 1], local_bot_masked
        )

    canopy_bottom = np.minimum(canopy_bottom, canopy_top)
    return canopy_top, canopy_bottom




