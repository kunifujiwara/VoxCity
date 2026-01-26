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


def create_canopy_grids_from_tree_gdf(tree_gdf, meshsize, rectangle_vertices):
    """
    Create canopy top and bottom height grids from a tree GeoDataFrame.
    
    Supports both Point geometries (individual trees with ellipsoid crowns) and
    Polygon geometries (forest/wood areas with flat canopy).
    
    Args:
        tree_gdf: GeoDataFrame with columns:
            - geometry: Point or Polygon/MultiPolygon
            - top_height: Height to canopy top (meters)
            - bottom_height: Height to canopy bottom (meters)
            - crown_diameter: Crown diameter (meters, used for Point geometries)
            - geometry_type (optional): 'point' or 'polygon' to distinguish geometry types
        meshsize: Grid cell size in meters
        rectangle_vertices: List of (lon, lat) tuples defining the area
        
    Returns:
        tuple: (canopy_height_grid, canopy_bottom_height_grid)
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

    # Separate point trees and polygon trees
    has_geometry_type = 'geometry_type' in tree_gdf.columns
    
    # Process polygon geometries first (they form the base layer)
    if has_geometry_type:
        polygon_gdf = tree_gdf[tree_gdf['geometry_type'] == 'polygon']
    else:
        # Detect polygon geometries by geometry type
        polygon_mask = tree_gdf.geometry.apply(
            lambda g: g is not None and g.geom_type in ['Polygon', 'MultiPolygon']
        )
        polygon_gdf = tree_gdf[polygon_mask]
    
    if len(polygon_gdf) > 0:
        # Rasterize polygon geometries
        # Note: We need to match the coordinate system used by individual trees
        # The grid uses origin = rectangle_vertices[0] (typically SW corner)
        # with i increasing along side_1 and j increasing along side_2
        from rasterio import features
        from affine import Affine
        
        # Get bounding box
        min_lon = min(coord[0] for coord in rectangle_vertices)
        max_lon = max(coord[0] for coord in rectangle_vertices)
        min_lat = min(coord[1] for coord in rectangle_vertices)
        max_lat = max(coord[1] for coord in rectangle_vertices)
        
        # Create affine transform (top-left origin for rasterio convention)
        # Rasterio produces a north-up grid (row 0 = north/max_lat)
        pixel_width = (max_lon - min_lon) / ny  # ny = columns (x direction)
        pixel_height = (max_lat - min_lat) / nx  # nx = rows (y direction)
        raster_transform = Affine(pixel_width, 0, min_lon, 0, -pixel_height, max_lat)
        
        # OPTIMIZATION: Group polygons by height to batch rasterize
        # This reduces the number of rasterization calls significantly
        height_groups = {}
        for _, row in polygon_gdf.iterrows():
            geom = row['geometry']
            if geom is None or geom.is_empty:
                continue
            
            top_h = float(row.get('top_height', 0.0) or 0.0)
            bot_h = float(row.get('bottom_height', 0.0) or 0.0)
            
            if top_h <= 0:
                continue
            if bot_h < 0:
                bot_h = 0.0
            if bot_h > top_h:
                top_h, bot_h = bot_h, top_h
            
            height_key = (top_h, bot_h)
            if height_key not in height_groups:
                height_groups[height_key] = []
            
            # Collect geometries for this height group
            if geom.geom_type == 'Polygon':
                height_groups[height_key].append(geom)
            else:  # MultiPolygon
                height_groups[height_key].extend(geom.geoms)
        
        # Batch rasterize each height group
        for (top_h, bot_h), geometries in height_groups.items():
            if not geometries:
                continue
            
            try:
                # Create shapes list with value 1 for all geometries in this group
                shapes = [(geom, 1) for geom in geometries]
                
                mask = np.zeros((nx, ny), dtype=np.uint8)
                features.rasterize(
                    shapes=shapes,
                    out=mask,
                    transform=raster_transform,
                    all_touched=False
                )
                
                # CRITICAL: Flip the mask vertically to match the grid coordinate system
                # Rasterio produces north-up (row 0 = north), but the grid uses
                # origin at rectangle_vertices[0] (typically SW, so row 0 = south)
                mask = np.flipud(mask)
                
                # Apply heights where mask is set (using maximum to preserve higher trees)
                polygon_cells = mask == 1
                canopy_top = np.where(
                    polygon_cells & (top_h > canopy_top),
                    top_h,
                    canopy_top
                )
                canopy_bottom = np.where(
                    polygon_cells & (canopy_top > 0) & ((canopy_bottom == 0) | (bot_h > canopy_bottom)),
                    bot_h,
                    canopy_bottom
                )
            except Exception:
                # Skip this height group if rasterization fails
                continue

    # Process point geometries (individual trees with ellipsoid crowns)
    if has_geometry_type:
        point_gdf = tree_gdf[tree_gdf['geometry_type'] == 'point']
    else:
        point_mask = tree_gdf.geometry.apply(
            lambda g: g is not None and g.geom_type == 'Point'
        )
        point_gdf = tree_gdf[point_mask]

    if len(point_gdf) > 0:
        # OPTIMIZATION: Pre-extract all data as numpy arrays to avoid repeated attribute access
        # Filter valid trees first
        valid_mask = (
            point_gdf.geometry.notna() & 
            (point_gdf['crown_diameter'] > 0) & 
            (point_gdf['top_height'] > 0)
        )
        valid_gdf = point_gdf[valid_mask]
        
        if len(valid_gdf) > 0:
            # Extract coordinates and attributes as arrays
            coords = np.array([(g.x, g.y) for g in valid_gdf.geometry])
            top_heights = valid_gdf['top_height'].fillna(0).values.astype(float)
            bot_heights = valid_gdf['bottom_height'].fillna(0).values.astype(float)
            diameters = valid_gdf['crown_diameter'].fillna(0).values.astype(float)
            
            # Fix bottom heights
            bot_heights = np.clip(bot_heights, 0, None)
            swap_mask = bot_heights > top_heights
            top_heights[swap_mask], bot_heights[swap_mask] = bot_heights[swap_mask], top_heights[swap_mask]
            
            # Vectorized coordinate transformation for all trees at once
            deltas = coords - origin  # Shape: (n_trees, 2)
            alpha_beta = (transform_inv @ deltas.T).T  # Shape: (n_trees, 2)
            alpha_m_all = alpha_beta[:, 0]
            beta_m_all = alpha_beta[:, 1]
            
            # Process each tree (loop still needed due to variable bounding box sizes)
            for idx in range(len(valid_gdf)):
                top_h = top_heights[idx]
                bot_h = bot_heights[idx]
                dia = diameters[idx]
                alpha_m = alpha_m_all[idx]
                beta_m = beta_m_all[idx]
                
                R = dia / 2.0
                a = max((top_h - bot_h) / 2.0, 0.0)
                z0 = (top_h + bot_h) / 2.0
                
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




