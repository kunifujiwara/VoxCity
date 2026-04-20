import warnings
from typing import List, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
from pyproj import CRS, Transformer

from ..utils import initialize_geod, calculate_distance, normalize_to_one_meter
from .core import calculate_grid_size, compute_grid_geometry, compute_cell_center_coords, normalize_gdf_crs
from ...utils.orientation import ensure_orientation, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP
from ...utils.logging import get_logger

_logger = get_logger(__name__)


def _validate_meshsize(meshsize: float) -> None:
    if meshsize < 0.1:
        warnings.warn(
            f"meshsize={meshsize} is very small (< 0.1 m). Check that you are not passing degrees.",
            UserWarning, stacklevel=3,
        )
    elif meshsize > 1000:
        warnings.warn(
            f"meshsize={meshsize} is very large (> 1000 m). Check units.",
            UserWarning, stacklevel=3,
        )


def _ensure_active_geometry(gdf):
    """Ensure *gdf* has an active geometry column, returning the (possibly fixed) GeoDataFrame."""
    try:
        _ = gdf.crs  # fast check – succeeds when geometry column is active
        return gdf
    except AttributeError:
        if 'geometry' in gdf.columns:
            return gdf.set_geometry('geometry')
        raise


def create_vegetation_height_grid_from_gdf_polygon(veg_gdf, mesh_size, polygon):
    """
    Create a vegetation height grid from a GeoDataFrame of vegetation polygons/objects.
    Cells with vegetation take the max height of intersecting features.
    Returns north-up grid (row 0 = north).

    Uses :func:`compute_cell_center_coords` so rotated rectangles are handled
    correctly.
    """
    _validate_meshsize(mesh_size)
    veg_gdf = normalize_gdf_crs(_ensure_active_geometry(veg_gdf))

    if 'height' not in veg_gdf.columns:
        raise ValueError("Vegetation GeoDataFrame must have a 'height' column.")

    if isinstance(polygon, Polygon):
        rectangle_vertices = list(polygon.exterior.coords[:-1])
    elif isinstance(polygon, list):
        rectangle_vertices = polygon
    else:
        raise ValueError("polygon must be a list of (lon, lat) or a shapely Polygon.")

    cc = compute_cell_center_coords(rectangle_vertices, mesh_size)
    if cc is None:
        warnings.warn("Rectangle is smaller than mesh_size; returning empty array.")
        return np.array([])

    nx, ny = cc["grid_size"]
    dx, dy = cc["adj_mesh"]
    if nx <= 1 and ny <= 1 and dx < mesh_size and dy < mesh_size:
        warnings.warn("Rectangle is smaller than mesh_size; returning empty array.")
        return np.array([])

    center_lons = cc["lons"].ravel()
    center_lats = cc["lats"].ravel()

    grid_points = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in zip(center_lons, center_lats)],
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

    veg_grid = np.zeros(nx * ny, dtype=float)
    for i, row_data in joined_agg.iterrows():
        if not np.isnan(row_data['height']):
            veg_grid[i] = row_data['height']

    veg_grid = veg_grid.reshape(nx, ny)
    return veg_grid


def create_dem_grid_from_gdf_polygon(terrain_gdf, mesh_size, polygon):
    """
    Create a height grid from a terrain GeoDataFrame using nearest-neighbor sampling.
    Returns north-up grid.

    Uses :func:`compute_cell_center_coords` so rotated rectangles are handled
    correctly.
    """
    _validate_meshsize(mesh_size)
    terrain_gdf = normalize_gdf_crs(_ensure_active_geometry(terrain_gdf))

    if isinstance(polygon, Polygon):
        rectangle_vertices = list(polygon.exterior.coords[:-1])
    elif isinstance(polygon, list):
        rectangle_vertices = polygon
    else:
        raise ValueError("`polygon` must be a list of (lon, lat) or a shapely Polygon.")

    cc = compute_cell_center_coords(rectangle_vertices, mesh_size)
    if cc is None:
        warnings.warn("Rectangle is smaller than mesh_size; returning empty array.")
        return np.array([])

    nx, ny = cc["grid_size"]
    dx, dy = cc["adj_mesh"]
    if nx <= 1 and ny <= 1 and dx < mesh_size and dy < mesh_size:
        warnings.warn("Rectangle is smaller than mesh_size; returning empty array.")
        return np.array([])

    center_lons = cc["lons"].ravel()
    center_lats = cc["lats"].ravel()

    grid_points = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in zip(center_lons, center_lats)],
        crs="EPSG:4326"
    )

    if 'elevation' not in terrain_gdf.columns:
        raise ValueError("terrain_gdf must have an 'elevation' column.")

    try:
        clon = float(np.mean(center_lons))
        clat = float(np.mean(center_lats))
        zone = int((clon + 180.0) // 6) + 1
        epsg_proj = 32600 + zone if clat >= 0 else 32700 + zone
        terrain_proj = terrain_gdf.to_crs(epsg=epsg_proj)
        grid_points_proj = grid_points.to_crs(epsg=epsg_proj)

        grid_points_elev = gpd.sjoin_nearest(
            grid_points_proj,
            terrain_proj[['elevation', 'geometry']],
            how="left",
            distance_col="dist_to_terrain"
        )
        grid_points_elev.index = grid_points.index
    except Exception:  # pyproj CRS transform can fail for edge-case zones; fall back to unprojected join
        _logger.debug("Projected sjoin_nearest failed; retrying in original CRS", exc_info=True)
        grid_points_elev = gpd.sjoin_nearest(
            grid_points,
            terrain_gdf[['elevation', 'geometry']],
            how="left",
            distance_col="dist_to_terrain"
        )

    dem_grid = np.full(nx * ny, np.nan, dtype=float)
    for i, elevation_val in zip(grid_points_elev.index, grid_points_elev['elevation']):
        dem_grid[i] = elevation_val

    dem_grid = dem_grid.reshape(nx, ny)
    return dem_grid


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

    _validate_meshsize(meshsize)
    required_cols = ['top_height', 'bottom_height', 'crown_diameter', 'geometry']
    for col in required_cols:
        if col not in tree_gdf.columns:
            raise ValueError(f"tree_gdf must contain '{col}' column.")

    tree_gdf = normalize_gdf_crs(_ensure_active_geometry(tree_gdf))

    geom = compute_grid_geometry(rectangle_vertices, meshsize)
    origin = geom["origin"]
    side_1, side_2 = geom["side_1"], geom["side_2"]
    u_vec, v_vec = geom["u_vec"], geom["v_vec"]
    grid_size, adjusted_meshsize = geom["grid_size"], geom["adj_mesh"]
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
                mask = ensure_orientation(mask, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP)
                
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
            except (ValueError, RuntimeError):
                _logger.debug("Skipping height group during canopy rasterization", exc_info=True)
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




