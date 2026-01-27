"""
Terrain/DEM grid processing functions.
"""
import warnings

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
from pyproj import Geod
from scipy.interpolate import griddata

from ..utils import (
    initialize_geod,
    calculate_distance,
    normalize_to_one_meter,
)
from .core import calculate_grid_size


def create_dem_grid_from_gdf_kdtree(terrain_gdf, mesh_size, polygon):
    """
    Create a height grid from a terrain GeoDataFrame using fast KDTree lookup.
    
    This is an optimized version that uses scipy.spatial.cKDTree for fast
    nearest-neighbor interpolation, which is significantly faster than
    scipy.interpolate.griddata for large datasets (100k+ points).
    
    Args:
        terrain_gdf: GeoDataFrame with Point geometries and 'elevation' column.
        mesh_size: Grid cell size in meters.
        polygon: List of (lon, lat) tuples or a shapely Polygon defining the area.
            Should be a list of 4 vertices in [SW, NW, NE, SE] order for consistency
            with other VoxCity grid functions.
    
    Returns:
        np.ndarray: Grid of elevation values with same orientation as buildings/landcover.
    """
    from scipy.spatial import cKDTree
    
    if terrain_gdf is None or len(terrain_gdf) == 0:
        warnings.warn("Empty terrain GeoDataFrame; returning None.")
        return None
    
    if terrain_gdf.crs is None:
        warnings.warn("terrain_gdf has no CRS. Assuming EPSG:4326.")
        terrain_gdf = terrain_gdf.set_crs(epsg=4326)
    elif terrain_gdf.crs.to_epsg() != 4326:
        terrain_gdf = terrain_gdf.to_crs(epsg=4326)

    # Use polygon list as rectangle_vertices if provided as list
    if isinstance(polygon, list):
        rectangle_vertices = polygon
        poly = Polygon(polygon)
    elif isinstance(polygon, Polygon):
        poly = polygon
        # Extract vertices from polygon (may not be in expected order)
        rectangle_vertices = list(poly.exterior.coords)[:-1]  # Remove closing point
    else:
        raise ValueError("`polygon` must be a list of (lon, lat) or a shapely Polygon.")

    # Use the same grid calculation approach as buildings.py for consistency
    geod = initialize_geod()
    vertex_0, vertex_1, vertex_3 = rectangle_vertices[0], rectangle_vertices[1], rectangle_vertices[3]
    
    dist_side_1 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_1[0], vertex_1[1])
    dist_side_2 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_3[0], vertex_3[1])
    
    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)
    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)
    
    grid_size, adjusted_meshsize = calculate_grid_size(side_1, side_2, u_vec, v_vec, mesh_size)
    num_cells_0, num_cells_1 = grid_size  # (side_1 direction, side_2 direction)
    
    if num_cells_0 < 1 or num_cells_1 < 1:
        warnings.warn("Polygon bounding box is smaller than mesh_size; returning empty array.")
        return np.array([])

    if 'elevation' not in terrain_gdf.columns:
        raise ValueError("terrain_gdf must have an 'elevation' column.")

    # Project to UTM for accurate distance-based KDTree operations
    centroid = poly.centroid
    lon_c, lat_c = float(centroid.x), float(centroid.y)
    zone = int((lon_c + 180.0) // 6) + 1
    epsg_proj = 32600 + zone if lat_c >= 0 else 32700 + zone
    
    try:
        terrain_proj = terrain_gdf.to_crs(epsg=epsg_proj)
        poly_proj = gpd.GeoSeries([poly], crs="EPSG:4326").to_crs(epsg=epsg_proj)[0]
        # Also project the origin point for grid construction
        origin_gdf = gpd.GeoDataFrame(geometry=[Point(vertex_0[0], vertex_0[1])], crs="EPSG:4326")
        origin_proj = origin_gdf.to_crs(epsg=epsg_proj).geometry[0]
    except Exception:
        # Fallback to WGS84 if projection fails
        terrain_proj = terrain_gdf
        poly_proj = poly
        origin_proj = Point(vertex_0[0], vertex_0[1])
    
    # Extract terrain points and elevations (vectorized)
    terrain_points = np.array([
        (geom.x, geom.y) for geom in terrain_proj.geometry if geom is not None
    ])
    terrain_elevations = terrain_proj['elevation'].values
    
    # Filter out invalid points
    valid_mask = ~np.isnan(terrain_elevations)
    terrain_points = terrain_points[valid_mask]
    terrain_elevations = terrain_elevations[valid_mask]
    
    if len(terrain_points) < 1:
        warnings.warn("No valid terrain points found; returning zeros.")
        return np.zeros((num_cells_0, num_cells_1), dtype=np.float32)
    
    # Build KDTree (very fast, O(n log n))
    kdtree = cKDTree(terrain_points)
    
    # Create grid of query points using the same convention as buildings.py
    # Grid origin is at vertex_0, with i increasing along side_1 and j along side_2
    # This matches the cell iteration in buildings.py _process_with_geometry_intersection
    origin_x, origin_y = origin_proj.x, origin_proj.y
    
    # Calculate projected vectors for u_vec and v_vec
    # side_1 = vertex_1 - vertex_0, side_2 = vertex_3 - vertex_0
    v1_gdf = gpd.GeoDataFrame(geometry=[Point(vertex_1[0], vertex_1[1])], crs="EPSG:4326")
    v3_gdf = gpd.GeoDataFrame(geometry=[Point(vertex_3[0], vertex_3[1])], crs="EPSG:4326")
    try:
        v1_proj = v1_gdf.to_crs(epsg=epsg_proj).geometry[0]
        v3_proj = v3_gdf.to_crs(epsg=epsg_proj).geometry[0]
    except Exception:
        v1_proj = Point(vertex_1[0], vertex_1[1])
        v3_proj = Point(vertex_3[0], vertex_3[1])
    
    side_1_proj = np.array([v1_proj.x - origin_x, v1_proj.y - origin_y])
    side_2_proj = np.array([v3_proj.x - origin_x, v3_proj.y - origin_y])
    
    # Unit vectors in projected coordinates
    u_proj = side_1_proj / (np.linalg.norm(side_1_proj) + 1e-12) * adjusted_meshsize[0]
    v_proj = side_2_proj / (np.linalg.norm(side_2_proj) + 1e-12) * adjusted_meshsize[1]
    
    # Create query points at cell centers (matching buildings.py convention)
    # Grid cell (i, j) has center at origin + (i + 0.5) * u_proj + (j + 0.5) * v_proj
    query_points = []
    for i in range(num_cells_0):
        for j in range(num_cells_1):
            cx = origin_x + (i + 0.5) * u_proj[0] + (j + 0.5) * v_proj[0]
            cy = origin_y + (i + 0.5) * u_proj[1] + (j + 0.5) * v_proj[1]
            query_points.append([cx, cy])
    query_points = np.array(query_points)
    
    # Query all points at once (very fast, O(n log n))
    _, indices = kdtree.query(query_points, k=1)
    
    # Reshape results back to grid (num_cells_0, num_cells_1) to match buildings.py
    dem_grid = terrain_elevations[indices].reshape(num_cells_0, num_cells_1).astype(np.float32)
    
    print(f"    DEM grid created using KDTree ({len(terrain_points):,} points -> {num_cells_0}x{num_cells_1} grid)")
    
    # No flipud needed - grid is now in same orientation as buildings.py
    # Grid origin is at vertex_0 (SW), i increases along side_1 (typically N), j along side_2 (typically E)
    return dem_grid


def create_dem_grid_from_gdf_polygon(terrain_gdf, mesh_size, polygon, interpolation=True, method='linear'):
    """
    Create a height grid from a terrain GeoDataFrame.
    
    Args:
        terrain_gdf: GeoDataFrame with Point geometries and 'elevation' column.
        mesh_size: Grid cell size in meters.
        polygon: List of (lon, lat) tuples or a shapely Polygon defining the area.
            Should be a list of 4 vertices in [SW, NW, NE, SE] order for consistency
            with other VoxCity grid functions.
        interpolation: If True, use interpolation; if False, use nearest-neighbor.
            Default is True.
        method: Interpolation method - 'linear', 'cubic', or 'nearest'.
            Default is 'linear' which works well for TIN-derived point data.
            'cubic' may produce smoother results but can overshoot/undershoot.
    
    Returns:
        np.ndarray: Grid of elevation values with same orientation as buildings/landcover.
    """
    if terrain_gdf.crs is None:
        warnings.warn("terrain_gdf has no CRS. Assuming EPSG:4326. ")
        terrain_gdf = terrain_gdf.set_crs(epsg=4326)
    else:
        if terrain_gdf.crs.to_epsg() != 4326:
            terrain_gdf = terrain_gdf.to_crs(epsg=4326)

    # Use polygon list as rectangle_vertices if provided as list
    if isinstance(polygon, list):
        rectangle_vertices = polygon
        poly = Polygon(polygon)
    elif isinstance(polygon, Polygon):
        poly = polygon
        rectangle_vertices = list(poly.exterior.coords)[:-1]
    else:
        raise ValueError("`polygon` must be a list of (lon, lat) or a shapely Polygon.")

    # Use the same grid calculation approach as buildings.py for consistency
    geod = initialize_geod()
    vertex_0, vertex_1, vertex_3 = rectangle_vertices[0], rectangle_vertices[1], rectangle_vertices[3]
    
    dist_side_1 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_1[0], vertex_1[1])
    dist_side_2 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_3[0], vertex_3[1])
    
    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)
    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)
    
    grid_size, adjusted_meshsize = calculate_grid_size(side_1, side_2, u_vec, v_vec, mesh_size)
    num_cells_0, num_cells_1 = grid_size
    
    if num_cells_0 < 1 or num_cells_1 < 1:
        warnings.warn("Polygon bounding box is smaller than mesh_size; returning empty array.")
        return np.array([])

    if 'elevation' not in terrain_gdf.columns:
        raise ValueError("terrain_gdf must have an 'elevation' column.")

    # Project to UTM for accurate distance-based operations
    centroid = poly.centroid
    lon_c, lat_c = float(centroid.x), float(centroid.y)
    zone = int((lon_c + 180.0) // 6) + 1
    epsg_proj = 32600 + zone if lat_c >= 0 else 32700 + zone
    
    try:
        terrain_proj = terrain_gdf.to_crs(epsg=epsg_proj)
        origin_gdf = gpd.GeoDataFrame(geometry=[Point(vertex_0[0], vertex_0[1])], crs="EPSG:4326")
        origin_proj = origin_gdf.to_crs(epsg=epsg_proj).geometry[0]
        v1_gdf = gpd.GeoDataFrame(geometry=[Point(vertex_1[0], vertex_1[1])], crs="EPSG:4326")
        v3_gdf = gpd.GeoDataFrame(geometry=[Point(vertex_3[0], vertex_3[1])], crs="EPSG:4326")
        v1_proj = v1_gdf.to_crs(epsg=epsg_proj).geometry[0]
        v3_proj = v3_gdf.to_crs(epsg=epsg_proj).geometry[0]
    except Exception:
        terrain_proj = terrain_gdf
        origin_proj = Point(vertex_0[0], vertex_0[1])
        v1_proj = Point(vertex_1[0], vertex_1[1])
        v3_proj = Point(vertex_3[0], vertex_3[1])
        epsg_proj = 4326

    origin_x, origin_y = origin_proj.x, origin_proj.y
    side_1_proj = np.array([v1_proj.x - origin_x, v1_proj.y - origin_y])
    side_2_proj = np.array([v3_proj.x - origin_x, v3_proj.y - origin_y])
    u_proj = side_1_proj / (np.linalg.norm(side_1_proj) + 1e-12) * adjusted_meshsize[0]
    v_proj = side_2_proj / (np.linalg.norm(side_2_proj) + 1e-12) * adjusted_meshsize[1]

    if interpolation:
        # Extract terrain points and elevations
        terrain_points = np.array([
            (geom.x, geom.y) for geom in terrain_proj.geometry if geom is not None
        ])
        terrain_elevations = terrain_proj['elevation'].values
        
        # Filter out invalid points
        valid_mask = ~np.isnan(terrain_elevations)
        terrain_points = terrain_points[valid_mask]
        terrain_elevations = terrain_elevations[valid_mask]
        
        if len(terrain_points) < 4:
            warnings.warn("Not enough valid terrain points for interpolation; falling back to nearest.")
            interpolation = False
        else:
            # Create grid query points matching buildings.py convention
            query_points = []
            for i in range(num_cells_0):
                for j in range(num_cells_1):
                    cx = origin_x + (i + 0.5) * u_proj[0] + (j + 0.5) * v_proj[0]
                    cy = origin_y + (i + 0.5) * u_proj[1] + (j + 0.5) * v_proj[1]
                    query_points.append([cx, cy])
            query_points = np.array(query_points)
            
            # Interpolate using specified method
            dem_values = griddata(
                terrain_points,
                terrain_elevations,
                query_points,
                method=method
            )
            
            # Fill NaN values with nearest neighbor
            nan_mask = np.isnan(dem_values)
            if np.any(nan_mask):
                nearest_values = griddata(
                    terrain_points,
                    terrain_elevations,
                    query_points,
                    method='nearest'
                )
                dem_values[nan_mask] = nearest_values[nan_mask]
            
            dem_grid = dem_values.reshape(num_cells_0, num_cells_1).astype(np.float32)
            return dem_grid
    
    # Nearest-neighbor approach using spatial join
    query_points_geom = []
    for i in range(num_cells_0):
        for j in range(num_cells_1):
            cx = origin_x + (i + 0.5) * u_proj[0] + (j + 0.5) * v_proj[0]
            cy = origin_y + (i + 0.5) * u_proj[1] + (j + 0.5) * v_proj[1]
            query_points_geom.append(Point(cx, cy))

    grid_points = gpd.GeoDataFrame(geometry=query_points_geom, crs=f"EPSG:{epsg_proj}")

    try:
        grid_points_elev = gpd.sjoin_nearest(
            grid_points,
            terrain_proj[['elevation', 'geometry']],
            how="left",
            distance_col="dist_to_terrain"
        )
    except Exception:
        grid_points_elev = gpd.sjoin_nearest(
            grid_points.to_crs("EPSG:4326"),
            terrain_gdf[['elevation', 'geometry']],
            how="left",
            distance_col="dist_to_terrain"
        )

    dem_grid = np.full((num_cells_0, num_cells_1), np.nan, dtype=float)
    for idx, elevation_val in enumerate(grid_points_elev['elevation']):
        i = idx // num_cells_1
        j = idx % num_cells_1
        dem_grid[i, j] = elevation_val
    
    return dem_grid.astype(np.float32)
