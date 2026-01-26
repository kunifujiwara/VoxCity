"""
Terrain/DEM grid processing functions.
"""
import warnings

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
from pyproj import Geod
from scipy.interpolate import griddata


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
    
    Returns:
        np.ndarray: North-up grid (row 0 = north) of elevation values.
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
    except Exception:
        # Fallback to WGS84 if projection fails
        terrain_proj = terrain_gdf
        poly_proj = poly
    
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
        return np.zeros((num_cells_y, num_cells_x), dtype=np.float32)
    
    # Build KDTree (very fast, O(n log n))
    kdtree = cKDTree(terrain_points)
    
    # Create grid of query points in projected coordinates
    left_p, bottom_p, right_p, top_p = poly_proj.bounds
    
    # Cell centers: X increases left to right, Y decreases top to bottom (for north-up output)
    xs_proj = np.linspace(left_p + mesh_size/2, right_p - mesh_size/2, num_cells_x)
    ys_proj = np.linspace(top_p - mesh_size/2, bottom_p + mesh_size/2, num_cells_y)
    
    # Create meshgrid of all query points
    X_proj, Y_proj = np.meshgrid(xs_proj, ys_proj)
    query_points = np.column_stack([X_proj.ravel(), Y_proj.ravel()])
    
    # Query all points at once (very fast, O(n log n))
    _, indices = kdtree.query(query_points, k=1)
    
    # Reshape results back to grid
    dem_grid = terrain_elevations[indices].reshape(num_cells_y, num_cells_x).astype(np.float32)
    
    print(f"    DEM grid created using KDTree ({len(terrain_points):,} points -> {num_cells_x}x{num_cells_y} grid)")
    
    # Flip to match the expected orientation (same as create_dem_grid_from_gdf_polygon)
    return np.flipud(dem_grid)


def create_dem_grid_from_gdf_polygon(terrain_gdf, mesh_size, polygon, interpolation=True, method='linear'):
    """
    Create a height grid from a terrain GeoDataFrame.
    
    Args:
        terrain_gdf: GeoDataFrame with Point geometries and 'elevation' column.
        mesh_size: Grid cell size in meters.
        polygon: List of (lon, lat) tuples or a shapely Polygon defining the area.
        interpolation: If True, use interpolation; if False, use nearest-neighbor.
            Default is True.
        method: Interpolation method - 'linear', 'cubic', or 'nearest'.
            Default is 'linear' which works well for TIN-derived point data.
            'cubic' may produce smoother results but can overshoot/undershoot.
    
    Returns:
        np.ndarray: North-up grid (row 0 = north) of elevation values.
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

    if 'elevation' not in terrain_gdf.columns:
        raise ValueError("terrain_gdf must have an 'elevation' column.")

    # Project to UTM for accurate distance-based operations
    centroid = poly.centroid
    lon_c, lat_c = float(centroid.x), float(centroid.y)
    zone = int((lon_c + 180.0) // 6) + 1
    epsg_proj = 32600 + zone if lat_c >= 0 else 32700 + zone
    
    try:
        terrain_proj = terrain_gdf.to_crs(epsg=epsg_proj)
    except Exception:
        # Fallback to WGS84 if projection fails
        terrain_proj = terrain_gdf
        epsg_proj = 4326

    if interpolation:
        # Use scipy griddata for interpolation
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
            # Create grid in projected coordinates
            poly_proj = gpd.GeoSeries([poly], crs="EPSG:4326").to_crs(epsg=epsg_proj)[0]
            left_p, bottom_p, right_p, top_p = poly_proj.bounds
            
            xs_proj = np.linspace(left_p, right_p, num_cells_x)
            ys_proj = np.linspace(top_p, bottom_p, num_cells_y)
            X_proj, Y_proj = np.meshgrid(xs_proj, ys_proj)
            
            # Interpolate using specified method
            dem_grid = griddata(
                terrain_points,
                terrain_elevations,
                (X_proj, Y_proj),
                method=method
            )
            
            # Fill NaN values (outside convex hull or edge cases) with nearest neighbor
            nan_mask = np.isnan(dem_grid)
            if np.any(nan_mask):
                nearest_grid = griddata(
                    terrain_points,
                    terrain_elevations,
                    (X_proj, Y_proj),
                    method='nearest'
                )
                dem_grid[nan_mask] = nearest_grid[nan_mask]
            
            return np.flipud(dem_grid)
    
    # Nearest-neighbor approach using spatial join
    xs = np.linspace(left, right, num_cells_x)
    ys = np.linspace(top, bottom, num_cells_y)
    X, Y = np.meshgrid(xs, ys)
    xs_flat = X.ravel()
    ys_flat = Y.ravel()

    grid_points = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in zip(xs_flat, ys_flat)],
        crs="EPSG:4326"
    )

    try:
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
