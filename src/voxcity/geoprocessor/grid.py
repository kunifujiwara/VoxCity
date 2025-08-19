"""
This module provides functions for creating and manipulating grids of building heights, land cover, and elevation data.
It includes functionality for:
- Grid creation and manipulation for various data types (buildings, land cover, elevation)
- Coordinate transformations and spatial operations
- Data interpolation and aggregation
- Vector to raster conversion
"""

import numpy as np
import pandas as pd
import os
from shapely.geometry import Polygon, Point, MultiPolygon, box, mapping
from scipy.ndimage import label, generate_binary_structure
from pyproj import Geod, Transformer, CRS
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from affine import Affine
import geopandas as gpd
from collections import defaultdict
from scipy.interpolate import griddata
from shapely.errors import GEOSException
from rtree import index
import warnings

from .utils import (
    initialize_geod,
    calculate_distance,
    normalize_to_one_meter,
    create_building_polygons,
    convert_format_lat_lon
)
from ..geoprocessor.polygon import (
    filter_buildings, 
    extract_building_heights_from_geotiff, 
    extract_building_heights_from_gdf,
    complement_building_heights_from_gdf,
    process_building_footprints_by_overlap
)
from ..utils.lc import (
    get_class_priority, 
    create_land_cover_polygons, 
    get_dominant_class,
)
from ..downloader.gee import (
    get_roi,
    save_geotiff_open_buildings_temporal
)

def apply_operation(arr, meshsize):
    """
    Applies a sequence of operations to an array based on a mesh size to normalize and discretize values.
    
    This function performs the following sequence of operations:
    1. Divides array by mesh size to normalize values
    2. Adds 0.5 to round values to nearest integer
    3. Floors the result to get integer values
    4. Scales back to original units by multiplying by mesh size
    
    Args:
        arr (numpy.ndarray): Input array to transform
        meshsize (float): Size of mesh to use for calculations
        
    Returns:
        numpy.ndarray: Transformed array after applying operations
        
    Example:
        >>> arr = np.array([1.2, 2.7, 3.4])
        >>> meshsize = 0.5
        >>> result = apply_operation(arr, meshsize)
    """
    # Divide array by mesh size to normalize values
    step1 = arr / meshsize
    # Add 0.5 to round values to nearest integer
    step2 = step1 + 0.5  
    # Floor to get integer values
    step3 = np.floor(step2)
    # Scale back to original units
    return step3 * meshsize

def translate_array(input_array, translation_dict):
    """
    Translates values in an array according to a dictionary mapping.
    
    This function creates a new array where each value from the input array
    is replaced by its corresponding value from the translation dictionary.
    Values not found in the dictionary are replaced with empty strings.
    
    Args:
        input_array (numpy.ndarray): Array containing values to translate
        translation_dict (dict): Dictionary mapping input values to output values
        
    Returns:
        numpy.ndarray: Array with translated values, with same shape as input array
        
    Example:
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> trans_dict = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        >>> result = translate_array(arr, trans_dict)
        >>> # result = array([['A', 'B'], ['C', 'D']], dtype=object)
    """
    # Create empty array of same shape that can hold objects (e.g. strings)
    translated_array = np.empty_like(input_array, dtype=object)
    # Iterate through array and replace values using dictionary
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            value = input_array[i, j]
            # Use dict.get() to handle missing keys, defaulting to empty string
            translated_array[i, j] = translation_dict.get(value, '')
    return translated_array

def group_and_label_cells(array):
    """
    Convert non-zero numbers in a 2D numpy array to sequential IDs starting from 1.
    
    This function creates a new array where all non-zero values are replaced with
    sequential IDs (1, 2, 3, etc.) while preserving zero values. This is useful
    for labeling distinct regions or features in a grid.
    
    Args:
        array (numpy.ndarray): Input 2D array with non-zero values to be labeled
        
    Returns:
        numpy.ndarray: Array with non-zero values converted to sequential IDs,
                      maintaining the same shape as input array
        
    Example:
        >>> arr = np.array([[0, 5, 5], [0, 5, 8], [0, 0, 8]])
        >>> result = group_and_label_cells(arr)
        >>> # result = array([[0, 1, 1], [0, 1, 2], [0, 0, 2]])
    """
    # Create a copy to avoid modifying input
    result = array.copy()
    
    # Get sorted set of unique non-zero values
    unique_values = sorted(set(array.flatten()) - {0})
    
    # Create mapping from original values to sequential IDs (1, 2, 3, etc)
    value_to_id = {value: idx + 1 for idx, value in enumerate(unique_values)}
    
    # Replace each non-zero value with its new sequential ID
    for value in unique_values:
        result[array == value] = value_to_id[value]
    
    return result

def process_grid_optimized(grid_bi, dem_grid):
    """
    Optimized version that computes per-building averages without allocating
    huge arrays when building IDs are large and sparse.
    """
    result = dem_grid.copy()

    # Only process if there are non-zero values
    if np.any(grid_bi != 0):
        # Convert to integer IDs (handle NaN for float arrays)
        if grid_bi.dtype.kind == 'f':
            grid_bi_int = np.nan_to_num(grid_bi, nan=0).astype(np.int64)
        else:
            grid_bi_int = grid_bi.astype(np.int64)

        # Work only on non-zero cells
        flat_ids = grid_bi_int.ravel()
        flat_dem = dem_grid.ravel()
        nz_mask = flat_ids != 0
        if np.any(nz_mask):
            ids_nz = flat_ids[nz_mask]
            vals_nz = flat_dem[nz_mask]

            # Densify IDs via inverse indices to avoid np.bincount on large max(id)
            unique_ids, inverse_idx = np.unique(ids_nz, return_inverse=True)
            sums = np.bincount(inverse_idx, weights=vals_nz)
            counts = np.bincount(inverse_idx)
            counts[counts == 0] = 1
            means = sums / counts

            # Scatter means back to result for non-zero cells
            result.ravel()[nz_mask] = means[inverse_idx]

    return result - np.min(result)

def process_grid(grid_bi, dem_grid):
    """
    Safe version that tries optimization first, then falls back to original method.
    """
    try:
        # Try the optimized version first
        return process_grid_optimized(grid_bi, dem_grid)
    except Exception as e:
        print(f"Optimized process_grid failed: {e}, using original method")
        # Fall back to original implementation
        unique_ids = np.unique(grid_bi[grid_bi != 0])
        result = dem_grid.copy()
        
        for id_num in unique_ids:
            mask = (grid_bi == id_num)
            avg_value = np.mean(dem_grid[mask])
            result[mask] = avg_value
        
        return result - np.min(result)
    """
    Optimized version that avoids converting to Python lists.
    Works directly with numpy arrays.
    """
    if not isinstance(arr, np.ndarray):
        return arr
    
    # Create output array
    result = np.empty_like(arr, dtype=object)
    
    # Vectorized operation for empty cells
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            cell = arr[i, j]
            
            if cell is None or (isinstance(cell, list) and len(cell) == 0):
                result[i, j] = []
            elif isinstance(cell, list):
                # Process list without converting entire array
                new_cell = []
                for segment in cell:
                    if isinstance(segment, (list, np.ndarray)):
                        # Use numpy operations where possible
                        if isinstance(segment, np.ndarray):
                            new_segment = np.where(np.isnan(segment), replace_value, segment).tolist()
                        else:
                            new_segment = [replace_value if (isinstance(v, float) and np.isnan(v)) else v for v in segment]
                        new_cell.append(new_segment)
                    else:
                        new_cell.append(segment)
                result[i, j] = new_cell
            else:
                result[i, j] = cell
    
    return result

def calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize):
    """
    Calculate grid size and adjusted mesh size based on input parameters.
    
    This function determines the number of grid cells needed in each direction and
    adjusts the mesh size to exactly fit the desired area. The calculation takes into
    account the input vectors and desired mesh size to ensure proper coverage.
    
    Args:
        side_1 (numpy.ndarray): First side vector defining the grid extent
        side_2 (numpy.ndarray): Second side vector defining the grid extent
        u_vec (numpy.ndarray): Unit vector in first direction
        v_vec (numpy.ndarray): Unit vector in second direction
        meshsize (float): Desired mesh size in the same units as the vectors
        
    Returns:
        tuple: A tuple containing:
            - grid_size (tuple of ints): Number of cells in each direction (nx, ny)
            - adjusted_mesh_size (tuple of floats): Actual mesh sizes that fit the area exactly
        
    Example:
        >>> side1 = np.array([100, 0])  # 100 units in x direction
        >>> side2 = np.array([0, 50])   # 50 units in y direction
        >>> u = np.array([1, 0])        # Unit vector in x
        >>> v = np.array([0, 1])        # Unit vector in y
        >>> mesh = 10                    # Desired 10-unit mesh
        >>> grid_size, adj_mesh = calculate_grid_size(side1, side2, u, v, mesh)
    """
    # Calculate total side lengths in meters using the relationship between side vectors and unit vectors
    # u_vec and v_vec represent degrees per meter along each side direction
    dist_side_1_m = np.linalg.norm(side_1) / (np.linalg.norm(u_vec) + 1e-12)
    dist_side_2_m = np.linalg.norm(side_2) / (np.linalg.norm(v_vec) + 1e-12)

    # Calculate number of cells (nx along u, ny along v), rounding to nearest integer and ensuring at least 1
    grid_size_0 = max(1, int(dist_side_1_m / meshsize + 0.5))
    grid_size_1 = max(1, int(dist_side_2_m / meshsize + 0.5))

    # Adjust mesh sizes (in meters) to exactly fit the sides with the calculated number of cells
    adjusted_mesh_size_0 = dist_side_1_m / grid_size_0
    adjusted_mesh_size_1 = dist_side_2_m / grid_size_1

    return (grid_size_0, grid_size_1), (adjusted_mesh_size_0, adjusted_mesh_size_1)

def create_coordinate_mesh(origin, grid_size, adjusted_meshsize, u_vec, v_vec):
    """
    Create a coordinate mesh based on input parameters.
    
    This function generates a 3D array representing a coordinate mesh, where each point
    in the mesh is calculated by adding scaled vectors to the origin point. The mesh
    is created using the specified grid size and adjusted mesh sizes.
    
    Args:
        origin (numpy.ndarray): Origin point coordinates (shape: (2,) or (3,))
        grid_size (tuple): Size of grid in each dimension (nx, ny)
        adjusted_meshsize (tuple): Adjusted mesh size in each dimension (dx, dy)
        u_vec (numpy.ndarray): Unit vector in first direction
        v_vec (numpy.ndarray): Unit vector in second direction
        
    Returns:
        numpy.ndarray: 3D array of shape (coord_dim, ny, nx) containing the coordinates
                      of each point in the mesh. coord_dim is the same as the
                      dimensionality of the input vectors.
        
    Example:
        >>> origin = np.array([0, 0])
        >>> grid_size = (5, 4)
        >>> mesh_size = (10, 10)
        >>> u = np.array([1, 0])
        >>> v = np.array([0, 1])
        >>> coords = create_coordinate_mesh(origin, grid_size, mesh_size, u, v)
    """
    # Create evenly spaced points along each axis
    x = np.linspace(0, grid_size[0], grid_size[0])
    y = np.linspace(0, grid_size[1], grid_size[1])
    
    # Create 2D coordinate grids
    xx, yy = np.meshgrid(x, y)

    # Calculate coordinates of each cell by adding scaled vectors
    cell_coords = origin[:, np.newaxis, np.newaxis] + \
                  xx[np.newaxis, :, :] * adjusted_meshsize[0] * u_vec[:, np.newaxis, np.newaxis] + \
                  yy[np.newaxis, :, :] * adjusted_meshsize[1] * v_vec[:, np.newaxis, np.newaxis]

    return cell_coords

def create_cell_polygon(origin, i, j, adjusted_meshsize, u_vec, v_vec):
    """
    Create a polygon representing a grid cell.
    
    This function generates a rectangular polygon for a specific grid cell by calculating
    its four corners based on the cell indices and grid parameters. The polygon is
    created in counter-clockwise order starting from the bottom-left corner.
    
    Args:
        origin (numpy.ndarray): Origin point coordinates (shape: (2,) or (3,))
        i (int): Row index of the cell
        j (int): Column index of the cell
        adjusted_meshsize (tuple): Adjusted mesh size in each dimension (dx, dy)
        u_vec (numpy.ndarray): Unit vector in first direction
        v_vec (numpy.ndarray): Unit vector in second direction
        
    Returns:
        shapely.geometry.Polygon: Polygon representing the grid cell, with vertices
                                ordered counter-clockwise from bottom-left
        
    Example:
        >>> origin = np.array([0, 0])
        >>> i, j = 1, 2  # Cell at row 1, column 2
        >>> mesh_size = (10, 10)
        >>> u = np.array([1, 0])
        >>> v = np.array([0, 1])
        >>> cell_poly = create_cell_polygon(origin, i, j, mesh_size, u, v)
    """
    # Calculate the four corners of the cell by adding scaled vectors
    bottom_left = origin + i * adjusted_meshsize[0] * u_vec + j * adjusted_meshsize[1] * v_vec
    bottom_right = origin + (i + 1) * adjusted_meshsize[0] * u_vec + j * adjusted_meshsize[1] * v_vec
    top_right = origin + (i + 1) * adjusted_meshsize[0] * u_vec + (j + 1) * adjusted_meshsize[1] * v_vec
    top_left = origin + i * adjusted_meshsize[0] * u_vec + (j + 1) * adjusted_meshsize[1] * v_vec
    
    # Create polygon from corners in counter-clockwise order
    return Polygon([bottom_left, bottom_right, top_right, top_left])

def tree_height_grid_from_land_cover(land_cover_grid_ori):
    """
    Convert a land cover grid to a tree height grid.
    
    This function transforms a land cover classification grid into a grid of tree heights
    by mapping land cover classes to predefined tree heights. The function first flips
    the input grid vertically and adjusts class values, then applies a translation
    dictionary to convert classes to heights.
    
    Land cover class to tree height mapping:
    - Class 4 (Forest): 10m height
    - All other classes: 0m height
    
    Args:
        land_cover_grid_ori (numpy.ndarray): Original land cover grid with class values
        
    Returns:
        numpy.ndarray: Grid of tree heights in meters, with same dimensions as input
        
    Example:
        >>> lc_grid = np.array([[1, 4, 2], [4, 3, 4], [2, 1, 3]])
        >>> tree_heights = tree_height_grid_from_land_cover(lc_grid)
        >>> # Result: array([[0, 10, 0], [10, 0, 10], [0, 0, 0]])
    """
    # Flip array vertically and add 1 to all values
    land_cover_grid = np.flipud(land_cover_grid_ori) + 1

    # Define mapping from land cover classes to tree heights
    tree_translation_dict = {
        1: 0,  # No trees
        2: 0,  # No trees
        3: 0,  # No trees
        4: 10, # Forest - 10m height
        5: 0,  # No trees
        6: 0,  # No trees
        7: 0,  # No trees
        8: 0,  # No trees
        9: 0,  # No trees
        10: 0  # No trees
    }
    
    # Convert land cover classes to tree heights and flip back
    tree_height_grid = translate_array(np.flipud(land_cover_grid), tree_translation_dict).astype(int)

    return tree_height_grid

def create_land_cover_grid_from_geotiff_polygon(tiff_path, mesh_size, land_cover_classes, polygon):
    """
    Create a land cover grid from a GeoTIFF file within a polygon boundary.
    
    Args:
        tiff_path (str): Path to GeoTIFF file
        mesh_size (float): Size of mesh cells
        land_cover_classes (dict): Dictionary mapping land cover classes
        polygon (list): List of polygon vertices
        
    Returns:
        numpy.ndarray: Grid of land cover classes within the polygon
    """
    with rasterio.open(tiff_path) as src:
        # Read RGB bands from GeoTIFF
        img = src.read((1,2,3))
        left, bottom, right, top = src.bounds
        src_crs = src.crs
        
        # Create a Shapely polygon from input coordinates
        poly = Polygon(polygon)
        
        # Get bounds of the polygon in WGS84 coordinates
        left_wgs84, bottom_wgs84, right_wgs84, top_wgs84 = poly.bounds
        # print(left, bottom, right, top)

        # Calculate width and height using geodesic calculations for accuracy
        geod = Geod(ellps="WGS84")
        _, _, width = geod.inv(left_wgs84, bottom_wgs84, right_wgs84, bottom_wgs84)
        _, _, height = geod.inv(left_wgs84, bottom_wgs84, left_wgs84, top_wgs84)
        
        # Calculate number of grid cells based on mesh size
        num_cells_x = int(width / mesh_size + 0.5)
        num_cells_y = int(height / mesh_size + 0.5)
        
        # Adjust mesh_size to fit the image exactly
        adjusted_mesh_size_x = (right - left) / num_cells_x
        adjusted_mesh_size_y = (top - bottom) / num_cells_y
        
        # Create affine transform for mapping between pixel and world coordinates
        new_affine = Affine(adjusted_mesh_size_x, 0, left, 0, -adjusted_mesh_size_y, top)
        
        # Create coordinate grids for the new mesh
        cols, rows = np.meshgrid(np.arange(num_cells_x), np.arange(num_cells_y))
        xs, ys = new_affine * (cols, rows)
        xs_flat, ys_flat = xs.flatten(), ys.flatten()
        
        # Convert world coordinates to image pixel indices
        row, col = src.index(xs_flat, ys_flat)
        row, col = np.array(row), np.array(col)
        
        # Filter out indices that fall outside the image bounds
        valid = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)
        row, col = row[valid], col[valid]
        
        # Initialize output grid with 'No Data' values
        grid = np.full((num_cells_y, num_cells_x), 'No Data', dtype=object)
        
        # Fill grid with dominant land cover classes
        for i, (r, c) in enumerate(zip(row, col)):
            cell_data = img[:, r, c]
            dominant_class = get_dominant_class(cell_data, land_cover_classes)
            grid_row, grid_col = np.unravel_index(i, (num_cells_y, num_cells_x))
            grid[grid_row, grid_col] = dominant_class
    
    # Flip grid vertically to match geographic orientation
    return np.flipud(grid)
    
def create_land_cover_grid_from_gdf_polygon(gdf, meshsize, source, rectangle_vertices, default_class='Developed space'):
    """Create a grid of land cover classes from GeoDataFrame polygon data.

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing land cover polygons
        meshsize (float): Size of each grid cell in meters
        source (str): Source of the land cover data to determine class priorities
        rectangle_vertices (list): List of 4 (lon,lat) coordinate pairs defining the rectangle bounds
        default_class (str, optional): Default land cover class for cells with no intersecting polygons.
                                     Defaults to 'Developed space'.

    Returns:
        numpy.ndarray: 2D grid of land cover classes as strings

    The function creates a regular grid over the given rectangle area and determines the dominant
    land cover class for each cell based on polygon intersections. Classes are assigned based on
    priority rules and majority area coverage.
    """

    # Default priority mapping for land cover classes (lower number = higher priority)
    class_priority = { 
        'Bareland': 4, 
        'Rangeland': 6, 
        'Developed space': 8, 
        'Road': 1,  # Roads have highest priority
        'Tree': 7, 
        'Water': 3, 
        'Agriculture land': 5, 
        'Building': 2  # Buildings have second highest priority
    }

    # Get source-specific priority mapping if available
    class_priority = get_class_priority(source)
    
    # Calculate grid dimensions and normalize direction vectors
    geod = initialize_geod()
    vertex_0, vertex_1, vertex_3 = rectangle_vertices[0], rectangle_vertices[1], rectangle_vertices[3]

    # Calculate actual distances between vertices using geodesic calculations
    dist_side_1 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_1[0], vertex_1[1])
    dist_side_2 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_3[0], vertex_3[1])

    # Create vectors representing the sides of the rectangle
    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)

    # Normalize vectors to represent 1 meter in each direction
    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)

    origin = np.array(rectangle_vertices[0])
    grid_size, adjusted_meshsize = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)  

    print(f"Adjusted mesh size: {adjusted_meshsize}")

    # Initialize grid with default land cover class
    grid = np.full(grid_size, default_class, dtype=object)

    # Calculate bounding box for spatial indexing
    extent = [min(coord[1] for coord in rectangle_vertices), max(coord[1] for coord in rectangle_vertices),
              min(coord[0] for coord in rectangle_vertices), max(coord[0] for coord in rectangle_vertices)]
    plotting_box = box(extent[2], extent[0], extent[3], extent[1])

    # Create spatial index for efficient polygon lookup
    land_cover_polygons = []
    idx = index.Index()
    for i, row in gdf.iterrows():
        polygon = row.geometry
        land_cover_class = row['class']
        land_cover_polygons.append((polygon, land_cover_class))
        idx.insert(i, polygon.bounds)

    # Iterate through each grid cell
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            land_cover_class = default_class
            cell = create_cell_polygon(origin, i, j, adjusted_meshsize, u_vec, v_vec)
            
            # Check intersections with polygons that could overlap this cell
            for k in idx.intersection(cell.bounds):
                polygon, land_cover_class_temp = land_cover_polygons[k]
                try:
                    if cell.intersects(polygon):
                        intersection = cell.intersection(polygon)
                        # If polygon covers more than 50% of cell, consider its land cover class
                        if intersection.area > cell.area/2:
                            rank = class_priority[land_cover_class]
                            rank_temp = class_priority[land_cover_class_temp]
                            # Update cell class if new class has higher priority (lower rank)
                            if rank_temp < rank:
                                land_cover_class = land_cover_class_temp
                                grid[i, j] = land_cover_class
                except GEOSException as e:
                    print(f"GEOS error at grid cell ({i}, {j}): {str(e)}")
                    # Attempt to fix invalid polygon geometry
                    try:
                        fixed_polygon = polygon.buffer(0)
                        if cell.intersects(fixed_polygon):
                            intersection = cell.intersection(fixed_polygon)
                            if intersection.area > cell.area/2:
                                rank = class_priority[land_cover_class]
                                rank_temp = class_priority[land_cover_class_temp]
                                if rank_temp < rank:
                                    land_cover_class = land_cover_class_temp
                                    grid[i, j] = land_cover_class
                    except Exception as fix_error:
                        print(f"Failed to fix polygon at grid cell ({i}, {j}): {str(fix_error)}")
                    continue 
    return grid

def create_height_grid_from_geotiff_polygon(tiff_path, mesh_size, polygon):
    """
    Create a height grid from a GeoTIFF file within a polygon boundary.
    
    Args:
        tiff_path (str): Path to GeoTIFF file
        mesh_size (float): Size of mesh cells
        polygon (list): List of polygon vertices
        
    Returns:
        numpy.ndarray: Grid of heights within the polygon
    """
    with rasterio.open(tiff_path) as src:
        # Read height data
        img = src.read(1)
        left, bottom, right, top = src.bounds
        src_crs = src.crs

        # Create polygon from input coordinates
        poly = Polygon(polygon)
        
        # Get polygon bounds in WGS84
        left_wgs84, bottom_wgs84, right_wgs84, top_wgs84 = poly.bounds
        # print(left, bottom, right, top)
        # print(left_wgs84, bottom_wgs84, right_wgs84, top_wgs84)

        # Calculate actual distances using geodesic methods
        geod = Geod(ellps="WGS84")
        _, _, width = geod.inv(left_wgs84, bottom_wgs84, right_wgs84, bottom_wgs84)
        _, _, height = geod.inv(left_wgs84, bottom_wgs84, left_wgs84, top_wgs84)

        # Calculate grid dimensions and adjust mesh size
        num_cells_x = int(width / mesh_size + 0.5)
        num_cells_y = int(height / mesh_size + 0.5)

        adjusted_mesh_size_x = (right - left) / num_cells_x
        adjusted_mesh_size_y = (top - bottom) / num_cells_y

        # Create affine transform for coordinate mapping
        new_affine = Affine(adjusted_mesh_size_x, 0, left, 0, -adjusted_mesh_size_y, top)

        # Generate coordinate grids
        cols, rows = np.meshgrid(np.arange(num_cells_x), np.arange(num_cells_y))
        xs, ys = new_affine * (cols, rows)
        xs_flat, ys_flat = xs.flatten(), ys.flatten()

        # Convert to image coordinates
        row, col = src.index(xs_flat, ys_flat)
        row, col = np.array(row), np.array(col)

        # Filter valid indices
        valid = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)
        row, col = row[valid], col[valid]

        # Create output grid and fill with height values
        grid = np.full((num_cells_y, num_cells_x), np.nan)
        flat_indices = np.ravel_multi_index((row, col), img.shape)
        np.put(grid, np.ravel_multi_index((rows.flatten()[valid], cols.flatten()[valid]), grid.shape), img.flat[flat_indices])

    return np.flipud(grid)

def create_building_height_grid_from_gdf_polygon(
    gdf,
    meshsize,
    rectangle_vertices,
    overlapping_footprint=False,
    gdf_comp=None,
    geotiff_path_comp=None,
    complement_building_footprints=None,
    complement_height=None
):
    """
    Create a building height grid from GeoDataFrame data within a polygon boundary.
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame containing building information
        meshsize (float): Size of mesh cells
        rectangle_vertices (list): List of rectangle vertices defining the boundary
        overlapping_footprint (bool): If True, use precise geometry-based processing for overlaps.
                                    If False, use faster rasterio-based approach.
        gdf_comp (geopandas.GeoDataFrame, optional): Complementary GeoDataFrame
        geotiff_path_comp (str, optional): Path to complementary GeoTIFF file
        complement_building_footprints (bool, optional): Whether to complement footprints
        complement_height (float, optional): Height value to use for buildings with height=0
        
    Returns:
        tuple: (building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings)
            - building_height_grid (numpy.ndarray): Grid of building heights
            - building_min_height_grid (numpy.ndarray): Grid of min building heights (list per cell)
            - building_id_grid (numpy.ndarray): Grid of building IDs
            - filtered_buildings (geopandas.GeoDataFrame): The buildings used (filtered_gdf)
    """
    # --------------------------------------------------------------------------
    # 1) COMMON INITIAL SETUP AND DATA FILTERING
    # --------------------------------------------------------------------------
    geod = initialize_geod()
    vertex_0, vertex_1, vertex_3 = rectangle_vertices[0], rectangle_vertices[1], rectangle_vertices[3]
    
    # Distances for each side
    dist_side_1 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_1[0], vertex_1[1])
    dist_side_2 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_3[0], vertex_3[1])
    
    # Normalized vectors
    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)
    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)
    
    # Grid parameters
    origin = np.array(rectangle_vertices[0])
    grid_size, adjusted_meshsize = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)
    
    # Filter the input GeoDataFrame by bounding box
    extent = [
        min(coord[1] for coord in rectangle_vertices),
        max(coord[1] for coord in rectangle_vertices),
        min(coord[0] for coord in rectangle_vertices),
        max(coord[0] for coord in rectangle_vertices)
    ]
    plotting_box = box(extent[2], extent[0], extent[3], extent[1])
    filtered_gdf = gdf[gdf.geometry.intersects(plotting_box)].copy()
    
    # Count buildings with height=0 or NaN
    zero_height_count = len(filtered_gdf[filtered_gdf['height'] == 0])
    nan_height_count = len(filtered_gdf[filtered_gdf['height'].isna()])
    print(f"{zero_height_count+nan_height_count} of the total {len(filtered_gdf)} building footprint from the base data source did not have height data.")
    
    # Optionally merge heights from complementary sources
    if gdf_comp is not None:
        filtered_gdf_comp = gdf_comp[gdf_comp.geometry.intersects(plotting_box)].copy()
        if complement_building_footprints:
            filtered_gdf = complement_building_heights_from_gdf(filtered_gdf, filtered_gdf_comp)
        else:
            filtered_gdf = extract_building_heights_from_gdf(filtered_gdf, filtered_gdf_comp)
    elif geotiff_path_comp:
        filtered_gdf = extract_building_heights_from_geotiff(geotiff_path_comp, filtered_gdf)
    
    # After filtering and complementing heights, process overlapping buildings
    filtered_gdf = process_building_footprints_by_overlap(filtered_gdf, overlap_threshold=0.5)
    
    # --------------------------------------------------------------------------
    # 2) BRANCH BASED ON OVERLAPPING_FOOTPRINT PARAMETER
    # --------------------------------------------------------------------------
    
    if overlapping_footprint:
        # Use precise geometry-based approach for better overlap handling
        return _process_with_geometry_intersection(
            filtered_gdf, grid_size, adjusted_meshsize, origin, u_vec, v_vec, complement_height
        )
    else:
        # Use faster rasterio-based approach
        return _process_with_rasterio(
            filtered_gdf, grid_size, adjusted_meshsize, origin, u_vec, v_vec, 
            rectangle_vertices, complement_height
        )


def _process_with_geometry_intersection(filtered_gdf, grid_size, adjusted_meshsize, origin, u_vec, v_vec, complement_height):
    """
    Process buildings using precise geometry intersection approach.
    Better for handling overlapping footprints but slower.
    """
    # Initialize output grids
    building_height_grid = np.zeros(grid_size)
    building_id_grid = np.zeros(grid_size)
    
    # Use a Python list-of-lists or object array for min_height tracking
    building_min_height_grid = np.empty(grid_size, dtype=object)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            building_min_height_grid[i, j] = []
    
    # --------------------------------------------------------------------------
    # PREPARE BUILDING POLYGONS & SPATIAL INDEX
    # --------------------------------------------------------------------------
    building_polygons = []
    for idx_b, row in filtered_gdf.iterrows():
        polygon = row.geometry
        height = row.get('height', None)
        
        # Replace height=0 with complement_height if specified
        if complement_height is not None and (height == 0 or height is None):
            height = complement_height
            
        min_height = row.get('min_height', 0)
        if pd.isna(min_height):
            min_height = 0
            
        is_inner = row.get('is_inner', False)
        feature_id = row.get('id', idx_b)
        
        # Fix invalid geometry
        if not polygon.is_valid:
            try:
                polygon = polygon.buffer(0)
                if not polygon.is_valid:
                    polygon = polygon.simplify(1e-8)
            except Exception as e:
                pass
                
        bounding_box = polygon.bounds  # (minx, miny, maxx, maxy)
        building_polygons.append((
            polygon, bounding_box, height, min_height, is_inner, feature_id
        ))
    
    # Build R-tree index using bounding boxes
    idx = index.Index()
    for i_b, (poly, bbox, _, _, _, _) in enumerate(building_polygons):
        idx.insert(i_b, bbox)
    
    # --------------------------------------------------------------------------
    # MAIN GRID LOOP WITH PRECISE INTERSECTION
    # --------------------------------------------------------------------------
    INTERSECTION_THRESHOLD = 0.3
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Create the cell polygon once
            cell = create_cell_polygon(origin, i, j, adjusted_meshsize, u_vec, v_vec)
            if not cell.is_valid:
                cell = cell.buffer(0)
            cell_area = cell.area
            
            # Find possible intersections from the index
            potential = list(idx.intersection(cell.bounds))
            if not potential:
                continue
            
            # Sort buildings by height descending
            cell_buildings = []
            for k in potential:
                bpoly, bbox, height, minh, inr, fid = building_polygons[k]
                sort_val = height if (height is not None) else -float('inf')
                cell_buildings.append((k, bpoly, bbox, height, minh, inr, fid, sort_val))
            cell_buildings.sort(key=lambda x: x[-1], reverse=True)
            
            found_intersection = False
            all_zero_or_nan = True
            
            for (k, polygon, bbox, height, min_height, is_inner, feature_id, _) in cell_buildings:
                try:
                    # Quick bounding-box check
                    minx_p, miny_p, maxx_p, maxy_p = bbox
                    minx_c, miny_c, maxx_c, maxy_c = cell.bounds
                    
                    # Overlap bounding box
                    overlap_minx = max(minx_p, minx_c)
                    overlap_miny = max(miny_p, miny_c)
                    overlap_maxx = min(maxx_p, maxx_c)
                    overlap_maxy = min(maxy_p, maxy_c)
                    
                    if (overlap_maxx <= overlap_minx) or (overlap_maxy <= overlap_miny):
                        continue
                    
                    # Area of bounding-box intersection
                    bbox_intersect_area = (overlap_maxx - overlap_minx) * (overlap_maxy - overlap_miny)
                    if bbox_intersect_area < INTERSECTION_THRESHOLD * cell_area:
                        continue
                    
                    # Ensure valid geometry
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
                    
                    if cell.intersects(polygon):
                        intersection = cell.intersection(polygon)
                        inter_area = intersection.area
                        
                        # If the fraction of cell covered > threshold
                        if (inter_area / cell_area) > INTERSECTION_THRESHOLD:
                            found_intersection = True
                            
                            # If not an inner courtyard
                            if not is_inner:
                                building_min_height_grid[i, j].append([min_height, height])
                                building_id_grid[i, j] = feature_id
                                
                                # Update building height if valid
                                if (height is not None and not np.isnan(height) and height > 0):
                                    all_zero_or_nan = False
                                    current_height = building_height_grid[i, j]
                                    
                                    # Replace if we had 0, nan, or smaller height
                                    if (current_height == 0 or np.isnan(current_height) or current_height < height):
                                        building_height_grid[i, j] = height
                            else:
                                # Inner courtyards => override with 0
                                building_min_height_grid[i, j] = [[0, 0]]
                                building_height_grid[i, j] = 0
                                found_intersection = True
                                all_zero_or_nan = False
                                break
                                
                except (GEOSException, ValueError) as e:
                    # Attempt fallback fix
                    try:
                        simplified_polygon = polygon.simplify(1e-8)
                        if simplified_polygon.is_valid:
                            intersection = cell.intersection(simplified_polygon)
                            inter_area = intersection.area
                            if (inter_area / cell_area) > INTERSECTION_THRESHOLD:
                                found_intersection = True
                                if not is_inner:
                                    building_min_height_grid[i, j].append([min_height, height])
                                    building_id_grid[i, j] = feature_id
                                    if (height is not None and not np.isnan(height) and height > 0):
                                        all_zero_or_nan = False
                                        if (building_height_grid[i, j] == 0 or 
                                            np.isnan(building_height_grid[i, j]) or 
                                            building_height_grid[i, j] < height):
                                            building_height_grid[i, j] = height
                                else:
                                    building_min_height_grid[i, j] = [[0, 0]]
                                    building_height_grid[i, j] = 0
                                    found_intersection = True
                                    all_zero_or_nan = False
                                    break
                    except Exception as fix_error:
                        print(f"Failed to process cell ({i}, {j}) - Building {k}: {str(fix_error)}")
                        continue
            
            # If we found intersecting buildings but all were zero/NaN, mark as NaN
            if found_intersection and all_zero_or_nan:
                building_height_grid[i, j] = np.nan
    
    return building_height_grid, building_min_height_grid, building_id_grid, filtered_gdf


def _process_with_rasterio(filtered_gdf, grid_size, adjusted_meshsize, origin, u_vec, v_vec, rectangle_vertices, complement_height):
    """
    Process buildings using fast rasterio-based approach.
    Faster but less precise for overlapping footprints.
    """
    # Set up transform for rasterio using rotated basis defined by u_vec and v_vec
    # Step vectors in coordinate units (degrees) per cell
    u_step = adjusted_meshsize[0] * u_vec
    v_step = adjusted_meshsize[1] * v_vec

    # Define the top-left corner so that row=0 is the northern edge
    top_left = origin + grid_size[1] * v_step

    # Affine transform mapping (col, row) -> (x, y)
    # x = a*col + b*row + c ; y = d*col + e*row + f
    # col increases along u_step; row increases southward, hence -v_step
    transform = Affine(u_step[0], -v_step[0], top_left[0],
                       u_step[1], -v_step[1], top_left[1])
    
    # Process buildings data
    filtered_gdf = filtered_gdf.copy()
    if complement_height is not None:
        mask = (filtered_gdf['height'] == 0) | (filtered_gdf['height'].isna())
        filtered_gdf.loc[mask, 'height'] = complement_height
    
    # Add missing columns with defaults
    filtered_gdf['min_height'] = 0
    
    if 'is_inner' not in filtered_gdf.columns:
        filtered_gdf['is_inner'] = False
    
    if 'id' not in filtered_gdf.columns:
        filtered_gdf['id'] = range(len(filtered_gdf))
    
    # Sort by height for proper layering
    regular_buildings = filtered_gdf[~filtered_gdf['is_inner']].copy()
    regular_buildings = regular_buildings.sort_values('height', ascending=True, na_position='first')
    
    # Temporary raster grids in rasterio's (rows=ny, cols=nx) order
    height_raster = np.zeros((grid_size[1], grid_size[0]), dtype=np.float64)
    id_raster = np.zeros((grid_size[1], grid_size[0]), dtype=np.float64)
    
    # Vectorized rasterization
    if len(regular_buildings) > 0:
        valid_buildings = regular_buildings[regular_buildings.geometry.is_valid].copy()
        
        if len(valid_buildings) > 0:
            # Height grid
            height_shapes = [(mapping(geom), height) for geom, height in 
                           zip(valid_buildings.geometry, valid_buildings['height']) 
                           if pd.notna(height) and height > 0]
            
            if height_shapes:
                height_raster = features.rasterize(
                    height_shapes,
                    out_shape=(grid_size[1], grid_size[0]),
                    transform=transform,
                    fill=0,
                    dtype=np.float64
                )
            
            # ID grid  
            id_shapes = [(mapping(geom), id_val) for geom, id_val in 
                        zip(valid_buildings.geometry, valid_buildings['id'])]
            
            if id_shapes:
                id_raster = features.rasterize(
                    id_shapes,
                    out_shape=(grid_size[1], grid_size[0]),
                    transform=transform,
                    fill=0,
                    dtype=np.float64
                )
    
    # Handle inner courtyards
    inner_buildings = filtered_gdf[filtered_gdf['is_inner']].copy()
    if len(inner_buildings) > 0:
        inner_shapes = [(mapping(geom), 1) for geom in inner_buildings.geometry if geom.is_valid]
        if inner_shapes:
            inner_mask = features.rasterize(
                inner_shapes,
                out_shape=(grid_size[1], grid_size[0]),
                transform=transform,
                fill=0,
                dtype=np.uint8
            )
            height_raster[inner_mask > 0] = 0
            id_raster[inner_mask > 0] = 0
    
    # Simplified min_height grid 
    building_min_height_grid = np.empty(grid_size, dtype=object)
    min_heights_raster = np.zeros((grid_size[1], grid_size[0]), dtype=np.float64)
    
    if len(regular_buildings) > 0:
        valid_buildings = regular_buildings[regular_buildings.geometry.is_valid].copy()
        if len(valid_buildings) > 0:
            min_height_shapes = [(mapping(geom), min_h) for geom, min_h in 
                               zip(valid_buildings.geometry, valid_buildings['min_height']) 
                               if pd.notna(min_h)]
            
            if min_height_shapes:
                min_heights_raster = features.rasterize(
                    min_height_shapes,
                    out_shape=(grid_size[1], grid_size[0]),
                    transform=transform,
                    fill=0,
                    dtype=np.float64
                )
    
    # Convert to list format (simplified)
    # Convert raster (ny, nx) to internal orientation (nx, ny) with north-up
    building_height_grid = np.flipud(height_raster).T
    building_id_grid = np.flipud(id_raster).T
    min_heights = np.flipud(min_heights_raster).T

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if building_height_grid[i, j] > 0:
                building_min_height_grid[i, j] = [[min_heights[i, j], building_height_grid[i, j]]]
            else:
                building_min_height_grid[i, j] = []

    return building_height_grid, building_min_height_grid, building_id_grid, filtered_gdf

def create_building_height_grid_from_open_building_temporal_polygon(meshsize, rectangle_vertices, output_dir):        
    """
    Create a building height grid from OpenBuildings temporal data within a polygon.
    
    Args:
        meshsize (float): Size of mesh cells
        rectangle_vertices (list): List of rectangle vertices defining the boundary
        output_dir (str): Directory to save intermediate GeoTIFF files
        
    Returns:
        tuple: (building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings)
    """
    # Get region of interest from vertices
    roi = get_roi(rectangle_vertices)
    
    # Create output directory and save intermediate GeoTIFF
    os.makedirs(output_dir, exist_ok=True)
    geotiff_path = os.path.join(output_dir, "building_height.tif")
    save_geotiff_open_buildings_temporal(roi, geotiff_path)
    
    # Create height grid from GeoTIFF
    building_height_grid = create_height_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices)
    
    # Initialize min height grid with appropriate height ranges
    building_min_height_grid = np.empty(building_height_grid.shape, dtype=object)
    for i in range(building_height_grid.shape[0]):
        for j in range(building_height_grid.shape[1]):
            if building_height_grid[i, j] <= 0:
                building_min_height_grid[i, j] = []
            else:
                building_min_height_grid[i, j] = [[0, building_height_grid[i, j]]]
    
    # Create building ID grid with sequential numbering for non-zero heights
    filtered_buildings = gpd.GeoDataFrame()
    building_id_grid = np.zeros_like(building_height_grid, dtype=int)        
    non_zero_positions = np.nonzero(building_height_grid)        
    sequence = np.arange(1, len(non_zero_positions[0]) + 1)        
    building_id_grid[non_zero_positions] = sequence

    return building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings

def create_dem_grid_from_geotiff_polygon(tiff_path, mesh_size, rectangle_vertices, dem_interpolation=False):
    """
    Create a Digital Elevation Model (DEM) grid from a GeoTIFF file within a polygon boundary.
    
    Args:
        tiff_path (str): Path to GeoTIFF file
        mesh_size (float): Size of mesh cells
        rectangle_vertices (list): List of rectangle vertices defining the boundary
        dem_interpolation (bool): Whether to use cubic interpolation for smoother results
        
    Returns:
        numpy.ndarray: Grid of elevation values
    """
    # Convert vertex coordinates to lat/lon format
    converted_coords = convert_format_lat_lon(rectangle_vertices)
    roi_shapely = Polygon(converted_coords)

    with rasterio.open(tiff_path) as src:
        # Read DEM data and handle no-data values
        dem = src.read(1)
        dem = np.where(dem < -1000, 0, dem)  # Replace extreme negative values with 0
        transform = src.transform
        src_crs = src.crs

        # Handle coordinate system conversion
        if src_crs.to_epsg() != 3857:
            transformer_to_3857 = Transformer.from_crs(src_crs, CRS.from_epsg(3857), always_xy=True)
        else:
            transformer_to_3857 = lambda x, y: (x, y)

        # Transform ROI bounds to EPSG:3857 (Web Mercator)
        roi_bounds = roi_shapely.bounds
        roi_left, roi_bottom = transformer_to_3857.transform(roi_bounds[0], roi_bounds[1])
        roi_right, roi_top = transformer_to_3857.transform(roi_bounds[2], roi_bounds[3])

        # Convert to WGS84 for accurate distance calculations
        wgs84 = CRS.from_epsg(4326)
        transformer_to_wgs84 = Transformer.from_crs(CRS.from_epsg(3857), wgs84, always_xy=True)
        roi_left_wgs84, roi_bottom_wgs84 = transformer_to_wgs84.transform(roi_left, roi_bottom)
        roi_right_wgs84, roi_top_wgs84 = transformer_to_wgs84.transform(roi_right, roi_top)

        # Calculate actual distances using geodesic methods
        geod = Geod(ellps="WGS84")
        _, _, roi_width_m = geod.inv(roi_left_wgs84, roi_bottom_wgs84, roi_right_wgs84, roi_bottom_wgs84)
        _, _, roi_height_m = geod.inv(roi_left_wgs84, roi_bottom_wgs84, roi_left_wgs84, roi_top_wgs84)

        # Calculate grid dimensions
        num_cells_x = int(roi_width_m / mesh_size + 0.5)
        num_cells_y = int(roi_height_m / mesh_size + 0.5)

        # Create coordinate grid in EPSG:3857
        x = np.linspace(roi_left, roi_right, num_cells_x, endpoint=False)
        y = np.linspace(roi_top, roi_bottom, num_cells_y, endpoint=False)
        xx, yy = np.meshgrid(x, y)

        # Transform original DEM coordinates to EPSG:3857
        rows, cols = np.meshgrid(range(dem.shape[0]), range(dem.shape[1]), indexing='ij')
        orig_x, orig_y = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
        orig_x, orig_y = transformer_to_3857.transform(orig_x, orig_y)

        # Interpolate DEM values onto new grid
        points = np.column_stack((orig_x, orig_y))
        values = dem.ravel()
        if dem_interpolation:
            # Use cubic interpolation for smoother results
            grid = griddata(points, values, (xx, yy), method='cubic')
        else:
            # Use nearest neighbor interpolation for raw data
            grid = griddata(points, values, (xx, yy), method='nearest')

    return np.flipud(grid)

def grid_to_geodataframe(grid_ori, rectangle_vertices, meshsize):
    """
    Converts a 2D grid to a GeoDataFrame with cell polygons and values.
    
    This function transforms a regular grid into a GeoDataFrame where each cell is
    represented as a polygon. The transformation handles coordinate systems properly,
    converting between WGS84 (EPSG:4326) and Web Mercator (EPSG:3857) for accurate
    distance calculations.
    
    Args:
        grid_ori (numpy.ndarray): 2D array containing grid values
        rectangle_vertices (list): List of [lon, lat] coordinates defining area corners.
                                 Should be in WGS84 (EPSG:4326) format.
        meshsize (float): Size of each grid cell in meters
        
    Returns:
        GeoDataFrame: A GeoDataFrame with columns:
            - geometry: Polygon geometry of each grid cell in WGS84 (EPSG:4326)
            - value: Value from the original grid
            
    Example:
        >>> grid = np.array([[1, 2], [3, 4]])
        >>> vertices = [[lon1, lat1], [lon2, lat2], [lon3, lat3], [lon4, lat4]]
        >>> mesh_size = 100  # 100 meters
        >>> gdf = grid_to_geodataframe(grid, vertices, mesh_size)
    
    Notes:
        - The input grid is flipped vertically before processing to match geographic
          orientation (north at top)
        - The output GeoDataFrame uses WGS84 (EPSG:4326) coordinate system
    """
    grid = np.flipud(grid_ori.copy())
    
    # Extract bounds from rectangle vertices
    min_lon = min(v[0] for v in rectangle_vertices)
    max_lon = max(v[0] for v in rectangle_vertices)
    min_lat = min(v[1] for v in rectangle_vertices)
    max_lat = max(v[1] for v in rectangle_vertices)
    
    rows, cols = grid.shape
    
    # Set up transformers for accurate coordinate calculations
    wgs84 = CRS.from_epsg(4326)
    web_mercator = CRS.from_epsg(3857)
    transformer_to_mercator = Transformer.from_crs(wgs84, web_mercator, always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs(web_mercator, wgs84, always_xy=True)
    
    # Convert bounds to Web Mercator for accurate distance calculations
    min_x, min_y = transformer_to_mercator.transform(min_lon, min_lat)
    max_x, max_y = transformer_to_mercator.transform(max_lon, max_lat)
    
    # Calculate cell sizes in Web Mercator coordinates
    cell_size_x = (max_x - min_x) / cols
    cell_size_y = (max_y - min_y) / rows
    
    # Create lists to store data
    polygons = []
    values = []
    
    # Create grid cells
    for i in range(rows):
        for j in range(cols):
            # Calculate cell bounds in Web Mercator
            cell_min_x = min_x + j * cell_size_x
            cell_max_x = min_x + (j + 1) * cell_size_x
            # Flip vertical axis since grid is stored with origin at top-left
            cell_min_y = max_y - (i + 1) * cell_size_y
            cell_max_y = max_y - i * cell_size_y
            
            # Convert cell corners back to WGS84
            cell_min_lon, cell_min_lat = transformer_to_wgs84.transform(cell_min_x, cell_min_y)
            cell_max_lon, cell_max_lat = transformer_to_wgs84.transform(cell_max_x, cell_max_y)
            
            # Create polygon for cell
            cell_poly = box(cell_min_lon, cell_min_lat, cell_max_lon, cell_max_lat)
            
            polygons.append(cell_poly)
            values.append(grid[i, j])
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'geometry': polygons,
        'value': values
    }, crs=CRS.from_epsg(4326))
    
    return gdf

def grid_to_point_geodataframe(grid_ori, rectangle_vertices, meshsize):
    """
    Converts a 2D grid to a GeoDataFrame with point geometries at cell centers and values.
    
    This function transforms a regular grid into a GeoDataFrame where each cell is
    represented by a point at its center. The transformation handles coordinate systems
    properly, converting between WGS84 (EPSG:4326) and Web Mercator (EPSG:3857) for
    accurate distance calculations.
    
    Args:
        grid_ori (numpy.ndarray): 2D array containing grid values
        rectangle_vertices (list): List of [lon, lat] coordinates defining area corners.
                                 Should be in WGS84 (EPSG:4326) format.
        meshsize (float): Size of each grid cell in meters
        
    Returns:
        GeoDataFrame: A GeoDataFrame with columns:
            - geometry: Point geometry at center of each grid cell in WGS84 (EPSG:4326)
            - value: Value from the original grid
            
    Example:
        >>> grid = np.array([[1, 2], [3, 4]])
        >>> vertices = [[lon1, lat1], [lon2, lat2], [lon3, lat3], [lon4, lat4]]
        >>> mesh_size = 100  # 100 meters
        >>> gdf = grid_to_point_geodataframe(grid, vertices, mesh_size)
    
    Notes:
        - The input grid is flipped vertically before processing to match geographic
          orientation (north at top)
        - The output GeoDataFrame uses WGS84 (EPSG:4326) coordinate system
        - Points are placed at the center of each grid cell
    """
    grid = np.flipud(grid_ori.copy())
    
    # Extract bounds from rectangle vertices
    min_lon = min(v[0] for v in rectangle_vertices)
    max_lon = max(v[0] for v in rectangle_vertices)
    min_lat = min(v[1] for v in rectangle_vertices)
    max_lat = max(v[1] for v in rectangle_vertices)
    
    rows, cols = grid.shape
    
    # Set up transformers for accurate coordinate calculations
    wgs84 = CRS.from_epsg(4326)
    web_mercator = CRS.from_epsg(3857)
    transformer_to_mercator = Transformer.from_crs(wgs84, web_mercator, always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs(web_mercator, wgs84, always_xy=True)
    
    # Convert bounds to Web Mercator for accurate distance calculations
    min_x, min_y = transformer_to_mercator.transform(min_lon, min_lat)
    max_x, max_y = transformer_to_mercator.transform(max_lon, max_lat)
    
    # Calculate cell sizes in Web Mercator coordinates
    cell_size_x = (max_x - min_x) / cols
    cell_size_y = (max_y - min_y) / rows
    
    # Create lists to store data
    points = []
    values = []
    
    # Create grid points at cell centers
    for i in range(rows):
        for j in range(cols):
            # Calculate cell center in Web Mercator
            cell_center_x = min_x + (j + 0.5) * cell_size_x
            # Flip vertical axis since grid is stored with origin at top-left
            cell_center_y = max_y - (i + 0.5) * cell_size_y
            
            # Convert cell center back to WGS84
            center_lon, center_lat = transformer_to_wgs84.transform(cell_center_x, cell_center_y)
            
            # Create point for cell center
            from shapely.geometry import Point
            cell_point = Point(center_lon, center_lat)
            
            points.append(cell_point)
            values.append(grid[i, j])
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'geometry': points,
        'value': values
    }, crs=CRS.from_epsg(4326))
    
    return gdf

def create_vegetation_height_grid_from_gdf_polygon(veg_gdf, mesh_size, polygon):
    """
    Create a vegetation height grid from a GeoDataFrame of vegetation polygons/objects
    within the bounding box of a given polygon, at a specified mesh spacing.
    Cells that intersect one or more vegetation polygons receive the
    (by default) maximum vegetation height among intersecting polygons.
    Cells that do not intersect any vegetation are set to 0.

    Args:
        veg_gdf (GeoDataFrame): A GeoDataFrame containing vegetation features
                                (usually polygons) with a 'height' column
                                (or a similarly named attribute). Must be in
                                EPSG:4326 or reprojectable to it.
        mesh_size (float):      Desired grid spacing in meters.
        polygon (list or Polygon):
          - If a list of (lon, lat) coords, will be converted to a shapely Polygon
            in EPSG:4326.
          - If a shapely Polygon, it must be in or reprojectable to EPSG:4326.

    Returns:
        np.ndarray: 2D array of vegetation height values covering the bounding box
                    of the polygon. The array is indexed [row, col] from top row
                    (north) to bottom row (south). Cells with no intersecting
                    vegetation are set to 0.
    """
    # ------------------------------------------------------------------------
    # 1. Ensure veg_gdf is in WGS84 (EPSG:4326)
    # ------------------------------------------------------------------------
    if veg_gdf.crs is None:
        warnings.warn("veg_gdf has no CRS. Assuming EPSG:4326. "
                      "If this is incorrect, please set the correct CRS and re-run.")
        veg_gdf = veg_gdf.set_crs(epsg=4326)
    else:
        if veg_gdf.crs.to_epsg() != 4326:
            veg_gdf = veg_gdf.to_crs(epsg=4326)

    # Must have a 'height' column (or change to your column name)
    if 'height' not in veg_gdf.columns:
        raise ValueError("Vegetation GeoDataFrame must have a 'height' column.")

    # ------------------------------------------------------------------------
    # 2. Convert input polygon to shapely Polygon in WGS84
    # ------------------------------------------------------------------------
    if isinstance(polygon, list):
        poly = Polygon(polygon)
    elif isinstance(polygon, Polygon):
        poly = polygon
    else:
        raise ValueError("polygon must be a list of (lon, lat) or a shapely Polygon.")

    # ------------------------------------------------------------------------
    # 3. Compute bounding box & grid dimensions
    # ------------------------------------------------------------------------
    left, bottom, right, top = poly.bounds
    geod = Geod(ellps="WGS84")

    # Horizontal (width) distance in meters
    _, _, width_m = geod.inv(left, bottom, right, bottom)
    # Vertical (height) distance in meters
    _, _, height_m = geod.inv(left, bottom, left, top)

    # Number of cells horizontally and vertically
    num_cells_x = int(width_m / mesh_size + 0.5)
    num_cells_y = int(height_m / mesh_size + 0.5)

    if num_cells_x < 1 or num_cells_y < 1:
        warnings.warn("Polygon bounding box is smaller than mesh_size; returning empty array.")
        return np.array([])

    # ------------------------------------------------------------------------
    # 4. Generate the grid (cell centers) covering the bounding box
    # ------------------------------------------------------------------------
    xs = np.linspace(left, right, num_cells_x)
    ys = np.linspace(top, bottom, num_cells_y)  # top→bottom
    X, Y = np.meshgrid(xs, ys)

    # Flatten these for convenience
    xs_flat = X.ravel()
    ys_flat = Y.ravel()

    # Create cell-center points as a GeoDataFrame
    grid_points = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in zip(xs_flat, ys_flat)],
        crs="EPSG:4326"
    )

    # ------------------------------------------------------------------------
    # 5. Spatial join (INTERSECTION) to find which vegetation objects each cell intersects
    #    - We only fill the cell if the point is actually inside (or intersects) a vegetation polygon
    #      If your data is more consistent with "contains" or "within", adjust the predicate accordingly.
    # ------------------------------------------------------------------------
    # NOTE:
    #   * If your vegetation is polygons, "predicate='intersects'" or "contains"
    #     can be used. Typically we check whether the cell center is inside the polygon.
    #   * If your vegetation is a point layer, you might do "predicate='within'"
    #     or similar. Adjust as needed.
    #
    # We'll do a left join so that unmatched cells remain in the result with NaN values.
    # Then we group by the index of the original grid_points to handle multiple intersects.
    # The 'index_right' is from the vegetation layer.
    # ------------------------------------------------------------------------

    joined = gpd.sjoin(
        grid_points,
        veg_gdf[['height', 'geometry']],
        how='left',
        predicate='intersects'
    )

    # Because one cell (row in grid_points) can intersect multiple polygons,
    # we need to aggregate them. We'll take the *maximum* height by default.
    joined_agg = (
        joined.groupby(joined.index)   # group by the index from grid_points
              .agg({'height': 'max'})  # or 'mean' if you prefer an average
    )

    # joined_agg is now a DataFrame with the same index as grid_points.
    # If a row didn't intersect any polygon, 'height' is NaN.

    # ------------------------------------------------------------------------
    # 6. Build the 2D height array, initializing with zeros
    # ------------------------------------------------------------------------
    veg_grid = np.zeros((num_cells_y, num_cells_x), dtype=float)

    # The row, col in the final array corresponds to how we built 'grid_points':
    #   row = i // num_cells_x
    #   col = i % num_cells_x
    for i, row_data in joined_agg.iterrows():
        if not np.isnan(row_data['height']):  # Only set values for cells with vegetation
            row_idx = i // num_cells_x
            col_idx = i % num_cells_x
            veg_grid[row_idx, col_idx] = row_data['height']

    # Result: row=0 is the top-most row, row=-1 is bottom.
    return np.flipud(veg_grid)

def create_dem_grid_from_gdf_polygon(terrain_gdf, mesh_size, polygon):
    """
    Create a height grid from a terrain GeoDataFrame within the bounding box
    of the given polygon, using nearest-neighbor sampling of elevations.
    Edges of the bounding box will also receive a nearest elevation,
    so there should be no NaNs around edges if data coverage is sufficient.

    Args:
        terrain_gdf (GeoDataFrame): A GeoDataFrame containing terrain features
                                    (points or centroids) with an 'elevation' column.
                                    Must be in EPSG:4326 or reprojectable to it.
        mesh_size (float):          Desired grid spacing in meters.
        polygon (list or Polygon):  Polygon specifying the region of interest.
                                    - If list of (lon, lat), will be made into a Polygon.
                                    - If a shapely Polygon, must be in WGS84 (EPSG:4326)
                                      or reprojected to it.

    Returns:
        np.ndarray: 2D array of height values covering the bounding box of the polygon,
                    from top row (north) to bottom row (south). Any location not
                    matched by terrain_gdf data remains NaN, but edges will not
                    automatically be NaN if terrain coverage exists.
    """

    # ------------------------------------------------------------------------
    # 1. Ensure terrain_gdf is in WGS84 (EPSG:4326)
    # ------------------------------------------------------------------------
    if terrain_gdf.crs is None:
        warnings.warn("terrain_gdf has no CRS. Assuming EPSG:4326. "
                      "If this is incorrect, please set the correct CRS and re-run.")
        terrain_gdf = terrain_gdf.set_crs(epsg=4326)
    else:
        # Reproject if needed
        if terrain_gdf.crs.to_epsg() != 4326:
            terrain_gdf = terrain_gdf.to_crs(epsg=4326)

    # Convert input polygon to shapely Polygon in WGS84
    if isinstance(polygon, list):
        poly = Polygon(polygon)  # assume coords are (lon, lat) in EPSG:4326
    elif isinstance(polygon, Polygon):
        poly = polygon
    else:
        raise ValueError("`polygon` must be a list of (lon, lat) or a shapely Polygon.")

    # ------------------------------------------------------------------------
    # 2. Compute bounding box and number of grid cells
    # ------------------------------------------------------------------------
    left, bottom, right, top = poly.bounds
    geod = Geod(ellps="WGS84")

    # Geodesic distances in meters
    _, _, width_m = geod.inv(left, bottom, right, bottom)
    _, _, height_m = geod.inv(left, bottom, left, top)

    # Number of cells in X and Y directions
    num_cells_x = int(width_m / mesh_size + 0.5)
    num_cells_y = int(height_m / mesh_size + 0.5)

    if num_cells_x < 1 or num_cells_y < 1:
        warnings.warn("Polygon bounding box is smaller than mesh_size; returning empty array.")
        return np.array([])

    # ------------------------------------------------------------------------
    # 3. Generate grid points covering the bounding box
    #    (all points, not just inside the polygon)
    # ------------------------------------------------------------------------
    xs = np.linspace(left, right, num_cells_x)
    ys = np.linspace(top, bottom, num_cells_y)  # top→bottom
    X, Y = np.meshgrid(xs, ys)

    # Flatten for convenience
    xs_flat = X.ravel()
    ys_flat = Y.ravel()

    # Create GeoDataFrame of all bounding-box points
    grid_points = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in zip(xs_flat, ys_flat)],
        crs="EPSG:4326"
    )

    # ------------------------------------------------------------------------
    # 4. Nearest-neighbor join from terrain_gdf to grid points
    # ------------------------------------------------------------------------
    if 'elevation' not in terrain_gdf.columns:
        raise ValueError("terrain_gdf must have an 'elevation' column.")

    # Nearest spatial join (requires GeoPandas >= 0.10)
    # This will assign each grid point the nearest terrain_gdf elevation.
    grid_points_elev = gpd.sjoin_nearest(
        grid_points,
        terrain_gdf[['elevation', 'geometry']],
        how="left",
        distance_col="dist_to_terrain"
    )

    # ------------------------------------------------------------------------
    # 5. Build the final 2D height array
    #    (rows: top->bottom, columns: left->right)
    # ------------------------------------------------------------------------
    dem_grid = np.full((num_cells_y, num_cells_x), np.nan, dtype=float)

    # The index mapping of grid_points_elev is the same as grid_points, so:
    # row = i // num_cells_x, col = i % num_cells_x
    for i, elevation_val in zip(grid_points_elev.index, grid_points_elev['elevation']):
        row = i // num_cells_x
        col = i % num_cells_x
        dem_grid[row, col] = elevation_val  # could be NaN if no data

    # By default, row=0 is the "north/top" row, row=-1 is "south/bottom" row.
    # If you prefer the bottom row as index=0, you'd do: np.flipud(dem_grid)

    return np.flipud(dem_grid)