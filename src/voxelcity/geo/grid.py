"""
This module provides functions for creating and manipulating grids of building heights, land cover, and elevation data.
"""

import numpy as np
import os
from shapely.geometry import Polygon
from scipy.ndimage import label, generate_binary_structure
from pyproj import Geod, Transformer, CRS
import rasterio
from affine import Affine
from shapely.geometry import box
from scipy.interpolate import griddata
from shapely.errors import GEOSException

from .utils import (
    initialize_geod,
    calculate_distance,
    normalize_to_one_meter,
    create_building_polygons,
    convert_format_lat_lon
)
from ..file.geojson import (
    filter_buildings, 
    extract_building_heights_from_geotiff, 
    extract_building_heights_from_geojson,
    complement_building_heights_from_geojson
)
from ..utils.lc import (
    get_class_priority, 
    create_land_cover_polygons, 
    get_dominant_class,
)
from ..download.gee import (
    get_roi,
    save_geotiff_open_buildings_temporal
)

def apply_operation(arr, meshsize):
    """
    Applies a sequence of operations to an array based on a mesh size.
    
    Args:
        arr (numpy.ndarray): Input array to transform
        meshsize (float): Size of mesh to use for calculations
        
    Returns:
        numpy.ndarray: Transformed array after applying operations
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
    
    Args:
        input_array (numpy.ndarray): Array containing values to translate
        translation_dict (dict): Dictionary mapping input values to output values
        
    Returns:
        numpy.ndarray: Array with translated values
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
    Zero values remain unchanged.
    
    Args:
        array (numpy.ndarray): Input 2D array
        
    Returns:
        numpy.ndarray: Array with non-zero values converted to sequential IDs
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

def process_grid(grid_bi, dem_grid):
    """
    Process a binary grid and DEM grid to create averaged elevation values.
    
    Args:
        grid_bi (numpy.ndarray): Binary grid indicating regions
        dem_grid (numpy.ndarray): Grid of elevation values
        
    Returns:
        numpy.ndarray: Processed grid with averaged elevation values
    """
    # Get unique non-zero region IDs
    unique_ids = np.unique(grid_bi[grid_bi != 0])
    result = dem_grid.copy()
    
    # For each region, calculate mean elevation and assign to all cells in region
    for id_num in unique_ids:
        mask = (grid_bi == id_num)
        avg_value = np.mean(dem_grid[mask])
        result[mask] = avg_value
    
    # Normalize by subtracting minimum value
    return result - np.min(result)

def calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize):
    """
    Calculate grid size and adjusted mesh size based on input parameters.
    
    Args:
        side_1 (numpy.ndarray): First side vector
        side_2 (numpy.ndarray): Second side vector
        u_vec (numpy.ndarray): Unit vector in first direction
        v_vec (numpy.ndarray): Unit vector in second direction
        meshsize (float): Desired mesh size
        
    Returns:
        tuple: Grid size (tuple of ints) and adjusted mesh size (tuple of floats)
    """
    # Calculate number of cells needed in each direction, rounding to nearest integer
    grid_size_0 = int(np.linalg.norm(side_1) / np.linalg.norm(meshsize * u_vec) + 0.5)
    grid_size_1 = int(np.linalg.norm(side_2) / np.linalg.norm(meshsize * v_vec) + 0.5)
    
    # Adjust mesh sizes to exactly fit the desired area with the calculated number of cells
    adjusted_mesh_size_0 = meshsize *  np.linalg.norm(meshsize * u_vec) * grid_size_0 / np.linalg.norm(side_1)
    adjusted_mesh_size_1 = meshsize *  np.linalg.norm(meshsize * v_vec) * grid_size_1 / np.linalg.norm(side_2)
    
    return (grid_size_0, grid_size_1), (adjusted_mesh_size_0, adjusted_mesh_size_1)

def create_coordinate_mesh(origin, grid_size, adjusted_meshsize, u_vec, v_vec):
    """
    Create a coordinate mesh based on input parameters.
    
    Args:
        origin (numpy.ndarray): Origin point coordinates
        grid_size (tuple): Size of grid in each dimension
        adjusted_meshsize (tuple): Adjusted mesh size in each dimension
        u_vec (numpy.ndarray): Unit vector in first direction
        v_vec (numpy.ndarray): Unit vector in second direction
        
    Returns:
        numpy.ndarray: Coordinate mesh
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
    
    Args:
        origin (numpy.ndarray): Origin point coordinates
        i (int): Row index
        j (int): Column index
        adjusted_meshsize (tuple): Adjusted mesh size in each dimension
        u_vec (numpy.ndarray): Unit vector in first direction
        v_vec (numpy.ndarray): Unit vector in second direction
        
    Returns:
        shapely.geometry.Polygon: Polygon representing the grid cell
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
    
    Args:
        land_cover_grid_ori (numpy.ndarray): Original land cover grid
        
    Returns:
        numpy.ndarray: Grid of tree heights
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

def create_land_cover_grid_from_geotiff(tiff_path, mesh_size, land_cover_classes):
    """
    Create a land cover grid from a GeoTIFF file.
    
    Args:
        tiff_path (str): Path to GeoTIFF file
        mesh_size (float): Size of mesh cells
        land_cover_classes (dict): Dictionary mapping land cover classes
        
    Returns:
        numpy.ndarray: Grid of land cover classes
    """
    with rasterio.open(tiff_path) as src:
        # Read RGB bands
        img = src.read((1,2,3))
        left, bottom, right, top = src.bounds
        src_crs = src.crs
        
        # Handle different coordinate reference systems
        if src_crs.to_epsg() == 3857:  # Web Mercator
            # Convert bounds from Web Mercator to WGS84 for accurate distance calculations
            wgs84 = CRS.from_epsg(4326)
            transformer = Transformer.from_crs(src_crs, wgs84, always_xy=True)
            left_wgs84, bottom_wgs84 = transformer.transform(left, bottom)
            right_wgs84, top_wgs84 = transformer.transform(right, top)
        
            # Calculate actual distances using geodesic calculations
            geod = Geod(ellps="WGS84")
            _, _, width = geod.inv(left_wgs84, bottom_wgs84, right_wgs84, bottom_wgs84)
            _, _, height = geod.inv(left_wgs84, bottom_wgs84, left_wgs84, top_wgs84)
        else:
            # For projections already in meters, use simple subtraction
            width = right - left
            height = top - bottom
        
        # Calculate grid dimensions based on mesh size
        num_cells_x = int(width / mesh_size + 0.5)
        num_cells_y = int(height / mesh_size + 0.5)
        
        # Adjust mesh size to fit image exactly
        adjusted_mesh_size_x = (right - left) / num_cells_x
        adjusted_mesh_size_y = (top - bottom) / num_cells_y
        
        # Create affine transform for new grid
        new_affine = Affine(adjusted_mesh_size_x, 0, left, 0, -adjusted_mesh_size_y, top)
        
        # Create coordinate grids
        cols, rows = np.meshgrid(np.arange(num_cells_x), np.arange(num_cells_y))
        xs, ys = new_affine * (cols, rows)
        xs_flat, ys_flat = xs.flatten(), ys.flatten()
        
        # Convert coordinates to image indices
        row, col = src.index(xs_flat, ys_flat)
        row, col = np.array(row), np.array(col)
        
        # Filter out invalid indices
        valid = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)
        row, col = row[valid], col[valid]
        
        # Create output grid and fill with land cover classes
        grid = np.full((num_cells_y, num_cells_x), 'No Data', dtype=object)
        
        for i, (r, c) in enumerate(zip(row, col)):
            cell_data = img[:, r, c]
            dominant_class = get_dominant_class(cell_data, land_cover_classes)
            grid_row, grid_col = np.unravel_index(i, (num_cells_y, num_cells_x))
            grid[grid_row, grid_col] = dominant_class
    
    # Flip grid vertically before returning
    return np.flipud(grid)

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
        bottom_wgs84, left_wgs84, top_wgs84, right_wgs84 = poly.bounds
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

def create_canopy_height_grid_from_geotiff(tiff_path, mesh_size):
    """
    Create a canopy height grid from a GeoTIFF file.
    
    Args:
        tiff_path (str): Path to GeoTIFF file
        mesh_size (float): Size of mesh cells
        
    Returns:
        numpy.ndarray: Grid of canopy heights
    """
    with rasterio.open(tiff_path) as src:
        # Read single band height data
        img = src.read(1)
        left, bottom, right, top = src.bounds
        src_crs = src.crs

        # Handle coordinate system conversion and distance calculations
        if src_crs.to_epsg() == 3857:  # Web Mercator projection
            # Convert bounds to WGS84 for accurate distance calculation
            wgs84 = CRS.from_epsg(4326)
            transformer = Transformer.from_crs(src_crs, wgs84, always_xy=True)
            left_wgs84, bottom_wgs84 = transformer.transform(left, bottom)
            right_wgs84, top_wgs84 = transformer.transform(right, top)
        
            # Calculate actual distances using geodesic methods
            geod = Geod(ellps="WGS84")
            _, _, width = geod.inv(left_wgs84, bottom_wgs84, right_wgs84, bottom_wgs84)
            _, _, height = geod.inv(left_wgs84, bottom_wgs84, left_wgs84, top_wgs84)
        else:
            # For projections already in meters, use simple subtraction
            width = right - left
            height = top - bottom

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
        bottom_wgs84, left_wgs84, top_wgs84, right_wgs84 = poly.bounds
        print(left, bottom, right, top)

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

def create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices, geojson_data_comp=None, geotiff_path_comp=None, complement_building_footprints=None):
    """
    Create a building height grid from GeoJSON data within a polygon boundary.
    
    Args:
        geojson_data (dict): GeoJSON data containing building information
        meshsize (float): Size of mesh cells
        rectangle_vertices (list): List of rectangle vertices defining the boundary
        geojson_data_comp (dict, optional): Complementary GeoJSON data
        geotiff_path_comp (str, optional): Path to complementary GeoTIFF file
        complement_building_footprints (bool, optional): Whether to complement building footprints
        
    Returns:
        tuple: (building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings)
            - building_height_grid (numpy.ndarray): Grid of building heights
            - building_min_height_grid (numpy.ndarray): Grid of minimum building heights
            - building_id_grid (numpy.ndarray): Grid of building IDs
            - filtered_buildings (list): List of filtered building features
    """
    # Initialize geodesic calculator and extract vertices
    geod = initialize_geod()
    vertex_0, vertex_1, vertex_3 = rectangle_vertices[0], rectangle_vertices[1], rectangle_vertices[3]

    # Calculate distances between vertices
    dist_side_1 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_1[1], vertex_1[0])
    dist_side_2 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_3[1], vertex_3[0])

    # Calculate normalized vectors for grid orientation
    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)
    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)

    # Set up grid parameters
    origin = np.array(rectangle_vertices[0])
    grid_size, adjusted_meshsize = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)

    # Initialize output grids
    building_height_grid = np.zeros(grid_size)
    building_id_grid = np.zeros(grid_size)
    building_min_height_grid = np.empty(grid_size, dtype=object)
    
    # Initialize min height grid with empty lists
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            building_min_height_grid[i, j] = []

    # Create bounding box for filtering buildings
    extent = [min(coord[1] for coord in rectangle_vertices), max(coord[1] for coord in rectangle_vertices),
              min(coord[0] for coord in rectangle_vertices), max(coord[0] for coord in rectangle_vertices)]
    plotting_box = box(extent[2], extent[0], extent[3], extent[1])

    # Filter and process buildings
    filtered_buildings = filter_buildings(geojson_data, plotting_box)

    # Handle complementary data sources
    if geojson_data_comp:
        filtered_geojson_data_comp = filter_buildings(geojson_data_comp, plotting_box)
        if complement_building_footprints:
            filtered_buildings_comp = complement_building_heights_from_geojson(filtered_buildings, filtered_geojson_data_comp)
        else:
            filtered_buildings_comp = extract_building_heights_from_geojson(filtered_buildings, filtered_geojson_data_comp)
    elif geotiff_path_comp:
        filtered_buildings_comp = extract_building_heights_from_geotiff(geotiff_path_comp, filtered_buildings)
    else:
        filtered_buildings_comp = filtered_buildings

    # Create building polygons and spatial index
    building_polygons, idx = create_building_polygons(filtered_buildings_comp)

    # Process each grid cell
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            cell = create_cell_polygon(origin, i, j, adjusted_meshsize, u_vec, v_vec)
            # Ensure cell geometry is valid
            if not cell.is_valid:
                cell = cell.buffer(0)
            
            # Get potential intersecting buildings using spatial index
            potential_intersections = list(idx.intersection(cell.bounds))
            
            if not potential_intersections:
                continue
                
            # Sort buildings by height for proper layering
            cell_buildings = [(k, building_polygons[k]) for k in potential_intersections]
            cell_buildings.sort(key=lambda x: x[1][1] if x[1][1] is not None else -float('inf'), reverse=True)
            
            # Track intersection status
            found_intersection = False
            all_zero_or_nan = True
            
            # Process each potential building intersection
            for k, (polygon, height, min_height, is_inner, feature_id) in cell_buildings:
                try:
                    # Ensure valid geometry
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
                    
                    # Check for intersection
                    if cell.intersects(polygon):
                        intersection = cell.intersection(polygon)
                        intersection_ratio = intersection.area / cell.area
                        
                        INTERSECTION_THRESHOLD = 0.3
                        
                        if intersection_ratio > INTERSECTION_THRESHOLD:
                            found_intersection = True
                            
                            if not is_inner:
                                # Store building information
                                building_min_height_grid[i, j].append([min_height, height])
                                building_id_grid[i, j] = feature_id
                                
                                # Update height if valid
                                has_valid_height = height is not None and not np.isnan(height) and height > 0
                                if has_valid_height:
                                    all_zero_or_nan = False
                                    
                                    current_height = building_height_grid[i, j]
                                    if (current_height == 0 or 
                                        current_height < height or 
                                        np.isnan(current_height)):
                                        building_height_grid[i, j] = height
                            else:
                                # Handle inner courtyards
                                building_min_height_grid[i, j] = [[0, 0]]
                                building_height_grid[i, j] = 0
                                found_intersection = True
                                all_zero_or_nan = False
                                break
                                
                except (GEOSException, ValueError) as e:
                    # Attempt to fix topology errors
                    try:
                        simplified_polygon = polygon.simplify(1e-8)
                        if simplified_polygon.is_valid:
                            intersection = cell.intersection(simplified_polygon)
                            intersection_ratio = intersection.area / cell.area
                            
                            if intersection_ratio > INTERSECTION_THRESHOLD:
                                found_intersection = True
                                
                                if not is_inner:
                                    building_min_height_grid[i, j].append([min_height, height])
                                    building_id_grid[i, j] = feature_id
                                    
                                    has_valid_height = height is not None and not np.isnan(height) and height > 0
                                    if has_valid_height:
                                        all_zero_or_nan = False
                                        
                                        if (building_height_grid[i, j] == 0 or 
                                            building_height_grid[i, j] < height or 
                                            np.isnan(building_height_grid[i, j])):
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
            
            # Set cell to NaN if all intersecting buildings had zero/NaN height
            if found_intersection and all_zero_or_nan:
                building_height_grid[i, j] = np.nan

    return building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings

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
    filtered_buildings = []
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