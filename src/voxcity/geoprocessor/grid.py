"""
This module provides functions for creating and manipulating grids of building heights, land cover, and elevation data.
"""

import numpy as np
import os
from shapely.geometry import Polygon, box
from scipy.ndimage import label, generate_binary_structure
from pyproj import Geod, Transformer, CRS
import rasterio
from affine import Affine
from shapely.geometry import box
from scipy.interpolate import griddata
from shapely.errors import GEOSException
import geopandas as gpd
from rtree import index

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
    complement_building_heights_from_gdf
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
    
def create_land_cover_grid_from_gdf_polygon(gdf, meshsize, source, rectangle_vertices):
    """Create a grid of land cover classes from GeoDataFrame polygon data.

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing land cover polygons
        meshsize (float): Size of each grid cell in meters
        source (str): Source of the land cover data to determine class priorities
        rectangle_vertices (list): List of 4 (lon,lat) coordinate pairs defining the rectangle bounds

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
    grid = np.full(grid_size, 'Developed space', dtype=object)

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
            land_cover_class = 'Developed space'
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
    gdf_comp=None,
    geotiff_path_comp=None,
    complement_building_footprints=None
):
    """
    Create a building height grid from GeoDataFrame data within a polygon boundary.
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame containing building information
        meshsize (float): Size of mesh cells
        rectangle_vertices (list): List of rectangle vertices defining the boundary
        gdf_comp (geopandas.GeoDataFrame, optional): Complementary GeoDataFrame
        geotiff_path_comp (str, optional): Path to complementary GeoTIFF file
        complement_building_footprints (bool, optional): Whether to complement footprints

    Returns:
        tuple: (building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings)
            - building_height_grid (numpy.ndarray): Grid of building heights
            - building_min_height_grid (numpy.ndarray): Grid of min building heights (list per cell)
            - building_id_grid (numpy.ndarray): Grid of building IDs
            - filtered_buildings (geopandas.GeoDataFrame): The buildings used (filtered_gdf)
    """
    # --------------------------------------------------------------------------
    # 1) INITIAL SETUP AND DATA FILTERING
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

    # Initialize output grids
    building_height_grid = np.zeros(grid_size)
    building_id_grid = np.zeros(grid_size)
    # Use a Python list-of-lists or object array for min_height tracking
    building_min_height_grid = np.empty(grid_size, dtype=object)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            building_min_height_grid[i, j] = []

    # Filter the input GeoDataFrame by bounding box
    extent = [
        min(coord[1] for coord in rectangle_vertices),
        max(coord[1] for coord in rectangle_vertices),
        min(coord[0] for coord in rectangle_vertices),
        max(coord[0] for coord in rectangle_vertices)
    ]
    plotting_box = box(extent[2], extent[0], extent[3], extent[1])
    filtered_gdf = gdf[gdf.geometry.intersects(plotting_box)].copy()

    # Optionally merge heights from complementary sources
    if gdf_comp is not None:
        filtered_gdf_comp = gdf_comp[gdf_comp.geometry.intersects(plotting_box)].copy()
        if complement_building_footprints:
            filtered_gdf = complement_building_heights_gdf(filtered_gdf, filtered_gdf_comp)
        else:
            filtered_gdf = extract_building_heights_from_gdf(filtered_gdf, filtered_gdf_comp)
    elif geotiff_path_comp:
        filtered_gdf = extract_building_heights_from_geotiff(geotiff_path_comp, filtered_gdf)

    # --------------------------------------------------------------------------
    # 2) PREPARE BUILDING POLYGONS & SPATIAL INDEX
    #    - fix geometries if needed
    #    - store bounding boxes
    # --------------------------------------------------------------------------
    building_polygons = []
    for idx_b, row in filtered_gdf.iterrows():
        polygon = row.geometry
        height = row.get('height', None)
        min_height = row.get('min_height', 0)
        is_inner = row.get('is_inner', False)
        feature_id = row.get('id', idx_b)

        # Fix invalid geometry (buffer(0) or simplify if needed)
        # Doing this once per building avoids repeated overhead per cell.
        if not polygon.is_valid:
            try:
                # Attempt simple fix
                polygon = polygon.buffer(0)
                if not polygon.is_valid:
                    # Attempt a small simplify if buffer(0) didn't fix it
                    polygon = polygon.simplify(1e-8)
            except Exception as e:
                # If still invalid, we keep it as-is; intersection attempts
                # will be caught in the main loop's try/except
                pass

        bounding_box = polygon.bounds  # (minx, miny, maxx, maxy)
        building_polygons.append(
            (
                polygon,
                bounding_box,
                height,
                min_height,
                is_inner,
                feature_id
            )
        )

    # Build R-tree index using bounding boxes
    idx = index.Index()
    for i_b, (poly, bbox, _, _, _, _) in enumerate(building_polygons):
        idx.insert(i_b, bbox)

    # --------------------------------------------------------------------------
    # 3) MAIN GRID LOOP
    # --------------------------------------------------------------------------
    INTERSECTION_THRESHOLD = 0.3

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Create the cell polygon once
            cell = create_cell_polygon(origin, i, j, adjusted_meshsize, u_vec, v_vec)
            if not cell.is_valid:
                cell = cell.buffer(0)

            cell_area = cell.area  # reused for intersection ratio checks

            # Find possible intersections from the index
            potential = list(idx.intersection(cell.bounds))
            if not potential:
                continue

            # Sort buildings by height descending
            # (height=None => treat as -inf so it sorts last)
            cell_buildings = []
            for k in potential:
                bpoly, bbox, height, minh, inr, fid = building_polygons[k]
                # We only sort by height for a stable layering
                sort_val = height if (height is not None) else -float('inf')
                cell_buildings.append((k, bpoly, bbox, height, minh, inr, fid, sort_val))

            cell_buildings.sort(key=lambda x: x[-1], reverse=True)

            found_intersection = False
            all_zero_or_nan = True

            for (
                k,
                polygon,
                bbox,
                height,
                min_height,
                is_inner,
                feature_id,
                _
            ) in cell_buildings:
                try:
                    # Quick bounding-box check to skip expensive geometry intersection
                    # if bounding boxes cannot possibly meet the threshold
                    # --------------------------------------------------------
                    minx_p, miny_p, maxx_p, maxy_p = bbox
                    minx_c, miny_c, maxx_c, maxy_c = cell.bounds

                    # Overlap bounding box
                    overlap_minx = max(minx_p, minx_c)
                    overlap_miny = max(miny_p, miny_c)
                    overlap_maxx = min(maxx_p, maxx_c)
                    overlap_maxy = min(maxy_p, maxy_c)

                    if (overlap_maxx <= overlap_minx) or (overlap_maxy <= overlap_miny):
                        # No bounding-box overlap => skip
                        continue

                    # Area of bounding-box intersection
                    bbox_intersect_area = (overlap_maxx - overlap_minx) * (overlap_maxy - overlap_miny)
                    if bbox_intersect_area < INTERSECTION_THRESHOLD * cell_area:
                        # Even if the polygons overlap, they can't exceed threshold
                        continue
                    # --------------------------------------------------------

                    # Ensure valid geometry
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
                        # If still invalid, let intersection attempt fail

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
                                if (
                                    height is not None
                                    and not np.isnan(height)
                                    and height > 0
                                ):
                                    all_zero_or_nan = False
                                    current_height = building_height_grid[i, j]

                                    # Replace if we had 0, nan, or smaller height
                                    if (
                                        current_height == 0
                                        or np.isnan(current_height)
                                        or current_height < height
                                    ):
                                        building_height_grid[i, j] = height

                            else:
                                # Inner courtyards => override with 0
                                building_min_height_grid[i, j] = [[0, 0]]
                                building_height_grid[i, j] = 0
                                found_intersection = True
                                all_zero_or_nan = False
                                # Because it's an inner courtyard, we can break
                                break

                except (GEOSException, ValueError) as e:
                    # Attempt to fix in a fallback
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

                                    if (
                                        height is not None
                                        and not np.isnan(height)
                                        and height > 0
                                    ):
                                        all_zero_or_nan = False
                                        if (
                                            building_height_grid[i, j] == 0
                                            or np.isnan(building_height_grid[i, j])
                                            or building_height_grid[i, j] < height
                                        ):
                                            building_height_grid[i, j] = height
                                else:
                                    building_min_height_grid[i, j] = [[0, 0]]
                                    building_height_grid[i, j] = 0
                                    found_intersection = True
                                    all_zero_or_nan = False
                                    break
                    except Exception as fix_error:
                        # Log and skip
                        print(f"Failed to process cell ({i}, {j}) - Building {k}: {str(fix_error)}")
                        continue

            # If we found intersecting buildings but all were zero/NaN, mark as NaN
            if found_intersection and all_zero_or_nan:
                building_height_grid[i, j] = np.nan

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
    """Converts a 2D grid to a GeoDataFrame with cell polygons and values.
    
    Args:
        grid: 2D numpy array containing grid values
        rectangle_vertices: List of [lon, lat] coordinates defining area corners
        meshsize: Size of each grid cell in meters
        
    Returns:
        GeoDataFrame with columns:
            - geometry: Polygon geometry of each grid cell
            - value: Value from the grid
    """
    grid = np.flipud(grid_ori.copy())
    
    # Extract bounds from rectangle vertices
    min_lon = min(v[0] for v in rectangle_vertices)
    max_lon = max(v[0] for v in rectangle_vertices)
    min_lat = min(v[1] for v in rectangle_vertices)
    max_lat = max(v[1] for v in rectangle_vertices)
    
    rows, cols = grid.shape
    
    # Calculate cell sizes in degrees (approximate)
    # 111,111 meters = 1 degree at equator
    cell_size_lon = meshsize / (111111 * np.cos(np.mean([min_lat, max_lat]) * np.pi / 180))
    cell_size_lat = meshsize / 111111
    
    # Create lists to store data
    polygons = []
    values = []
    
    # Create grid cells
    for i in range(rows):
        for j in range(cols):
            # Calculate cell bounds
            cell_min_lon = min_lon + j * cell_size_lon
            cell_max_lon = min_lon + (j + 1) * cell_size_lon
            # Flip vertical axis since grid is stored with origin at top-left
            cell_min_lat = max_lat - (i + 1) * cell_size_lat
            cell_max_lat = max_lat - i * cell_size_lat
            
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