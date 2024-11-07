import numpy as np
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
    extract_building_heights_from_geojson
)
from ..utils.lc import (
    get_class_priority, 
    create_land_cover_polygons, 
    get_dominant_class,
)
# from ..download.mbfp import get_mbfp_geojson
# from ..download.gee import (
#     get_roi,
#     save_geotiff_open_buildings_temporal
# )

def apply_operation(arr, meshsize):
    step1 = arr / meshsize
    step2 = step1 + 0.5
    step3 = np.floor(step2)
    return step3 * meshsize

def translate_array(input_array, translation_dict):
    translated_array = np.empty_like(input_array, dtype=object)
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            value = input_array[i, j]
            translated_array[i, j] = translation_dict.get(value, '')
    return translated_array

def group_and_label_cells(input_array):
    binary_array = input_array > 0
    structure = generate_binary_structure(2, 1)
    labeled_array, num_features = label(binary_array, structure=structure)
    result = np.zeros_like(input_array, dtype=int)
    for i in range(1, num_features + 1):
        result[labeled_array == i] = i
    return result

def process_grid(grid_bi, dem_grid):
    unique_ids = np.unique(grid_bi[grid_bi != 0])
    result = dem_grid.copy()
    for id_num in unique_ids:
        mask = (grid_bi == id_num)
        avg_value = np.mean(dem_grid[mask])
        result[mask] = avg_value
    return result - np.min(result)

def calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize):
    grid_size_0 = int(np.linalg.norm(side_1) / np.linalg.norm(meshsize * u_vec) + 0.5)
    grid_size_1 = int(np.linalg.norm(side_2) / np.linalg.norm(meshsize * v_vec) + 0.5)
    adjusted_mesh_size_0 = meshsize *  np.linalg.norm(meshsize * u_vec) * grid_size_0 / np.linalg.norm(side_1)
    adjusted_mesh_size_1 = meshsize *  np.linalg.norm(meshsize * v_vec) * grid_size_1 / np.linalg.norm(side_2)
    return (grid_size_0, grid_size_1), (adjusted_mesh_size_0, adjusted_mesh_size_1)

def create_coordinate_mesh(origin, grid_size, adjusted_meshsize, u_vec, v_vec):
    x = np.linspace(0, grid_size[0], grid_size[0])
    y = np.linspace(0, grid_size[1], grid_size[1])
    xx, yy = np.meshgrid(x, y)

    cell_coords = origin[:, np.newaxis, np.newaxis] + \
                  xx[np.newaxis, :, :] * adjusted_meshsize[0] * u_vec[:, np.newaxis, np.newaxis] + \
                  yy[np.newaxis, :, :] * adjusted_meshsize[1] * v_vec[:, np.newaxis, np.newaxis]

    return cell_coords

def create_cell_polygon(origin, i, j, adjusted_meshsize, u_vec, v_vec):
    bottom_left = origin + i * adjusted_meshsize[0] * u_vec + j * adjusted_meshsize[1] * v_vec
    bottom_right = origin + (i + 1) * adjusted_meshsize[0] * u_vec + j * adjusted_meshsize[1] * v_vec
    top_right = origin + (i + 1) * adjusted_meshsize[0] * u_vec + (j + 1) * adjusted_meshsize[1] * v_vec
    top_left = origin + i * adjusted_meshsize[0] * u_vec + (j + 1) * adjusted_meshsize[1] * v_vec
    return Polygon([bottom_left, bottom_right, top_right, top_left])

def tree_height_grid_from_land_cover(land_cover_grid_ori):

    land_cover_grid = np.flipud(land_cover_grid_ori) + 1

    tree_translation_dict = {
        1: 0,
        2: 0,
        3: 0,
        4: 10,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0
    }
    tree_height_grid = translate_array(np.flipud(land_cover_grid), tree_translation_dict).astype(int)

    return tree_height_grid

def create_land_cover_grid_from_geotiff(tiff_path, mesh_size, land_cover_classes):
    with rasterio.open(tiff_path) as src:
        img = src.read((1,2,3))
        left, bottom, right, top = src.bounds
        src_crs = src.crs
        
        # Calculate width and height in meters
        if src_crs.to_epsg() == 3857:  # Web Mercator
            # Convert bounds to WGS84
            wgs84 = CRS.from_epsg(4326)
            transformer = Transformer.from_crs(src_crs, wgs84, always_xy=True)
            left_wgs84, bottom_wgs84 = transformer.transform(left, bottom)
            right_wgs84, top_wgs84 = transformer.transform(right, top)
        
            # Use geodesic calculations for accuracy
            geod = Geod(ellps="WGS84")
            _, _, width = geod.inv(left_wgs84, bottom_wgs84, right_wgs84, bottom_wgs84)
            _, _, height = geod.inv(left_wgs84, bottom_wgs84, left_wgs84, top_wgs84)
        else:
            # For other projections, assume units are already in meters
            width = right - left
            height = top - bottom
        
        # Display width and height in meters
        # print(f"ROI Width: {width:.2f} meters")
        # print(f"ROI Height: {height:.2f} meters")
        
        num_cells_x = int(width / mesh_size + 0.5)
        num_cells_y = int(height / mesh_size + 0.5)
        
        # Adjust mesh_size to fit the image exactly
        adjusted_mesh_size_x = (right - left) / num_cells_x
        adjusted_mesh_size_y = (top - bottom) / num_cells_y
        
        # Create a new affine transformation for the new grid
        new_affine = Affine(adjusted_mesh_size_x, 0, left, 0, -adjusted_mesh_size_y, top)
        
        cols, rows = np.meshgrid(np.arange(num_cells_x), np.arange(num_cells_y))
        xs, ys = new_affine * (cols, rows)
        xs_flat, ys_flat = xs.flatten(), ys.flatten()
        
        row, col = src.index(xs_flat, ys_flat)
        row, col = np.array(row), np.array(col)
        valid = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)
        row, col = row[valid], col[valid]
        
        grid = np.full((num_cells_y, num_cells_x), 'No Data', dtype=object)
        
        for i, (r, c) in enumerate(zip(row, col)):
            cell_data = img[:, r, c]
            dominant_class = get_dominant_class(cell_data, land_cover_classes)
            grid_row, grid_col = np.unravel_index(i, (num_cells_y, num_cells_x))
            grid[grid_row, grid_col] = dominant_class
    
    return np.flipud(grid)

def create_land_cover_grid_from_geotiff_polygon(tiff_path, mesh_size, land_cover_classes, polygon):
    with rasterio.open(tiff_path) as src:
        img = src.read((1,2,3))
        left, bottom, right, top = src.bounds
        src_crs = src.crs
        
        # Create a Shapely polygon from input coordinates
        poly = Polygon(polygon)
        
        # Get bounds of the polygon
        bottom_wgs84, left_wgs84, top_wgs84, right_wgs84 = poly.bounds
        # print(left, bottom, right, top)

        # Use geodesic calculations for accuracy
        geod = Geod(ellps="WGS84")
        _, _, width = geod.inv(left_wgs84, bottom_wgs84, right_wgs84, bottom_wgs84)
        _, _, height = geod.inv(left_wgs84, bottom_wgs84, left_wgs84, top_wgs84)
        
        # Display width and height in meters
        # print(f"ROI Width: {width:.2f} meters")
        # print(f"ROI Height: {height:.2f} meters")
        
        num_cells_x = int(width / mesh_size + 0.5)
        num_cells_y = int(height / mesh_size + 0.5)
        
        # Adjust mesh_size to fit the image exactly
        adjusted_mesh_size_x = (right - left) / num_cells_x
        adjusted_mesh_size_y = (top - bottom) / num_cells_y
        
        # Create a new affine transformation for the new grid
        new_affine = Affine(adjusted_mesh_size_x, 0, left, 0, -adjusted_mesh_size_y, top)
        
        cols, rows = np.meshgrid(np.arange(num_cells_x), np.arange(num_cells_y))
        xs, ys = new_affine * (cols, rows)
        xs_flat, ys_flat = xs.flatten(), ys.flatten()
        
        row, col = src.index(xs_flat, ys_flat)
        row, col = np.array(row), np.array(col)
        valid = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)
        row, col = row[valid], col[valid]
        
        grid = np.full((num_cells_y, num_cells_x), 'No Data', dtype=object)
        
        for i, (r, c) in enumerate(zip(row, col)):
            cell_data = img[:, r, c]
            dominant_class = get_dominant_class(cell_data, land_cover_classes)
            grid_row, grid_col = np.unravel_index(i, (num_cells_y, num_cells_x))
            grid[grid_row, grid_col] = dominant_class
    
    return np.flipud(grid)

def create_land_cover_grid_from_geojson_polygon(geojson_data, meshsize, source, rectangle_vertices):

    class_priority = { 
        'Bareland': 4, 
        'Rangeland': 6, 
        'Developed space': 8, 
        'Road': 1, 
        'Tree': 7, 
        'Water': 3, 
        'Agriculture land': 5, 
        'Building': 2 
    }

    class_priority = get_class_priority(source)
    
    # Calculate grid and normalize vectors
    geod = initialize_geod()
    vertex_0, vertex_1, vertex_3 = rectangle_vertices[0], rectangle_vertices[1], rectangle_vertices[3]

    dist_side_1 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_1[1], vertex_1[0])
    dist_side_2 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_3[1], vertex_3[0])

    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)

    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)

    origin = np.array(rectangle_vertices[0])
    grid_size, adjusted_meshsize = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)  

    # print(f"Calculated grid size: {grid_size}")
    print(f"Adjusted mesh size: {adjusted_meshsize}")

    # Create the grid
    # grid = np.zeros(grid_size)
    # grid = np.full(grid_size, 0)
    grid = np.full(grid_size, 'Developed space', dtype=object)

    # Setup transformer and plotting extent
    extent = [min(coord[1] for coord in rectangle_vertices), max(coord[1] for coord in rectangle_vertices),
              min(coord[0] for coord in rectangle_vertices), max(coord[0] for coord in rectangle_vertices)]
    plotting_box = box(extent[2], extent[0], extent[3], extent[1])

    land_cover_polygons, idx = create_land_cover_polygons(geojson_data) 

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            land_cover_class = 'Developed space'
            # grid[i, j] = 3
            # grid[i, j] = class_mapping[land_cover_class]
            cell = create_cell_polygon(origin, i, j, adjusted_meshsize, u_vec, v_vec)
            for k in idx.intersection(cell.bounds):
                polygon, land_cover_class_temp = land_cover_polygons[k]
                try:
                    if cell.intersects(polygon):
                        # print("intersection")
                        intersection = cell.intersection(polygon)
                        if intersection.area > cell.area/2:
                            rank = class_priority[land_cover_class]
                            rank_temp = class_priority[land_cover_class_temp]
                            if rank_temp < rank:
                                land_cover_class = land_cover_class_temp
                                grid[i, j] = land_cover_class
                            # break
                except GEOSException as e:
                    print(f"GEOS error at grid cell ({i}, {j}): {str(e)}")
                    # Attempt to fix the polygon
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
                                # break
                    except Exception as fix_error:
                        print(f"Failed to fix polygon at grid cell ({i}, {j}): {str(fix_error)}")
                    continue 
    return grid

def create_canopy_height_grid_from_geotiff(tiff_path, mesh_size):
    with rasterio.open(tiff_path) as src:
        img = src.read(1)
        left, bottom, right, top = src.bounds
        src_crs = src.crs

        # Calculate width and height in meters
        if src_crs.to_epsg() == 3857:  # Web Mercator
            # Convert bounds to WGS84
            wgs84 = CRS.from_epsg(4326)
            transformer = Transformer.from_crs(src_crs, wgs84, always_xy=True)
            left_wgs84, bottom_wgs84 = transformer.transform(left, bottom)
            right_wgs84, top_wgs84 = transformer.transform(right, top)
        
            # Use geodesic calculations for accuracy
            geod = Geod(ellps="WGS84")
            _, _, width = geod.inv(left_wgs84, bottom_wgs84, right_wgs84, bottom_wgs84)
            _, _, height = geod.inv(left_wgs84, bottom_wgs84, left_wgs84, top_wgs84)
        else:
            # For other projections, assume units are already in meters
            width = right - left
            height = top - bottom

        # Display width and height in meters
        # print(f"ROI Width: {width:.2f} meters")
        # print(f"ROI Height: {height:.2f} meters")

        num_cells_x = int(width / mesh_size + 0.5)
        num_cells_y = int(height / mesh_size + 0.5)

        # Adjust mesh_size to fit the image exactly
        adjusted_mesh_size_x = (right - left) / num_cells_x
        adjusted_mesh_size_y = (top - bottom) / num_cells_y

        # Create a new affine transformation for the new grid
        new_affine = Affine(adjusted_mesh_size_x, 0, left, 0, -adjusted_mesh_size_y, top)

        cols, rows = np.meshgrid(np.arange(num_cells_x), np.arange(num_cells_y))
        xs, ys = new_affine * (cols, rows)
        xs_flat, ys_flat = xs.flatten(), ys.flatten()

        row, col = src.index(xs_flat, ys_flat)
        row, col = np.array(row), np.array(col)

        valid = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)
        row, col = row[valid], col[valid]

        grid = np.full((num_cells_y, num_cells_x), np.nan)
        flat_indices = np.ravel_multi_index((row, col), img.shape)
        np.put(grid, np.ravel_multi_index((rows.flatten()[valid], cols.flatten()[valid]), grid.shape), img.flat[flat_indices])

    return np.flipud(grid)

def create_height_grid_from_geotiff_polygon(tiff_path, mesh_size, polygon):
    with rasterio.open(tiff_path) as src:
        img = src.read(1)
        left, bottom, right, top = src.bounds
        src_crs = src.crs

        # Create a Shapely polygon from input coordinates
        poly = Polygon(polygon)
        
        # Get bounds of the polygon
        bottom_wgs84, left_wgs84, top_wgs84, right_wgs84 = poly.bounds
        print(left, bottom, right, top)

        # Use geodesic calculations for accuracy
        geod = Geod(ellps="WGS84")
        _, _, width = geod.inv(left_wgs84, bottom_wgs84, right_wgs84, bottom_wgs84)
        _, _, height = geod.inv(left_wgs84, bottom_wgs84, left_wgs84, top_wgs84)
        
        # Display width and height in meters
        # print(f"ROI Width: {width:.2f} meters")
        # print(f"ROI Height: {height:.2f} meters")

        num_cells_x = int(width / mesh_size + 0.5)
        num_cells_y = int(height / mesh_size + 0.5)

        # Adjust mesh_size to fit the image exactly
        adjusted_mesh_size_x = (right - left) / num_cells_x
        adjusted_mesh_size_y = (top - bottom) / num_cells_y

        # Create a new affine transformation for the new grid
        new_affine = Affine(adjusted_mesh_size_x, 0, left, 0, -adjusted_mesh_size_y, top)

        cols, rows = np.meshgrid(np.arange(num_cells_x), np.arange(num_cells_y))
        xs, ys = new_affine * (cols, rows)
        xs_flat, ys_flat = xs.flatten(), ys.flatten()

        row, col = src.index(xs_flat, ys_flat)
        row, col = np.array(row), np.array(col)

        valid = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)
        row, col = row[valid], col[valid]

        grid = np.full((num_cells_y, num_cells_x), np.nan)
        flat_indices = np.ravel_multi_index((row, col), img.shape)
        np.put(grid, np.ravel_multi_index((rows.flatten()[valid], cols.flatten()[valid]), grid.shape), img.flat[flat_indices])

    return np.flipud(grid)

def create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices, geojson_data_comp=None, geotiff_path_comp=None):
    # Calculate grid and normalize vectors
    geod = initialize_geod()
    vertex_0, vertex_1, vertex_3 = rectangle_vertices[0], rectangle_vertices[1], rectangle_vertices[3]

    dist_side_1 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_1[1], vertex_1[0])
    dist_side_2 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_3[1], vertex_3[0])

    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)

    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)

    origin = np.array(rectangle_vertices[0])
    grid_size, adjusted_meshsize = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)

    # print(f"Calculated grid size: {grid_size}")
    # print(f"Adjusted mesh size: {adjusted_meshsize}")

    # Create the grid
    building_height_grid = np.zeros(grid_size)

    # Create object array that can store lists of varying lengths
    building_min_height_grid = np.empty(grid_size, dtype=object)
    # Initialize each cell with an empty list
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            building_min_height_grid[i, j] = []

    # Setup transformer and plotting extent
    extent = [min(coord[1] for coord in rectangle_vertices), max(coord[1] for coord in rectangle_vertices),
              min(coord[0] for coord in rectangle_vertices), max(coord[0] for coord in rectangle_vertices)]
    plotting_box = box(extent[2], extent[0], extent[3], extent[1])

    # Filter polygons and create building polygons
    filtered_buildings = filter_buildings(geojson_data, plotting_box)

    if geojson_data_comp:
        filtered_geojson_data_comp = filter_buildings(geojson_data_comp, plotting_box)
        filtered_buildings_comp = extract_building_heights_from_geojson(filtered_buildings, filtered_geojson_data_comp)
    elif geotiff_path_comp:
        filtered_buildings_comp = extract_building_heights_from_geotiff(geotiff_path_comp, filtered_buildings)
    else:
        filtered_buildings_comp = filtered_buildings

    building_polygons, idx = create_building_polygons(filtered_buildings_comp)

    # Calculate building heights for each grid cell
    # buildings_found = 0
    # for i in range(grid_size[0]):
    #     for j in range(grid_size[1]):
    #         cell = create_cell_polygon(origin, i, j, adjusted_meshsize, u_vec, v_vec)
    #         for k in idx.intersection(cell.bounds):
    #             polygon, height = building_polygons[k]
    #             if cell.intersects(polygon) and cell.intersection(polygon).area > cell.area/2:
    #                 grid[i, j] = height
    #                 buildings_found += 1
    #                 break

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            cell = create_cell_polygon(origin, i, j, adjusted_meshsize, u_vec, v_vec)
            for k in idx.intersection(cell.bounds):
                polygon, height, min_height = building_polygons[k]
                try:
                    if cell.intersects(polygon):
                        intersection = cell.intersection(polygon)
                        if intersection.area > cell.area/2:
                            building_min_height_grid[i, j].append([min_height, height])
                            if (building_height_grid[i, j] == 0) or (building_height_grid[i, j] < height) or (building_height_grid[i, j] == np.nan):
                                building_height_grid[i, j] = height
                            # break
                except GEOSException as e:
                    print(f"GEOS error at grid cell ({i}, {j}): {str(e)}")
                    # Attempt to fix the polygon
                    try:
                        fixed_polygon = polygon.buffer(0)
                        if cell.intersects(fixed_polygon):
                            intersection = cell.intersection(fixed_polygon)
                            if intersection.area > cell.area/2:
                                building_min_height_grid[i, j].append([min_height, height])
                                if (building_height_grid[i, j] == 0) or (building_height_grid[i, j] < height) or (building_height_grid[i, j] == np.nan):
                                    building_height_grid[i, j] = height
                                # break
                    except Exception as fix_error:
                        print(f"Failed to fix polygon at grid cell ({i}, {j}): {str(fix_error)}")
                    continue

    return building_height_grid, building_min_height_grid, filtered_buildings

def create_dem_grid_from_geotiff_polygon(tiff_path, mesh_size, rectangle_vertices, dem_interpolation=False):

    converted_coords = convert_format_lat_lon(rectangle_vertices)
    roi_shapely = Polygon(converted_coords)

    with rasterio.open(tiff_path) as src:
        dem = src.read(1)
        dem = np.where(dem < -1000, 0, dem)
        transform = src.transform
        src_crs = src.crs

        # Ensure we're working with EPSG:3857
        if src_crs.to_epsg() != 3857:
            transformer_to_3857 = Transformer.from_crs(src_crs, CRS.from_epsg(3857), always_xy=True)
        else:
            transformer_to_3857 = lambda x, y: (x, y)

        # Transform ROI bounds to EPSG:3857
        roi_bounds = roi_shapely.bounds
        roi_left, roi_bottom = transformer_to_3857.transform(roi_bounds[0], roi_bounds[1])
        roi_right, roi_top = transformer_to_3857.transform(roi_bounds[2], roi_bounds[3])

        # Calculate width and height in meters using geodesic methods
        wgs84 = CRS.from_epsg(4326)
        transformer_to_wgs84 = Transformer.from_crs(CRS.from_epsg(3857), wgs84, always_xy=True)
        roi_left_wgs84, roi_bottom_wgs84 = transformer_to_wgs84.transform(roi_left, roi_bottom)
        roi_right_wgs84, roi_top_wgs84 = transformer_to_wgs84.transform(roi_right, roi_top)

        geod = Geod(ellps="WGS84")
        _, _, roi_width_m = geod.inv(roi_left_wgs84, roi_bottom_wgs84, roi_right_wgs84, roi_bottom_wgs84)
        _, _, roi_height_m = geod.inv(roi_left_wgs84, roi_bottom_wgs84, roi_left_wgs84, roi_top_wgs84)

        # Display width and height in meters
        # print(f"ROI Width: {roi_width_m:.2f} meters")
        # print(f"ROI Height: {roi_height_m:.2f} meters")

        num_cells_x = int(roi_width_m / mesh_size + 0.5)
        num_cells_y = int(roi_height_m / mesh_size + 0.5)

        # # Adjust mesh_size to fit the ROI exactly
        # adjusted_mesh_size_x = roi_width_m / num_cells_x
        # adjusted_mesh_size_y = roi_height_m / num_cells_y

        # Create grid in EPSG:3857
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