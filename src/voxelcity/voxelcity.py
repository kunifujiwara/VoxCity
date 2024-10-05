"""Main module."""

import numpy as np
import os

# Local application/library specific imports
# from .download.urbanwatch import get_geotif_urbanwatch
from .download.mbfp import get_mbfp_geojson
from .download.osm import load_geojsons_from_openstreetmap
# from .download.utils import download_file
from .download.oemj import save_oemj_as_geotiff
# from .download.nasadem import (
#     download_nasa_dem,
#     interpolate_dem,
#     get_utm_crs
# )
from .download.gee import (
    initialize_earth_engine,
    get_roi,
    get_image_collection,
    save_geotiff,
    get_dem_image,
    save_geotiff_esa_land_cover,
    save_geotiff_open_buildings_temporal
)
from .geo.utils import convert_land_cover_array
from .geo.grid import (
    group_and_label_cells, 
    process_grid,
    create_land_cover_grid_from_geotiff_polygon,
    create_height_grid_from_geotiff_polygon,
    create_building_height_grid_from_geojson_polygon,
    create_dem_grid_from_geotiff_polygon
)
from .utils.visualization import (
    get_land_cover_classes,
    visualize_land_cover_grid,
    visualize_numerical_grid,
    convert_land_cover,
    visualize_land_cover_grid_on_map,
    visualize_numerical_grid_on_map,
    visualize_building_height_grid_on_map,
    visualize_3d_voxel
)

def get_land_cover_grid(rectangle_vertices, meshsize, source = 'Urbanwatch', output_dir="output", visualization=True):

    print("Creating Land Use Land Cover grid\n ")
    print(f"Data source: {source}")
    
    # Initialize Earth Engine
    initialize_earth_engine()

    os.makedirs(output_dir, exist_ok=True)
    geotiff_path = os.path.join(output_dir, "land_cover.tif")

    if source == 'Urbanwatch':
        roi = get_roi(rectangle_vertices)
        collection_name = "projects/sat-io/open-datasets/HRLC/urban-watch-cities"
        image = get_image_collection(collection_name, roi)
        save_geotiff(image, geotiff_path)
    elif source == 'ESA WorldCover':
        roi = get_roi(rectangle_vertices)
        save_geotiff_esa_land_cover(roi, geotiff_path)
    elif source == 'OpenEarthMapJapan':
        save_oemj_as_geotiff(rectangle_vertices, geotiff_path)   
    
    land_cover_classes = get_land_cover_classes(source)

    land_cover_grid_str = create_land_cover_grid_from_geotiff_polygon(geotiff_path, meshsize, land_cover_classes, rectangle_vertices)
    color_map = {cls: [r/255, g/255, b/255] for (r,g,b), cls in land_cover_classes.items()}
    # color_map['No Data'] = [0.5, 0.5, 0.5]

    if visualization:
        visualize_land_cover_grid(np.flipud(land_cover_grid_str), meshsize, color_map, land_cover_classes)
    land_cover_grid_int = convert_land_cover_array(land_cover_grid_str, land_cover_classes)

    return land_cover_grid_int

def get_building_height_grid(rectangle_vertices, meshsize, source = 'Microsoft Building Footprints', output_dir="output", visualization=True):

    print("Creating Building Height grid\n ")
    print(f"Data source: {source}")

    os.makedirs(output_dir, exist_ok=True)

    if source == 'Microsoft Building Footprints':
        geojson_data = get_mbfp_geojson(output_dir, rectangle_vertices)
        building_height_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    elif source == 'OpenStreetMap':
        geojson_data = load_geojsons_from_openstreetmap(rectangle_vertices)
        building_height_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    elif source == "Open Building 2.5D Temporal":
        roi = get_roi(rectangle_vertices)
        os.makedirs(output_dir, exist_ok=True)
        geotiff_path = os.path.join(output_dir, "building_height.tif")
        save_geotiff_open_buildings_temporal(roi, geotiff_path)
        building_height_grid = create_height_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices)
        filtered_buildings = []
    elif source == "OpenStreetMap & Microsoft Building Footprints":
        geojson_data = load_geojsons_from_openstreetmap(rectangle_vertices)
        geojson_data_comp = get_mbfp_geojson(output_dir, rectangle_vertices)
        building_height_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices, geojson_data_comp=geojson_data_comp)
    elif source == "OpenStreetMap & Open Building 2.5D Temporal":
        geojson_data = load_geojsons_from_openstreetmap(rectangle_vertices)
        roi = get_roi(rectangle_vertices)
        os.makedirs(output_dir, exist_ok=True)
        geotiff_path_comp = os.path.join(output_dir, "building_height.tif")
        save_geotiff_open_buildings_temporal(roi, geotiff_path_comp)
        building_height_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices, geotiff_path_comp=geotiff_path_comp)   

    if visualization:
        visualize_numerical_grid(np.flipud(building_height_grid), meshsize, "building height (m)", cmap='viridis', label='Value')

    return building_height_grid, filtered_buildings#, buildings_found

def get_canopy_height_grid(rectangle_vertices, meshsize, output_dir="output", visualization=True):

    print("Creating Canopy Height grid\n ")
    print(f"Data source: High Resolution Canopy Height Maps by WRI and Meta")
    
    # Initialize Earth Engine
    initialize_earth_engine()

    os.makedirs(output_dir, exist_ok=True)
    geotiff_path = os.path.join(output_dir, "canopy_height.tif")
    
    roi = get_roi(rectangle_vertices)
    collection_name = "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"
    image = get_image_collection(collection_name, roi)
    save_geotiff(image, geotiff_path)  

    canopy_height_grid = create_height_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices)

    if visualization:
        visualize_numerical_grid(np.flipud(canopy_height_grid), meshsize, "Tree canopy height", cmap='Greens', label='Tree canopy height (m)')

    return canopy_height_grid

def get_dem_grid(rectangle_vertices, meshsize, source='NASA', output_dir="output", visualization=True):

    print("Creating Digital Elevation Model (DEM) grid\n ")
    print(f"Data source: {source}")

    # Initialize Earth Engine
    initialize_earth_engine()

    os.makedirs(output_dir, exist_ok=True)
    geotiff_path = os.path.join(output_dir, "dem.tif")

    buffer_distance = 100
    roi = get_roi(rectangle_vertices)
    roi_buffered = roi.buffer(buffer_distance)
    image = get_dem_image(roi_buffered, source)
    save_geotiff(image, geotiff_path, scale=30, region=roi_buffered)

    dem_grid = create_dem_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices)
    if visualization:
        visualize_numerical_grid(np.flipud(dem_grid), meshsize, title='Digital Elevation Model', cmap='terrain', label='Elevation (m)')

    return dem_grid

# def get_grid_dem(rectangle_vertices, meshsize, output_dir, api_key, **kwargs):
#     geod = initialize_geod()

#     center_lat = sum(vertex[0] for vertex in rectangle_vertices) / len(rectangle_vertices)
#     center_lon = sum(vertex[1] for vertex in rectangle_vertices) / len(rectangle_vertices)
#     dst_crs = get_utm_crs(center_lat, center_lon)

#     vertex_0 = rectangle_vertices[0]
#     vertex_1 = rectangle_vertices[1]
#     vertex_3 = rectangle_vertices[3]

#     dist_side_1 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_1[1], vertex_1[0])
#     dist_side_2 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_3[1], vertex_3[0])

#     side_1 = np.array(vertex_1) - np.array(vertex_0)
#     side_2 = np.array(vertex_3) - np.array(vertex_0)

#     u_vec = normalize_to_one_meter(side_1, dist_side_1)
#     v_vec = normalize_to_one_meter(side_2, dist_side_2)

#     origin = np.array(rectangle_vertices[0])
#     grid_size = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)

#     cell_coords = create_coordinate_mesh(origin, grid_size, meshsize, u_vec, v_vec)

#     cell_coords_flat = cell_coords.reshape(2, -1).T
#     transformer = setup_transformer(CRS.from_epsg(4326), dst_crs)
#     transformed_coords = np.array([transform_coords(transformer, lon, lat) for lat, lon in cell_coords_flat])

#     transformed_coords = np.array([coord for coord in transformed_coords if coord is not None])

#     all_coords = np.array(rectangle_vertices)
#     buffer = 0.01
#     bbox = [
#         min(all_coords[:, 1]) - buffer,
#         min(all_coords[:, 0]) - buffer,
#         max(all_coords[:, 1]) + buffer,
#         max(all_coords[:, 0]) + buffer
#     ]

#     dem_data = download_nasa_dem(bbox, api_key)

#     interpolated_dem = interpolate_dem(dem_data, transformed_coords, dst_crs)
#     dem_grid = interpolated_dem.reshape(grid_size[::-1]).T

#     plot_grid(dem_grid, origin, meshsize, u_vec, v_vec, transformer, dst_crs, rectangle_vertices, 'dem')

#     return dem_grid

# # Main function to handle land cover, building height, and DEM
# def get_grid_data(rectangle_vertices, data_type, meshsize=4, output_dir='output', **kwargs):
#     if data_type == 'land_cover':
#         land_cover_classes = kwargs.pop('land_cover_classes', None)
#         if land_cover_classes is None:
#             raise ValueError("land_cover_classes is required for land_cover analysis")
#         return get_grid_land_cover(rectangle_vertices, land_cover_classes, meshsize, output_dir, **kwargs)
#     elif data_type == 'building_height':
#         return get_grid_building_height(rectangle_vertices, meshsize, output_dir, **kwargs)
#     elif data_type == 'dem':
#         api_key = kwargs.pop('api_key', None)
#         if api_key is None:
#             raise ValueError("api_key is required for DEM analysis")
#         return get_grid_dem(rectangle_vertices, meshsize, output_dir, api_key, **kwargs)
#     else:
#         raise ValueError("Invalid data_type. Choose 'land_cover', 'building_height', or 'dem'.")
    
def create_3d_voxel(building_height_grid_ori, land_cover_grid_ori, dem_grid_ori, tree_grid_ori, voxel_size, land_cover_source):
    print("Generating 3D voxel data")
    if land_cover_source != 'OpenEarthMapJapan':
        land_cover_grid_converted = convert_land_cover(land_cover_grid_ori, land_cover_source=land_cover_source)  
    else:
        land_cover_grid_converted = land_cover_grid_ori      

    # Prepare grids
    building_height_grid = np.flipud(building_height_grid_ori.copy())
    land_cover_grid = np.flipud(land_cover_grid_converted.copy()) + 1
    dem_grid = np.flipud(dem_grid_ori.copy()) - np.min(dem_grid_ori)
    building_nr_grid = group_and_label_cells(np.flipud(building_height_grid_ori.copy()))
    dem_grid = process_grid(building_nr_grid, dem_grid)
    tree_grid = np.flipud(tree_grid_ori.copy())

    # Ensure all input grids have the same shape
    assert building_height_grid.shape == land_cover_grid.shape == dem_grid.shape == tree_grid.shape, "Input grids must have the same shape"

    # Get the dimensions of the input grids
    rows, cols = building_height_grid.shape

    # Calculate the maximum height needed for the 3D array
    max_height = int(np.ceil(np.max(building_height_grid + dem_grid + tree_grid) / voxel_size))

    # Create an empty 3D array
    voxel_grid = np.zeros((rows, cols, max_height), dtype=np.int32)

    # Fill the 3D array
    for i in range(rows):
        for j in range(cols):
            ground_level = int(dem_grid[i, j] / voxel_size + 0.5)
            building_height = int(building_height_grid[i, j] / voxel_size + 0.5)
            tree_height = int(tree_grid[i, j] / voxel_size + 0.5)
            land_cover = land_cover_grid[i, j]

            # Fill underground cells with -1
            voxel_grid[i, j, :ground_level] = -1

            # Set ground level cell to land cover
            voxel_grid[i, j, ground_level] = land_cover

            # Fill tree crown with value -2
            if tree_height > 0:
                tree_start = ground_level
                tree_end = ground_level + tree_height
                voxel_grid[i, j, tree_start:tree_end+1] = -2

            # Fill building with value -3
            if building_height > 0:
                voxel_grid[i, j, ground_level+1:ground_level+building_height+1] = -3

    return voxel_grid


def get_voxelcity(rectangle_vertices, building_source, land_cover_source, dem_source, meshsize, remove_perimeter_object=None, mapvis=False, voxelvis=False):
    #prepare of grid data
    land_cover_grid = get_land_cover_grid(rectangle_vertices, meshsize, source = land_cover_source)
    building_height_grid, building_geojson = get_building_height_grid(rectangle_vertices, meshsize, source = building_source)
    canopy_height_grid = get_canopy_height_grid(rectangle_vertices, meshsize)
    if dem_source == "Flat":
        dem_grid = np.zeros_like(land_cover_grid)
    else:
        dem_grid = get_dem_grid(rectangle_vertices, meshsize, source = dem_source)

    if remove_perimeter_object:
        w_peri = int(remove_perimeter_object * building_height_grid.shape[0] + 0.5)
        h_peri = int(remove_perimeter_object * building_height_grid.shape[1] + 0.5)
        building_height_grid[:w_peri, :] = building_height_grid[-w_peri:, :] = building_height_grid[:, :h_peri] = building_height_grid[:, -h_peri:] = 0
        canopy_height_grid[:w_peri, :] = canopy_height_grid[-w_peri:, :] = canopy_height_grid[:, :h_peri] = canopy_height_grid[:, -h_peri:] = 0

    #display grid data on basemap
    if mapvis:
        visualize_land_cover_grid_on_map(land_cover_grid, rectangle_vertices, meshsize, source = land_cover_source)
        visualize_building_height_grid_on_map(building_height_grid, building_geojson, rectangle_vertices, meshsize)
        visualize_numerical_grid_on_map(canopy_height_grid, rectangle_vertices, meshsize, "canopy_height")
        visualize_numerical_grid_on_map(dem_grid, rectangle_vertices, meshsize, "dem")

    #prepare 3D voxel grid  
    voxelcity_grid = create_3d_voxel(building_height_grid, land_cover_grid, dem_grid, canopy_height_grid, meshsize, land_cover_source)

    #display grid data in 3D
    if voxelvis:      
        visualize_3d_voxel(voxelcity_grid, voxel_size=meshsize)

    return voxelcity_grid, building_height_grid, canopy_height_grid, land_cover_grid, dem_grid