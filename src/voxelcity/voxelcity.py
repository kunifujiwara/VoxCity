"""Main module."""

import numpy as np
import os

# Local application/library specific imports
# from .download.urbanwatch import get_geotif_urbanwatch
from .download.mbfp import get_mbfp_geojson
from .download.osm import load_geojsons_from_openstreetmap, load_geojsons_from_osmbuildings, load_land_cover_geojson_from_osm
# from .download.utils import download_file
from .download.oemj import save_oemj_as_geotiff
from .download.omt import load_geojsons_from_openmaptiles
from .download.eubucco import load_geojson_from_eubucco
from .download.overture import load_geojsons_from_overture
# from .download.nasadem import (
#     download_nasa_dem,
#     interpolate_dem,
#     get_utm_crs
# )
from .download.gee import (
    initialize_earth_engine,
    get_roi,
    get_ee_image_collection,
    get_ee_image,
    save_geotiff,
    get_dem_image,
    save_geotiff_esa_land_cover,
    save_geotiff_esri_landcover,
    save_geotiff_dynamic_world_v1,
    save_geotiff_open_buildings_temporal
)
from .geo.grid import (
    group_and_label_cells, 
    process_grid,
    create_land_cover_grid_from_geotiff_polygon,
    create_height_grid_from_geotiff_polygon,
    create_building_height_grid_from_geojson_polygon,
    create_dem_grid_from_geotiff_polygon,
    create_land_cover_grid_from_geojson_polygon,
    create_building_height_grid_from_open_building_temporal_polygon
)
from .utils.lc import convert_land_cover, convert_land_cover_array
from .file.geojson import get_geojson_from_gpkg, save_geojson
from .utils.visualization import (
    get_land_cover_classes,
    visualize_land_cover_grid,
    visualize_numerical_grid,
    visualize_land_cover_grid_on_map,
    visualize_numerical_grid_on_map,
    visualize_building_height_grid_on_map,
    visualize_3d_voxel
)

# def get_land_cover_grid(rectangle_vertices, meshsize, source, output_dir="output", visualization=True):
def get_land_cover_grid(rectangle_vertices, meshsize, source, output_dir, **kwargs):

    print("Creating Land Use Land Cover grid\n ")
    print(f"Data source: {source}")
    
    # Initialize Earth Engine
    initialize_earth_engine()

    os.makedirs(output_dir, exist_ok=True)
    geotiff_path = os.path.join(output_dir, "land_cover.tif")

    if source == 'Urbanwatch':
        roi = get_roi(rectangle_vertices)
        collection_name = "projects/sat-io/open-datasets/HRLC/urban-watch-cities"
        image = get_ee_image_collection(collection_name, roi)
        save_geotiff(image, geotiff_path)
    elif source == 'ESA WorldCover':
        roi = get_roi(rectangle_vertices)
        save_geotiff_esa_land_cover(roi, geotiff_path)
    elif source == 'ESRI 10m Annual Land Cover':
        esri_landcover_year = kwargs.get("esri_landcover_year")
        roi = get_roi(rectangle_vertices)
        save_geotiff_esri_landcover(roi, geotiff_path, year=esri_landcover_year)
    elif source == 'Dynamic World V1':
        dynamic_world_date = kwargs.get("dynamic_world_date")
        roi = get_roi(rectangle_vertices)
        save_geotiff_dynamic_world_v1(roi, geotiff_path, dynamic_world_date)
    elif source == 'OpenEarthMapJapan':
        save_oemj_as_geotiff(rectangle_vertices, geotiff_path)   
    elif source == 'OpenStreetMap':
        land_cover_geojson = load_land_cover_geojson_from_osm(rectangle_vertices)
    
    land_cover_classes = get_land_cover_classes(source)

    if source == 'OpenStreetMap':
        land_cover_grid_str = create_land_cover_grid_from_geojson_polygon(land_cover_geojson, meshsize, source, rectangle_vertices)
    else:
        land_cover_grid_str = create_land_cover_grid_from_geotiff_polygon(geotiff_path, meshsize, land_cover_classes, rectangle_vertices)

    color_map = {cls: [r/255, g/255, b/255] for (r,g,b), cls in land_cover_classes.items()}
    # color_map['No Data'] = [0.5, 0.5, 0.5]

    if kwargs["gridvis"]:
        visualize_land_cover_grid(np.flipud(land_cover_grid_str), meshsize, color_map, land_cover_classes)
    land_cover_grid_int = convert_land_cover_array(land_cover_grid_str, land_cover_classes)

    return land_cover_grid_int

# def get_building_height_grid(rectangle_vertices, meshsize, source, output_dir="output", visualization=True, maptiler_API_key=None, file_path=None):
def get_building_height_grid(rectangle_vertices, meshsize, source, output_dir, **kwargs):

    # Initialize Earth Engine
    initialize_earth_engine()

    print("Creating Building Height grid\n ")
    print(f"Data source: {source}")

    os.makedirs(output_dir, exist_ok=True)
    
    if source == 'Microsoft Building Footprints':
        geojson_data = get_mbfp_geojson(output_dir, rectangle_vertices)
        # building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    elif source == 'OpenStreetMap':
        geojson_data = load_geojsons_from_openstreetmap(rectangle_vertices)
        # building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    elif source == 'OSM Buildings':
        geojson_data = load_geojsons_from_osmbuildings(rectangle_vertices)
        # building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    elif source == "Open Building 2.5D Temporal":
        building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_open_building_temporal_polygon(meshsize, rectangle_vertices, output_dir)
    elif source == 'EUBUCCO v0.1':
        geojson_data = load_geojson_from_eubucco(rectangle_vertices, output_dir)
        # building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    elif source == "OpenMapTiles":
        geojson_data = load_geojsons_from_openmaptiles(rectangle_vertices, kwargs["maptiler_API_key"])
        # building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    elif source == "Overture":
        geojson_data = load_geojsons_from_overture(rectangle_vertices)
        # building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    elif source == "Local file":
        _, extension = os.path.splitext(kwargs["building_path"])
        if extension == ".gpkg":
            geojson_data = get_geojson_from_gpkg(kwargs["building_path"], rectangle_vertices)
            # building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    
    building_complementary_source = kwargs.get("building_complementary_source") 

    if (building_complementary_source is None) or (building_complementary_source=='None'):
        if source != "Open Building 2.5D Temporal":
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    else:
        if building_complementary_source == "Open Building 2.5D Temporal":
            roi = get_roi(rectangle_vertices)
            os.makedirs(output_dir, exist_ok=True)
            geotiff_path_comp = os.path.join(output_dir, "building_height.tif")
            save_geotiff_open_buildings_temporal(roi, geotiff_path_comp)
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices, geotiff_path_comp=geotiff_path_comp)   
        else:
            if building_complementary_source == 'Microsoft Building Footprints':
                geojson_data_comp = get_mbfp_geojson(output_dir, rectangle_vertices)
                # building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
            elif building_complementary_source == 'OpenStreetMap':
                geojson_data_comp = load_geojsons_from_openstreetmap(rectangle_vertices)
                # building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
            elif building_complementary_source == 'OSM Buildings':
                geojson_data_comp = load_geojsons_from_osmbuildings(rectangle_vertices)
                # building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
            elif building_complementary_source == 'EUBUCCO v0.1':
                geojson_data_comp = load_geojson_from_eubucco(rectangle_vertices, output_dir)
                # building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
            elif building_complementary_source == "OpenMapTiles":
                geojson_data_comp = load_geojsons_from_openmaptiles(rectangle_vertices, kwargs["maptiler_API_key"])
                # building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
            elif building_complementary_source == "Overture":
                geojson_data_comp = load_geojsons_from_overture(rectangle_vertices)
                # building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
            elif building_complementary_source == "Local file":
                _, extension = os.path.splitext(kwargs["building_complementary_path"])
                if extension == ".gpkg":
                    geojson_data_comp = get_geojson_from_gpkg(kwargs["building_complementary_path"], rectangle_vertices)
            complement_building_footprints = kwargs.get("complement_building_footprints")
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices, geojson_data_comp=geojson_data_comp, complement_building_footprints=complement_building_footprints)
        
    # if source == 'Microsoft Building Footprints':
    #     geojson_data = get_mbfp_geojson(output_dir, rectangle_vertices)
    #     building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    # elif source == 'OpenStreetMap':
    #     geojson_data = load_geojsons_from_openstreetmap(rectangle_vertices)
    #     building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    # elif source == 'OSM Buildings':
    #     geojson_data = load_geojsons_from_osmbuildings(rectangle_vertices)
    #     building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    # elif source == "Open Building 2.5D Temporal":
    #     roi = get_roi(rectangle_vertices)
    #     os.makedirs(output_dir, exist_ok=True)
    #     geotiff_path = os.path.join(output_dir, "building_height.tif")
    #     save_geotiff_open_buildings_temporal(roi, geotiff_path)
    #     building_height_grid = create_height_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices)
    #     building_min_height_grid = np.empty(building_height_grid.shape, dtype=object)
    #     for i in range(building_height_grid.shape[0]):
    #         for j in range(building_height_grid.shape[1]):
    #             if building_height_grid[i, j] <= 0:
    #                 building_min_height_grid[i, j] = []
    #             else:
    #                 building_min_height_grid[i, j] = [[0, building_height_grid[i, j]]]
    #     filtered_buildings = []
    #     building_id_grid = np.zeros_like(building_height_grid, dtype=int)        
    #     # Get positions of non-zero elements
    #     non_zero_positions = np.nonzero(building_height_grid)        
    #     # Create sequential integers starting from 1
    #     num_non_zeros = len(non_zero_positions[0])
    #     sequence = np.arange(1, num_non_zeros + 1)        
    #     # Place sequential integers at non-zero positions
    #     building_id_grid[non_zero_positions] = sequence
    # elif source == 'EUBUCCO v0.1':
    #     geojson_data = load_geojson_from_eubucco(rectangle_vertices, output_dir)
    #     building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    # elif source == "OpenMapTiles":
    #     geojson_data = load_geojsons_from_openmaptiles(rectangle_vertices, kwargs["maptiler_API_key"])
    #     print(geojson_data)
    #     building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    # elif source == "Overture":
    #     geojson_data = load_geojsons_from_overture(rectangle_vertices)
    #     building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    # elif source == "OpenStreetMap & Microsoft Building Footprints":
    #     geojson_data = load_geojsons_from_openstreetmap(rectangle_vertices)
    #     geojson_data_comp = get_mbfp_geojson(output_dir, rectangle_vertices)
    #     building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices, geojson_data_comp=geojson_data_comp)
    # elif source == "OpenStreetMap & Open Building 2.5D Temporal":
    #     geojson_data = load_geojsons_from_openstreetmap(rectangle_vertices)
    #     roi = get_roi(rectangle_vertices)
    #     os.makedirs(output_dir, exist_ok=True)
    #     geotiff_path_comp = os.path.join(output_dir, "building_height.tif")
    #     save_geotiff_open_buildings_temporal(roi, geotiff_path_comp)
    #     building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices, geotiff_path_comp=geotiff_path_comp)   
    # elif source == "Local file":
    #     _, extension = os.path.splitext(kwargs["building_path"])
    #     if extension == ".gpkg":
    #         geojson_data = get_geojson_from_gpkg(kwargs["building_path"], rectangle_vertices)
    #         building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    # elif source == "OpenStreetMap & Local file":
    #     geojson_data = load_geojsons_from_openstreetmap(rectangle_vertices)
    #     _, extension = os.path.splitext(kwargs["building_path"])
    #     if extension == ".gpkg":
    #         geojson_data_comp = get_geojson_from_gpkg(kwargs["building_path"], rectangle_vertices)
    #     complement_polygon = kwargs.get("complement_polygon")
    #     building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices, geojson_data_comp=geojson_data_comp, complement_polygon=complement_polygon)

    if kwargs["gridvis"]:
        visualize_numerical_grid(np.flipud(building_height_grid), meshsize, "building height (m)", cmap='viridis', label='Value')

    return building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings#, buildings_found

def get_canopy_height_grid(rectangle_vertices, meshsize, source, output_dir, **kwargs):
# def get_canopy_height_grid(rectangle_vertices, meshsize, source, output_dir="output", visualization=True):

    print("Creating Canopy Height grid\n ")
    print(f"Data source: High Resolution Canopy Height Maps by WRI and Meta")
    
    # Initialize Earth Engine
    initialize_earth_engine()

    os.makedirs(output_dir, exist_ok=True)
    geotiff_path = os.path.join(output_dir, "canopy_height.tif")
    
    roi = get_roi(rectangle_vertices)
    if source == 'High Resolution 1m Global Canopy Height Maps':
        collection_name = "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"  
        image = get_ee_image_collection(collection_name, roi)      
    elif source == 'ETH Global Sentinel-2 10m Canopy Height (2020)':
        collection_name = "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"
        image = get_ee_image(collection_name, roi)
    save_geotiff(image, geotiff_path, resolution=meshsize)  

    canopy_height_grid = create_height_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices)

    if kwargs["gridvis"]:
        visualize_numerical_grid(np.flipud(canopy_height_grid), meshsize, "Tree canopy height", cmap='Greens', label='Tree canopy height (m)')

    return canopy_height_grid

# def get_dem_grid(rectangle_vertices, meshsize, source, output_dir="output", visualization=True):
def get_dem_grid(rectangle_vertices, meshsize, source, output_dir, **kwargs):

    print("Creating Digital Elevation Model (DEM) grid\n ")
    print(f"Data source: {source}")

    # Initialize Earth Engine
    initialize_earth_engine()

    geotiff_path = os.path.join(output_dir, "dem.tif")

    buffer_distance = 100
    roi = get_roi(rectangle_vertices)
    roi_buffered = roi.buffer(buffer_distance)
    image = get_dem_image(roi_buffered, source)
    # if source in ["England 1m DTM", 'USGS 3DEP 1m', 'DEM France 1m']:
    #     save_geotiff(image, geotiff_path, scale=1, region=roi_buffered, crs='EPSG:4326')
    # elif source in ['DEM France 5m', 'AUSTRALIA 5M DEM']:
    #     save_geotiff(image, geotiff_path, scale=5, region=roi_buffered, crs='EPSG:4326')
    if source in ["England 1m DTM", 'USGS 3DEP 1m', 'DEM France 1m', 'DEM France 5m', 'AUSTRALIA 5M DEM']:
        save_geotiff(image, geotiff_path, scale=meshsize, region=roi_buffered, crs='EPSG:4326')
    else:
        save_geotiff(image, geotiff_path, scale=30, region=roi_buffered)

    dem_interpolation = kwargs.get("dem_interpolation")
    dem_grid = create_dem_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices, dem_interpolation=dem_interpolation)
    if kwargs["gridvis"]:
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
    
def create_3d_voxel(building_height_grid_ori, building_min_height_grid_ori, building_id_grid_ori, land_cover_grid_ori, dem_grid_ori, tree_grid_ori, voxel_size, land_cover_source, **kwargs):

    # building_min_height_grid_ori = kwargs['building_min_height_grid_ori']

    print("Generating 3D voxel data")
    if (land_cover_source == 'OpenStreetMap'):
        land_cover_grid_converted = land_cover_grid_ori
    else:
        land_cover_grid_converted = convert_land_cover(land_cover_grid_ori, land_cover_source=land_cover_source)

    # Prepare grids
    building_height_grid = np.flipud(np.nan_to_num(building_height_grid_ori, nan=10.0))#set 10m height to nan
    building_min_height_grid = np.flipud(replace_nan_in_nested(building_min_height_grid_ori))#set 10m height to nan
    building_id_grid = np.flipud(building_id_grid_ori)
    land_cover_grid = np.flipud(land_cover_grid_converted.copy()) + 1
    dem_grid = np.flipud(dem_grid_ori.copy()) - np.min(dem_grid_ori)
    # building_nr_grid = group_and_label_cells(building_id_grid)
    dem_grid = process_grid(building_id_grid, dem_grid)
    tree_grid = np.flipud(tree_grid_ori.copy())

    # Ensure all input grids have the same shape
    assert building_height_grid.shape == land_cover_grid.shape == dem_grid.shape == tree_grid.shape, "Input grids must have the same shape"

    # Get the dimensions of the input grids
    rows, cols = building_height_grid.shape

    # Calculate the maximum height needed for the 3D array
    max_height = int(np.ceil(np.max(building_height_grid + dem_grid + tree_grid) / voxel_size))+1

    # Create an empty 3D array
    voxel_grid = np.zeros((rows, cols, max_height), dtype=np.int32)

    trunk_height_ratio = kwargs.get("trunk_height_ratio")
    if trunk_height_ratio is None:
        trunk_height_ratio = 11.76 / 19.98

    # Fill the 3D array
    for i in range(rows):
        for j in range(cols):
            ground_level = int(dem_grid[i, j] / voxel_size + 0.5) + 1 

            tree_height = tree_grid[i, j]
            land_cover = land_cover_grid[i, j]

            # Fill underground cells with -1
            voxel_grid[i, j, :ground_level] = -1

            # Set ground level cell to land cover
            if land_cover == 6:
                voxel_grid[i, j, :ground_level] = land_cover
            else:
                voxel_grid[i, j, ground_level-1] = land_cover

            # Fill tree crown with value -2
            if tree_height > 0:
                crown_base_height = (tree_height * trunk_height_ratio)
                crown_base_height_level = int(crown_base_height / voxel_size + 0.5)
                crown_top_height = tree_height
                crown_top_height_level = int(crown_top_height / voxel_size + 0.5)
                if (crown_top_height_level == crown_base_height_level) and (crown_base_height_level>0):
                    crown_base_height_level -= 1
                tree_start = ground_level + crown_base_height_level
                tree_end = ground_level + crown_top_height_level
                voxel_grid[i, j, tree_start:tree_end] = -2
                # print(f"crown_top_height: {crown_top_height}")
                # print(f"crown_base_height: {crown_base_height}")
                # print(f"crown_top_height_level: {crown_top_height_level}")
                # print(f"crown_base_height_level: {crown_base_height_level}")
                # print(f"tree_start: {tree_start}")
                # print(f"tree_end: {tree_end}")
                # print(voxel_grid[i, j, :tree_end])

            # Fill building with value -3
            for k in building_min_height_grid[i, j]:
                building_min_height = int(k[0] / voxel_size + 0.5)
                building_height = int(k[1] / voxel_size + 0.5)
                voxel_grid[i, j, ground_level+building_min_height:ground_level+building_height] = -3

    return voxel_grid

def create_3d_voxel_individuals(building_height_grid_ori, land_cover_grid_ori, dem_grid_ori, tree_grid_ori, voxel_size, land_cover_source, layered_interval=None):
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
    land_cover_voxel_grid = np.zeros((rows, cols, max_height), dtype=np.int32)
    building_voxel_grid = np.zeros((rows, cols, max_height), dtype=np.int32)
    tree_voxel_grid = np.zeros((rows, cols, max_height), dtype=np.int32)
    dem_voxel_grid = np.zeros((rows, cols, max_height), dtype=np.int32)

    # Fill the 3D array
    for i in range(rows):
        for j in range(cols):
            ground_level = int(dem_grid[i, j] / voxel_size + 0.5)
            building_height = int(building_height_grid[i, j] / voxel_size + 0.5)
            tree_height = int(tree_grid[i, j] / voxel_size + 0.5)
            land_cover = land_cover_grid[i, j]

            # Fill underground cells with -1
            dem_voxel_grid[i, j, :ground_level+1] = -1

            # Set ground level cell to land cover
            land_cover_voxel_grid[i, j, 0] = land_cover

            # Fill tree crown with value -2
            if tree_height > 0:
                tree_voxel_grid[i, j, :tree_height] = -2

            # Fill building with value -3
            if building_height > 0:
                building_voxel_grid[i, j, :building_height] = -3
    
    if not layered_interval:
        layered_interval = max(max_height, int(dem_grid.shape[0]/4 + 0.5))

    extract_height = min(layered_interval, max_height)

    layered_voxel_grid = np.zeros((rows, cols, layered_interval*4), dtype=np.int32)
    layered_voxel_grid[:, :, :extract_height] = dem_voxel_grid[:, :, :extract_height]
    layered_voxel_grid[:, :, layered_interval:layered_interval+extract_height] = land_cover_voxel_grid[:, :, :extract_height]
    layered_voxel_grid[:, :, 2*layered_interval:2*layered_interval+extract_height] = building_voxel_grid[:, :, :extract_height]
    layered_voxel_grid[:, :, 3*layered_interval:3*layered_interval+extract_height] = tree_voxel_grid[:, :, :extract_height]

    return land_cover_voxel_grid, building_voxel_grid, tree_voxel_grid, dem_voxel_grid, layered_voxel_grid

# def get_voxelcity(rectangle_vertices, building_source, land_cover_source, canopy_height_source, dem_source, meshsize, remove_perimeter_object=None, mapvis=False, voxelvis=False, maptiler_API_key=None, img_save_path=None):
def get_voxelcity(rectangle_vertices, building_source, land_cover_source, canopy_height_source, dem_source, meshsize, **kwargs):

    # if kwargs["output_dir"]:
    #     output_dir = kwargs["output_dir"]
    # else:
    #     output_dir = 'output'
    output_dir = kwargs.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
        
    # Remove 'output_dir' from kwargs to prevent duplication
    kwargs.pop('output_dir', None)

    #prepare of grid data
    land_cover_grid = get_land_cover_grid(rectangle_vertices, meshsize, land_cover_source, output_dir, **kwargs)
    building_height_grid, building_min_height_grid, building_id_grid, building_geojson = get_building_height_grid(rectangle_vertices, meshsize, building_source, output_dir, **kwargs)
    save_path = f"{output_dir}/building.geojson"
    
    # print(building_geojson[0])
    save_geojson(building_geojson, save_path)
    canopy_height_grid = get_canopy_height_grid(rectangle_vertices, meshsize, canopy_height_source, output_dir, **kwargs)
    if dem_source == "Flat":
        dem_grid = np.zeros_like(land_cover_grid)
    else:
        dem_grid = get_dem_grid(rectangle_vertices, meshsize, dem_source, output_dir, **kwargs)

    min_canopy_height = kwargs.get("min_canopy_height")
    if min_canopy_height is not None:
        canopy_height_grid[canopy_height_grid < kwargs["min_canopy_height"]] = 0        

    remove_perimeter_object = kwargs.get("remove_perimeter_object")
    if remove_perimeter_object > 0:
        w_peri = int(remove_perimeter_object * building_height_grid.shape[0] + 0.5)
        h_peri = int(remove_perimeter_object * building_height_grid.shape[1] + 0.5)
        # building_height_grid[:w_peri, :] = building_height_grid[-w_peri:, :] = building_height_grid[:, :h_peri] = building_height_grid[:, -h_peri:] = 0
        # building_min_height_grid[:w_peri, :] = building_min_height_grid[-w_peri:, :] = building_min_height_grid[:, :h_peri] = building_min_height_grid[:, -h_peri:] = []
        canopy_height_grid[:w_peri, :] = canopy_height_grid[-w_peri:, :] = canopy_height_grid[:, :h_peri] = canopy_height_grid[:, -h_peri:] = 0

        # Using np.concatenate()
        ids1 = np.unique(building_id_grid[:w_peri, :][building_id_grid[:w_peri, :] > 0])
        ids2 = np.unique(building_id_grid[-w_peri:, :][building_id_grid[-w_peri:, :] > 0])
        ids3 = np.unique(building_id_grid[:, :h_peri][building_id_grid[:, :h_peri] > 0])
        ids4 = np.unique(building_id_grid[:, -h_peri:][building_id_grid[:, -h_peri:] > 0])
        remove_ids = np.concatenate((ids1, ids2, ids3, ids4))
        for remove_id in remove_ids:
            positions = np.where(building_id_grid == remove_id)
            building_height_grid[positions] = 0
            building_min_height_grid[positions] = [[] for _ in range(len(building_min_height_grid[positions]))]

    #display grid data on basemap
    mapvis = kwargs.get("mapvis")
    if mapvis:
        visualize_land_cover_grid_on_map(land_cover_grid, rectangle_vertices, meshsize, source = land_cover_source)
        visualize_building_height_grid_on_map(building_height_grid, building_geojson, rectangle_vertices, meshsize)
        visualize_numerical_grid_on_map(canopy_height_grid, rectangle_vertices, meshsize, "canopy_height")
        visualize_numerical_grid_on_map(dem_grid, rectangle_vertices, meshsize, "dem")

    #prepare 3D voxel grid  
    voxelcity_grid = create_3d_voxel(building_height_grid, building_min_height_grid, building_id_grid, land_cover_grid, dem_grid, canopy_height_grid, meshsize, land_cover_source)

    voxelvis = kwargs.get("voxelvis")
    #display grid data in 3D
    if voxelvis:
        new_height = int(550/meshsize+0.5)     
        voxelcity_grid_vis = np.zeros((voxelcity_grid.shape[0], voxelcity_grid.shape[1], new_height))
        voxelcity_grid_vis[:, :, :voxelcity_grid.shape[2]] = voxelcity_grid
        voxelcity_grid_vis[-1, -1, -1] = -99 #To fix camera location and angle of view
        visualize_3d_voxel(voxelcity_grid_vis, voxel_size=meshsize, save_path=kwargs["voxelvis_img_save_path"])
        # visualize_3d_voxel(voxelcity_grid, voxel_size=meshsize, save_path=img_save_path)

    return voxelcity_grid, building_height_grid, building_min_height_grid, building_id_grid, canopy_height_grid, land_cover_grid, dem_grid, building_geojson

def replace_nan_in_nested(arr, replace_value=10.0):
    # Convert array to list for easier manipulation
    arr = arr.tolist()
    
    # Iterate through all dimensions
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            # Check if the element is a list
            if arr[i][j]:  # if not empty list
                for k in range(len(arr[i][j])):
                    # For each innermost list
                    if isinstance(arr[i][j][k], list):
                        for l in range(len(arr[i][j][k])):
                            if isinstance(arr[i][j][k][l], float) and np.isnan(arr[i][j][k][l]):
                                arr[i][j][k][l] = replace_value
    
    return np.array(arr, dtype=object)