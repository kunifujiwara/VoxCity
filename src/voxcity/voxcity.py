"""Main module for voxcity.

This module provides functions to generate 3D voxel representations of cities using various data sources.
It handles land cover, building heights, canopy heights, and digital elevation models to create detailed
3D city models.

The main functions are:
- get_land_cover_grid: Creates a grid of land cover classifications
- get_building_height_grid: Creates a grid of building heights 
- get_canopy_height_grid: Creates a grid of tree canopy heights
- get_dem_grid: Creates a digital elevation model grid
- create_3d_voxel: Combines the grids into a 3D voxel representation
- create_3d_voxel_individuals: Creates separate voxel grids for each component
- get_voxcity: Main function to generate a complete voxel city model
"""

import numpy as np
import os

# Local application/library specific imports
from .download.mbfp import get_mbfp_geojson
from .download.osm import load_geojsons_from_openstreetmap, load_land_cover_geojson_from_osm
from .download.oemj import save_oemj_as_geotiff
from .download.omt import load_geojsons_from_openmaptiles
from .download.eubucco import load_geojson_from_eubucco
from .download.overture import load_geojsons_from_overture
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

def get_land_cover_grid(rectangle_vertices, meshsize, source, output_dir, **kwargs):
    """Creates a grid of land cover classifications.

    Args:
        rectangle_vertices: List of coordinates defining the area of interest
        meshsize: Size of each grid cell in meters
        source: Data source for land cover (e.g. 'ESA WorldCover', 'OpenStreetMap')
        output_dir: Directory to save output files
        **kwargs: Additional arguments including:
            - esri_landcover_year: Year for ESRI land cover data
            - dynamic_world_date: Date for Dynamic World data
            - gridvis: Whether to visualize the grid

    Returns:
        numpy.ndarray: Grid of land cover classifications as integer values
    """

    print("Creating Land Use Land Cover grid\n ")
    print(f"Data source: {source}")
    
    # Initialize Earth Engine for accessing satellite data
    initialize_earth_engine()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    geotiff_path = os.path.join(output_dir, "land_cover.tif")

    # Get land cover data based on selected source
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
        # For OSM, we get data directly as GeoJSON instead of GeoTIFF
        land_cover_geojson = load_land_cover_geojson_from_osm(rectangle_vertices)
    
    # Get mapping of land cover classes for the selected source
    land_cover_classes = get_land_cover_classes(source)

    # Create grid from either GeoJSON (OSM) or GeoTIFF (other sources)
    if source == 'OpenStreetMap':
        land_cover_grid_str = create_land_cover_grid_from_geojson_polygon(land_cover_geojson, meshsize, source, rectangle_vertices)
    else:
        land_cover_grid_str = create_land_cover_grid_from_geotiff_polygon(geotiff_path, meshsize, land_cover_classes, rectangle_vertices)

    # Create color map for visualization, scaling RGB values to 0-1 range
    color_map = {cls: [r/255, g/255, b/255] for (r,g,b), cls in land_cover_classes.items()}

    # Visualize grid if requested
    grid_vis = kwargs.get("gridvis", True)    
    if grid_vis:
        visualize_land_cover_grid(np.flipud(land_cover_grid_str), meshsize, color_map, land_cover_classes)
    
    # Convert string labels to integer codes
    land_cover_grid_int = convert_land_cover_array(land_cover_grid_str, land_cover_classes)

    return land_cover_grid_int

# def get_building_height_grid(rectangle_vertices, meshsize, source, output_dir="output", visualization=True, maptiler_API_key=None, file_path=None):
def get_building_height_grid(rectangle_vertices, meshsize, source, output_dir, **kwargs):
    """Creates a grid of building heights.

    Args:
        rectangle_vertices: List of coordinates defining the area of interest
        meshsize: Size of each grid cell in meters
        source: Data source for buildings (e.g. 'OpenStreetMap', 'Microsoft Building Footprints')
        output_dir: Directory to save output files
        **kwargs: Additional arguments including:
            - maptiler_API_key: API key for MapTiler
            - building_path: Path to local building data file
            - building_complementary_source: Additional building data source
            - gridvis: Whether to visualize the grid

    Returns:
        tuple:
            - numpy.ndarray: Grid of building heights
            - numpy.ndarray: Grid of building minimum heights
            - numpy.ndarray: Grid of building IDs
            - list: Filtered building features
    """

    # Initialize Earth Engine for accessing satellite data
    initialize_earth_engine()

    print("Creating Building Height grid\n ")
    print(f"Data source: {source}")

    os.makedirs(output_dir, exist_ok=True)
    
    # Get building data from primary source
    if source == 'Microsoft Building Footprints':
        geojson_data = get_mbfp_geojson(output_dir, rectangle_vertices)
    elif source == 'OpenStreetMap':
        geojson_data = load_geojsons_from_openstreetmap(rectangle_vertices)
    elif source == "Open Building 2.5D Temporal":
        # Special case: directly creates grids without intermediate GeoJSON
        building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_open_building_temporal_polygon(meshsize, rectangle_vertices, output_dir)
    elif source == 'EUBUCCO v0.1':
        geojson_data = load_geojson_from_eubucco(rectangle_vertices, output_dir)
    elif source == "OpenMapTiles":
        geojson_data = load_geojsons_from_openmaptiles(rectangle_vertices, kwargs["maptiler_API_key"])
    elif source == "Overture":
        geojson_data = load_geojsons_from_overture(rectangle_vertices)
    elif source == "Local file":
        # Handle local GPKG files
        _, extension = os.path.splitext(kwargs["building_path"])
        if extension == ".gpkg":
            geojson_data = get_geojson_from_gpkg(kwargs["building_path"], rectangle_vertices)
    
    # Check for complementary building data source
    building_complementary_source = kwargs.get("building_complementary_source") 

    if (building_complementary_source is None) or (building_complementary_source=='None'):
        # Use only primary source
        if source != "Open Building 2.5D Temporal":
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices)
    else:
        # Handle complementary source
        if building_complementary_source == "Open Building 2.5D Temporal":
            # Special case: use temporal height data as complement
            roi = get_roi(rectangle_vertices)
            os.makedirs(output_dir, exist_ok=True)
            geotiff_path_comp = os.path.join(output_dir, "building_height.tif")
            save_geotiff_open_buildings_temporal(roi, geotiff_path_comp)
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices, geotiff_path_comp=geotiff_path_comp)   
        else:
            # Get complementary data from other sources
            if building_complementary_source == 'Microsoft Building Footprints':
                geojson_data_comp = get_mbfp_geojson(output_dir, rectangle_vertices)
            elif building_complementary_source == 'OpenStreetMap':
                geojson_data_comp = load_geojsons_from_openstreetmap(rectangle_vertices)
            elif building_complementary_source == 'OSM Buildings':
                geojson_data_comp = load_geojsons_from_osmbuildings(rectangle_vertices)
            elif building_complementary_source == 'EUBUCCO v0.1':
                geojson_data_comp = load_geojson_from_eubucco(rectangle_vertices, output_dir)
            elif building_complementary_source == "OpenMapTiles":
                geojson_data_comp = load_geojsons_from_openmaptiles(rectangle_vertices, kwargs["maptiler_API_key"])
            elif building_complementary_source == "Overture":
                geojson_data_comp = load_geojsons_from_overture(rectangle_vertices)
            elif building_complementary_source == "Local file":
                _, extension = os.path.splitext(kwargs["building_complementary_path"])
                if extension == ".gpkg":
                    geojson_data_comp = get_geojson_from_gpkg(kwargs["building_complementary_path"], rectangle_vertices)
            
            # Option to complement footprints only or both footprints and heights
            complement_building_footprints = kwargs.get("complement_building_footprints")
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_geojson_polygon(geojson_data, meshsize, rectangle_vertices, geojson_data_comp=geojson_data_comp, complement_building_footprints=complement_building_footprints)

    # Visualize grid if requested
    grid_vis = kwargs.get("gridvis", True)    
    if grid_vis:
        visualize_numerical_grid(np.flipud(building_height_grid), meshsize, "building height (m)", cmap='viridis', label='Value')

    return building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings

def get_canopy_height_grid(rectangle_vertices, meshsize, source, output_dir, **kwargs):
    """Creates a grid of tree canopy heights.

    Args:
        rectangle_vertices: List of coordinates defining the area of interest
        meshsize: Size of each grid cell in meters
        source: Data source for canopy heights
        output_dir: Directory to save output files
        **kwargs: Additional arguments including:
            - gridvis: Whether to visualize the grid

    Returns:
        numpy.ndarray: Grid of canopy heights
    """

    print("Creating Canopy Height grid\n ")
    print(f"Data source: High Resolution Canopy Height Maps by WRI and Meta")
    
    # Initialize Earth Engine for accessing satellite data
    initialize_earth_engine()

    os.makedirs(output_dir, exist_ok=True)
    geotiff_path = os.path.join(output_dir, "canopy_height.tif")
    
    # Get region of interest and canopy height data
    roi = get_roi(rectangle_vertices)
    if source == 'High Resolution 1m Global Canopy Height Maps':
        collection_name = "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"  
        image = get_ee_image_collection(collection_name, roi)      
    elif source == 'ETH Global Sentinel-2 10m Canopy Height (2020)':
        collection_name = "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"
        image = get_ee_image(collection_name, roi)
    
    # Save canopy height data as GeoTIFF
    save_geotiff(image, geotiff_path, resolution=meshsize)  

    # Create height grid from GeoTIFF
    canopy_height_grid = create_height_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices)

    # Visualize grid if requested
    grid_vis = kwargs.get("gridvis", True)    
    if grid_vis:
        visualize_numerical_grid(np.flipud(canopy_height_grid), meshsize, "Tree canopy height", cmap='Greens', label='Tree canopy height (m)')

    return canopy_height_grid

def get_dem_grid(rectangle_vertices, meshsize, source, output_dir, **kwargs):
    """Creates a digital elevation model grid.

    Args:
        rectangle_vertices: List of coordinates defining the area of interest
        meshsize: Size of each grid cell in meters
        source: Data source for DEM
        output_dir: Directory to save output files
        **kwargs: Additional arguments including:
            - dem_interpolation: Interpolation method for DEM
            - gridvis: Whether to visualize the grid

    Returns:
        numpy.ndarray: Grid of elevation values
    """

    print("Creating Digital Elevation Model (DEM) grid\n ")
    print(f"Data source: {source}")

    # Initialize Earth Engine for accessing elevation data
    initialize_earth_engine()

    geotiff_path = os.path.join(output_dir, "dem.tif")

    # Add buffer around ROI to ensure smooth interpolation at edges
    buffer_distance = 100
    roi = get_roi(rectangle_vertices)
    roi_buffered = roi.buffer(buffer_distance)
    
    # Get DEM data
    image = get_dem_image(roi_buffered, source)
    
    # Save DEM data with appropriate resolution based on source
    if source in ["England 1m DTM", 'USGS 3DEP 1m', 'DEM France 1m', 'DEM France 5m', 'AUSTRALIA 5M DEM']:
        save_geotiff(image, geotiff_path, scale=meshsize, region=roi_buffered, crs='EPSG:4326')
    else:
        # Default to 30m resolution for other sources
        save_geotiff(image, geotiff_path, scale=30, region=roi_buffered)

    # Create DEM grid with optional interpolation method
    dem_interpolation = kwargs.get("dem_interpolation")
    dem_grid = create_dem_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices, dem_interpolation=dem_interpolation)

    # Visualize grid if requested
    grid_vis = kwargs.get("gridvis", True)    
    if grid_vis:
        visualize_numerical_grid(np.flipud(dem_grid), meshsize, title='Digital Elevation Model', cmap='terrain', label='Elevation (m)')

    return dem_grid

def create_3d_voxel(building_height_grid_ori, building_min_height_grid_ori, building_id_grid_ori, land_cover_grid_ori, dem_grid_ori, tree_grid_ori, voxel_size, land_cover_source, **kwargs):
    """Creates a 3D voxel representation combining all input grids.

    Args:
        building_height_grid_ori: Grid of building heights
        building_min_height_grid_ori: Grid of building minimum heights
        building_id_grid_ori: Grid of building IDs
        land_cover_grid_ori: Grid of land cover classifications
        dem_grid_ori: Grid of elevation values
        tree_grid_ori: Grid of tree heights
        voxel_size: Size of each voxel in meters
        land_cover_source: Source of land cover data
        **kwargs: Additional arguments including:
            - trunk_height_ratio: Ratio of trunk height to total tree height

    Returns:
        numpy.ndarray: 3D voxel grid with encoded values for different features
    """

    print("Generating 3D voxel data")
    
    # Convert land cover values if not from OpenStreetMap
    if (land_cover_source == 'OpenStreetMap'):
        land_cover_grid_converted = land_cover_grid_ori
    else:
        land_cover_grid_converted = convert_land_cover(land_cover_grid_ori, land_cover_source=land_cover_source)

    # Prepare and flip all input grids vertically for consistent orientation
    building_height_grid = np.flipud(np.nan_to_num(building_height_grid_ori, nan=10.0)) # Replace NaN values with 10m height
    building_min_height_grid = np.flipud(replace_nan_in_nested(building_min_height_grid_ori)) # Replace NaN in nested arrays
    building_id_grid = np.flipud(building_id_grid_ori)
    land_cover_grid = np.flipud(land_cover_grid_converted.copy()) + 1 # Add 1 to avoid 0 values in land cover
    dem_grid = np.flipud(dem_grid_ori.copy()) - np.min(dem_grid_ori) # Normalize DEM to start at 0
    dem_grid = process_grid(building_id_grid, dem_grid) # Process DEM based on building footprints
    tree_grid = np.flipud(tree_grid_ori.copy())

    # Validate input dimensions
    assert building_height_grid.shape == land_cover_grid.shape == dem_grid.shape == tree_grid.shape, "Input grids must have the same shape"

    rows, cols = building_height_grid.shape

    # Calculate required height for 3D grid - add 1 to ensure enough space
    max_height = int(np.ceil(np.max(building_height_grid + dem_grid + tree_grid) / voxel_size))+1

    # Initialize empty 3D grid
    voxel_grid = np.zeros((rows, cols, max_height), dtype=np.int32)

    # Get trunk height ratio for trees, default based on typical tree proportions
    trunk_height_ratio = kwargs.get("trunk_height_ratio")
    if trunk_height_ratio is None:
        trunk_height_ratio = 11.76 / 19.98  # Default ratio based on typical tree proportions

    # Fill the 3D grid cell by cell
    for i in range(rows):
        for j in range(cols):
            # Calculate ground level in voxel units (+1 to ensure space for surface features)
            ground_level = int(dem_grid[i, j] / voxel_size + 0.5) + 1 

            tree_height = tree_grid[i, j]
            land_cover = land_cover_grid[i, j]

            # Fill underground voxels with -1
            voxel_grid[i, j, :ground_level] = -1

            # Set surface land cover value
            voxel_grid[i, j, ground_level-1] = land_cover

            # Process trees - split into trunk and crown sections
            if tree_height > 0:
                # Calculate crown base and top heights
                crown_base_height = (tree_height * trunk_height_ratio)
                crown_base_height_level = int(crown_base_height / voxel_size + 0.5)
                crown_top_height = tree_height
                crown_top_height_level = int(crown_top_height / voxel_size + 0.5)
                
                # Ensure minimum crown height of 1 voxel
                if (crown_top_height_level == crown_base_height_level) and (crown_base_height_level>0):
                    crown_base_height_level -= 1
                    
                # Calculate tree start and end positions relative to ground level
                tree_start = ground_level + crown_base_height_level
                tree_end = ground_level + crown_top_height_level
                
                # Fill tree crown voxels with -2
                voxel_grid[i, j, tree_start:tree_end] = -2

            # Process buildings - handle multiple height segments
            for k in building_min_height_grid[i, j]:
                building_min_height = int(k[0] / voxel_size + 0.5)  # Lower height of building segment
                building_height = int(k[1] / voxel_size + 0.5)      # Upper height of building segment
                # Fill building voxels with -3
                voxel_grid[i, j, ground_level+building_min_height:ground_level+building_height] = -3

    return voxel_grid

def create_3d_voxel_individuals(building_height_grid_ori, land_cover_grid_ori, dem_grid_ori, tree_grid_ori, voxel_size, land_cover_source, layered_interval=None):
    """Creates separate 3D voxel grids for each component.

    Args:
        building_height_grid_ori: Grid of building heights
        land_cover_grid_ori: Grid of land cover classifications
        dem_grid_ori: Grid of elevation values
        tree_grid_ori: Grid of tree heights
        voxel_size: Size of each voxel in meters
        land_cover_source: Source of land cover data
        layered_interval: Interval for layered output

    Returns:
        tuple:
            - numpy.ndarray: Land cover voxel grid
            - numpy.ndarray: Building voxel grid
            - numpy.ndarray: Tree voxel grid
            - numpy.ndarray: DEM voxel grid
            - numpy.ndarray: Combined layered voxel grid
    """

    print("Generating 3D voxel data")
    # Convert land cover values if not from OpenEarthMapJapan
    if land_cover_source != 'OpenEarthMapJapan':
        land_cover_grid_converted = convert_land_cover(land_cover_grid_ori, land_cover_source=land_cover_source)  
    else:
        land_cover_grid_converted = land_cover_grid_ori      

    # Prepare and flip all input grids vertically
    building_height_grid = np.flipud(building_height_grid_ori.copy())
    land_cover_grid = np.flipud(land_cover_grid_converted.copy()) + 1  # Add 1 to avoid 0 values
    dem_grid = np.flipud(dem_grid_ori.copy()) - np.min(dem_grid_ori)  # Normalize DEM to start at 0
    building_nr_grid = group_and_label_cells(np.flipud(building_height_grid_ori.copy()))
    dem_grid = process_grid(building_nr_grid, dem_grid)  # Process DEM based on building footprints
    tree_grid = np.flipud(tree_grid_ori.copy())

    # Validate input dimensions
    assert building_height_grid.shape == land_cover_grid.shape == dem_grid.shape == tree_grid.shape, "Input grids must have the same shape"

    rows, cols = building_height_grid.shape

    # Calculate required height for 3D grid
    max_height = int(np.ceil(np.max(building_height_grid + dem_grid + tree_grid) / voxel_size))

    # Initialize empty 3D grids for each component
    land_cover_voxel_grid = np.zeros((rows, cols, max_height), dtype=np.int32)
    building_voxel_grid = np.zeros((rows, cols, max_height), dtype=np.int32)
    tree_voxel_grid = np.zeros((rows, cols, max_height), dtype=np.int32)
    dem_voxel_grid = np.zeros((rows, cols, max_height), dtype=np.int32)

    # Fill individual component grids
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
    
    # Set default layered interval if not provided
    if not layered_interval:
        layered_interval = max(max_height, int(dem_grid.shape[0]/4 + 0.5))

    # Create combined layered visualization
    extract_height = min(layered_interval, max_height)
    layered_voxel_grid = np.zeros((rows, cols, layered_interval*4), dtype=np.int32)
    
    # Stack components in layers with equal spacing
    layered_voxel_grid[:, :, :extract_height] = dem_voxel_grid[:, :, :extract_height]
    layered_voxel_grid[:, :, layered_interval:layered_interval+extract_height] = land_cover_voxel_grid[:, :, :extract_height]
    layered_voxel_grid[:, :, 2*layered_interval:2*layered_interval+extract_height] = building_voxel_grid[:, :, :extract_height]
    layered_voxel_grid[:, :, 3*layered_interval:3*layered_interval+extract_height] = tree_voxel_grid[:, :, :extract_height]

    return land_cover_voxel_grid, building_voxel_grid, tree_voxel_grid, dem_voxel_grid, layered_voxel_grid

def get_voxcity(rectangle_vertices, building_source, land_cover_source, canopy_height_source, dem_source, meshsize, **kwargs):
    """Main function to generate a complete voxel city model.

    Args:
        rectangle_vertices: List of coordinates defining the area of interest
        building_source: Source for building height data (e.g. 'OSM', 'EUBUCCO')
        land_cover_source: Source for land cover data (e.g. 'ESA', 'ESRI') 
        canopy_height_source: Source for tree canopy height data
        dem_source: Source for digital elevation model data ('Flat' or other source)
        meshsize: Size of each grid cell in meters
        **kwargs: Additional keyword arguments including:
            - output_dir: Directory to save output files (default: 'output')
            - min_canopy_height: Minimum height threshold for tree canopy
            - remove_perimeter_object: Factor to remove objects near perimeter
            - mapvis: Whether to visualize grids on map
            - voxelvis: Whether to visualize 3D voxel model
            - voxelvis_img_save_path: Path to save 3D visualization

    Returns:
        tuple containing:
            - voxcity_grid: 3D voxel grid of the complete city model
            - building_height_grid: 2D grid of building heights
            - building_min_height_grid: 2D grid of minimum building heights
            - building_id_grid: 2D grid of building IDs
            - canopy_height_grid: 2D grid of tree canopy heights
            - land_cover_grid: 2D grid of land cover classifications
            - dem_grid: 2D grid of ground elevation
            - building_geojson: GeoJSON of building footprints and metadata
    """
    # Create output directory if it doesn't exist
    output_dir = kwargs.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
        
    # Remove 'output_dir' from kwargs to prevent duplication
    kwargs.pop('output_dir', None)

    # Generate all required 2D grids
    land_cover_grid = get_land_cover_grid(rectangle_vertices, meshsize, land_cover_source, output_dir, **kwargs)
    building_height_grid, building_min_height_grid, building_id_grid, building_geojson = get_building_height_grid(rectangle_vertices, meshsize, building_source, output_dir, **kwargs)
    
    # Save building data to GeoJSON
    save_path = f"{output_dir}/building.geojson"
    save_geojson(building_geojson, save_path)
    
    # Get canopy height data
    canopy_height_grid = get_canopy_height_grid(rectangle_vertices, meshsize, canopy_height_source, output_dir, **kwargs)
    
    # Handle DEM - either flat or from source
    if dem_source == "Flat":
        dem_grid = np.zeros_like(land_cover_grid)
    else:
        dem_grid = get_dem_grid(rectangle_vertices, meshsize, dem_source, output_dir, **kwargs)

    # Apply minimum canopy height threshold if specified
    min_canopy_height = kwargs.get("min_canopy_height")
    if min_canopy_height is not None:
        canopy_height_grid[canopy_height_grid < kwargs["min_canopy_height"]] = 0        

    # Remove objects near perimeter if specified
    remove_perimeter_object = kwargs.get("remove_perimeter_object")
    if (remove_perimeter_object is not None) and (remove_perimeter_object > 0):
        # Calculate perimeter width based on grid dimensions
        w_peri = int(remove_perimeter_object * building_height_grid.shape[0] + 0.5)
        h_peri = int(remove_perimeter_object * building_height_grid.shape[1] + 0.5)
        
        # Clear canopy heights in perimeter
        canopy_height_grid[:w_peri, :] = canopy_height_grid[-w_peri:, :] = canopy_height_grid[:, :h_peri] = canopy_height_grid[:, -h_peri:] = 0

        # Find building IDs in perimeter regions
        ids1 = np.unique(building_id_grid[:w_peri, :][building_id_grid[:w_peri, :] > 0])
        ids2 = np.unique(building_id_grid[-w_peri:, :][building_id_grid[-w_peri:, :] > 0])
        ids3 = np.unique(building_id_grid[:, :h_peri][building_id_grid[:, :h_peri] > 0])
        ids4 = np.unique(building_id_grid[:, -h_peri:][building_id_grid[:, -h_peri:] > 0])
        remove_ids = np.concatenate((ids1, ids2, ids3, ids4))
        
        # Remove buildings in perimeter
        for remove_id in remove_ids:
            positions = np.where(building_id_grid == remove_id)
            building_height_grid[positions] = 0
            building_min_height_grid[positions] = [[] for _ in range(len(building_min_height_grid[positions]))]

    # Visualize 2D grids on map if requested
    mapvis = kwargs.get("mapvis")
    if mapvis:
        visualize_land_cover_grid_on_map(land_cover_grid, rectangle_vertices, meshsize, source = land_cover_source)
        visualize_building_height_grid_on_map(building_height_grid, building_geojson, rectangle_vertices, meshsize)
        visualize_numerical_grid_on_map(canopy_height_grid, rectangle_vertices, meshsize, "canopy_height")
        visualize_numerical_grid_on_map(dem_grid, rectangle_vertices, meshsize, "dem")

    # Generate 3D voxel grid
    voxcity_grid = create_3d_voxel(building_height_grid, building_min_height_grid, building_id_grid, land_cover_grid, dem_grid, canopy_height_grid, meshsize, land_cover_source)

    # Visualize 3D model if requested
    voxelvis = kwargs.get("voxelvis")
    if voxelvis:
        # Create taller visualization grid with fixed height
        new_height = int(550/meshsize+0.5)     
        voxcity_grid_vis = np.zeros((voxcity_grid.shape[0], voxcity_grid.shape[1], new_height))
        voxcity_grid_vis[:, :, :voxcity_grid.shape[2]] = voxcity_grid
        voxcity_grid_vis[-1, -1, -1] = -99  # Add marker to fix camera location and angle of view
        visualize_3d_voxel(voxcity_grid_vis, voxel_size=meshsize, save_path=kwargs["voxelvis_img_save_path"])

    return voxcity_grid, building_height_grid, building_min_height_grid, building_id_grid, canopy_height_grid, land_cover_grid, dem_grid, building_geojson

def replace_nan_in_nested(arr, replace_value=10.0):
    """Replace NaN values in a nested array structure with a specified value.

    Args:
        arr: Numpy array containing nested lists and potentially NaN values
        replace_value: Value to replace NaN with (default: 10.0)

    Returns:
        Numpy array with NaN values replaced
    """
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