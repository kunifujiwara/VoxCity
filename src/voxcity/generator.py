"""Main module for voxcity.

This module provides functions to generate 3D voxel representations of cities using various data sources.
It handles land cover, building heights, canopy heights, and digital elevation models to create detailed
3D city models.

Orientation contract:
- All 2D grids in this module use the canonical internal orientation "north_up":
  row 0 is the northern/top row and the last row is the southern/bottom row.
- Columns increase eastward: column 0 is the western/leftmost column and
  indices increase toward the east/right.
- Processing functions accept and return grids in north_up orientation.
- Visualization utilities may vertically flip arrays for display purposes only.
- 3D indexing follows (row, col, z) = (north→south, west→east, ground→up).

The main functions are:
- get_land_cover_grid: Creates a grid of land cover classifications
- get_building_height_grid: Creates a grid of building heights (supports GeoDataFrame input)
- get_canopy_height_grid: Creates a grid of tree canopy heights
- get_dem_grid: Creates a digital elevation model grid
- create_3d_voxel: Combines the grids into a 3D voxel representation
- create_3d_voxel_individuals: Creates separate voxel grids for each component
- get_voxcity: Main function to generate a complete voxel city model (supports GeoDataFrame input)

Key Features:
- Support for multiple data sources (OpenStreetMap, ESA WorldCover, Google Earth Engine, etc.)
- Direct GeoDataFrame input for building data (useful for custom datasets)
- 3D voxel generation with configurable resolution
- Visualization capabilities for both 2D grids and 3D models
- Data export in various formats (GeoTIFF, GeoJSON, pickle)
"""

# Standard library imports
import numpy as np
import os
try:
    from numba import jit, prange
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available. Using optimized version without JIT compilation.")
    # Define dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Local application/library specific imports

# Data downloaders - modules for fetching geospatial data from various sources
from .downloader.mbfp import get_mbfp_gdf
from .downloader.osm import load_gdf_from_openstreetmap, load_land_cover_gdf_from_osm
from .downloader.oemj import save_oemj_as_geotiff
# from .downloader.omt import load_gdf_from_openmaptiles
from .downloader.eubucco import load_gdf_from_eubucco
from .downloader.overture import load_gdf_from_overture
from .downloader.citygml import load_buid_dem_veg_from_citygml
from .downloader.gba import load_gdf_from_gba

# Google Earth Engine related imports - for satellite and elevation data
from .downloader.gee import (
    initialize_earth_engine,
    get_roi,
    get_ee_image_collection,
    get_ee_image,
    save_geotiff,
    get_dem_image,
    save_geotiff_esa_land_cover,
    save_geotiff_esri_landcover,
    save_geotiff_dynamic_world_v1,
    save_geotiff_open_buildings_temporal,
    save_geotiff_dsm_minus_dtm
)

# Grid processing functions - for converting geodata to raster grids
from .geoprocessor.grid import (
    group_and_label_cells, 
    process_grid,
    create_land_cover_grid_from_geotiff_polygon,
    create_height_grid_from_geotiff_polygon,
    create_building_height_grid_from_gdf_polygon,
    create_dem_grid_from_geotiff_polygon,
    create_land_cover_grid_from_gdf_polygon,
    create_building_height_grid_from_open_building_temporal_polygon,
    create_vegetation_height_grid_from_gdf_polygon,
    create_canopy_grids_from_tree_gdf,
    create_dem_grid_from_gdf_polygon
)

# Utility functions
from .utils.lc import convert_land_cover, convert_land_cover_array
from .geoprocessor.polygon import get_gdf_from_gpkg, save_geojson

# Visualization functions - for creating plots and 3D visualizations
from .utils.visualization import (
    get_land_cover_classes,
    visualize_land_cover_grid,
    visualize_numerical_grid,
    visualize_landcover_grid_on_basemap,
    visualize_numerical_grid_on_basemap,
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
            - default_land_cover_class: Default class for grid cells with no intersecting polygons (default: 'Developed space')

    Orientation:
        All inputs and the returned grid use north_up orientation (row 0 = north/top),
        columns increase eastward (col 0 = west/left). Downstream visualization may
        flip vertically for display, but processing remains north_up/eastward.

    Returns:
        numpy.ndarray: Grid of land cover classifications as integer values
    """

    print("Creating Land Use Land Cover grid\n ")
    print(f"Data source: {source}")
    
    # Initialize Earth Engine for satellite-based data sources
    # Skip initialization for local/vector data sources
    if source not in ["OpenStreetMap", "OpenEarthMapJapan"]:
        initialize_earth_engine()

    # Ensure output directory exists for saving intermediate files
    os.makedirs(output_dir, exist_ok=True)
    geotiff_path = os.path.join(output_dir, "land_cover.tif")

    # Handle different data sources - each requires specific processing
    # Satellite/raster-based sources are saved as GeoTIFF files
    if source == 'Urbanwatch':
        # Urban-focused land cover from satellite imagery
        roi = get_roi(rectangle_vertices)
        collection_name = "projects/sat-io/open-datasets/HRLC/urban-watch-cities"
        image = get_ee_image_collection(collection_name, roi)
        save_geotiff(image, geotiff_path)
    elif source == 'ESA WorldCover':
        # Global land cover from European Space Agency
        roi = get_roi(rectangle_vertices)
        save_geotiff_esa_land_cover(roi, geotiff_path)
    elif source == 'ESRI 10m Annual Land Cover':
        # High-resolution annual land cover from ESRI
        esri_landcover_year = kwargs.get("esri_landcover_year")
        roi = get_roi(rectangle_vertices)
        save_geotiff_esri_landcover(roi, geotiff_path, year=esri_landcover_year)
    elif source == 'Dynamic World V1':
        # Near real-time land cover from Google's Dynamic World
        dynamic_world_date = kwargs.get("dynamic_world_date")
        roi = get_roi(rectangle_vertices)
        save_geotiff_dynamic_world_v1(roi, geotiff_path, dynamic_world_date)
    elif source == 'OpenEarthMapJapan':
        # Japan-specific land cover dataset
        # Allow SSL/HTTP options to be passed through kwargs
        ssl_verify = kwargs.get('ssl_verify', kwargs.get('verify', True))
        allow_insecure_ssl = kwargs.get('allow_insecure_ssl', False)
        allow_http_fallback = kwargs.get('allow_http_fallback', False)
        timeout_s = kwargs.get('timeout', 30)

        save_oemj_as_geotiff(
            rectangle_vertices,
            geotiff_path,
            ssl_verify=ssl_verify,
            allow_insecure_ssl=allow_insecure_ssl,
            allow_http_fallback=allow_http_fallback,
            timeout_s=timeout_s,
        )
        # Ensure the file was actually created before proceeding
        if not os.path.exists(geotiff_path):
            raise FileNotFoundError(
                f"OEMJ download failed; expected GeoTIFF not found: {geotiff_path}. "
                "You can try setting ssl_verify=False or allow_http_fallback=True in kwargs."
            )   
    elif source == 'OpenStreetMap':
        # Vector-based land cover from OpenStreetMap
        # This bypasses the GeoTIFF workflow and gets data directly as GeoJSON
        land_cover_gdf = load_land_cover_gdf_from_osm(rectangle_vertices)
    
    # Get the classification scheme for the selected data source
    # Each source has its own land cover categories and color coding
    land_cover_classes = get_land_cover_classes(source)

    # Convert geospatial data to regular grid format
    # Different processing for vector vs raster data sources
    if source == 'OpenStreetMap':
        # Process vector data directly from GeoDataFrame
        default_class = kwargs.get('default_land_cover_class', 'Developed space')
        land_cover_grid_str = create_land_cover_grid_from_gdf_polygon(land_cover_gdf, meshsize, source, rectangle_vertices, default_class=default_class)
    else:
        # Process raster data from GeoTIFF file
        land_cover_grid_str = create_land_cover_grid_from_geotiff_polygon(geotiff_path, meshsize, land_cover_classes, rectangle_vertices)

    # Prepare color mapping for visualization
    # Convert RGB values from 0-255 range to 0-1 range for matplotlib
    color_map = {cls: [r/255, g/255, b/255] for (r,g,b), cls in land_cover_classes.items()}

    # Generate visualization if requested
    grid_vis = kwargs.get("gridvis", True)    
    if grid_vis:
        # Flip grid vertically for correct display orientation
        visualize_land_cover_grid(np.flipud(land_cover_grid_str), meshsize, color_map, land_cover_classes)
    
    # Convert string-based land cover labels to integer codes for processing
    # This enables efficient numerical operations on the grid
    land_cover_grid_int = convert_land_cover_array(land_cover_grid_str, land_cover_classes)

    return land_cover_grid_int

# def get_building_height_grid(rectangle_vertices, meshsize, source, output_dir="output", visualization=True, maptiler_API_key=None, file_path=None):
def get_building_height_grid(rectangle_vertices, meshsize, source, output_dir, building_gdf=None, **kwargs):
    """Creates a grid of building heights.

    Args:
        rectangle_vertices: List of coordinates defining the area of interest
        meshsize: Size of each grid cell in meters
        source: Data source for buildings (e.g. 'OpenStreetMap', 'Microsoft Building Footprints', 'GeoDataFrame')
        output_dir: Directory to save output files
        building_gdf: Optional GeoDataFrame with building footprint, height and other information
        **kwargs: Additional arguments including:
            - maptiler_API_key: API key for MapTiler
            - building_path: Path to local building data file
            - building_complementary_source: Additional building data source
            - gridvis: Whether to visualize the grid

    Orientation:
        All inputs and outputs use north_up orientation (row 0 = north/top),
        columns increase eastward (col 0 = west/left).

    Returns:
        tuple:
            - numpy.ndarray: Grid of building heights
            - numpy.ndarray: Grid of building minimum heights
            - numpy.ndarray: Grid of building IDs
            - list: Filtered building features
    """

    # Initialize Earth Engine only for building sources that require it
    ee_required_sources = {"Open Building 2.5D Temporal"}
    if source in ee_required_sources:
        initialize_earth_engine()

    print("Creating Building Height grid\n ")
    print(f"Data source: {source}")

    os.makedirs(output_dir, exist_ok=True)
    
    # If building_gdf is provided, use it directly
    if building_gdf is not None:
        gdf = building_gdf
        print("Using provided GeoDataFrame for building data")
    else:
        # Fetch building data from primary source
        # Each source has different data formats and processing requirements
        # Floor height (m) for inferring heights from floors/levels
        floor_height = kwargs.get("floor_height", 3.0)
        if source == 'Microsoft Building Footprints':
            # Machine learning-derived building footprints from satellite imagery
            gdf = get_mbfp_gdf(output_dir, rectangle_vertices)
        elif source == 'OpenStreetMap':
            # Crowd-sourced building data with varying completeness
            gdf = load_gdf_from_openstreetmap(rectangle_vertices, floor_height=floor_height)
        elif source == "Open Building 2.5D Temporal":
            # Special case: this source provides both footprints and heights
            # Skip GeoDataFrame processing and create grids directly
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_open_building_temporal_polygon(meshsize, rectangle_vertices, output_dir)
        elif source == 'EUBUCCO v0.1':
            # European building database with height information
            gdf = load_gdf_from_eubucco(rectangle_vertices, output_dir)
        # elif source == "OpenMapTiles":
        #     # Vector tiles service for building data
        #     gdf = load_gdf_from_openmaptiles(rectangle_vertices, kwargs["maptiler_API_key"])
        elif source == "Overture":
            # Open building dataset from Overture Maps Foundation
            gdf = load_gdf_from_overture(rectangle_vertices, floor_height=floor_height)
        elif source in ("GBA", "Global Building Atlas"):
            # Global Building Atlas LOD1 polygons (GeoParquet tiles)
            clip_gba = kwargs.get("gba_clip", False)
            gba_download_dir = kwargs.get("gba_download_dir")
            gdf = load_gdf_from_gba(rectangle_vertices, download_dir=gba_download_dir, clip_to_rectangle=clip_gba)
        elif source == "Local file":
            # Handle user-provided local building data files
            _, extension = os.path.splitext(kwargs["building_path"])
            if extension == ".gpkg":
                gdf = get_gdf_from_gpkg(kwargs["building_path"], rectangle_vertices)
        elif source == "GeoDataFrame":
            # This case is handled by the building_gdf parameter above
            raise ValueError("When source is 'GeoDataFrame', building_gdf parameter must be provided")
    
    # Handle complementary data sources to fill gaps or provide additional information
    # This allows combining multiple sources for better coverage or accuracy
    building_complementary_source = kwargs.get("building_complementary_source") 
    building_complement_height = kwargs.get("building_complement_height")
    overlapping_footprint = kwargs.get("overlapping_footprint")

    if (building_complementary_source is None) or (building_complementary_source=='None'):
        # Use only the primary data source
        if source != "Open Building 2.5D Temporal":
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_gdf_polygon(gdf, meshsize, rectangle_vertices, complement_height=building_complement_height, overlapping_footprint=overlapping_footprint)
    else:
        # Combine primary source with complementary data
        if building_complementary_source == "Open Building 2.5D Temporal":
            # Use temporal height data to complement footprint data
            roi = get_roi(rectangle_vertices)
            os.makedirs(output_dir, exist_ok=True)
            geotiff_path_comp = os.path.join(output_dir, "building_height.tif")
            save_geotiff_open_buildings_temporal(roi, geotiff_path_comp)
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_gdf_polygon(gdf, meshsize, rectangle_vertices, geotiff_path_comp=geotiff_path_comp, complement_height=building_complement_height, overlapping_footprint=overlapping_footprint)   
        elif building_complementary_source in ["England 1m DSM - DTM", "Netherlands 0.5m DSM - DTM"]:
            # Use digital surface model minus digital terrain model for height estimation
            roi = get_roi(rectangle_vertices)
            os.makedirs(output_dir, exist_ok=True)
            geotiff_path_comp = os.path.join(output_dir, "building_height.tif")
            save_geotiff_dsm_minus_dtm(roi, geotiff_path_comp, meshsize, building_complementary_source)
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_gdf_polygon(gdf, meshsize, rectangle_vertices, geotiff_path_comp=geotiff_path_comp, complement_height=building_complement_height, overlapping_footprint=overlapping_footprint)
        else:
            # Fetch complementary data from another vector source
            if building_complementary_source == 'Microsoft Building Footprints':
                gdf_comp = get_mbfp_gdf(output_dir, rectangle_vertices)
            elif building_complementary_source == 'OpenStreetMap':
                gdf_comp = load_gdf_from_openstreetmap(rectangle_vertices, floor_height=floor_height)
            elif building_complementary_source == 'EUBUCCO v0.1':
                gdf_comp = load_gdf_from_eubucco(rectangle_vertices, output_dir)
            # elif building_complementary_source == "OpenMapTiles":
            #     gdf_comp = load_gdf_from_openmaptiles(rectangle_vertices, kwargs["maptiler_API_key"])
            elif building_complementary_source == "Overture":
                gdf_comp = load_gdf_from_overture(rectangle_vertices, floor_height=floor_height)
            elif building_complementary_source in ("GBA", "Global Building Atlas"):
                clip_gba = kwargs.get("gba_clip", False)
                gba_download_dir = kwargs.get("gba_download_dir")
                gdf_comp = load_gdf_from_gba(rectangle_vertices, download_dir=gba_download_dir, clip_to_rectangle=clip_gba)
            elif building_complementary_source == "Local file":
                _, extension = os.path.splitext(kwargs["building_complementary_path"])
                if extension == ".gpkg":
                    gdf_comp = get_gdf_from_gpkg(kwargs["building_complementary_path"], rectangle_vertices)
            
            # Configure how to combine the complementary data
            # Can complement footprints only or both footprints and heights
            complement_building_footprints = kwargs.get("complement_building_footprints")
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_gdf_polygon(gdf, meshsize, rectangle_vertices, gdf_comp=gdf_comp, complement_building_footprints=complement_building_footprints, complement_height=building_complement_height, overlapping_footprint=overlapping_footprint)

    # Generate visualization if requested
    grid_vis = kwargs.get("gridvis", True)    
    if grid_vis:
        # Replace zeros with NaN for better visualization (don't show empty areas)
        building_height_grid_nan = building_height_grid.copy()
        building_height_grid_nan[building_height_grid_nan == 0] = np.nan
        # Flip grid vertically for correct display orientation
        visualize_numerical_grid(np.flipud(building_height_grid_nan), meshsize, "building height (m)", cmap='viridis', label='Value')

    return building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings

def get_canopy_height_grid(rectangle_vertices, meshsize, source, output_dir, **kwargs):
    """Creates canopy top and bottom height grids.

    Supports satellite sources and a GeoDataFrame source (from draw_additional_trees).

    Args:
        rectangle_vertices: List of coordinates defining the area of interest
        meshsize: Size of each grid cell in meters
        source: Data source for canopy heights. Use 'GeoDataFrame' or 'tree_gdf' for tree_gdf path/object.
        output_dir: Directory to save output files
        **kwargs: Additional arguments including:
            - gridvis: Whether to visualize the grid
            - tree_gdf: GeoDataFrame of trees (optional when source='GeoDataFrame')
            - tree_gdf_path: Path to a local file (e.g., .gpkg) with trees when source='GeoDataFrame'
            - trunk_height_ratio: Fraction of top height used as canopy bottom for non-GDF sources

    Orientation:
        Inputs and returned grids use north_up orientation (row 0 = north/top),
        columns increase eastward (col 0 = west/left).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: (canopy_top_height_grid, canopy_bottom_height_grid)
    """

    print("Creating Canopy Height grid\n ")
    print(f"Data source: {source}")

    os.makedirs(output_dir, exist_ok=True)

    # Branch: compute from a provided GeoDataFrame (draw_additional_trees output)
    if source in ('GeoDataFrame', 'tree_gdf', 'Tree_GeoDataFrame', 'GDF'):
        tree_gdf = kwargs.get('tree_gdf')
        tree_gdf_path = kwargs.get('tree_gdf_path')
        if tree_gdf is None and tree_gdf_path is not None:
            _, ext = os.path.splitext(tree_gdf_path)
            if ext.lower() == '.gpkg':
                tree_gdf = get_gdf_from_gpkg(tree_gdf_path, rectangle_vertices)
            else:
                raise ValueError("Unsupported tree file format. Use .gpkg or pass a GeoDataFrame.")
        if tree_gdf is None:
            raise ValueError("When source='GeoDataFrame', provide 'tree_gdf' or 'tree_gdf_path'.")

        canopy_top, canopy_bottom = create_canopy_grids_from_tree_gdf(tree_gdf, meshsize, rectangle_vertices)

        # Visualization
        grid_vis = kwargs.get("gridvis", True)
        if grid_vis:
            vis = canopy_top.copy()
            vis[vis == 0] = np.nan
            visualize_numerical_grid(np.flipud(vis), meshsize, "Tree canopy height (top)", cmap='Greens', label='Tree canopy height (m)')

        return canopy_top, canopy_bottom

    # Default: satellite/remote sensing sources
    print("Data source: High Resolution Canopy Height Maps by WRI and Meta")
    initialize_earth_engine()

    geotiff_path = os.path.join(output_dir, "canopy_height.tif")

    roi = get_roi(rectangle_vertices)
    if source == 'High Resolution 1m Global Canopy Height Maps':
        collection_name = "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"
        image = get_ee_image_collection(collection_name, roi)
    elif source == 'ETH Global Sentinel-2 10m Canopy Height (2020)':
        collection_name = "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"
        image = get_ee_image(collection_name, roi)
    else:
        raise ValueError(f"Unsupported canopy source: {source}")

    save_geotiff(image, geotiff_path, resolution=meshsize)
    canopy_height_grid = create_height_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices)

    # Derive bottom grid using trunk_height_ratio (consistent with create_3d_voxel)
    trunk_height_ratio = kwargs.get("trunk_height_ratio")
    if trunk_height_ratio is None:
        trunk_height_ratio = 11.76 / 19.98
    canopy_bottom_grid = canopy_height_grid * float(trunk_height_ratio)

    grid_vis = kwargs.get("gridvis", True)
    if grid_vis:
        canopy_height_grid_nan = canopy_height_grid.copy()
        canopy_height_grid_nan[canopy_height_grid_nan == 0] = np.nan
        visualize_numerical_grid(np.flipud(canopy_height_grid_nan), meshsize, "Tree canopy height", cmap='Greens', label='Tree canopy height (m)')
    return canopy_height_grid, canopy_bottom_grid

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

    Orientation:
        Returned DEM grid uses north_up orientation (row 0 = north/top),
        columns increase eastward (col 0 = west/left).

    Returns:
        numpy.ndarray: Grid of elevation values
    """

    print("Creating Digital Elevation Model (DEM) grid\n ")
    print(f"Data source: {source}")

    if source == "Local file":
        # Use user-provided local DEM file
        geotiff_path = kwargs["dem_path"]
    else:
        # Fetch DEM data from various satellite/government sources
        initialize_earth_engine()

        geotiff_path = os.path.join(output_dir, "dem.tif")

        # Add buffer around region of interest to ensure smooth interpolation at edges
        # This prevents edge artifacts in the final grid
        buffer_distance = 100
        roi = get_roi(rectangle_vertices)
        roi_buffered = roi.buffer(buffer_distance)
        
        # Fetch elevation data from selected source
        image = get_dem_image(roi_buffered, source)
        
        # Save DEM data with appropriate resolution based on source capabilities
        if source in ["England 1m DTM", 'DEM France 1m', 'DEM France 5m', 'AUSTRALIA 5M DEM', 'Netherlands 0.5m DTM']:
            # High-resolution elevation models - use specified mesh size
            save_geotiff(image, geotiff_path, scale=meshsize, region=roi_buffered, crs='EPSG:4326')
        elif source == 'USGS 3DEP 1m':
            # US Geological Survey 3D Elevation Program
            # Ensure minimum scale of 1.25m due to data limitations
            scale = max(meshsize, 1.25)
            save_geotiff(image, geotiff_path, scale=scale, region=roi_buffered, crs='EPSG:4326')
        else:
            # Default to 30m resolution for global/lower resolution sources
            save_geotiff(image, geotiff_path, scale=30, region=roi_buffered)

    # Convert GeoTIFF to regular grid with optional interpolation
    # Interpolation helps fill gaps and smooth transitions
    dem_interpolation = kwargs.get("dem_interpolation")
    dem_grid = create_dem_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices, dem_interpolation=dem_interpolation)

    # Generate visualization if requested
    grid_vis = kwargs.get("gridvis", True)    
    if grid_vis:
        # Use terrain color scheme appropriate for elevation data
        visualize_numerical_grid(np.flipud(dem_grid), meshsize, title='Digital Elevation Model', cmap='terrain', label='Elevation (m)')

    return dem_grid

def create_3d_voxel(building_height_grid_ori, building_min_height_grid_ori, 
                    building_id_grid_ori, land_cover_grid_ori, dem_grid_ori, 
                    tree_grid_ori, voxel_size, land_cover_source, canopy_bottom_height_grid_ori=None, **kwargs):
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
        kwargs: Additional arguments including:
            - trunk_height_ratio: Ratio of trunk height to total tree height
    Orientation:
        All 2D inputs must be north_up (row 0 = north/top) with columns increasing
        eastward (col 0 = west/left). 3D indexing follows (row, col, z) =
        (north→south, west→east, ground→up).

    Returns:
        numpy.ndarray: 3D voxel grid with encoded values for different features
    """
    print("Generating 3D voxel data")

    # Convert land cover values to standardized format if needed
    # OpenStreetMap data is already in the correct format
    if (land_cover_source == 'OpenStreetMap'):
        land_cover_grid_converted = land_cover_grid_ori
    else:
        land_cover_grid_converted = convert_land_cover(land_cover_grid_ori, land_cover_source=land_cover_source)
    # Prepare all input grids for 3D processing
    # Flip vertically to align with standard geographic orientation (north-up)
    # Handle missing data appropriately for each grid type
    building_height_grid = np.flipud(np.nan_to_num(building_height_grid_ori, nan=10.0)) # Replace NaN values with 10m height
    building_min_height_grid = np.flipud(replace_nan_in_nested(building_min_height_grid_ori)) # Replace NaN in nested arrays
    building_id_grid = np.flipud(building_id_grid_ori)
    land_cover_grid = np.flipud(land_cover_grid_converted.copy()) + 1 # Add 1 to avoid 0 values in land cover
    dem_grid = np.flipud(dem_grid_ori.copy()) - np.min(dem_grid_ori) # Normalize DEM to start at 0
    dem_grid = process_grid(building_id_grid, dem_grid) # Process DEM based on building footprints
    tree_grid = np.flipud(tree_grid_ori.copy())
    canopy_bottom_grid = None
    if canopy_bottom_height_grid_ori is not None:
        canopy_bottom_grid = np.flipud(canopy_bottom_height_grid_ori.copy())
    # Validate that all input grids have consistent dimensions
    assert building_height_grid.shape == land_cover_grid.shape == dem_grid.shape == tree_grid.shape, "Input grids must have the same shape"
    rows, cols = building_height_grid.shape
    # Calculate the required height for the 3D voxel grid
    # Add 1 voxel layer to ensure sufficient vertical space
    max_height = int(np.ceil(np.max(building_height_grid + dem_grid + tree_grid) / voxel_size))+1
    # Initialize the 3D voxel grid with zeros
    # Use int8 by default to reduce memory (values range from about -99 to small positives)
    # Allow override via kwarg 'voxel_dtype'
    voxel_dtype = kwargs.get("voxel_dtype", np.int8)

    # Optional: estimate memory and allow a soft limit before allocating
    try:
        bytes_per_elem = np.dtype(voxel_dtype).itemsize
        est_mb = rows * cols * max_height * bytes_per_elem / (1024 ** 2)
        print(f"Voxel grid shape: ({rows}, {cols}, {max_height}), dtype: {voxel_dtype}, ~{est_mb:.1f} MB")
        max_ram_mb = kwargs.get("max_voxel_ram_mb")
        if (max_ram_mb is not None) and (est_mb > max_ram_mb):
            raise MemoryError(f"Estimated voxel grid memory {est_mb:.1f} MB exceeds limit {max_ram_mb} MB. Increase mesh size or restrict ROI.")
    except Exception:
        pass

    voxel_grid = np.zeros((rows, cols, max_height), dtype=voxel_dtype)
    # Configure tree trunk-to-crown ratio
    # This determines how much of the tree is trunk vs canopy
    trunk_height_ratio = kwargs.get("trunk_height_ratio")
    if trunk_height_ratio is None:
        trunk_height_ratio = 11.76 / 19.98  # Default ratio based on typical tree proportions
    # Process each grid cell to build the 3D voxel representation
    for i in range(rows):
        for j in range(cols):
            # Calculate ground level in voxel units
            # Add 1 to ensure space for surface features
            ground_level = int(dem_grid[i, j] / voxel_size + 0.5) + 1 
            # Extract current cell values
            tree_height = tree_grid[i, j]
            land_cover = land_cover_grid[i, j]
            # Fill underground voxels with -1 (represents subsurface)
            voxel_grid[i, j, :ground_level] = -1
            # Set the ground surface to the land cover type
            voxel_grid[i, j, ground_level-1] = land_cover
            # Process tree canopy if trees are present
            if tree_height > 0:
                # Calculate tree structure: trunk base to crown base to crown top
                if canopy_bottom_grid is not None:
                    crown_base_height = canopy_bottom_grid[i, j]
                else:
                    crown_base_height = (tree_height * trunk_height_ratio)
                crown_base_height_level = int(crown_base_height / voxel_size + 0.5)
                crown_top_height = tree_height
                crown_top_height_level = int(crown_top_height / voxel_size + 0.5)

                # Ensure minimum crown height of 1 voxel
                # Prevent crown base and top from being at the same level
                if (crown_top_height_level == crown_base_height_level) and (crown_base_height_level>0):
                    crown_base_height_level -= 1

                # Calculate absolute positions relative to ground level
                tree_start = ground_level + crown_base_height_level
                tree_end = ground_level + crown_top_height_level

                # Fill tree crown voxels with -2 (represents vegetation canopy)
                voxel_grid[i, j, tree_start:tree_end] = -2
            # Process buildings - handle multiple height segments per building
            # Some buildings may have multiple levels or complex height profiles
            for k in building_min_height_grid[i, j]:
                building_min_height = int(k[0] / voxel_size + 0.5)  # Lower height of building segment
                building_height = int(k[1] / voxel_size + 0.5)      # Upper height of building segment
                # Fill building voxels with -3 (represents built structures)
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

    Orientation:
        All 2D inputs must be north_up (row 0 = north/top) with columns increasing
        eastward (col 0 = west/left). Each returned 3D grid preserves this X,Y
        orientation, with z increasing upward.

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

def get_voxcity(rectangle_vertices, building_source, land_cover_source, canopy_height_source, dem_source, meshsize, building_gdf=None, terrain_gdf=None, **kwargs):
    """Main function to generate a complete voxel city model.

    Args:
        rectangle_vertices: List of coordinates defining the area of interest
        building_source: Source for building height data (e.g. 'OSM', 'EUBUCCO', 'GeoDataFrame')
        land_cover_source: Source for land cover data (e.g. 'ESA', 'ESRI') 
        canopy_height_source: Source for tree canopy height data
        dem_source: Source for digital elevation model data ('Flat' or other source)
        meshsize: Size of each grid cell in meters
        building_gdf: Optional GeoDataFrame with building footprint, height and other information
        terrain_gdf: Optional GeoDataFrame with terrain elements including an 'elevation' column
        **kwargs: Additional keyword arguments including:
            - output_dir: Directory to save output files (default: 'output')
            - min_canopy_height: Minimum height threshold for tree canopy
            - remove_perimeter_object: Factor to remove objects near perimeter
            - mapvis: Whether to visualize grids on map
            - voxelvis: Whether to visualize 3D voxel model
            - voxelvis_img_save_path: Path to save 3D visualization
            - default_land_cover_class: Default class for land cover grid cells with no intersecting polygons (default: 'Developed space')

    Orientation:
        All intermediate and final 2D grids use north_up (row 0 = north/top) with
        columns increasing eastward (col 0 = west/left). Visual previews may flip
        vertically for display only.

    Returns:
        tuple containing:
            - voxcity_grid: 3D voxel grid of the complete city model
            - building_height_grid: 2D grid of building heights
            - building_min_height_grid: 2D grid of minimum building heights
            - building_id_grid: 2D grid of building IDs
            - canopy_height_grid: 2D grid of tree canopy top heights
            - canopy_bottom_height_grid: 2D grid of tree canopy bottom heights
            - land_cover_grid: 2D grid of land cover classifications
            - dem_grid: 2D grid of ground elevation
            - building_geojson: GeoJSON of building footprints and metadata
    """
    # Set up output directory for intermediate and final files
    output_dir = kwargs.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
        
    # Remove 'output_dir' from kwargs to prevent duplication in function calls
    kwargs.pop('output_dir', None)

    # STEP 1: Generate all required 2D grids from various data sources
    # These grids form the foundation for the 3D voxel model
    
    # Land cover classification grid (e.g., urban, forest, water, agriculture)
    land_cover_grid = get_land_cover_grid(rectangle_vertices, meshsize, land_cover_source, output_dir, **kwargs)
    
    # Building footprints and height information
    building_height_grid, building_min_height_grid, building_id_grid, building_gdf = get_building_height_grid(rectangle_vertices, meshsize, building_source, output_dir, building_gdf=building_gdf, **kwargs)
    
    # Save building data to file for later analysis or visualization
    if not building_gdf.empty:
        save_path = f"{output_dir}/building.gpkg"
        building_gdf.to_file(save_path, driver='GPKG')
    
    # STEP 2: Handle canopy height data
    # Either use static values or fetch from satellite sources
    if canopy_height_source == "Static":
        # Create uniform canopy height for all tree-covered areas (top grid)
        canopy_height_grid = np.zeros_like(land_cover_grid, dtype=float)
        static_tree_height = kwargs.get("static_tree_height", 10.0)
        # Determine tree class indices based on source-specific class names
        _classes = get_land_cover_classes(land_cover_source)
        _class_to_int = {name: i for i, name in enumerate(_classes.values())}
        _tree_labels = ["Tree", "Trees", "Tree Canopy"]
        _tree_indices = [_class_to_int[label] for label in _tree_labels if label in _class_to_int]
        tree_mask = np.isin(land_cover_grid, _tree_indices) if _tree_indices else np.zeros_like(land_cover_grid, dtype=bool)
        canopy_height_grid[tree_mask] = static_tree_height

        # Derive bottom from trunk_height_ratio
        trunk_height_ratio = kwargs.get("trunk_height_ratio")
        if trunk_height_ratio is None:
            trunk_height_ratio = 11.76 / 19.98
        canopy_bottom_height_grid = canopy_height_grid * float(trunk_height_ratio)
    else:
        # Fetch canopy top/bottom from source
        canopy_height_grid, canopy_bottom_height_grid = get_canopy_height_grid(rectangle_vertices, meshsize, canopy_height_source, output_dir, **kwargs)
    
    # STEP 3: Handle digital elevation model (terrain)
    if terrain_gdf is not None:
        # Build DEM directly from provided terrain GeoDataFrame (e.g., precomputed from CityGML)
        dem_grid = create_dem_grid_from_gdf_polygon(terrain_gdf, meshsize, rectangle_vertices)
    else:
        if dem_source == "Flat":
            # Create flat terrain for simplified modeling
            dem_grid = np.zeros_like(land_cover_grid)
        else:
            # Fetch terrain elevation from various sources
            dem_grid = get_dem_grid(rectangle_vertices, meshsize, dem_source, output_dir, **kwargs)

    # STEP 4: Apply optional data filtering and cleaning
    
    # Filter out low vegetation that may be noise in the data
    min_canopy_height = kwargs.get("min_canopy_height")
    if min_canopy_height is not None:
        canopy_height_grid[canopy_height_grid < kwargs["min_canopy_height"]] = 0        
        if 'canopy_bottom_height_grid' in locals():
            canopy_bottom_height_grid[canopy_height_grid == 0] = 0

    # Remove objects near the boundary to avoid edge effects
    # This is useful when the area of interest is part of a larger urban area
    remove_perimeter_object = kwargs.get("remove_perimeter_object")
    if (remove_perimeter_object is not None) and (remove_perimeter_object > 0):
        print("apply perimeter removal")
        # Calculate perimeter width based on grid dimensions
        w_peri = int(remove_perimeter_object * building_height_grid.shape[0] + 0.5)
        h_peri = int(remove_perimeter_object * building_height_grid.shape[1] + 0.5)
        
        # Clear canopy heights in perimeter areas
        canopy_height_grid[:w_peri, :] = canopy_height_grid[-w_peri:, :] = canopy_height_grid[:, :h_peri] = canopy_height_grid[:, -h_peri:] = 0
        if 'canopy_bottom_height_grid' in locals():
            canopy_bottom_height_grid[:w_peri, :] = canopy_bottom_height_grid[-w_peri:, :] = canopy_bottom_height_grid[:, :h_peri] = canopy_bottom_height_grid[:, -h_peri:] = 0

        # Identify buildings that intersect with perimeter areas
        ids1 = np.unique(building_id_grid[:w_peri, :][building_id_grid[:w_peri, :] > 0])
        ids2 = np.unique(building_id_grid[-w_peri:, :][building_id_grid[-w_peri:, :] > 0])
        ids3 = np.unique(building_id_grid[:, :h_peri][building_id_grid[:, :h_peri] > 0])
        ids4 = np.unique(building_id_grid[:, -h_peri:][building_id_grid[:, -h_peri:] > 0])
        remove_ids = np.concatenate((ids1, ids2, ids3, ids4))
        
        # Remove identified buildings from all grids
        for remove_id in remove_ids:
            positions = np.where(building_id_grid == remove_id)
            building_height_grid[positions] = 0
            building_min_height_grid[positions] = [[] for _ in range(len(building_min_height_grid[positions]))]

        # Visualize grids after optional perimeter removal
        grid_vis = kwargs.get("gridvis", True)
        if grid_vis:
            # Building height grid visualization (zeros hidden)
            building_height_grid_nan = building_height_grid.copy()
            building_height_grid_nan[building_height_grid_nan == 0] = np.nan
            visualize_numerical_grid(
                np.flipud(building_height_grid_nan),
                meshsize,
                "building height (m)",
                cmap='viridis',
                label='Value'
            )

            # Canopy height grid visualization (zeros hidden)
            canopy_height_grid_nan = canopy_height_grid.copy()
            canopy_height_grid_nan[canopy_height_grid_nan == 0] = np.nan
            visualize_numerical_grid(
                np.flipud(canopy_height_grid_nan),
                meshsize,
                "Tree canopy height (m)",
                cmap='Greens',
                label='Tree canopy height (m)'
            )

    # STEP 5: Generate optional 2D visualizations on interactive maps
    mapvis = kwargs.get("mapvis")
    if mapvis:
        # Create map-based visualizations of all data layers
        # These help users understand the input data before 3D modeling
        
        # Visualize land cover using the new function
        visualize_landcover_grid_on_basemap(
            land_cover_grid, 
            rectangle_vertices, 
            meshsize, 
            source=land_cover_source,
            alpha=0.7,
            figsize=(12, 8),
            basemap='CartoDB light',
            show_edge=False
        )
        
        # Visualize building heights using the new function
        visualize_numerical_grid_on_basemap(
            building_height_grid,
            rectangle_vertices,
            meshsize,
            value_name="Building Heights (m)",
            cmap='viridis',
            alpha=0.7,
            figsize=(12, 8),
            basemap='CartoDB light',
            show_edge=False
        )
        
        # Visualize canopy heights using the new function
        visualize_numerical_grid_on_basemap(
            canopy_height_grid,
            rectangle_vertices,
            meshsize,
            value_name="Canopy Heights (m)",
            cmap='Greens',
            alpha=0.7,
            figsize=(12, 8),
            basemap='CartoDB light',
            show_edge=False
        )
        
        # Visualize DEM using the new function
        visualize_numerical_grid_on_basemap(
            dem_grid,
            rectangle_vertices,
            meshsize,
            value_name="Terrain Elevation (m)",
            cmap='terrain',
            alpha=0.7,
            figsize=(12, 8),
            basemap='CartoDB light',
            show_edge=False
        )

    # STEP 6: Generate the final 3D voxel model
    # This combines all 2D grids into a comprehensive 3D representation
    voxcity_grid = create_3d_voxel(building_height_grid, building_min_height_grid, building_id_grid, land_cover_grid, dem_grid, canopy_height_grid, meshsize, land_cover_source, canopy_bottom_height_grid)

    # STEP 7: Generate optional 3D visualization
    # voxelvis = kwargs.get("voxelvis")
    # if voxelvis:
    #     # Create a taller grid for better visualization
    #     # Fixed height ensures consistent camera positioning
    #     new_height = int(550/meshsize+0.5)     
    #     voxcity_grid_vis = np.zeros((voxcity_grid.shape[0], voxcity_grid.shape[1], new_height))
    #     voxcity_grid_vis[:, :, :voxcity_grid.shape[2]] = voxcity_grid
    #     voxcity_grid_vis[-1, -1, -1] = -99  # Add marker to fix camera location and angle of view
    #     visualize_3d_voxel(voxcity_grid_vis, voxel_size=meshsize, save_path=kwargs["voxelvis_img_save_path"])

    # STEP 8: Save all generated data for future use
    save_voxcity = kwargs.get("save_voxctiy_data", True)
    if save_voxcity:
        save_path = kwargs.get("save_data_path", f"{output_dir}/voxcity_data.pkl")
        save_voxcity_data(save_path, voxcity_grid, building_height_grid, building_min_height_grid, 
                         building_id_grid, canopy_height_grid, land_cover_grid, dem_grid, 
                         building_gdf, meshsize, rectangle_vertices)
    
    return voxcity_grid, building_height_grid, building_min_height_grid, building_id_grid, canopy_height_grid, canopy_bottom_height_grid, land_cover_grid, dem_grid, building_gdf

def get_voxcity_CityGML(rectangle_vertices, land_cover_source, canopy_height_source, meshsize, url_citygml=None, citygml_path=None, **kwargs):
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

    Orientation:
        All intermediate and final 2D grids use north_up (row 0 = north/top) with
        columns increasing eastward (col 0 = west/left). Visual previews may flip
        vertically for display only.

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

    # SSL/HTTP options for CityGML download (optional)
    # Backward compatible: accept 'verify' but prefer 'ssl_verify'
    ssl_verify = kwargs.pop('ssl_verify', kwargs.pop('verify', True))
    ca_bundle = kwargs.pop('ca_bundle', None)
    timeout = kwargs.pop('timeout', 60)

    # get all required gdfs    
    building_gdf, terrain_gdf, vegetation_gdf = load_buid_dem_veg_from_citygml(
        url=url_citygml,
        citygml_path=citygml_path,
        base_dir=output_dir,
        rectangle_vertices=rectangle_vertices,
        ssl_verify=ssl_verify,
        ca_bundle=ca_bundle,
        timeout=timeout
    )

    # Normalize CRS to WGS84 (EPSG:4326) to ensure consistent operations downstream
    try:
        import geopandas as gpd  # noqa: F401
        if building_gdf is not None:
            if building_gdf.crs is None:
                building_gdf = building_gdf.set_crs(epsg=4326)
            elif getattr(building_gdf.crs, 'to_epsg', lambda: None)() != 4326 and building_gdf.crs != "EPSG:4326":
                building_gdf = building_gdf.to_crs(epsg=4326)
        if terrain_gdf is not None:
            if terrain_gdf.crs is None:
                terrain_gdf = terrain_gdf.set_crs(epsg=4326)
            elif getattr(terrain_gdf.crs, 'to_epsg', lambda: None)() != 4326 and terrain_gdf.crs != "EPSG:4326":
                terrain_gdf = terrain_gdf.to_crs(epsg=4326)
        if vegetation_gdf is not None:
            if vegetation_gdf.crs is None:
                vegetation_gdf = vegetation_gdf.set_crs(epsg=4326)
            elif getattr(vegetation_gdf.crs, 'to_epsg', lambda: None)() != 4326 and vegetation_gdf.crs != "EPSG:4326":
                vegetation_gdf = vegetation_gdf.to_crs(epsg=4326)
    except Exception:
        pass

    land_cover_grid = get_land_cover_grid(rectangle_vertices, meshsize, land_cover_source, output_dir, **kwargs)        

    # building_height_grid, building_min_height_grid, building_id_grid, building_gdf = get_building_height_grid(rectangle_vertices, meshsize, building_source, output_dir, **kwargs)
    print("Creating building height grid")
    # Prepare complementary building source if provided
    building_complementary_source = kwargs.get("building_complementary_source")
    gdf_comp = None
    geotiff_path_comp = None
    complement_building_footprints = kwargs.get("complement_building_footprints")
    # Default to complement footprints when a complementary source is specified
    if complement_building_footprints is None and (building_complementary_source not in (None, "None")):
        complement_building_footprints = True

    if (building_complementary_source is not None) and (building_complementary_source != "None"):
        # Vector complementary sources
        floor_height = kwargs.get("floor_height", 3.0)
        if building_complementary_source == 'Microsoft Building Footprints':
            gdf_comp = get_mbfp_gdf(kwargs.get("output_dir", "output"), rectangle_vertices)
        elif building_complementary_source == 'OpenStreetMap':
            gdf_comp = load_gdf_from_openstreetmap(rectangle_vertices, floor_height=floor_height)
        elif building_complementary_source == 'EUBUCCO v0.1':
            gdf_comp = load_gdf_from_eubucco(rectangle_vertices, kwargs.get("output_dir", "output"))
        elif building_complementary_source == 'Overture':
            gdf_comp = load_gdf_from_overture(rectangle_vertices, floor_height=floor_height)
        elif building_complementary_source in ("GBA", "Global Building Atlas"):
            clip_gba = kwargs.get("gba_clip", False)
            gba_download_dir = kwargs.get("gba_download_dir")
            gdf_comp = load_gdf_from_gba(rectangle_vertices, download_dir=gba_download_dir, clip_to_rectangle=clip_gba)
        elif building_complementary_source == 'Local file':
            comp_path = kwargs.get("building_complementary_path")
            if comp_path is not None:
                _, extension = os.path.splitext(comp_path)
                if extension == ".gpkg":
                    gdf_comp = get_gdf_from_gpkg(comp_path, rectangle_vertices)
        # Ensure complementary GDF uses WGS84
        if gdf_comp is not None:
            try:
                if gdf_comp.crs is None:
                    gdf_comp = gdf_comp.set_crs(epsg=4326)
                elif getattr(gdf_comp.crs, 'to_epsg', lambda: None)() != 4326 and gdf_comp.crs != "EPSG:4326":
                    gdf_comp = gdf_comp.to_crs(epsg=4326)
            except Exception:
                pass
        # Raster complementary sources (height only)
        elif building_complementary_source == "Open Building 2.5D Temporal":
            roi = get_roi(rectangle_vertices)
            os.makedirs(kwargs.get("output_dir", "output"), exist_ok=True)
            geotiff_path_comp = os.path.join(kwargs.get("output_dir", "output"), "building_height.tif")
            save_geotiff_open_buildings_temporal(roi, geotiff_path_comp)
        elif building_complementary_source in ["England 1m DSM - DTM", "Netherlands 0.5m DSM - DTM"]:
            roi = get_roi(rectangle_vertices)
            os.makedirs(kwargs.get("output_dir", "output"), exist_ok=True)
            geotiff_path_comp = os.path.join(kwargs.get("output_dir", "output"), "building_height.tif")
            save_geotiff_dsm_minus_dtm(roi, geotiff_path_comp, meshsize, building_complementary_source)

    # Filter and assemble kwargs accepted by the grid function
    _allowed_building_kwargs = {
        "overlapping_footprint",
        "gdf_comp",
        "geotiff_path_comp",
        "complement_building_footprints",
        "complement_height",
    }
    _building_kwargs = {k: v for k, v in kwargs.items() if k in _allowed_building_kwargs}
    if gdf_comp is not None:
        _building_kwargs["gdf_comp"] = gdf_comp
    if geotiff_path_comp is not None:
        _building_kwargs["geotiff_path_comp"] = geotiff_path_comp
    if complement_building_footprints is not None:
        _building_kwargs["complement_building_footprints"] = complement_building_footprints

    # Map user-provided building_complement_height -> complement_height for grid builder
    comp_height_user = kwargs.get("building_complement_height")
    if comp_height_user is not None:
        _building_kwargs["complement_height"] = comp_height_user
    # If footprints are being complemented and no height provided, default to 10
    if _building_kwargs.get("complement_building_footprints") and ("complement_height" not in _building_kwargs):
        _building_kwargs["complement_height"] = 10.0

    building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_gdf_polygon(
        building_gdf, meshsize, rectangle_vertices, **_building_kwargs
    )

    # Visualize grid if requested
    grid_vis = kwargs.get("gridvis", True)    
    if grid_vis:
        building_height_grid_nan = building_height_grid.copy()
        building_height_grid_nan[building_height_grid_nan == 0] = np.nan
        visualize_numerical_grid(np.flipud(building_height_grid_nan), meshsize, "building height (m)", cmap='viridis', label='Value')
    
    # Save building data to GeoJSON
    if not building_gdf.empty:
        save_path = f"{output_dir}/building.gpkg"
        building_gdf.to_file(save_path, driver='GPKG')
    
    # Get canopy height data
    if canopy_height_source == "Static":
        # Create canopy height grid with same shape as land cover grid
        canopy_height_grid_comp = np.zeros_like(land_cover_grid, dtype=float)
        
        # Set default static height for trees (20 meters is a typical average tree height)
        static_tree_height = kwargs.get("static_tree_height", 10.0)
        # Determine tree class indices based on source-specific class names
        _classes = get_land_cover_classes(land_cover_source)
        _class_to_int = {name: i for i, name in enumerate(_classes.values())}
        _tree_labels = ["Tree", "Trees", "Tree Canopy"]
        _tree_indices = [_class_to_int[label] for label in _tree_labels if label in _class_to_int]
        tree_mask = np.isin(land_cover_grid, _tree_indices) if _tree_indices else np.zeros_like(land_cover_grid, dtype=bool)
        
        # Set static height for tree cells
        canopy_height_grid_comp[tree_mask] = static_tree_height

        # Bottom comp from trunk ratio
        trunk_height_ratio = kwargs.get("trunk_height_ratio")
        if trunk_height_ratio is None:
            trunk_height_ratio = 11.76 / 19.98
        canopy_bottom_height_grid_comp = canopy_height_grid_comp * float(trunk_height_ratio)
    else:
        canopy_height_grid_comp, canopy_bottom_height_grid_comp = get_canopy_height_grid(rectangle_vertices, meshsize, canopy_height_source, output_dir, **kwargs)
    
    # In the get_voxcity_CityGML function, modify it to handle None vegetation_gdf
    if vegetation_gdf is not None:
        canopy_height_grid = create_vegetation_height_grid_from_gdf_polygon(vegetation_gdf, meshsize, rectangle_vertices)
        # Base bottom grid from ratio
        trunk_height_ratio = kwargs.get("trunk_height_ratio")
        if trunk_height_ratio is None:
            trunk_height_ratio = 11.76 / 19.98
        canopy_bottom_height_grid = canopy_height_grid * float(trunk_height_ratio)
    else:
        # Create an empty canopy_height_grid with the same shape as your other grids
        # This depends on the expected shape, you might need to adjust
        canopy_height_grid = np.zeros_like(building_height_grid)
        canopy_bottom_height_grid = np.zeros_like(building_height_grid)

    mask = (canopy_height_grid == 0) & (canopy_height_grid_comp != 0)
    canopy_height_grid[mask] = canopy_height_grid_comp[mask]
    # Apply same complementation to bottom grid
    mask_b = (canopy_bottom_height_grid == 0) & (canopy_bottom_height_grid_comp != 0)
    canopy_bottom_height_grid[mask_b] = canopy_bottom_height_grid_comp[mask_b]

    # Ensure bottom <= top
    canopy_bottom_height_grid = np.minimum(canopy_bottom_height_grid, canopy_height_grid)
    
    # Handle DEM - either flat or from source
    if kwargs.pop('flat_dem', None):
        dem_grid = np.zeros_like(land_cover_grid)
    else:
        print("Creating Digital Elevation Model (DEM) grid")
        dem_grid = create_dem_grid_from_gdf_polygon(terrain_gdf, meshsize, rectangle_vertices)
        
        # Visualize grid if requested
        grid_vis = kwargs.get("gridvis", True)    
        if grid_vis:
            visualize_numerical_grid(np.flipud(dem_grid), meshsize, title='Digital Elevation Model', cmap='terrain', label='Elevation (m)')
        

    # Apply minimum canopy height threshold if specified
    min_canopy_height = kwargs.get("min_canopy_height")
    if min_canopy_height is not None:
        canopy_height_grid[canopy_height_grid < kwargs["min_canopy_height"]] = 0        
        canopy_bottom_height_grid[canopy_height_grid == 0] = 0

    # Remove objects near perimeter if specified
    remove_perimeter_object = kwargs.get("remove_perimeter_object")
    if (remove_perimeter_object is not None) and (remove_perimeter_object > 0):
        print("apply perimeter removal")
        # Calculate perimeter width based on grid dimensions
        w_peri = int(remove_perimeter_object * building_height_grid.shape[0] + 0.5)
        h_peri = int(remove_perimeter_object * building_height_grid.shape[1] + 0.5)
        
        # Clear canopy heights in perimeter
        canopy_height_grid[:w_peri, :] = canopy_height_grid[-w_peri:, :] = canopy_height_grid[:, :h_peri] = canopy_height_grid[:, -h_peri:] = 0
        canopy_bottom_height_grid[:w_peri, :] = canopy_bottom_height_grid[-w_peri:, :] = canopy_bottom_height_grid[:, :h_peri] = canopy_bottom_height_grid[:, -h_peri:] = 0

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

        # Visualize grids after optional perimeter removal
        grid_vis = kwargs.get("gridvis", True)
        if grid_vis:
            # Building height grid visualization (zeros hidden)
            building_height_grid_nan = building_height_grid.copy()
            building_height_grid_nan[building_height_grid_nan == 0] = np.nan
            visualize_numerical_grid(
                np.flipud(building_height_grid_nan),
                meshsize,
                "building height (m)",
                cmap='viridis',
                label='Value'
            )

            # Canopy height grid visualization (zeros hidden)
            canopy_height_grid_nan = canopy_height_grid.copy()
            canopy_height_grid_nan[canopy_height_grid_nan == 0] = np.nan
            visualize_numerical_grid(
                np.flipud(canopy_height_grid_nan),
                meshsize,
                "Tree canopy height (m)",
                cmap='Greens',
                label='Tree canopy height (m)'
            )

    # Generate 3D voxel grid
    voxcity_grid = create_3d_voxel(building_height_grid, building_min_height_grid, building_id_grid, land_cover_grid, dem_grid, canopy_height_grid, meshsize, land_cover_source, canopy_bottom_height_grid)

    # Save all data if a save path is provided
    save_voxcity = kwargs.get("save_voxctiy_data", True)
    if save_voxcity:
        save_path = kwargs.get("save_data_path", f"{output_dir}/voxcity_data.pkl")
        save_voxcity_data(save_path, voxcity_grid, building_height_grid, building_min_height_grid, 
                         building_id_grid, canopy_height_grid, land_cover_grid, dem_grid, 
                         building_gdf, meshsize, rectangle_vertices)
    
    return voxcity_grid, building_height_grid, building_min_height_grid, building_id_grid, canopy_height_grid, canopy_bottom_height_grid, land_cover_grid, dem_grid, filtered_buildings

def replace_nan_in_nested(arr, replace_value=10.0):
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

def save_voxcity_data(output_path, voxcity_grid, building_height_grid, building_min_height_grid, 
                     building_id_grid, canopy_height_grid, land_cover_grid, dem_grid, 
                     building_gdf, meshsize, rectangle_vertices):
    """Save voxcity data to a file for later loading.
    
    Args:
        output_path: Path to save the data file
        voxcity_grid: 3D voxel grid of the complete city model
        building_height_grid: 2D grid of building heights
        building_min_height_grid: 2D grid of minimum building heights
        building_id_grid: 2D grid of building IDs
        canopy_height_grid: 2D grid of tree canopy heights
        land_cover_grid: 2D grid of land cover classifications
        dem_grid: 2D grid of ground elevation
        building_gdf: GeoDataFrame of building footprints and metadata
        meshsize: Size of each grid cell in meters
        rectangle_vertices: List of coordinates defining the area of interest
    """
    import pickle
    import os
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a comprehensive dictionary containing all voxcity data
    # This preserves all components needed to reconstruct or analyze the model
    data_dict = {
        'voxcity_grid': voxcity_grid,
        'building_height_grid': building_height_grid,
        'building_min_height_grid': building_min_height_grid,
        'building_id_grid': building_id_grid,
        'canopy_height_grid': canopy_height_grid,
        'land_cover_grid': land_cover_grid,
        'dem_grid': dem_grid,
        'building_gdf': building_gdf,
        'meshsize': meshsize,
        'rectangle_vertices': rectangle_vertices
    }
    
    # Serialize and save the data using pickle for efficient storage
    # Pickle preserves exact data types and structures
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"Voxcity data saved to {output_path}")

def load_voxcity_data(input_path):
    """Load voxcity data from a saved file.
    
    Args:
        input_path: Path to the saved data file
        
    Returns:
        tuple: All the voxcity data components including:
            - voxcity_grid: 3D voxel grid of the complete city model
            - building_height_grid: 2D grid of building heights
            - building_min_height_grid: 2D grid of minimum building heights
            - building_id_grid: 2D grid of building IDs
            - canopy_height_grid: 2D grid of tree canopy heights
            - land_cover_grid: 2D grid of land cover classifications
            - dem_grid: 2D grid of ground elevation
            - building_gdf: GeoDataFrame of building footprints and metadata
            - meshsize: Size of each grid cell in meters
            - rectangle_vertices: List of coordinates defining the area of interest
    """
    import pickle
    
    # Deserialize the data from the saved file
    with open(input_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    print(f"Voxcity data loaded from {input_path}")
    
    # Return all components as a tuple in the same order as the main function
    # This ensures compatibility with existing code that expects this structure
    return (
        data_dict['voxcity_grid'],
        data_dict['building_height_grid'],
        data_dict['building_min_height_grid'],
        data_dict['building_id_grid'],
        data_dict['canopy_height_grid'],
        data_dict['land_cover_grid'],
        data_dict['dem_grid'],
        data_dict['building_gdf'],
        data_dict['meshsize'],
        data_dict['rectangle_vertices']
    )