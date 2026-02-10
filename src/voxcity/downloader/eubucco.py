"""
Module for downloading and processing building data from the EUBUCCO dataset.

This module provides functionality to download, extract, filter and convert building footprint data 
from the EUBUCCO (European Building Characteristics) dataset. It handles downloading zipped GeoPackage 
files, extracting building geometries and heights, and converting them to GeoJSON format.

The module supports:
- Downloading building data for specific European countries
- Extracting and processing GeoPackage files
- Converting coordinates between different coordinate reference systems (CRS)
- Filtering buildings by geographic area
- Handling building height data and confidence values
- Converting to standardized GeoJSON format

Key functions:
- filter_and_convert_gdf_to_geojson_eubucco(): Filters and converts GeoPackage data to GeoJSON
- download_extract_open_gpkg_from_eubucco(): Downloads and extracts EUBUCCO data
- get_gdf_from_eubucco(): Gets GeoDataFrame from EUBUCCO for a specific area
- load_gdf_from_eubucco(): Main interface for loading EUBUCCO building data

Dependencies:
- shapely: For geometric operations
- fiona: For reading GeoPackage files
- geopandas: For GeoDataFrame operations
- requests: For downloading data
"""

import json
from shapely.geometry import Polygon, mapping, shape, MultiPolygon
import requests
import zipfile
import os
from io import BytesIO
import fiona
from shapely.ops import transform
from fiona.transform import transform_geom
import logging
import shapely
import geopandas as gpd

from ..geoprocessor.utils import get_country_name

EUBUCCO_BASE_URL = "https://api.eubucco.com/v0.1"

# Dictionary mapping European countries to their EUBUCCO data download URLs
country_links = {}


def populate_country_links():
    global country_links

    # Populate the coutry_links variable
    response = requests.get(f"{EUBUCCO_BASE_URL}/countries")
    if response.status_code != 200:
        msg = f"Failed get the country-url lookup. Status code: {response.status_code}"
        logging.error(msg)
        raise Exception(msg)
    

    data = response.json()
    country_links = {country_data["name"]: country_data["gpkg"]["download_link"]
                                               for country_data in data}


# Get the country links when the current module is being imported
populate_country_links()


# ================================
# Configuration
# ================================

# Setup logging with timestamp and level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_and_convert_gdf_to_geojson_eubucco(gpkg_file, layer_name, rectangle_vertices, output_geojson):
    """
    Filters features in a GeoPackage that intersect with a given rectangle and writes them to a GeoJSON file.

    This function:
    1. Creates a polygon from the input rectangle vertices
    2. Handles coordinate system transformations if needed
    3. Filters buildings that intersect with the target area
    4. Processes building geometries and properties
    5. Writes filtered data to GeoJSON format

    Parameters:
    - gpkg_file (str): Path to the GeoPackage file containing building data
    - layer_name (str): Name of the layer within the GeoPackage to process
    - rectangle_vertices (list of tuples): List of (longitude, latitude) tuples defining the rectangle vertices
    - output_geojson (str): Path where the output GeoJSON file will be written

    Returns:
    None

    Notes:
    - The function assumes input coordinates are in WGS84 (EPSG:4326)
    - Building heights are stored in meters
    - Missing or invalid heights are assigned a default value of -1.0
    - A confidence value of -1.0 indicates no confidence data available
    """
    # Create polygon from rectangle vertices (already in lon,lat format)
    rectangle_polygon = Polygon(rectangle_vertices)

    # Get Shapely version for compatibility checks
    shapely_version = shapely.__version__
    major_version = int(shapely_version.split('.')[0])
    logging.info(f"Using Shapely version: {shapely_version}")

    # Process the GeoPackage
    with fiona.Env():
        with fiona.open(gpkg_file, layer=layer_name, mode='r') as src:
            # Verify CRS information
            src_crs = src.crs
            if not src_crs:
                logging.error("Input GeoPackage layer has no CRS defined.")
                raise ValueError("Input GeoPackage layer has no CRS defined.")

            # Check if coordinate transformation is needed
            target_crs = 'EPSG:4326'
            if src_crs.get('init') != 'epsg:4326' and 'epsg:4326' not in src_crs.values():
                transform_required = True
                logging.info("Transforming rectangle polygon to match source CRS.")
                rectangle_polygon_transformed = transform_geom(target_crs, src_crs, mapping(rectangle_polygon))
                rectangle_polygon = shape(rectangle_polygon_transformed)
            else:
                transform_required = False

            # Define schema for output GeoJSON
            output_schema = {
                'geometry': 'Polygon',
                'properties': {
                    'height': 'float',
                    'confidence': 'float'
                },
            }

            # Process features and write to output
            with fiona.open(
                output_geojson,
                'w',
                driver='GeoJSON',
                crs=target_crs,
                schema=output_schema
            ) as dst:
                feature_count = 0
                missing_height_count = 0
                
                # Iterate through all features in source file
                for idx, feature in enumerate(src):
                    # Progress logging
                    if idx % 10000 == 0 and idx > 0:
                        logging.info(f"Processed {idx} features...")

                    # Skip invalid geometries
                    geom = feature['geometry']
                    if not geom:
                        continue

                    shapely_geom = shape(geom)

                    # Check intersection with target rectangle
                    if not shapely_geom.intersects(rectangle_polygon):
                        continue

                    # Get intersection geometry
                    intersection = shapely_geom.intersection(rectangle_polygon)
                    if intersection.is_empty:
                        continue

                    # Handle different geometry types
                    if isinstance(intersection, Polygon):
                        polygons = [intersection]
                    elif isinstance(intersection, MultiPolygon):
                        if major_version < 2:
                            polygons = list(intersection)
                        else:
                            polygons = list(intersection.geoms)
                    else:
                        continue

                    # Process each polygon in the intersection
                    for poly in polygons:
                        # Transform coordinates if needed
                        if transform_required:
                            transformed_geom = transform_geom(src_crs, target_crs, mapping(poly))
                            shapely_transformed_poly = shape(transformed_geom)
                        else:
                            shapely_transformed_poly = poly

                        # Extract polygon coordinates (already in lon,lat format)
                        coords = []
                        coords.append(list(shapely_transformed_poly.exterior.coords))  # Exterior ring
                        for interior in shapely_transformed_poly.interiors:  # Interior rings (holes)
                            coords.append(list(interior.coords))

                        # Create GeoJSON geometry
                        out_geom = {
                            'type': 'Polygon',
                            'coordinates': coords
                        }

                        # Extract and validate height property
                        height = feature['properties'].get('height', -1.0)
                        if height is None:
                            height = -1.0
                            missing_height_count += 1
                            logging.debug(f"Feature ID {feature.get('id', 'N/A')} has missing 'height'. Assigned default value.")

                        try:
                            height = float(height)
                        except (TypeError, ValueError):
                            height = -1.0
                            missing_height_count += 1
                            logging.debug(f"Feature ID {feature.get('id', 'N/A')} has invalid 'height'. Assigned default value.")

                        # Create feature properties
                        properties = {
                            'height': height,
                            'confidence': -1.0  # Default confidence value
                        }

                        # Create and write output feature
                        out_feature = {
                            'geometry': out_geom,
                            'properties': properties,
                            'type': 'Feature'
                        }
                        dst.write(out_feature)
                        feature_count += 1

                # Log processing summary
                logging.info(f"Total features written to GeoJSON: {feature_count}")
                logging.info(f"Total features with missing or invalid 'height': {missing_height_count}")


def download_extract_open_gpkg_from_eubucco(url, output_dir):
    """
    Downloads a ZIP file from a URL, extracts the GeoPackage (.gpkg) file, and returns its path.

    This function:
    1. Downloads a ZIP file from the EUBUCCO API
    2. Extracts the contents to a specified directory
    3. Locates and returns the path to the GeoPackage file

    Parameters:
    - url (str): URL to download the ZIP file containing the GeoPackage
    - output_dir (str): Directory where extracted files will be stored

    Returns:
    - str: Absolute path to the extracted GeoPackage file

    Raises:
    - Exception: If download fails or no GeoPackage file is found
    - requests.exceptions.RequestException: For network-related errors

    Notes:
    - Creates a subdirectory 'EUBUCCO_raw' in the output directory
    - Logs progress and errors using the logging module
    """
    # Download ZIP file from URL
    logging.info("Downloading file...")
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to download file. Status code: {response.status_code}")
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

    # Extract contents of ZIP file
    logging.info("Extracting ZIP file...")
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(f"{output_dir}/EUBUCCO_raw")

    # Find GPKG file in extracted contents
    gpkg_file = None
    for root, dirs, files in os.walk(f"{output_dir}/EUBUCCO_raw"):
        for file in files:
            if file.endswith(".gpkg"):
                gpkg_file = os.path.join(root, file)
                break
        if gpkg_file:
            break

    if not gpkg_file:
        logging.error("No GPKG file found in the extracted files.")
        raise Exception("No GPKG file found in the extracted files.")

    logging.info(f"GeoPackage file found: {gpkg_file}")
    return gpkg_file

def get_gdf_from_eubucco(rectangle_vertices, country_links, output_dir, file_name):
    """
    Downloads, extracts, filters, and converts GeoPackage data to GeoJSON based on the rectangle vertices.

    This function:
    1. Determines the target country based on input coordinates
    2. Downloads and extracts EUBUCCO data for that country
    3. Reads the GeoPackage into a GeoDataFrame
    4. Ensures correct coordinate reference system
    5. Assigns unique IDs to buildings

    Parameters:
    - rectangle_vertices (list of tuples): List of (longitude, latitude) tuples defining the area of interest
    - country_links (dict): Dictionary mapping country names to their respective GeoPackage URLs
    - output_dir (str): Directory to save downloaded and processed files
    - file_name (str): Name for the output GeoJSON file

    Returns:
    - geopandas.GeoDataFrame: DataFrame containing building geometries and properties
        or None if the target area has no EUBUCCO data

    Notes:
    - Automatically transforms coordinates to WGS84 (EPSG:4326) if needed
    - Assigns sequential IDs to buildings starting from 0
    - Logs errors if target area is not covered by EUBUCCO
    """
    # Determine country based on first vertex
    country_name = get_country_name(rectangle_vertices[0][0], rectangle_vertices[0][1])  # Swap order for get_country_name
    if country_name in country_links:
        url = country_links[country_name]
    else:
        logging.error("Your target area does not have data in EUBUCCO.")
        return

    # Download and extract GPKG file
    gpkg_file = download_extract_open_gpkg_from_eubucco(url, output_dir)

    # Read GeoPackage file while preserving its CRS
    gdf = gpd.read_file(gpkg_file)
    
    # Only set CRS if not already set
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    # Transform to WGS84 if needed
    elif gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    # Replace id column with index numbers
    gdf['id'] = gdf.index
    
    return gdf

def load_gdf_from_eubucco(rectangle_vertices, output_dir):
    """
    Downloads EUBUCCO data and loads it as GeoJSON.

    This function serves as the main interface for loading EUBUCCO building data.
    It handles the complete workflow from downloading to processing the data.

    Parameters:
    - rectangle_vertices (list of tuples): List of (longitude, latitude) tuples defining the area
        The first vertex is used to determine which country's data to download
    - output_dir (str): Directory to save intermediate and output files
        Creates a subdirectory 'EUBUCCO_raw' for raw downloaded data

    Returns:
    - geopandas.GeoDataFrame: DataFrame containing:
        - geometry: Building footprint polygons
        - height: Building heights in meters
        - id: Unique identifier for each building
        or None if the target area has no EUBUCCO data

    Notes:
    - Output is always in WGS84 (EPSG:4326) coordinate system
    - Building heights are in meters
    - Buildings without height data are assigned a height of -1.0
    - The function automatically determines the appropriate country dataset
    """
    # Define output file path
    file_name = 'building.geojson' 
    file_path = f"{output_dir}/{file_name}"

    
    # Refresh the coutry links
    populate_country_links()

    # Download and save GeoJSON
    gdf = get_gdf_from_eubucco(rectangle_vertices, country_links, output_dir, file_name)   

    return gdf
