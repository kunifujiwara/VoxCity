"""
Module for downloading and processing Microsoft Building Footprints data.

This module provides functionality to download building footprint data from Microsoft's
open dataset, which contains building polygons extracted from satellite imagery using
AI. It handles downloading quadkey-based data files and converting them to GeoJSON format.

The data is organized using quadkeys, which are hierarchical spatial indexing strings
that identify tiles on the map at different zoom levels. Each quadkey corresponds to
a specific geographic area and zoom level.

Key Features:
- Downloads building footprint data from Microsoft's global buildings dataset
- Handles quadkey-based spatial queries
- Converts compressed data files to GeoJSON format
- Supports rectangular region queries using vertex coordinates
"""

import pandas as pd
import os
from .utils import download_file
from ..geoprocessor.utils import tile_from_lat_lon, quadkey_to_tile
from ..geoprocessor.polygon import load_gdf_from_multiple_gz, swap_coordinates

def get_geojson_links(output_dir):
    """Download and load the dataset links CSV file containing building footprint URLs.
    
    This function downloads a master CSV file from Microsoft's server that contains
    links to all available building footprint datasets. The CSV includes metadata
    such as location names, quadkeys, URLs, and file sizes for each dataset tile.
    
    Args:
        output_dir (str): Directory path where the CSV file will be saved
        
    Returns:
        pandas.DataFrame: DataFrame containing dataset links with columns:
            - Location: String identifier for the geographic region
            - QuadKey: String representing the tile's quadkey
            - Url: Direct download link for the GeoJSON data
            - Size: File size information
    
    Note:
        The CSV file is cached locally in the output directory for future use.
    """
    # URL for the master CSV file containing links to all building footprint data
    url = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
    filepath = os.path.join(output_dir, "dataset-links.csv")
    
    # Download the CSV file
    download_file(url, filepath)

    # Define data types for CSV columns to ensure proper loading
    data_types = {
        'Location': 'str',
        'QuadKey': 'str', 
        'Url': 'str',
        'Size': 'str'
    }

    # Load and return the CSV as a DataFrame
    df_links = pd.read_csv(filepath, dtype=data_types)
    return df_links

def find_row_for_location(df, lon, lat):
    """Find the dataset row containing building data for a given lon/lat coordinate.
    
    This function searches through the dataset links DataFrame to find the appropriate
    tile containing the specified geographic coordinates. It converts the input
    coordinates to tile coordinates at the same zoom level as each quadkey and
    checks for a match.
    
    Args:
        df (pandas.DataFrame): DataFrame containing dataset links from get_geojson_links()
        lon (float): Longitude coordinate to search for (-180 to 180)
        lat (float): Latitude coordinate to search for (-90 to 90)
        
    Returns:
        pandas.Series: Matching row from DataFrame containing the quadkey and download URL,
                      or None if no matching tile is found
    
    Note:
        The function handles invalid quadkeys gracefully by skipping them and
        continues searching through all available tiles.
    """
    for index, row in df.iterrows():
        quadkey = str(row['QuadKey'])
        if not isinstance(quadkey, str) or len(quadkey) == 0:
            continue
            
        try:
            # Convert lon/lat to tile coordinates at the quadkey's zoom level
            loc_tile_x, loc_tile_y = tile_from_lat_lon(lat, lon, len(quadkey))
            qk_tile_x, qk_tile_y, _ = quadkey_to_tile(quadkey)
            
            # Return row if tile coordinates match
            if loc_tile_x == qk_tile_x and loc_tile_y == qk_tile_y:
                return row
        except Exception as e:
            print(f"Error processing row {index}: {e}")
    return None

def get_mbfp_gdf(output_dir, rectangle_vertices):
    """Download and process building footprint data for a rectangular region.
    
    This function takes a list of coordinates defining a rectangular region and:
    1. Downloads the necessary building footprint data files covering the region
    2. Loads and combines the GeoJSON data from all relevant files
    3. Processes the data to ensure consistent coordinate ordering
    4. Assigns unique sequential IDs to each building
    
    Args:
        output_dir (str): Directory path where downloaded files will be saved
        rectangle_vertices (list): List of (lon, lat) tuples defining the rectangle corners.
                                 The coordinates should define a bounding box of the
                                 area of interest.
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing building footprints with columns:
            - geometry: Building polygon geometries
            - id: Sequential unique identifier for each building
            
    Note:
        - Files are downloaded only if not already present in the output directory
        - Coordinates in the input vertices should be in (longitude, latitude) order
        - The function handles cases where some vertices might not have available data
    """
    print("Downloading geojson files")
    df_links = get_geojson_links(output_dir)

    # Find and download files for each vertex of the rectangle
    filenames = []
    for vertex in rectangle_vertices:
        lon, lat = vertex
        row = find_row_for_location(df_links, lon, lat)
        if row is not None:
            # Construct filename and download if not already downloaded
            location = row["Location"]
            quadkey = row["QuadKey"]
            filename = os.path.join(output_dir, f"{location}_{quadkey}.gz")
            if filename not in filenames:
                filenames.append(filename)
                download_file(row["Url"], filename)
        else:
            print("No matching row found.")

    # Load GeoJSON data from downloaded files and fix coordinate ordering
    gdf = load_gdf_from_multiple_gz(filenames)    

    # Replace id column with index numbers
    gdf['id'] = gdf.index
    
    return gdf