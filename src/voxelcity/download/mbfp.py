"""
Module for downloading and processing Microsoft Building Footprints data.

This module provides functionality to download building footprint data from Microsoft's
open dataset, which contains building polygons extracted from satellite imagery using
AI. It handles downloading quadkey-based data files and converting them to GeoJSON format.
"""

import pandas as pd
import os
from .utils import download_file
from ..geo.utils import tile_from_lat_lon, quadkey_to_tile
from ..file.geojson import load_geojsons_from_multiple_gz, swap_coordinates

def get_geojson_links(output_dir):
    """Download and load the dataset links CSV file containing building footprint URLs.
    
    Args:
        output_dir: Directory to save the downloaded CSV file
        
    Returns:
        pandas.DataFrame: DataFrame containing dataset links and quadkey information
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

def find_row_for_location(df, lat, lon):
    """Find the dataset row containing building data for a given lat/lon coordinate.
    
    Args:
        df: DataFrame containing dataset links
        lat: Latitude coordinate to search for
        lon: Longitude coordinate to search for
        
    Returns:
        pandas.Series: Matching row from DataFrame, or None if no match found
    """
    for index, row in df.iterrows():
        quadkey = str(row['QuadKey'])
        if not isinstance(quadkey, str) or len(quadkey) == 0:
            continue
            
        try:
            # Convert lat/lon to tile coordinates at the quadkey's zoom level
            loc_tile_x, loc_tile_y = tile_from_lat_lon(lat, lon, len(quadkey))
            qk_tile_x, qk_tile_y, _ = quadkey_to_tile(quadkey)
            
            # Return row if tile coordinates match
            if loc_tile_x == qk_tile_x and loc_tile_y == qk_tile_y:
                return row
        except Exception as e:
            print(f"Error processing row {index}: {e}")
    return None

def get_mbfp_geojson(output_dir, rectangle_vertices):
    """Download and process building footprint data for a rectangular region.
    
    Args:
        output_dir: Directory to save downloaded files
        rectangle_vertices: List of (lat, lon) tuples defining the rectangle corners
        
    Returns:
        dict: GeoJSON data containing building footprints
    """
    print("Downloading geojson files")
    df_links = get_geojson_links(output_dir)

    # Find and download files for each vertex of the rectangle
    filenames = []
    for vertex in rectangle_vertices:
        lat, lon = vertex
        row = find_row_for_location(df_links, lat, lon)
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
    geojson_data = load_geojsons_from_multiple_gz(filenames)
    swap_coordinates(geojson_data)

    return geojson_data