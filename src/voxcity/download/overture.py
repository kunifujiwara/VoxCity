"""
Module for downloading and processing building footprint data from Overture Maps.

This module provides functionality to download and process building footprints,
handling the conversion of Overture Maps data to GeoJSON format with standardized properties.
"""

from overturemaps import core
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping

def convert_numpy_to_python(obj):
    """
    Recursively convert numpy types to native Python types.
    
    Args:
        obj: Object to convert, can be dict, list, tuple, numpy type or other
        
    Returns:
        Converted object with numpy types replaced by native Python types
    """
    # Handle dictionary case - recursively convert all values
    if isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    # Handle list case - recursively convert all items
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    # Handle tuple case - recursively convert all items
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    # Convert numpy integer types to Python int
    elif isinstance(obj, np.integer):
        return int(obj)
    # Convert numpy float types to Python float
    elif isinstance(obj, np.floating):
        return float(obj)
    # Convert numpy arrays to Python lists recursively
    elif isinstance(obj, np.ndarray):
        return convert_numpy_to_python(obj.tolist())
    # Keep native Python types and None as-is
    elif isinstance(obj, (bool, str, int, float)) or obj is None:
        return obj
    # Convert anything else to string
    else:
        return str(obj)

def is_valid_value(value):
    """
    Check if a value is valid (not NA/null) and handle array-like objects.
    
    Args:
        value: Value to check
        
    Returns:
        bool: True if value is valid (not NA/null or is array-like), False otherwise
    """
    # Arrays and lists are always considered valid since they may contain valid data
    if isinstance(value, (np.ndarray, list)):
        return True  # Always include arrays/lists
    # Use pandas notna() to check for NA/null values in a robust way
    return pd.notna(value)

def convert_gdf_to_geojson(gdf):
    """
    Convert GeoDataFrame to GeoJSON format with coordinates in (lat, lon) order.
    Extracts all columns as properties except for 'geometry' and 'bbox'.
    Sets height and min_height to 0 if not present and handles arrays.
    
    Args:
        gdf (GeoDataFrame): Input GeoDataFrame to convert
        
    Returns:
        list: List of GeoJSON feature dictionaries
    """
    features = []
    id_count = 1
    
    for idx, row in gdf.iterrows():
        # Convert Shapely geometry to GeoJSON format
        geom = mapping(row['geometry'])
        
        # Swap coordinate ordering from (lon, lat) to (lat, lon)
        if geom['type'] == 'Polygon':
            new_coordinates = []
            for ring in geom['coordinates']:
                new_ring = [[coord[1], coord[0]] for coord in ring]
                new_coordinates.append(new_ring)
            geom['coordinates'] = new_coordinates
        
        # Initialize properties dictionary for this feature
        properties = {}
        
        # Handle height values with defaults
        height_value = row.get('height')
        min_height_value = row.get('min_height')
        
        # Set height values, defaulting to 0.0 if invalid/missing
        properties['height'] = float(height_value) if is_valid_value(height_value) else 0.0
        properties['min_height'] = float(min_height_value) if is_valid_value(min_height_value) else 0.0
        
        # Process all other columns except excluded ones
        excluded_columns = {'geometry', 'bbox', 'height', 'min_height'}
        for column in gdf.columns:
            if column not in excluded_columns:
                value = row[column]
                # Convert value to Python native type if valid, otherwise set to None
                properties[column] = convert_numpy_to_python(value) if is_valid_value(value) else None
        
        # Add sequential ID to properties
        properties['id'] = convert_numpy_to_python(id_count)
        id_count += 1
        
        # Create GeoJSON feature object
        feature = {
            'type': 'Feature',
            'properties': convert_numpy_to_python(properties),
            'geometry': convert_numpy_to_python(geom)
        }
        
        features.append(feature)
    
    return features

def rectangle_to_bbox(vertices):
    """
    Convert rectangle vertices in (lat, lon) format to a GeoDataFrame bbox
    with Shapely box geometry in (minx, miny, maxx, maxy) format
    
    Args:
        vertices (list): List of tuples containing (lat, lon) coordinates
        
    Returns:
        tuple: Bounding box coordinates (min_lon, min_lat, max_lon, max_lat)
    """
    # Extract lat, lon values from vertices
    lats = [vertex[0] for vertex in vertices]
    lons = [vertex[1] for vertex in vertices]
    
    # Calculate bounding box extents
    min_lat = min(lats)
    max_lat = max(lats)
    min_lon = min(lons)
    max_lon = max(lons)

    # Return bbox in format expected by Overture Maps API
    return (min_lon, min_lat, max_lon, max_lat)

def join_gdfs_vertically(gdf1, gdf2):
    """
    Join two GeoDataFrames vertically, handling different columns.
    
    Args:
        gdf1 (GeoDataFrame): First GeoDataFrame
        gdf2 (GeoDataFrame): Second GeoDataFrame
        
    Returns:
        GeoDataFrame: Combined GeoDataFrame with all columns from both inputs
    """
    # Print diagnostic information about column differences
    print("GDF1 columns:", list(gdf1.columns))
    print("GDF2 columns:", list(gdf2.columns))
    print("\nColumns in GDF1 but not in GDF2:", set(gdf1.columns) - set(gdf2.columns))
    print("Columns in GDF2 but not in GDF1:", set(gdf2.columns) - set(gdf1.columns))
    
    # Get union of all columns from both dataframes
    all_columns = set(gdf1.columns) | set(gdf2.columns)
    
    # Add missing columns with None values to ensure compatible schemas
    for col in all_columns:
        if col not in gdf1.columns:
            gdf1[col] = None
        if col not in gdf2.columns:
            gdf2[col] = None
    
    # Vertically concatenate the GeoDataFrames
    combined_gdf = pd.concat([gdf1, gdf2], axis=0, ignore_index=True)
    
    # Convert back to GeoDataFrame to preserve geometry column
    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry')
    
    # Print summary statistics of combined dataset
    print("\nCombined GeoDataFrame info:")
    print(f"Total rows: {len(combined_gdf)}")
    print(f"Total columns: {len(combined_gdf.columns)}")
    
    return combined_gdf

def load_geojsons_from_overture(rectangle_vertices):
    """
    Download and process building footprint data from Overture Maps.
    
    Args:
        rectangle_vertices (list): List of (lat, lon) coordinates defining the bounding box
        
    Returns:
        list: List of GeoJSON features containing building footprints with standardized properties
    """
    # Convert vertices to bounding box format required by Overture Maps
    bbox = rectangle_to_bbox(rectangle_vertices)

    # Download building and building part data from Overture Maps
    building_gdf = core.geodataframe("building", bbox=bbox)
    building_part_gdf = core.geodataframe("building_part", bbox=bbox)
    
    # Combine building and building part data into single dataset
    joined_building_gdf = join_gdfs_vertically(building_gdf, building_part_gdf)

    # Convert combined dataset to GeoJSON format
    geojson_features = convert_gdf_to_geojson(joined_building_gdf)

    return geojson_features