"""
Module for downloading and processing building footprint data from Overture Maps.

This module provides functionality to download and process building footprints,
handling the conversion of Overture Maps data to GeoJSON format with standardized properties.

The module includes functions for:
- Converting data types between numpy and Python native types
- Processing and validating building footprint data
- Handling geometric operations and coordinate transformations
- Combining and standardizing building data from multiple sources

Main workflow:
1. Download building data from Overture Maps using a bounding box
2. Process and standardize the data format
3. Combine building and building part data
4. Add unique identifiers and standardize properties
"""

from overturemaps import core
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping

def convert_numpy_to_python(obj):
    """
    Recursively convert numpy types to native Python types.
    
    This function handles various numpy data types and complex nested structures,
    ensuring all data is converted to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert, can be:
            - dict: Dictionary with potentially nested numpy types
            - list/tuple: Sequence with potentially nested numpy types
            - numpy.ndarray: Numpy array to be converted to list
            - numpy.integer/numpy.floating: Numpy numeric types
            - native Python types (bool, str, int, float)
            - None values
        
    Returns:
        object: Converted object with all numpy types replaced by native Python types
        
    Examples:
        >>> convert_numpy_to_python(np.int64(42))
        42
        >>> convert_numpy_to_python({'a': np.array([1, 2, 3])})
        {'a': [1, 2, 3]}
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
    
    This function is used to validate data before processing, ensuring that
    null/NA values are handled appropriately while preserving array-like structures.
    
    Args:
        value: Value to check, can be:
            - numpy.ndarray: Always considered valid
            - list: Always considered valid
            - scalar values: Checked for NA/null status
        
    Returns:
        bool: True if value is valid (not NA/null or is array-like), False otherwise
        
    Note:
        Arrays and lists are always considered valid since they may contain
        valid data that needs to be processed individually.
    """
    # Arrays and lists are always considered valid since they may contain valid data
    if isinstance(value, (np.ndarray, list)):
        return True  # Always include arrays/lists
    # Use pandas notna() to check for NA/null values in a robust way
    return pd.notna(value)

def convert_gdf_to_geojson(gdf):
    """
    Convert GeoDataFrame to GeoJSON format with coordinates in (lon, lat) order.
    
    This function processes a GeoDataFrame containing building data and converts it
    to a standardized GeoJSON format. It handles special cases for height values
    and ensures all properties are properly converted to JSON-serializable types.
    
    Args:
        gdf (GeoDataFrame): Input GeoDataFrame containing building data with columns:
            - geometry: Shapely geometry objects
            - height: Building height (optional)
            - min_height: Minimum building height (optional)
            - Additional property columns
        
    Returns:
        list: List of GeoJSON feature dictionaries, each containing:
            - type: Always "Feature"
            - properties: Dictionary of building properties including:
                - height: Building height (defaults to 0.0)
                - min_height: Minimum height (defaults to 0.0)
                - id: Sequential unique identifier
                - All other columns from input GeoDataFrame
            - geometry: GeoJSON geometry object
            
    Note:
        - Height values default to 0.0 if missing or invalid
        - All numpy types are converted to native Python types
        - Sequential IDs are assigned starting from 1
    """
    features = []
    id_count = 1
    
    for idx, row in gdf.iterrows():
        # Convert Shapely geometry to GeoJSON format
        geom = mapping(row['geometry'])
        
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
    Convert rectangle vertices in (lon, lat) format to a bounding box.
    
    This function takes a list of coordinate pairs defining a rectangle and
    converts them to a bounding box format required by the Overture Maps API.
    
    Args:
        vertices (list): List of tuples containing (lon, lat) coordinates
            defining the corners of a rectangle
        
    Returns:
        tuple: Bounding box coordinates in format (min_lon, min_lat, max_lon, max_lat)
            suitable for use with Overture Maps API
            
    Note:
        The function calculates the minimum and maximum coordinates to ensure
        the bounding box encompasses all provided vertices.
    """
    # Extract lon, lat values from vertices
    lons = [vertex[0] for vertex in vertices]
    lats = [vertex[1] for vertex in vertices]
    
    # Calculate bounding box extents
    min_lat = min(lats)
    max_lat = max(lats)
    min_lon = min(lons)
    max_lon = max(lons)

    # Return bbox in format expected by Overture Maps API
    return (min_lon, min_lat, max_lon, max_lat)

def join_gdfs_vertically(gdf1, gdf2):
    """
    Join two GeoDataFrames vertically, handling different column structures.
    
    This function combines two GeoDataFrames that may have different columns,
    ensuring all columns from both datasets are preserved in the output.
    It provides diagnostic information about the combining process.
    
    Args:
        gdf1 (GeoDataFrame): First GeoDataFrame (e.g., buildings)
        gdf2 (GeoDataFrame): Second GeoDataFrame (e.g., building parts)
        
    Returns:
        GeoDataFrame: Combined GeoDataFrame containing:
            - All rows from both input GeoDataFrames
            - All columns from both inputs (filled with None where missing)
            - Preserved geometry column
            
    Note:
        - Prints diagnostic information about column differences
        - Handles missing columns by filling with None values
        - Preserves the geometry column for spatial operations
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

def load_gdf_from_overture(rectangle_vertices, floor_height=3.0):
    """
    Download and process building footprint data from Overture Maps.
    
    This function serves as the main entry point for downloading building data.
    It handles the complete workflow of downloading both building and building
    part data, combining them, and preparing them for further processing.
    
    Args:
        rectangle_vertices (list): List of (lon, lat) coordinates defining
            the bounding box for data download
        
    Returns:
        GeoDataFrame: Combined dataset containing:
            - Building and building part geometries
            - Standardized properties
            - Sequential numeric IDs
            
    Note:
        - Downloads both building and building_part data from Overture Maps
        - Combines the datasets while preserving all properties
        - Assigns sequential IDs based on the final dataset index
    """
    # Convert input vertices to Overture Maps API bounding box format
    bbox = rectangle_to_bbox(rectangle_vertices)

    # Download primary building footprints and additional building part data
    building_gdf = core.geodataframe("building", bbox=bbox)
    building_part_gdf = core.geodataframe("building_part", bbox=bbox)
    
    # Combine both datasets into a single comprehensive building dataset
    joined_building_gdf = join_gdfs_vertically(building_gdf, building_part_gdf)

    # Ensure numeric height and infer from floors when missing
    try:
        joined_building_gdf['height'] = pd.to_numeric(joined_building_gdf.get('height', None), errors='coerce')
    except Exception:
        # Create height column if missing
        joined_building_gdf['height'] = None
        joined_building_gdf['height'] = pd.to_numeric(joined_building_gdf['height'], errors='coerce')

    # Combine possible floors columns (first non-null among candidates)
    floors_candidates = []
    for col in ['building:levels', 'levels', 'num_floors', 'floors']:
        if col in joined_building_gdf.columns:
            floors_candidates.append(pd.to_numeric(joined_building_gdf[col], errors='coerce'))
    if floors_candidates:
        floors_series = floors_candidates[0]
        for s in floors_candidates[1:]:
            floors_series = floors_series.combine_first(s)
        # Infer height where height is NaN/<=0 and floors > 0
        mask_missing_height = (~joined_building_gdf['height'].notna()) | (joined_building_gdf['height'] <= 0)
        if isinstance(floor_height, (int, float)):
            inferred = floors_series * float(floor_height)
        else:
            inferred = floors_series * 3.0
        joined_building_gdf.loc[mask_missing_height & (floors_series > 0), 'height'] = inferred

    # Assign sequential IDs based on the final dataset index
    joined_building_gdf['id'] = joined_building_gdf.index

    return joined_building_gdf