from overturemaps import core
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping

def convert_numpy_to_python(obj):
    """
    Recursively convert numpy types to native Python types.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_to_python(obj.tolist())
    elif isinstance(obj, (bool, str, int, float)) or obj is None:
        return obj
    else:
        return str(obj)

def is_valid_value(value):
    """
    Check if a value is valid (not NA/null) and handle array-like objects.
    """
    if isinstance(value, (np.ndarray, list)):
        return True  # Always include arrays/lists
    return pd.notna(value)

def convert_gdf_to_geojson(gdf):
    """
    Convert GeoDataFrame to GeoJSON format with coordinates in (lat, lon) order.
    Extracts all columns as properties except for 'geometry' and 'bbox'.
    Sets height and min_height to 0 if not present and handles arrays.
    """
    features = []
    id_count = 1
    
    for idx, row in gdf.iterrows():
        # Get the geometry in GeoJSON format
        geom = mapping(row['geometry'])
        
        # Create a new coordinates structure with swapped (lat, lon) ordering
        if geom['type'] == 'Polygon':
            new_coordinates = []
            for ring in geom['coordinates']:
                new_ring = [[coord[1], coord[0]] for coord in ring]
                new_coordinates.append(new_ring)
            geom['coordinates'] = new_coordinates
        
        # Create properties dictionary
        properties = {}
        
        # First set height and min_height with default values of 0
        height_value = row.get('height')
        min_height_value = row.get('min_height')
        
        properties['height'] = float(height_value) if is_valid_value(height_value) else 0.0
        properties['min_height'] = float(min_height_value) if is_valid_value(min_height_value) else 0.0
        
        # Add all columns except 'geometry' and 'bbox'
        excluded_columns = {'geometry', 'bbox', 'height', 'min_height'}
        for column in gdf.columns:
            if column not in excluded_columns:
                value = row[column]
                # Always include the value, but convert it appropriately
                properties[column] = convert_numpy_to_python(value) if is_valid_value(value) else None
        
        # Add the index as id
        properties['id'] = convert_numpy_to_python(id_count)
        id_count += 1
        
        # Create feature dictionary
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
        GeoDataFrame: GeoDataFrame with bbox geometry
    """
    # Extract lat, lon values
    lats = [vertex[0] for vertex in vertices]
    lons = [vertex[1] for vertex in vertices]
    
    # Get min/max values
    min_lat = min(lats)
    max_lat = max(lats)
    min_lon = min(lons)
    max_lon = max(lons)

    return (min_lon, min_lat, max_lon, max_lat)

def join_gdfs_vertically(gdf1, gdf2):
    """
    Join two GeoDataFrames vertically, handling different columns.
    
    Args:
        gdf1 (GeoDataFrame): First GeoDataFrame
        gdf2 (GeoDataFrame): Second GeoDataFrame
        
    Returns:
        GeoDataFrame: Combined GeoDataFrame
    """
    # Print column differences for visibility
    print("GDF1 columns:", list(gdf1.columns))
    print("GDF2 columns:", list(gdf2.columns))
    print("\nColumns in GDF1 but not in GDF2:", set(gdf1.columns) - set(gdf2.columns))
    print("Columns in GDF2 but not in GDF1:", set(gdf2.columns) - set(gdf1.columns))
    
    # Create a set of all columns from both dataframes
    all_columns = set(gdf1.columns) | set(gdf2.columns)
    
    # Add missing columns to each GeoDataFrame with None values
    for col in all_columns:
        if col not in gdf1.columns:
            gdf1[col] = None
        if col not in gdf2.columns:
            gdf2[col] = None
    
    # Concatenate the GeoDataFrames
    combined_gdf = pd.concat([gdf1, gdf2], axis=0, ignore_index=True)
    
    # Ensure the result is a GeoDataFrame
    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry')
    
    # Print info about the combined GDF
    print("\nCombined GeoDataFrame info:")
    print(f"Total rows: {len(combined_gdf)}")
    print(f"Total columns: {len(combined_gdf.columns)}")
    
    return combined_gdf

def load_geojsons_from_overture(rectangle_vertices):

    # Convert to bbox
    bbox = rectangle_to_bbox(rectangle_vertices)

    building_gdf = core.geodataframe("building", bbox=bbox)
    building_part_gdf = core.geodataframe("building_part", bbox=bbox)
    joined_building_gdf = join_gdfs_vertically(building_gdf, building_part_gdf)

    geojson_features = convert_gdf_to_geojson(joined_building_gdf)

    return geojson_features