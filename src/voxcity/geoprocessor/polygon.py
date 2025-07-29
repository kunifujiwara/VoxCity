"""
Module for handling GeoJSON data related to building footprints and heights.

This module provides functionality for loading, filtering, transforming and saving GeoJSON data,
with a focus on building footprints and their height information. It includes functions for
coordinate transformations, spatial filtering, and height data extraction from various sources.
"""

# Required imports for GIS operations, data manipulation and file handling
import geopandas as gpd
import json
from shapely.geometry import Polygon, Point, shape
from shapely.errors import GEOSException, ShapelyError
import pandas as pd
import numpy as np
import gzip
from typing import List, Dict
from pyproj import Transformer, CRS
import rasterio
from rasterio.mask import mask
import copy
from rtree import index

from .utils import validate_polygon_coordinates

def filter_and_convert_gdf_to_geojson(gdf, rectangle_vertices):
    """
    Filter a GeoDataFrame by a bounding rectangle and convert to GeoJSON format.
    
    This function performs spatial filtering on a GeoDataFrame using a bounding rectangle,
    and converts the filtered data to GeoJSON format. It handles both Polygon and MultiPolygon
    geometries, splitting MultiPolygons into separate Polygon features.
    
    Args:
        gdf (GeoDataFrame): Input GeoDataFrame containing building data
            Must have 'geometry' and 'height' columns
            Any CRS is accepted, will be converted to WGS84 if needed
        rectangle_vertices (list): List of (lon, lat) tuples defining the bounding rectangle
            Must be in WGS84 (EPSG:4326) coordinate system
            Must form a valid rectangle (4 vertices, clockwise or counterclockwise)
        
    Returns:
        list: List of GeoJSON features within the bounding rectangle
            Each feature contains:
            - geometry: Polygon coordinates in WGS84
            - properties: Dictionary with 'height', 'confidence', and 'id'
            - type: Always "Feature"
    
    Memory Optimization:
        - Uses spatial indexing for efficient filtering
        - Downcasts numeric columns to save memory
        - Cleans up intermediate data structures
        - Splits MultiPolygons into separate features
    """
    # Reproject to WGS84 if necessary for consistent coordinate system
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs(epsg=4326)

    # Downcast 'height' to float32 to save memory
    gdf['height'] = pd.to_numeric(gdf['height'], downcast='float')

    # Add 'confidence' column with default value for height reliability
    gdf['confidence'] = -1.0

    # Create shapely polygon from rectangle vertices for spatial filtering
    rectangle_polygon = Polygon(rectangle_vertices)

    # Use spatial index to efficiently filter geometries that intersect with rectangle
    gdf.sindex  # Ensure spatial index is built
    possible_matches_index = list(gdf.sindex.intersection(rectangle_polygon.bounds))
    possible_matches = gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(rectangle_polygon)]
    filtered_gdf = precise_matches.copy()

    # Delete intermediate data to save memory
    del gdf, possible_matches, precise_matches

    # Create GeoJSON features from filtered geometries
    features = []
    feature_id = 1
    for idx, row in filtered_gdf.iterrows():
        geom = row['geometry'].__geo_interface__
        properties = {
            'height': row['height'],
            'confidence': row['confidence'],
            'id': feature_id
        }

        # Handle MultiPolygon by splitting into separate Polygon features
        if geom['type'] == 'MultiPolygon':
            for polygon_coords in geom['coordinates']:
                single_geom = {
                    'type': 'Polygon',
                    'coordinates': polygon_coords
                }
                feature = {
                    'type': 'Feature',
                    'properties': properties.copy(),  # Use copy to avoid shared references
                    'geometry': single_geom
                }
                features.append(feature)
                feature_id += 1
        elif geom['type'] == 'Polygon':
            feature = {
                'type': 'Feature',
                'properties': properties,
                'geometry': geom
            }
            features.append(feature)
            feature_id += 1
        else:
            pass  # Skip other geometry types

    # Create a FeatureCollection
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    # Clean up memory
    del filtered_gdf, features

    return geojson["features"]

def get_geojson_from_gpkg(gpkg_path, rectangle_vertices):
    """
    Read a GeoPackage file and convert it to GeoJSON format within a bounding rectangle.
    
    Args:
        gpkg_path (str): Path to the GeoPackage file
        rectangle_vertices (list): List of (lon, lat) tuples defining the bounding rectangle
        
    Returns:
        list: List of GeoJSON features within the bounding rectangle
    """
    # Open and read the GPKG file
    print(f"Opening GPKG file: {gpkg_path}")
    gdf = gpd.read_file(gpkg_path)
    geojson = filter_and_convert_gdf_to_geojson(gdf, rectangle_vertices)
    return geojson

def extract_building_heights_from_gdf(gdf_0: gpd.GeoDataFrame, gdf_1: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract building heights from one GeoDataFrame and apply them to another based on spatial overlap.
    
    This function transfers height information from a reference GeoDataFrame to a primary GeoDataFrame
    based on the spatial overlap between building footprints. For each building in the primary dataset
    that needs height data, it calculates a weighted average height from overlapping buildings in the
    reference dataset.
    
    Args:
        gdf_0 (gpd.GeoDataFrame): Primary GeoDataFrame to update with heights
            Must have 'geometry' column with building footprints
            Will be updated with height values where missing or zero
        gdf_1 (gpd.GeoDataFrame): Reference GeoDataFrame containing height data
            Must have 'geometry' column with building footprints
            Must have 'height' column with valid height values
        
    Returns:
        gpd.GeoDataFrame: Updated primary GeoDataFrame with extracted heights
            Buildings with overlapping reference data get weighted average heights
            Buildings without overlapping data retain original height or get NaN
    
    Statistics Tracked:
        - count_0: Number of buildings without height in primary dataset
        - count_1: Number of buildings successfully updated with height
        - count_2: Number of buildings where no reference height data found
    
    Note:
        - Uses R-tree spatial indexing for efficient overlap detection
        - Handles invalid geometries by attempting to fix them with buffer(0)
        - Weighted average is based on the area of overlap between buildings
    """
    # Make a copy of input GeoDataFrame to avoid modifying original
    gdf_primary = gdf_0.copy()
    gdf_ref = gdf_1.copy()

    # Make sure height columns exist with default values
    if 'height' not in gdf_primary.columns:
        gdf_primary['height'] = 0.0
    if 'height' not in gdf_ref.columns:
        gdf_ref['height'] = 0.0

    # Initialize counters for statistics
    count_0 = 0  # Buildings without height
    count_1 = 0  # Buildings updated with height
    count_2 = 0  # Buildings with no height data found

    # Create spatial index for reference buildings to speed up intersection tests
    spatial_index = index.Index()
    for i, geom in enumerate(gdf_ref.geometry):
        if geom.is_valid:
            spatial_index.insert(i, geom.bounds)

    # Process each building in primary dataset that needs height data
    for idx_primary, row in gdf_primary.iterrows():
        if row['height'] <= 0 or pd.isna(row['height']):
            count_0 += 1
            geom = row.geometry
            
            # Variables for weighted average height calculation
            overlapping_height_area = 0  # Sum of (height * overlap_area)
            overlapping_area = 0         # Total overlap area
            
            # Get potential intersecting buildings using spatial index
            potential_matches = list(spatial_index.intersection(geom.bounds))
            
            # Check intersections with reference buildings
            for ref_idx in potential_matches:
                if ref_idx >= len(gdf_ref):
                    continue
                    
                ref_row = gdf_ref.iloc[ref_idx]
                try:
                    # Calculate intersection if geometries overlap
                    if geom.intersects(ref_row.geometry):
                        overlap_area = geom.intersection(ref_row.geometry).area
                        overlapping_height_area += ref_row['height'] * overlap_area
                        overlapping_area += overlap_area
                except GEOSException:
                    # Try to fix invalid geometries using buffer(0)
                    try:
                        fixed_ref_geom = ref_row.geometry.buffer(0)
                        if geom.intersects(fixed_ref_geom):
                            overlap_area = geom.intersection(fixed_ref_geom).area
                            overlapping_height_area += ref_row['height'] * overlap_area
                            overlapping_area += overlap_area
                    except Exception:
                        print(f"Failed to fix polygon")
                    continue
            
            # Update height if overlapping buildings found
            if overlapping_height_area > 0:
                count_1 += 1
                # Calculate weighted average height based on overlap areas
                new_height = overlapping_height_area / overlapping_area
                gdf_primary.at[idx_primary, 'height'] = new_height
            else:
                count_2 += 1
                gdf_primary.at[idx_primary, 'height'] = np.nan
    
    # Print statistics about height updates
    if count_0 > 0:
        print(f"For {count_1} of these building footprints without height, values from the complementary source were assigned.")
        print(f"For {count_2} of these building footprints without height, no data exist in complementary data.")

    return gdf_primary

# from typing import List, Dict
# from shapely.geometry import shape
# from shapely.errors import GEOSException
# import numpy as np

# def complement_building_heights_from_geojson(geojson_data_0: List[Dict], geojson_data_1: List[Dict]) -> List[Dict]:
#     """
#     Complement building heights in one GeoJSON dataset with data from another and add non-intersecting buildings.
    
#     Args:
#         geojson_data_0 (List[Dict]): Primary GeoJSON features to update with heights
#         geojson_data_1 (List[Dict]): Reference GeoJSON features containing height data
        
#     Returns:
#         List[Dict]: Updated GeoJSON features with complemented heights and additional buildings
#     """
#     # Convert primary dataset to Shapely polygons for intersection checking
#     existing_buildings = []
#     for feature in geojson_data_0:
#         geom = shape(feature['geometry'])
#         existing_buildings.append(geom)
    
#     # Convert reference dataset to Shapely polygons with height info
#     reference_buildings = []
#     for feature in geojson_data_1:
#         geom = shape(feature['geometry'])
#         height = feature['properties']['height']
#         reference_buildings.append((geom, height, feature))
    
#     # Initialize counters for statistics
#     count_0 = 0  # Buildings without height
#     count_1 = 0  # Buildings updated with height
#     count_2 = 0  # Buildings with no height data found
#     count_3 = 0  # New non-intersecting buildings added
    
#     # Process primary dataset and update heights where needed
#     updated_geojson_data_0 = []
#     for feature in geojson_data_0:
#         geom = shape(feature['geometry'])
#         height = feature['properties']['height']
#         if height == 0:     
#             count_0 += 1       
#             # Calculate weighted average height based on overlapping areas
#             overlapping_height_area = 0
#             overlapping_area = 0
#             for ref_geom, ref_height, _ in reference_buildings:
#                 try:
#                     if geom.intersects(ref_geom):
#                         overlap_area = geom.intersection(ref_geom).area
#                         overlapping_height_area += ref_height * overlap_area
#                         overlapping_area += overlap_area
#                 except GEOSException as e:
#                     # Try to fix invalid geometries
#                     try:
#                         fixed_ref_geom = ref_geom.buffer(0)
#                         if geom.intersects(fixed_ref_geom):
#                             overlap_area = geom.intersection(ref_geom).area
#                             overlapping_height_area += ref_height * overlap_area
#                             overlapping_area += overlap_area
#                     except Exception as fix_error:
#                         print(f"Failed to fix polygon")
#                     continue
            
#             # Update height if overlapping buildings found
#             if overlapping_height_area > 0:
#                 count_1 += 1
#                 new_height = overlapping_height_area / overlapping_area
#                 feature['properties']['height'] = new_height
#             else:
#                 count_2 += 1
#                 feature['properties']['height'] = np.nan
        
#         updated_geojson_data_0.append(feature)
    
#     # Add non-intersecting buildings from reference dataset
#     for ref_geom, ref_height, ref_feature in reference_buildings:
#         has_intersection = False
#         try:
#             # Check if reference building intersects with any existing building
#             for existing_geom in existing_buildings:
#                 if ref_geom.intersects(existing_geom):
#                     has_intersection = True
#                     break
            
#             # Add building if it doesn't intersect with any existing ones
#             if not has_intersection:
#                 updated_geojson_data_0.append(ref_feature)
#                 count_3 += 1
                
#         except GEOSException as e:
#             # Try to fix invalid geometries
#             try:
#                 fixed_ref_geom = ref_geom.buffer(0)
#                 for existing_geom in existing_buildings:
#                     if fixed_ref_geom.intersects(existing_geom):
#                         has_intersection = True
#                         break
                
#                 if not has_intersection:
#                     updated_geojson_data_0.append(ref_feature)
#                     count_3 += 1
#             except Exception as fix_error:
#                 print(f"Failed to process non-intersecting building")
#             continue
    
#     # Print statistics about updates
#     if count_0 > 0:
#         print(f"{count_0} of the total {len(geojson_data_0)} building footprint from base source did not have height data.")
#         print(f"For {count_1} of these building footprints without height, values from complement source were assigned.")
#         print(f"{count_3} non-intersecting buildings from Microsoft Building Footprints were added to the output.")
    
#     return updated_geojson_data_0

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from shapely.errors import GEOSException

def geojson_to_gdf(geojson_data, id_col='id'):
    """
    Convert a list of GeoJSON-like dict features into a GeoDataFrame.
    
    This function takes a list of GeoJSON feature dictionaries (Fiona-like format)
    and converts them into a GeoDataFrame, handling geometry conversion and property
    extraction. It ensures each feature has a unique identifier.
    
    Args:
        geojson_data (List[Dict]): A list of feature dicts (Fiona-like)
            Each dict must have 'geometry' and 'properties' keys
            'geometry' must be a valid GeoJSON geometry
            'properties' can be empty but must be a dict if present
        id_col (str, optional): Name of property to use as an identifier
            Default is 'id'
            If not found in properties, a sequential ID will be created
            Must be a string that can be used as a column name
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with geometry and property columns
            Will have 'geometry' column with Shapely geometries
            Will have columns for all properties found in features
            Will have id_col with unique identifiers
            Will be set to WGS84 (EPSG:4326) coordinate system
    
    Note:
        - Handles missing properties gracefully
        - Creates sequential IDs if id_col not found
        - Converts GeoJSON geometries to Shapely objects
        - Sets WGS84 as coordinate system
        - Preserves all properties as columns
    """
    # Build lists for geometry and properties
    geometries = []
    all_props = []

    for i, feature in enumerate(geojson_data):
        # Extract geometry and convert to Shapely object
        geom = feature.get('geometry')
        shapely_geom = shape(geom) if geom else None

        # Extract properties, ensuring they exist
        props = feature.get('properties', {})
        
        # If specified ID column is missing, create sequential ID
        if id_col not in props:
            props[id_col] = i  # fallback ID

        # Capture geometry and all properties
        geometries.append(shapely_geom)
        all_props.append(props)

    # Create GeoDataFrame with geometries and properties
    gdf = gpd.GeoDataFrame(all_props, geometry=geometries, crs="EPSG:4326")
    return gdf


def complement_building_heights_from_gdf(gdf_0, gdf_1,
                                    primary_id='id', ref_id='id'):
    """
    Use a vectorized approach with GeoPandas to:
      1) Find intersections and compute weighted average heights
      2) Update heights in the primary dataset
      3) Add non-intersecting buildings from the reference dataset
    
    Args:
        gdf_0 (gpd.GeoDataFrame): Primary GeoDataFrame
        gdf_1 (gpd.GeoDataFrame): Reference GeoDataFrame
        primary_id (str): Name of the unique identifier in primary dataset's properties
        ref_id (str): Name of the unique identifier in reference dataset's properties

    Returns:
        gpd.GeoDataFrame: Updated GeoDataFrame (including new buildings).
    """
    # Make a copy of input GeoDataFrames to avoid modifying originals
    gdf_primary = gdf_0.copy()
    gdf_ref = gdf_1.copy()

    # Ensure both are in the same CRS, e.g. EPSG:4326 or some projected CRS
    # If needed, do something like:
    # gdf_primary = gdf_primary.to_crs("EPSG:xxxx")
    # gdf_ref = gdf_ref.to_crs("EPSG:xxxx")

    # Make sure height columns exist
    if 'height' not in gdf_primary.columns:
        gdf_primary['height'] = 0.0
    if 'height' not in gdf_ref.columns:
        gdf_ref['height'] = 0.0

    # ----------------------------------------------------------------
    # 1) Intersection to compute areas for overlapping buildings
    # ----------------------------------------------------------------
    # We'll rename columns to avoid collision after overlay
    gdf_primary = gdf_primary.rename(columns={'height': 'height_primary'})
    gdf_ref = gdf_ref.rename(columns={'height': 'height_ref'})

    # We perform an 'intersection' overlay to get the overlapping polygons
    intersect_gdf = gpd.overlay(gdf_primary, gdf_ref, how='intersection')

    # Compute intersection area
    intersect_gdf['intersect_area'] = intersect_gdf.area
    intersect_gdf['height_area'] = intersect_gdf['height_ref'] * intersect_gdf['intersect_area']

    # ----------------------------------------------------------------
    # 2) Aggregate to get weighted average height for each primary building
    # ----------------------------------------------------------------
    # We group by the primary building ID, summing up the area and the 'height_area'
    group_cols = {
        'height_area': 'sum',
        'intersect_area': 'sum'
    }
    grouped = intersect_gdf.groupby(f'{primary_id}_1').agg(group_cols)

    # Weighted average
    grouped['weighted_height'] = grouped['height_area'] / grouped['intersect_area']

    # ----------------------------------------------------------------
    # 3) Merge aggregated results back to the primary GDF
    # ----------------------------------------------------------------
    # After merging, the primary GDF will have a column 'weighted_height'
    gdf_primary = gdf_primary.merge(grouped['weighted_height'],
                                    left_on=primary_id,
                                    right_index=True,
                                    how='left')

    # Where primary had zero or missing height, we assign the new weighted height
    zero_or_nan_mask = (gdf_primary['height_primary'] == 0) | (gdf_primary['height_primary'].isna())
    
    # Only update heights where we have valid weighted heights
    valid_weighted_height_mask = zero_or_nan_mask & gdf_primary['weighted_height'].notna()
    gdf_primary.loc[valid_weighted_height_mask, 'height_primary'] = gdf_primary.loc[valid_weighted_height_mask, 'weighted_height']
    gdf_primary['height_primary'] = gdf_primary['height_primary'].fillna(np.nan)

    # ----------------------------------------------------------------
    # 4) Identify reference buildings that do not intersect any primary building
    # ----------------------------------------------------------------
    # Another overlay or spatial join can do this:
    # Option A: use 'difference' on reference to get non-overlapping parts, but that can chop polygons.
    # Option B: check building-level intersection. We'll do a bounding test with sjoin.
    
    # For building-level intersection, do a left join of ref onto primary.
    # Then we'll identify which reference IDs are missing from the intersection result.
    sjoin_gdf = gpd.sjoin(gdf_ref, gdf_primary, how='left', predicate='intersects')
    
    # Find reference buildings that don't intersect with any primary building
    non_intersect_mask = sjoin_gdf[f'{primary_id}_right'].isna()
    non_intersect_ids = sjoin_gdf[non_intersect_mask][f'{ref_id}_left'].unique()

    # Extract them from the original reference GDF
    gdf_ref_non_intersect = gdf_ref[gdf_ref[ref_id].isin(non_intersect_ids)]

    # We'll rename columns back to 'height' to be consistent
    gdf_ref_non_intersect = gdf_ref_non_intersect.rename(columns={'height_ref': 'height'})

    # Also rename any other properties you prefer. For clarity, keep an ID so you know they came from reference.

    # ----------------------------------------------------------------
    # 5) Combine the updated primary GDF with the new reference buildings
    # ----------------------------------------------------------------
    # First, rename columns in updated primary GDF
    gdf_primary = gdf_primary.rename(columns={'height_primary': 'height'})
    # Drop the 'weighted_height' column to clean up
    if 'weighted_height' in gdf_primary.columns:
        gdf_primary.drop(columns='weighted_height', inplace=True)

    # Concatenate
    final_gdf = pd.concat([gdf_primary, gdf_ref_non_intersect], ignore_index=True)

    # Calculate statistics
    count_total = len(gdf_primary)
    count_0 = len(gdf_primary[zero_or_nan_mask])
    count_1 = len(gdf_primary[valid_weighted_height_mask])
    count_2 = count_0 - count_1
    count_3 = len(gdf_ref_non_intersect)
    count_4 = count_3
    height_mask = gdf_ref_non_intersect['height'].notna() & (gdf_ref_non_intersect['height'] > 0)
    count_5 = len(gdf_ref_non_intersect[height_mask])
    count_6 = count_4 - count_5
    final_height_mask = final_gdf['height'].notna() & (final_gdf['height'] > 0)
    count_7 = len(final_gdf[final_height_mask])
    count_8 = len(final_gdf)

    # Print statistics if there were buildings without height data
    if count_0 > 0:
        print(f"{count_0} of the total {count_total} building footprints from base data source did not have height data.")
        print(f"For {count_1} of these building footprints without height, values from complementary data were assigned.")
        print(f"For the rest {count_2}, no data exists in complementary data.")
        print(f"Footprints of {count_3} buildings were added from the complementary source.")
        print(f"Of these {count_4} additional building footprints, {count_5} had height data while {count_6} had no height data.")
        print(f"In total, {count_7} buildings had height data out of {count_8} total building footprints.")

    return final_gdf


def gdf_to_geojson_dicts(gdf, id_col='id'):
    """
    Convert a GeoDataFrame to a list of dicts similar to GeoJSON features.
    
    This function converts a GeoDataFrame into a list of dictionary objects that
    follow the GeoJSON Feature format. Each feature will have geometry and properties,
    with an optional ID field handled separately from other properties.
    
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to convert
            Must have 'geometry' column with Shapely geometries
            All non-geometry columns will become properties
            Can optionally have id_col for unique identifiers
        id_col (str, optional): Name of column to use as feature ID
            Default is 'id'
            If present, will be excluded from properties
            If not present, features will not have explicit IDs
    
    Returns:
        list: List of GeoJSON-like feature dictionaries
            Each dict will have:
            - type: Always "Feature"
            - geometry: GeoJSON geometry from Shapely object
            - properties: All columns except geometry and ID
    
    Note:
        - Converts Shapely geometries to GeoJSON format
        - Preserves all non-geometry columns as properties
        - Handles missing ID column gracefully
        - Maintains original property types
        - Excludes ID from properties if specified
    """
    # Convert GeoDataFrame to dictionary records for easier processing
    records = gdf.to_dict(orient='records')
    features = []
    
    for rec in records:
        # Extract and convert geometry to GeoJSON format using __geo_interface__
        geom = rec.pop('geometry', None)
        if geom is not None:
            geom = geom.__geo_interface__
            
        # Extract ID if present and create properties dict excluding ID
        feature_id = rec.get(id_col, None)
        props = {k: v for k, v in rec.items() if k != id_col}
        
        # Create GeoJSON Feature object with type, properties, and geometry
        feature = {
            'type': 'Feature',
            'properties': props,
            'geometry': geom
        }
        features.append(feature)

    return features

def load_gdf_from_multiple_gz(file_paths):
    """
    Load GeoJSON features from multiple gzipped files into a single GeoDataFrame.
    
    This function reads multiple gzipped GeoJSON files, where each line in each file
    represents a single GeoJSON feature. It combines all features into a single
    GeoDataFrame, ensuring height properties are properly handled and coordinates
    are in WGS84.
    
    Args:
        file_paths (list): List of paths to gzipped GeoJSON files
            Each file should contain one GeoJSON feature per line
            Files should be readable as UTF-8 text
            Features should be in WGS84 coordinate system
        
    Returns:
        gpd.GeoDataFrame: Combined GeoDataFrame containing all features
            Will have 'geometry' column with building footprints
            Will have 'height' column (0 for missing values)
            Will be set to WGS84 (EPSG:4326) coordinate system
    
    Note:
        - Skips lines that cannot be parsed as valid JSON
        - Sets missing height values to 0
        - Assumes input coordinates are in WGS84
        - Memory usage scales with total number of features
        - Reports JSON parsing errors but continues processing
    """
    # Initialize list to store all GeoJSON features
    geojson_objects = []
    
    # Process each gzipped file
    for gz_file_path in file_paths:
        # Read each gzipped file line by line as UTF-8 text
        with gzip.open(gz_file_path, 'rt', encoding='utf-8') as file:
            for line in file:
                try:
                    # Parse each line as a GeoJSON feature
                    data = json.loads(line)
                    
                    # Ensure height property exists and has valid value
                    if 'properties' in data and 'height' in data['properties']:
                        if data['properties']['height'] is None:
                            data['properties']['height'] = 0
                    else:
                        # Create properties dict if missing
                        if 'properties' not in data:
                            data['properties'] = {}
                        # Set default height value
                        data['properties']['height'] = 0
                        
                    geojson_objects.append(data)
                except json.JSONDecodeError as e:
                    print(f"Skipping line in {gz_file_path} due to JSONDecodeError: {e}")
    
    # Convert list of GeoJSON features to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(geojson_objects)
    
    # Set coordinate reference system to WGS84
    gdf.set_crs(epsg=4326, inplace=True)
    
    return gdf

def filter_buildings(geojson_data, plotting_box):
    """
    Filter building features that intersect with a given bounding box.
    
    This function filters a list of GeoJSON building features to keep only those
    that intersect with a specified bounding box. It performs geometry validation
    and handles invalid geometries gracefully.
    
    Args:
        geojson_data (list): List of GeoJSON features representing buildings
            Each feature must have valid 'geometry' property
            Coordinates must be in same CRS as plotting_box
            Invalid geometries will be skipped with warning
        plotting_box (Polygon): Shapely polygon defining the bounding box
            Must be a valid Shapely Polygon object
            Must be in same coordinate system as geojson_data
            Used for spatial intersection testing
        
    Returns:
        list: Filtered list of GeoJSON features that intersect with the bounding box
            Features maintain their original structure
            Invalid features are excluded
            Order of features is preserved
    
    Note:
        - Validates polygon coordinates before processing
        - Skips features with invalid geometries
        - Reports validation and geometry errors
        - No coordinate system transformation is performed
        - Memory efficient as it creates new list only for valid features
    """
    # Initialize list for valid intersecting features
    filtered_features = []
    
    # Process each feature in the input data
    for feature in geojson_data:
        # Validate polygon coordinates before processing
        if not validate_polygon_coordinates(feature['geometry']):
            print("Skipping feature with invalid geometry")
            print(feature['geometry'])
            continue
            
        try:
            # Convert GeoJSON geometry to Shapely geometry for spatial operations
            geom = shape(feature['geometry'])
            
            # Skip invalid geometries that can't be fixed
            if not geom.is_valid:
                print("Skipping invalid geometry")
                print(geom)
                continue
                
            # Keep features that intersect with bounding box
            if plotting_box.intersects(geom):
                filtered_features.append(feature)
                
        except ShapelyError as e:
            # Log geometry errors but continue processing
            print(f"Skipping feature due to geometry error: {e}")
            
    return filtered_features

def extract_building_heights_from_geotiff(geotiff_path, gdf):
    """
    Extract building heights from a GeoTIFF raster for building footprints in a GeoDataFrame.
    
    This function processes building footprints to extract height information from a GeoTIFF
    raster file. It handles coordinate transformation between WGS84 (EPSG:4326) and the raster's
    CRS, and calculates average heights for each building footprint.
    
    Args:
        geotiff_path (str): Path to the GeoTIFF height raster file containing elevation data
        gdf (gpd.GeoDataFrame): GeoDataFrame containing building footprints with geometry column
            The GeoDataFrame should be in WGS84 (EPSG:4326) coordinate system
        
    Returns:
        gpd.GeoDataFrame: Updated GeoDataFrame with extracted heights in the 'height' column
            - Buildings with valid height data will have their height values updated
            - Buildings with no valid height data will have NaN values
            - Original buildings with existing valid heights are preserved
    
    Statistics Reported:
        - Total number of buildings without height data
        - Number of buildings successfully updated with height data
        - Number of buildings where no height data could be found
    
    Note:
        - The function only processes Polygon geometries (not MultiPolygons or other types)
        - Buildings are considered to need height processing if they have no height or height <= 0
        - Heights are calculated as the mean of all valid raster values within the building footprint
    """
    # Make a copy to avoid modifying the input
    gdf = gdf.copy()

    # Initialize counters for statistics
    count_0 = 0  # Buildings without height
    count_1 = 0  # Buildings updated with height
    count_2 = 0  # Buildings with no height data found

    # Open GeoTIFF and process buildings
    with rasterio.open(geotiff_path) as src:
        # Create coordinate transformer from WGS84 to raster CRS for geometry transformation
        transformer = Transformer.from_crs(CRS.from_epsg(4326), src.crs, always_xy=True)

        # Filter buildings that need height processing:
        # - Must be Polygon type (not MultiPolygon)
        # - Either has no height or height <= 0
        mask_condition = (gdf.geometry.geom_type == 'Polygon') & ((gdf.get('height', 0) <= 0) | gdf.get('height').isna())
        buildings_to_process = gdf[mask_condition]
        count_0 = len(buildings_to_process)

        for idx, row in buildings_to_process.iterrows():
            # Transform building polygon coordinates from WGS84 to raster CRS
            coords = list(row.geometry.exterior.coords)
            transformed_coords = [transformer.transform(lon, lat) for lon, lat in coords]
            polygon = shape({"type": "Polygon", "coordinates": [transformed_coords]})
            
            try:
                # Extract height values from raster within the building polygon
                # all_touched=True ensures we get all pixels that the polygon touches
                masked_data, _ = rasterio.mask.mask(src, [polygon], crop=True, all_touched=True)
                
                # Filter out nodata values from the raster
                heights = masked_data[0][masked_data[0] != src.nodata]
                
                # Calculate average height if valid samples exist
                if len(heights) > 0:
                    count_1 += 1
                    gdf.at[idx, 'height'] = float(np.mean(heights))
                else:
                    count_2 += 1
                    gdf.at[idx, 'height'] = np.nan
            except ValueError as e:
                print(f"Error processing building at index {idx}. Error: {str(e)}")
                gdf.at[idx, 'height'] = None

    # Print statistics about height updates
    if count_0 > 0:
        print(f"{count_0} of the total {len(gdf)} building footprint from OSM did not have height data.")
        print(f"For {count_1} of these building footprints without height, values from complementary data were assigned.")
        print(f"For {count_2} of these building footprints without height, no data exist in complementary data.")

    return gdf

def get_gdf_from_gpkg(gpkg_path, rectangle_vertices):
    """
    Read a GeoPackage file and convert it to a GeoDataFrame with consistent CRS.
    
    This function reads a GeoPackage file containing building footprints and ensures
    the data is properly formatted with WGS84 coordinate system and unique identifiers.
    It handles CRS conversion if needed and adds sequential IDs.
    
    Args:
        gpkg_path (str): Path to the GeoPackage file
            File must exist and be readable
            Must contain valid building footprint geometries
            Any coordinate system is accepted
        rectangle_vertices (list): List of (lon, lat) tuples defining the bounding rectangle
            Must be in WGS84 (EPSG:4326) coordinate system
            Used for spatial filtering (not implemented in this function)
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing building footprints
            Will have 'geometry' column with building geometries
            Will have 'id' column with sequential integers
            Will be in WGS84 (EPSG:4326) coordinate system
    
    Note:
        - Prints informative message when opening file
        - Sets CRS to WGS84 if not specified
        - Transforms to WGS84 if different CRS
        - Adds sequential IDs starting from 0
        - rectangle_vertices parameter is currently unused
    """
    # Open and read the GPKG file
    print(f"Opening GPKG file: {gpkg_path}")
    gdf = gpd.read_file(gpkg_path)

    # Only set CRS if not already set
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    # Transform to WGS84 if needed
    elif gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    # Replace id column with sequential index numbers
    gdf['id'] = gdf.index
    
    return gdf

def swap_coordinates(features):
    """
    Swap coordinate ordering in GeoJSON features from (lat, lon) to (lon, lat).
    
    This function modifies GeoJSON features in-place to swap the order of coordinates
    from (latitude, longitude) to (longitude, latitude). It handles both Polygon and
    MultiPolygon geometries, maintaining their structure while swapping coordinates.
    
    Args:
        features (list): List of GeoJSON features to process
            Features must have 'geometry' property
            Supported geometry types: 'Polygon', 'MultiPolygon'
            Coordinates must be in (lat, lon) order initially
    
    Returns:
        None: Features are modified in-place
    
    Note:
        - Modifies features directly (no copy created)
        - Handles both Polygon and MultiPolygon geometries
        - For Polygons: processes single coordinate ring
        - For MultiPolygons: processes multiple coordinate rings
        - Assumes input coordinates are in (lat, lon) order
        - Resulting coordinates will be in (lon, lat) order
    """
    # Process each feature based on geometry type
    for feature in features:
        if feature['geometry']['type'] == 'Polygon':
            # Swap coordinates for simple polygons
            # Each polygon is a list of rings (exterior and optional holes)
            new_coords = [[[lon, lat] for lat, lon in polygon] for polygon in feature['geometry']['coordinates']]
            feature['geometry']['coordinates'] = new_coords
        elif feature['geometry']['type'] == 'MultiPolygon':
            # Swap coordinates for multi-polygons (polygons with holes)
            # Each multipolygon is a list of polygons, each with its own rings
            new_coords = [[[[lon, lat] for lat, lon in polygon] for polygon in multipolygon] for multipolygon in feature['geometry']['coordinates']]
            feature['geometry']['coordinates'] = new_coords

def save_geojson(features, save_path):
    """
    Save GeoJSON features to a file with coordinate swapping and pretty printing.
    
    This function takes a list of GeoJSON features, swaps their coordinate ordering
    if needed, wraps them in a FeatureCollection, and saves to a file with proper
    JSON formatting. It creates a deep copy to avoid modifying the original data.
    
    Args:
        features (list): List of GeoJSON features to save
            Each feature should have valid GeoJSON structure
            Features can be Polygon or MultiPolygon type
            Coordinates will be swapped if in (lat, lon) order
        save_path (str): Path where the GeoJSON file should be saved
            Will overwrite existing file if present
            Directory must exist and be writable
            File will be created with UTF-8 encoding
        
    Returns:
        None
    
    Note:
        - Creates deep copy to preserve original feature data
        - Swaps coordinates from (lat, lon) to (lon, lat) order
        - Wraps features in a FeatureCollection object
        - Uses pretty printing with 2-space indentation
        - Handles both Polygon and MultiPolygon geometries
    """
    # Create deep copy to avoid modifying original data
    geojson_features = copy.deepcopy(features)
    
    # Swap coordinate ordering from (lat, lon) to (lon, lat)
    swap_coordinates(geojson_features)

    # Create FeatureCollection structure
    geojson = {
        "type": "FeatureCollection",
        "features": geojson_features
    }

    # Write to file with pretty printing (2-space indentation)
    with open(save_path, 'w') as f:
        json.dump(geojson, f, indent=2)

def find_building_containing_point(building_gdf, target_point):
    """
    Find building IDs that contain a given point in their footprint.
    
    This function identifies all buildings in a GeoDataFrame whose footprint contains
    a specified geographic point. Only Polygon geometries are considered, and the point
    must be fully contained within the building footprint (not just touching).
    
    Args:
        building_gdf (GeoDataFrame): GeoDataFrame containing building geometries and IDs
            Must have 'geometry' column with Polygon geometries
            Must have 'id' column or index will be used as fallback
            Geometries must be in same CRS as target_point coordinates
        target_point (tuple): Tuple of (lon, lat) coordinates to check
            Must be in same coordinate system as building_gdf geometries
            Order must be (longitude, latitude) if using WGS84
        
    Returns:
        list: List of building IDs containing the target point
            Empty list if no buildings contain the point
            Multiple IDs possible if buildings overlap
            IDs are in arbitrary order
    
    Note:
        - Only processes Polygon geometries (skips MultiPolygons and others)
        - Uses Shapely's contains() method which requires point to be fully inside polygon
        - No spatial indexing is used, performs linear search through all buildings
    """
    # Create Shapely point from input coordinates
    point = Point(target_point[0], target_point[1])
    
    # Initialize list to store matching building IDs
    id_list = []
    
    # Check each building in the GeoDataFrame
    for idx, row in building_gdf.iterrows():
        # Skip any geometry that is not a simple Polygon
        if not isinstance(row.geometry, Polygon):
            continue
            
        # Check if point is fully contained within building footprint
        if row.geometry.contains(point):
            # Use specified ID column or None if not found
            id_list.append(row.get('id', None))
    
    return id_list

def get_buildings_in_drawn_polygon(building_gdf, drawn_polygons, 
                                   operation='within'):
    """
    Find buildings that intersect with or are contained within user-drawn polygons.
    
    This function identifies buildings from a GeoDataFrame that have a specified spatial
    relationship with one or more polygons defined by user-drawn vertices. The relationship can be
    either intersection (building overlaps polygon) or containment (building fully within
    polygon).
    
    Args:
        building_gdf (GeoDataFrame): GeoDataFrame containing building footprints
            Must have 'geometry' column with Polygon geometries
            Must have 'id' column or index will be used as fallback
            Geometries must be in same CRS as drawn_polygons vertices
        drawn_polygons (list): List of dictionaries containing polygon data
            Each dictionary must have:
            - 'id': Unique polygon identifier (int)
            - 'vertices': List of (lon, lat) tuples defining polygon vertices
            - 'color': Color string (optional, for reference)
            Must be in same coordinate system as building_gdf geometries
            Must form valid polygons (3+ vertices, first != last)
            Order must be (longitude, latitude) if using WGS84
        operation (str, optional): Type of spatial relationship to check
            'within': buildings must be fully contained in drawn polygon (default)
            'intersect': buildings must overlap with drawn polygon
            
    Returns:
        list: List of building IDs that satisfy the spatial relationship with any of the drawn polygons
            Empty list if no buildings meet the criteria
            IDs are returned in order of processing
            May contain None values if buildings lack IDs
            Duplicate building IDs are removed (a building matching multiple polygons appears only once)
    
    Note:
        - Only processes Polygon geometries (skips MultiPolygons and others)
        - No spatial indexing is used, performs linear search through all buildings
        - Invalid operation parameter will raise ValueError
        - Does not validate polygon closure (first vertex = last vertex)
        - Buildings matching any of the drawn polygons are included in the result
    """
    if not drawn_polygons:
        return []
    
    # Initialize set to store matching building IDs (using set to avoid duplicates)
    included_building_ids = set()
    
    # Process each polygon
    for polygon_data in drawn_polygons:
        vertices = polygon_data['vertices']
        
        # Create Shapely Polygon from drawn vertices
        drawn_polygon_shapely = Polygon(vertices)
        
        # Check each building in the GeoDataFrame
        for idx, row in building_gdf.iterrows():
            # Skip any geometry that is not a simple Polygon
            if not isinstance(row.geometry, Polygon):
                continue
            
            # Check spatial relationship based on specified operation
            if operation == 'intersect':
                if row.geometry.intersects(drawn_polygon_shapely):
                    included_building_ids.add(row.get('id', None))
            elif operation == 'within':
                if row.geometry.within(drawn_polygon_shapely):
                    included_building_ids.add(row.get('id', None))
            else:
                raise ValueError("operation must be 'intersect' or 'within'")
    
    # Convert set back to list and return
    return list(included_building_ids)

def process_building_footprints_by_overlap(filtered_gdf, overlap_threshold=0.5):
    """
    Process building footprints to merge overlapping buildings based on area overlap ratio.
    
    This function identifies and merges building footprints that significantly overlap with each other.
    Buildings are processed in order of decreasing area, and smaller buildings that overlap significantly
    with larger ones are assigned the ID of the larger building, effectively merging them.
    
    Args:
        filtered_gdf (geopandas.GeoDataFrame): GeoDataFrame containing building footprints
            Must have 'geometry' column with building polygons
            If CRS is set, areas will be calculated in Web Mercator projection
        overlap_threshold (float, optional): Threshold for overlap ratio (0.0-1.0) to merge buildings
            Default is 0.5 (50% overlap)
            Higher values require more overlap for merging
            Lower values will result in more aggressive merging
        
    Returns:
        geopandas.GeoDataFrame: Processed GeoDataFrame with updated IDs
            Overlapping buildings will share the same ID
            Original geometries are preserved, only IDs are updated
            All other columns remain unchanged
    
    Note:
        - Uses R-tree spatial indexing for efficient overlap detection
        - Projects to Web Mercator (EPSG:3857) for accurate area calculation if CRS is set
        - Handles invalid geometries by attempting to fix them with buffer(0)
        - Processes buildings in order of decreasing area (largest first)
    """
    # Make a copy to avoid modifying the original
    gdf = filtered_gdf.copy()
    
    # Ensure 'id' column exists, use index if not present
    if 'id' not in gdf.columns:
        gdf['id'] = gdf.index
    
    # Project to Web Mercator for accurate area calculation if CRS is set
    if gdf.crs is None:
        # Work with original geometries if no CRS is set
        gdf_projected = gdf.copy()
    else:
        # Store original CRS to convert back later
        original_crs = gdf.crs
        # Project to Web Mercator for accurate area calculation
        gdf_projected = gdf.to_crs("EPSG:3857")
    
    # Calculate areas and sort by decreasing area for processing largest buildings first
    gdf_projected['area'] = gdf_projected.geometry.area
    gdf_projected = gdf_projected.sort_values(by='area', ascending=False)
    gdf_projected = gdf_projected.reset_index(drop=True)
    
    # Create spatial index for efficient querying of potential overlaps
    spatial_idx = index.Index()
    for i, geom in enumerate(gdf_projected.geometry):
        if geom.is_valid:
            spatial_idx.insert(i, geom.bounds)
        else:
            # Fix invalid geometries using buffer(0) technique
            fixed_geom = geom.buffer(0)
            if fixed_geom.is_valid:
                spatial_idx.insert(i, fixed_geom.bounds)
    
    # Track ID replacements to avoid repeated processing
    id_mapping = {}
    
    # Process each building (skip the largest one as it's our reference)
    for i in range(1, len(gdf_projected)):
        current_poly = gdf_projected.iloc[i].geometry
        current_area = gdf_projected.iloc[i].area
        current_id = gdf_projected.iloc[i]['id']
        
        # Skip if already mapped to another ID
        if current_id in id_mapping:
            continue
        
        # Ensure geometry is valid for processing
        if not current_poly.is_valid:
            current_poly = current_poly.buffer(0)
            if not current_poly.is_valid:
                continue
        
        # Find potential overlaps with larger polygons using spatial index
        potential_overlaps = [j for j in spatial_idx.intersection(current_poly.bounds) if j < i]
        
        for j in potential_overlaps:
            larger_poly = gdf_projected.iloc[j].geometry
            larger_id = gdf_projected.iloc[j]['id']
            
            # Follow ID mapping chain to get final ID
            if larger_id in id_mapping:
                larger_id = id_mapping[larger_id]
            
            # Ensure geometry is valid for intersection test
            if not larger_poly.is_valid:
                larger_poly = larger_poly.buffer(0)
                if not larger_poly.is_valid:
                    continue
            
            try:
                # Calculate overlap ratio relative to current building's area
                if current_poly.intersects(larger_poly):
                    overlap = current_poly.intersection(larger_poly)
                    overlap_ratio = overlap.area / current_area
                    
                    # Merge buildings if overlap exceeds threshold
                    if overlap_ratio > overlap_threshold:
                        id_mapping[current_id] = larger_id
                        gdf_projected.at[i, 'id'] = larger_id
                        break  # Stop at first significant overlap
            except (GEOSException, ValueError) as e:
                # Skip problematic geometries
                continue
    
    # Propagate ID changes through the original DataFrame
    for i, row in filtered_gdf.iterrows():
        orig_id = row.get('id')
        if orig_id in id_mapping:
            filtered_gdf.at[i, 'id'] = id_mapping[orig_id]
    
    return filtered_gdf

def merge_gdfs_with_id_conflict_resolution(gdf_1, gdf_2, id_columns=['id', 'building_id']):
    """
    Merge two GeoDataFrames while resolving ID conflicts by modifying IDs in the second GeoDataFrame.
    
    This function merges two GeoDataFrames containing building footprints, ensuring that
    when buildings from both datasets have the same ID or building_id, the IDs in the
    second GeoDataFrame are modified to maintain uniqueness across the merged dataset.
    
    Args:
        gdf_1 (gpd.GeoDataFrame): Primary GeoDataFrame containing building footprints
            Must have 'geometry' column with building polygons
            Must have 'id' and 'building_id' columns (or specified id_columns)
            Will remain unchanged during merging
        gdf_2 (gpd.GeoDataFrame): Secondary GeoDataFrame containing building footprints
            Must have 'geometry' column with building polygons
            Must have 'id' and 'building_id' columns (or specified id_columns)
            IDs will be modified if conflicts exist with gdf_1
        id_columns (list, optional): List of column names to check for ID conflicts
            Default is ['id', 'building_id']
            All specified columns must exist in both GeoDataFrames
    
    Returns:
        gpd.GeoDataFrame: Merged GeoDataFrame with resolved ID conflicts
            Contains all buildings from both input GeoDataFrames
            All ID columns are unique across the entire dataset
            Original geometries and other properties are preserved
            Missing columns are filled with None values
    
    Note:
        - Uses the maximum ID values from gdf_1 as the starting point for new IDs in gdf_2
        - Modifies all specified ID columns in gdf_2 to maintain consistency
        - Preserves all other columns and data from both GeoDataFrames
        - Assumes both GeoDataFrames have the same coordinate reference system
        - Handles missing ID columns gracefully by skipping them
        - Sets missing columns to None instead of NaN for better compatibility
    """
    # Make copies to avoid modifying original GeoDataFrames
    gdf_primary = gdf_1.copy()
    gdf_secondary = gdf_2.copy()
    
    # Validate that required ID columns exist in both GeoDataFrames
    missing_columns = []
    for col in id_columns:
        if col not in gdf_primary.columns:
            missing_columns.append(f"'{col}' missing from gdf_1")
        if col not in gdf_secondary.columns:
            missing_columns.append(f"'{col}' missing from gdf_2")
    
    if missing_columns:
        print(f"Warning: Missing ID columns: {', '.join(missing_columns)}")
        # Remove missing columns from the list to process
        id_columns = [col for col in id_columns 
                     if col in gdf_primary.columns and col in gdf_secondary.columns]
    
    if not id_columns:
        print("Warning: No valid ID columns found. Merging without ID conflict resolution.")
        # Handle missing columns before concatenation
        merged_gdf = _merge_gdfs_with_missing_columns(gdf_primary, gdf_secondary)
        return merged_gdf
    
    # Calculate the maximum ID values from the primary GeoDataFrame for each ID column
    max_ids = {}
    for col in id_columns:
        if gdf_primary[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            max_ids[col] = gdf_primary[col].max()
        else:
            # For non-numeric IDs, we'll use the length of the primary DataFrame
            max_ids[col] = len(gdf_primary)
    
    # Create a mapping for new IDs in the secondary GeoDataFrame
    id_mapping = {}
    next_ids = {col: max_ids[col] + 1 for col in id_columns}
    
    # Process each row in the secondary GeoDataFrame
    for idx, row in gdf_secondary.iterrows():
        needs_new_ids = False
        
        # Check if any ID column conflicts with the primary GeoDataFrame
        for col in id_columns:
            current_id = row[col]
            
            # Check if this ID exists in the primary GeoDataFrame
            if current_id in gdf_primary[col].values:
                needs_new_ids = True
                break
        
        # If conflicts found, assign new IDs
        if needs_new_ids:
            for col in id_columns:
                new_id = next_ids[col]
                gdf_secondary.at[idx, col] = new_id
                next_ids[col] += 1
    
    # Handle missing columns before merging
    merged_gdf = _merge_gdfs_with_missing_columns(gdf_primary, gdf_secondary)
    
    # Print statistics about the merge
    total_buildings = len(merged_gdf)
    primary_buildings = len(gdf_primary)
    secondary_buildings = len(gdf_secondary)
    modified_buildings = sum(1 for idx, row in gdf_secondary.iterrows() 
                           if any(row[col] != gdf_2.iloc[idx][col] for col in id_columns))
    
    print(f"Merged {primary_buildings} buildings from primary dataset with {secondary_buildings} buildings from secondary dataset.")
    print(f"Total buildings in merged dataset: {total_buildings}")
    if modified_buildings > 0:
        print(f"Modified IDs for {modified_buildings} buildings in secondary dataset to resolve conflicts.")
    
    return merged_gdf


def _merge_gdfs_with_missing_columns(gdf_1, gdf_2):
    """
    Helper function to merge two GeoDataFrames while handling missing columns.
    
    This function ensures that when one GeoDataFrame has columns that the other doesn't,
    those missing values are filled with None instead of NaN.
    
    Args:
        gdf_1 (gpd.GeoDataFrame): First GeoDataFrame
        gdf_2 (gpd.GeoDataFrame): Second GeoDataFrame
    
    Returns:
        gpd.GeoDataFrame: Merged GeoDataFrame with all columns from both inputs
    """
    # Find columns that exist in one GeoDataFrame but not the other
    columns_1 = set(gdf_1.columns)
    columns_2 = set(gdf_2.columns)
    
    # Columns only in gdf_1
    only_in_1 = columns_1 - columns_2
    # Columns only in gdf_2
    only_in_2 = columns_2 - columns_1
    
    # Add missing columns to gdf_1 with None values
    for col in only_in_2:
        gdf_1[col] = None
    
    # Add missing columns to gdf_2 with None values
    for col in only_in_1:
        gdf_2[col] = None
    
    # Ensure both GeoDataFrames have the same column order
    all_columns = sorted(list(columns_1.union(columns_2)))
    gdf_1 = gdf_1[all_columns]
    gdf_2 = gdf_2[all_columns]
    
    # Merge the GeoDataFrames
    merged_gdf = pd.concat([gdf_1, gdf_2], ignore_index=True)
    
    return merged_gdf