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

from ..geo.utils import validate_polygon_coordinates

def filter_and_convert_gdf_to_geojson(gdf, rectangle_vertices):
    """
    Filter a GeoDataFrame by a bounding rectangle and convert to GeoJSON format.
    
    Args:
        gdf (GeoDataFrame): Input GeoDataFrame containing building data
        rectangle_vertices (list): List of (lon, lat) tuples defining the bounding rectangle
        
    Returns:
        list: List of GeoJSON features within the bounding rectangle
    """
    # Reproject to WGS84 if necessary
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs(epsg=4326)

    # Downcast 'height' to save memory
    gdf['height'] = pd.to_numeric(gdf['height'], downcast='float')

    # Add 'confidence' column with default value
    gdf['confidence'] = -1.0

    # Rectangle vertices already in (lon,lat) format for shapely
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
    for idx, row in filtered_gdf.iterrows():
        geom = row['geometry'].__geo_interface__
        properties = {
            'height': row['height'],
            'confidence': row['confidence']
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
                    'properties': properties,
                    'geometry': single_geom
                }
                features.append(feature)
        elif geom['type'] == 'Polygon':
            feature = {
                'type': 'Feature',
                'properties': properties,
                'geometry': geom
            }
            features.append(feature)
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

def extract_building_heights_from_geojson(geojson_data_0: List[Dict], geojson_data_1: List[Dict]) -> List[Dict]:
    """
    Extract building heights from one GeoJSON dataset and apply them to another based on spatial overlap.
    
    Args:
        geojson_data_0 (List[Dict]): Primary GeoJSON features to update with heights
        geojson_data_1 (List[Dict]): Reference GeoJSON features containing height data
        
    Returns:
        List[Dict]: Updated primary GeoJSON features with extracted heights
    """
    # Convert reference dataset to Shapely polygons with height info
    reference_buildings = []
    for feature in geojson_data_1:
        geom = shape(feature['geometry'])
        height = feature['properties']['height']
        reference_buildings.append((geom, height))

    # Initialize counters for statistics
    count_0 = 0  # Buildings without height
    count_1 = 0  # Buildings updated with height
    count_2 = 0  # Buildings with no height data found

    # Process primary dataset and update heights where needed
    updated_geojson_data_0 = []
    for feature in geojson_data_0:
        geom = shape(feature['geometry'])
        height = feature['properties']['height']
        if height == 0:     
            count_0 += 1       
            # Calculate weighted average height based on overlapping areas
            overlapping_height_area = 0
            overlapping_area = 0
            for ref_geom, ref_height in reference_buildings:
                try:
                    if geom.intersects(ref_geom):
                        overlap_area = geom.intersection(ref_geom).area
                        overlapping_height_area += ref_height * overlap_area
                        overlapping_area += overlap_area
                except GEOSException as e:
                    # Try to fix invalid geometries using buffer(0)
                    print(f"GEOS error at a building polygon {ref_geom}")
                    try:
                        fixed_ref_geom = ref_geom.buffer(0)
                        if geom.intersects(fixed_ref_geom):
                            overlap_area = geom.intersection(ref_geom).area
                            overlapping_height_area += ref_height * overlap_area
                            overlapping_area += overlap_area
                    except Exception as fix_error:
                        print(f"Failed to fix polygon")
                    continue
            
            # Update height if overlapping buildings found
            if overlapping_height_area > 0:
                count_1 += 1
                # Calculate weighted average height
                new_height = overlapping_height_area / overlapping_area
                feature['properties']['height'] = new_height
            else:
                count_2 += 1
                feature['properties']['height'] = np.nan
        
        updated_geojson_data_0.append(feature)
    
    # Print statistics about height updates
    if count_0 > 0:
        print(f"{count_0} of the total {len(geojson_data_0)} building footprint from OSM did not have height data.")
        print(f"For {count_1} of these building footprints without height, values from Microsoft Building Footprints were assigned.")

    return updated_geojson_data_0

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
    
    Args:
        geojson_data (List[Dict]): A list of feature dicts (Fiona-like).
        id_col (str): Name of property to use as an identifier. If not found,
                      we'll try to create a unique ID.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with geometry and property columns.
    """
    # Build lists for geometry and properties
    geometries = []
    all_props = []

    for i, feature in enumerate(geojson_data):
        # Extract geometry
        geom = feature.get('geometry')
        shapely_geom = shape(geom) if geom else None

        # Extract properties
        props = feature.get('properties', {})
        
        # If an ID column is missing, create one
        if id_col not in props:
            props[id_col] = i  # fallback ID

        # Capture geometry and all props
        geometries.append(shapely_geom)
        all_props.append(props)

    gdf = gpd.GeoDataFrame(all_props, geometry=geometries, crs="EPSG:4326")
    return gdf


def complement_building_heights_gdf(geojson_data_0, geojson_data_1,
                                    primary_id='id', ref_id='id'):
    """
    Use a vectorized approach with GeoPandas to:
      1) Convert both datasets to GeoDataFrames
      2) Find intersections and compute weighted average heights
      3) Update heights in the primary dataset
      4) Add non-intersecting buildings from the reference dataset
    
    Args:
        geojson_data_0 (List[Dict]): Primary GeoJSON-like features
        geojson_data_1 (List[Dict]): Reference GeoJSON-like features
        primary_id (str): Name of the unique identifier in primary dataset's properties
        ref_id (str): Name of the unique identifier in reference dataset's properties

    Returns:
        gpd.GeoDataFrame: Updated GeoDataFrame (including new buildings).
                         You can convert it back to a list of dict features if needed.
    """
    # ----------------------------------------------------------------
    # 1) Convert primary and reference data to GeoDataFrames
    # ----------------------------------------------------------------
    gdf_primary = geojson_to_gdf(geojson_data_0, id_col=primary_id)
    gdf_ref = geojson_to_gdf(geojson_data_1, id_col=ref_id)

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
    # 2) Intersection to compute areas for overlapping buildings
    # ----------------------------------------------------------------
    # We'll rename columns to avoid collision after overlay
    gdf_primary = gdf_primary.rename(columns={'height': 'height_primary'})
    gdf_ref = gdf_ref.rename(columns={'height': 'height_ref'})

    # We perform an 'intersection' overlay to get the overlapping polygons
    intersect_gdf = gpd.overlay(gdf_primary, gdf_ref, how='intersection')

    # Compute intersection area
    intersect_gdf['intersect_area'] = intersect_gdf.area
    # Weighted area (height_ref * intersect_area)
    intersect_gdf['height_area'] = intersect_gdf['height_ref'] * intersect_gdf['intersect_area']

    # ----------------------------------------------------------------
    # 3) Aggregate to get weighted average height for each primary building
    # ----------------------------------------------------------------
    # We group by the primary building ID, summing up the area and the 'height_area'
    group_cols = {
        'height_area': 'sum',
        'intersect_area': 'sum'
    }
    grouped = intersect_gdf.groupby(gdf_primary[primary_id].name).agg(group_cols)

    # Weighted average
    grouped['weighted_height'] = grouped['height_area'] / grouped['intersect_area']

    # ----------------------------------------------------------------
    # 4) Merge aggregated results back to the primary GDF
    # ----------------------------------------------------------------
    # After merging, the primary GDF will have a column 'weighted_height'
    gdf_primary = gdf_primary.merge(grouped['weighted_height'],
                                    left_on=primary_id,
                                    right_index=True,
                                    how='left')

    # Where primary had zero or missing height, we assign the new weighted height
    zero_or_nan_mask = (gdf_primary['height_primary'] == 0) | (gdf_primary['height_primary'].isna())
    gdf_primary.loc[zero_or_nan_mask, 'height_primary'] = gdf_primary.loc[zero_or_nan_mask, 'weighted_height']

    # For any building that had no overlap, 'weighted_height' might be NaN.
    # Keep it as NaN or set to 0 if you prefer:
    gdf_primary['height_primary'] = gdf_primary['height_primary'].fillna(np.nan)

    # ----------------------------------------------------------------
    # 5) Identify reference buildings that do not intersect any primary building
    # ----------------------------------------------------------------
    # Another overlay or spatial join can do this:
    # Option A: use 'difference' on reference to get non-overlapping parts, but that can chop polygons.
    # Option B: check building-level intersection. We'll do a bounding test with sjoin.
    
    # For building-level intersection, do a left join of ref onto primary.
    # Then we'll identify which reference IDs are missing from the intersection result.
    sjoin_gdf = gpd.sjoin(gdf_ref, gdf_primary, how='left', op='intersects')

    # All reference buildings that did not intersect any primary building
    non_intersect_ids = sjoin_gdf.loc[sjoin_gdf[primary_id].isna(), ref_id].unique()

    # Extract them from the original reference GDF
    gdf_ref_non_intersect = gdf_ref[gdf_ref[ref_id].isin(non_intersect_ids)]

    # We'll rename columns back to 'height' to be consistent
    gdf_ref_non_intersect = gdf_ref_non_intersect.rename(columns={'height_ref': 'height'})

    # Also rename any other properties you prefer. For clarity, keep an ID so you know they came from reference.

    # ----------------------------------------------------------------
    # 6) Combine the updated primary GDF with the new reference buildings
    # ----------------------------------------------------------------
    # First, rename columns in updated primary GDF
    gdf_primary = gdf_primary.rename(columns={'height_primary': 'height'})
    # Drop the 'weighted_height' column to clean up
    if 'weighted_height' in gdf_primary.columns:
        gdf_primary.drop(columns='weighted_height', inplace=True)

    # Concatenate
    final_gdf = pd.concat([gdf_primary, gdf_ref_non_intersect], ignore_index=True)

    # ----------------------------------------------------------------
    # Return the combined GeoDataFrame
    # (You can convert it back to a list of GeoJSON-like dictionaries)
    # ----------------------------------------------------------------
    return final_gdf


def gdf_to_geojson_dicts(gdf, id_col='id'):
    """
    Convert a GeoDataFrame to a list of dicts similar to GeoJSON features.
    """
    records = gdf.to_dict(orient='records')
    features = []
    for rec in records:
        # geometry is separate
        geom = rec.pop('geometry', None)
        if geom is not None:
            geom = geom.__geo_interface__
        # use or set ID
        feature_id = rec.get(id_col, None)
        props = {k: v for k, v in rec.items() if k != id_col}
        # build GeoJSON-like feature dict
        feature = {
            'type': 'Feature',
            'properties': props,
            'geometry': geom
        }
        features.append(feature)

    return features


def complement_building_heights_from_geojson(geojson_data_0, geojson_data_1,
                                             primary_id='id', ref_id='id'):
    """
    High-level function that wraps the GeoPandas approach end-to-end.
    Returns a list of GeoJSON-like feature dicts.
    """
    # 1) Complement building heights using the GeoDataFrame approach
    final_gdf = complement_building_heights_gdf(
        geojson_data_0,
        geojson_data_1,
        primary_id=primary_id,
        ref_id=ref_id
    )

    # 2) Convert back to geojson-like dict format
    updated_features = gdf_to_geojson_dicts(final_gdf, id_col=primary_id)
    return updated_features

def load_geojsons_from_multiple_gz(file_paths):
    """
    Load GeoJSON features from multiple gzipped files.
    
    Args:
        file_paths (list): List of paths to gzipped GeoJSON files
        
    Returns:
        list: Combined list of GeoJSON features from all files
    """
    geojson_objects = []
    for gz_file_path in file_paths:
        # Read each gzipped file line by line
        with gzip.open(gz_file_path, 'rt', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    # Ensure height property exists and has valid value
                    if 'properties' in data and 'height' in data['properties']:
                        if data['properties']['height'] is None:
                            data['properties']['height'] = 0
                    else:
                        if 'properties' not in data:
                            data['properties'] = {}
                        data['properties']['height'] = 0
                    geojson_objects.append(data)
                except json.JSONDecodeError as e:
                    print(f"Skipping line in {gz_file_path} due to JSONDecodeError: {e}")
    return geojson_objects

def filter_buildings(geojson_data, plotting_box):
    """
    Filter building features that intersect with a given bounding box.
    
    Args:
        geojson_data (list): List of GeoJSON features
        plotting_box (Polygon): Shapely polygon defining the bounding box
        
    Returns:
        list: Filtered list of GeoJSON features that intersect with the bounding box
    """
    filtered_features = []
    for feature in geojson_data:
        # Validate polygon coordinates before processing
        if not validate_polygon_coordinates(feature['geometry']):
            print("Skipping feature with invalid geometry")
            print(feature['geometry'])
            continue
        try:
            # Convert GeoJSON geometry to Shapely geometry
            geom = shape(feature['geometry'])
            if not geom.is_valid:
                print("Skipping invalid geometry")
                print(geom)
                continue
            # Keep features that intersect with bounding box
            if plotting_box.intersects(geom):
                filtered_features.append(feature)
        except ShapelyError as e:
            print(f"Skipping feature due to geometry error: {e}")
    return filtered_features

def extract_building_heights_from_geotiff(geotiff_path, geojson_data):
    """
    Extract building heights from a GeoTIFF raster for GeoJSON building footprints.
    
    Args:
        geotiff_path (str): Path to the GeoTIFF height raster
        geojson_data (Union[str, list]): GeoJSON features or JSON string of features
        
    Returns:
        Union[str, list]: Updated GeoJSON features with extracted heights in same format as input
    """
    # Handle input format - convert string to object if needed
    if isinstance(geojson_data, str):
        geojson = json.loads(geojson_data)
        input_was_string = True
    else:
        geojson = geojson_data
        input_was_string = False

    # Initialize counters for statistics
    count_0 = 0  # Buildings without height
    count_1 = 0  # Buildings updated with height
    count_2 = 0  # Buildings with no height data found

    # Open GeoTIFF and process buildings
    with rasterio.open(geotiff_path) as src:
        # Create coordinate transformer from WGS84 to raster CRS
        transformer = Transformer.from_crs(CRS.from_epsg(4326), src.crs, always_xy=True)

        # Process each building feature
        for feature in geojson:
            if (feature['geometry']['type'] == 'Polygon') & (feature['properties']['height']<=0):
                count_0 += 1
                # Transform coordinates from (lon, lat) to raster CRS
                coords = feature['geometry']['coordinates'][0]
                transformed_coords = [transformer.transform(lon, lat) for lon, lat in coords]
                
                # Create polygon in raster CRS
                polygon = shape({"type": "Polygon", "coordinates": [transformed_coords]})
                
                try:
                    # Extract height values from raster within polygon
                    masked, mask_transform = mask(src, [polygon], crop=True, all_touched=True)
                    heights = masked[0][masked[0] != src.nodata]
                    
                    # Calculate average height if valid samples exist
                    if len(heights) > 0:
                        count_1 += 1
                        avg_height = np.mean(heights)
                        feature['properties']['height'] = float(avg_height)
                    else:
                        count_2 += 1
                        feature['properties']['height'] = 10
                        print(f"No valid height data for feature: {feature['properties']}")
                except ValueError as e:
                    print(f"Error processing feature: {feature['properties']}. Error: {str(e)}")
                    feature['properties']['extracted_height'] = None

    # Print statistics about height updates
    if count_0 > 0:
        print(f"{count_0} of the total {len(geojson_data)} building footprint from OSM did not have height data.")
        print(f"For {count_1} of these building footprints without height, values from Open Building 2.5D Temporal were assigned.")
        print(f"For {count_2} of these building footprints without height, no data exist in Open Building 2.5D Temporal. Height values of 10m were set instead")

    # Return result in same format as input
    if input_was_string:
        return json.dumps(geojson, indent=2)
    else:
        return geojson

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

def swap_coordinates(features):
    """
    Swap coordinate ordering in GeoJSON features from (lat, lon) to (lon, lat).
    
    Args:
        features (list): List of GeoJSON features to process
    """
    # Process each feature based on geometry type
    for feature in features:
        if feature['geometry']['type'] == 'Polygon':
            # Swap coordinates for simple polygons
            new_coords = [[[lon, lat] for lat, lon in polygon] for polygon in feature['geometry']['coordinates']]
            feature['geometry']['coordinates'] = new_coords
        elif feature['geometry']['type'] == 'MultiPolygon':
            # Swap coordinates for multi-polygons (polygons with holes)
            new_coords = [[[[lon, lat] for lat, lon in polygon] for polygon in multipolygon] for multipolygon in feature['geometry']['coordinates']]
            feature['geometry']['coordinates'] = new_coords

def save_geojson(features, save_path):
    """
    Save GeoJSON features to a file with swapped coordinates.
    
    Args:
        features (list): List of GeoJSON features to save
        save_path (str): Path where the GeoJSON file should be saved
    """
    # Create deep copy to avoid modifying original data
    geojson_features = copy.deepcopy(features)
    
    # Swap coordinate ordering
    swap_coordinates(geojson_features)

    # Create FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": geojson_features
    }

    # Write to file with pretty printing
    with open(save_path, 'w') as f:
        json.dump(geojson, f, indent=2)

def find_building_containing_point(features, target_point):
    """
    Find building IDs that contain a given point.
    
    Args:
        features (list): List of GeoJSON feature dictionaries
        target_point (tuple): Tuple of (lon, lat)
        
    Returns:
        list: List of building IDs containing the target point
    """
    # Create Shapely point
    point = Point(target_point[0], target_point[1])
    
    id_list = []
    for feature in features:
        # Get polygon coordinates and convert to Shapely polygon
        coords = feature['geometry']['coordinates'][0]
        polygon = Polygon(coords)
        
        # Check if point is within polygon
        if polygon.contains(point):
            id_list.append(feature['properties']['id'])
    
    return id_list

def get_buildings_in_drawn_polygon(building_geojson, drawn_polygon_vertices, 
                                   operation='within'):
    """
    Given a list of building footprints and a set of drawn polygon 
    vertices (in lon, lat), return the building IDs that fall within or 
    intersect the drawn polygon.

    Args:
        building_geojson (list): 
            A list of GeoJSON features, each feature is a dict with:
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [lon1, lat1], [lon2, lat2], ...
                        ]
                    ]
                },
                "properties": {
                    "id": ...
                    ...
                }
            }

        drawn_polygon_vertices (list): 
            A list of (lon, lat) tuples representing the polygon drawn by the user.

        operation (str):
            Determines how to include buildings. 
            Use "intersect" to include buildings that intersect the drawn polygon. 
            Use "within" to include buildings that lie entirely within the drawn polygon.

    Returns:
        list:
            A list of building IDs (strings or ints) that satisfy the condition.
    """
    # Create Shapely Polygon from drawn vertices
    drawn_polygon_shapely = Polygon(drawn_polygon_vertices)

    included_building_ids = []

    # Check each building in the GeoJSON
    for feature in building_geojson:
        # Skip any feature that is not Polygon
        if feature['geometry']['type'] != 'Polygon':
            continue

        # Extract coordinates
        coords = feature['geometry']['coordinates'][0]

        # Create a Shapely polygon for the building
        building_polygon = Polygon(coords)

        # Depending on the operation, check the relationship
        if operation == 'intersect':
            if building_polygon.intersects(drawn_polygon_shapely):
                included_building_ids.append(feature['properties'].get('id', None))
        elif operation == 'within':
            if building_polygon.within(drawn_polygon_shapely):
                included_building_ids.append(feature['properties'].get('id', None))
        else:
            raise ValueError("operation must be 'intersect' or 'within'")

    return included_building_ids