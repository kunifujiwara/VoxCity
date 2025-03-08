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
    
    Args:
        gdf_0 (gpd.GeoDataFrame): Primary GeoDataFrame to update with heights
        gdf_1 (gpd.GeoDataFrame): Reference GeoDataFrame containing height data
        
    Returns:
        gpd.GeoDataFrame: Updated primary GeoDataFrame with extracted heights
    """
    # Make a copy of input GeoDataFrame to avoid modifying original
    gdf_primary = gdf_0.copy()
    gdf_ref = gdf_1.copy()

    # Make sure height columns exist
    if 'height' not in gdf_primary.columns:
        gdf_primary['height'] = 0.0
    if 'height' not in gdf_ref.columns:
        gdf_ref['height'] = 0.0

    # Initialize counters for statistics
    count_0 = 0  # Buildings without height
    count_1 = 0  # Buildings updated with height
    count_2 = 0  # Buildings with no height data found

    # Create spatial index for reference buildings
    from rtree import index
    spatial_index = index.Index()
    for i, geom in enumerate(gdf_ref.geometry):
        if geom.is_valid:
            spatial_index.insert(i, geom.bounds)

    # Process each building in primary dataset that needs height data
    for idx_primary, row in gdf_primary.iterrows():
        if row['height'] == 0:
            count_0 += 1
            geom = row.geometry
            
            # Calculate weighted average height based on overlapping areas
            overlapping_height_area = 0
            overlapping_area = 0
            
            # Get potential intersecting buildings using spatial index
            potential_matches = list(spatial_index.intersection(geom.bounds))
            
            # Check intersections with reference buildings
            for ref_idx in potential_matches:
                if ref_idx >= len(gdf_ref):
                    continue
                    
                ref_row = gdf_ref.iloc[ref_idx]
                try:
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
                # Calculate weighted average height
                new_height = overlapping_height_area / overlapping_area
                gdf_primary.at[idx_primary, 'height'] = new_height
            else:
                count_2 += 1
                gdf_primary.at[idx_primary, 'height'] = np.nan
    
    # Print statistics about height updates
    if count_0 > 0:
        # print(f"{count_0} of the total {len(gdf_primary)} building footprint from OSM did not have height data.")
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

def load_gdf_from_multiple_gz(file_paths):
    """
    Load GeoJSON features from multiple gzipped files into a GeoDataFrame.
    
    Args:
        file_paths (list): List of paths to gzipped GeoJSON files
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing features from all files
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
    
    # Convert list of GeoJSON features to GeoDataFrame
    # swap_coordinates(geojson_objects)
    gdf = gpd.GeoDataFrame.from_features(geojson_objects)
    
    # Set CRS to WGS84 which is typically used for these files
    gdf.set_crs(epsg=4326, inplace=True)
    
    return gdf

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

def extract_building_heights_from_geotiff(geotiff_path, gdf):
    """
    Extract building heights from a GeoTIFF raster for building footprints in a GeoDataFrame.
    
    Args:
        geotiff_path (str): Path to the GeoTIFF height raster
        gdf (gpd.GeoDataFrame): GeoDataFrame containing building footprints
        
    Returns:
        gpd.GeoDataFrame: Updated GeoDataFrame with extracted heights
    """
    # Make a copy to avoid modifying the input
    gdf = gdf.copy()

    # Initialize counters for statistics
    count_0 = 0  # Buildings without height
    count_1 = 0  # Buildings updated with height
    count_2 = 0  # Buildings with no height data found

    # Open GeoTIFF and process buildings
    with rasterio.open(geotiff_path) as src:
        # Create coordinate transformer from WGS84 to raster CRS
        transformer = Transformer.from_crs(CRS.from_epsg(4326), src.crs, always_xy=True)

        # Filter buildings that need height processing
        mask_condition = (gdf.geometry.geom_type == 'Polygon') & (gdf.get('height', 0) <= 0)
        buildings_to_process = gdf[mask_condition]
        count_0 = len(buildings_to_process)

        for idx, row in buildings_to_process.iterrows():
            # Transform geometry to raster CRS
            coords = list(row.geometry.exterior.coords)
            transformed_coords = [transformer.transform(lon, lat) for lon, lat in coords]
            polygon = shape({"type": "Polygon", "coordinates": [transformed_coords]})
            
            try:
                # Extract height values from raster within polygon
                masked_data, _ = rasterio.mask.mask(src, [polygon], crop=True, all_touched=True)
                heights = masked_data[0][masked_data[0] != src.nodata]
                
                # Calculate average height if valid samples exist
                if len(heights) > 0:
                    count_1 += 1
                    gdf.at[idx, 'height'] = float(np.mean(heights))
                else:
                    count_2 += 1
                    gdf.at[idx, 'height'] = np.nan
                    # print(f"No valid height data for building at index {idx}")
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

    # Only set CRS if not already set
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    # Transform to WGS84 if needed
    elif gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    # Replace id column with index numbers
    gdf['id'] = gdf.index
    
    return gdf

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

def find_building_containing_point(building_gdf, target_point):
    """
    Find building IDs that contain a given point.
    
    Args:
        building_gdf (GeoDataFrame): GeoDataFrame containing building geometries and IDs
        target_point (tuple): Tuple of (lon, lat)
        
    Returns:
        list: List of building IDs containing the target point
    """
    # Create Shapely point
    point = Point(target_point[0], target_point[1])
    
    id_list = []
    for idx, row in building_gdf.iterrows():
        # Skip any geometry that is not Polygon
        if not isinstance(row.geometry, Polygon):
            continue
            
        # Check if point is within polygon
        if row.geometry.contains(point):
            id_list.append(row.get('id', None))
    
    return id_list

def get_buildings_in_drawn_polygon(building_gdf, drawn_polygon_vertices, 
                                   operation='within'):
    """
    Given a GeoDataFrame of building footprints and a set of drawn polygon 
    vertices (in lon, lat), return the building IDs that fall within or 
    intersect the drawn polygon.

    Args:
        building_gdf (GeoDataFrame): 
            A GeoDataFrame containing building footprints with:
            - geometry column containing Polygon geometries
            - id column containing building IDs

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

    # Check each building in the GeoDataFrame
    for idx, row in building_gdf.iterrows():
        # Skip any geometry that is not Polygon
        if not isinstance(row.geometry, Polygon):
            continue

        # Depending on the operation, check the relationship
        if operation == 'intersect':
            if row.geometry.intersects(drawn_polygon_shapely):
                included_building_ids.append(row.get('id', None))
        elif operation == 'within':
            if row.geometry.within(drawn_polygon_shapely):
                included_building_ids.append(row.get('id', None))
        else:
            raise ValueError("operation must be 'intersect' or 'within'")

    return included_building_ids

def process_building_footprints_by_overlap(filtered_gdf, overlap_threshold=0.5):
    """
    Process building footprints to merge overlapping buildings.
    
    Args:
        filtered_gdf (geopandas.GeoDataFrame): GeoDataFrame containing building footprints
        overlap_threshold (float): Threshold for overlap ratio (0.0-1.0) to merge buildings
        
    Returns:
        geopandas.GeoDataFrame: Processed GeoDataFrame with updated IDs
    """
    # Make a copy to avoid modifying the original
    gdf = filtered_gdf.copy()
    
    # Ensure 'id' column exists
    if 'id' not in gdf.columns:
        gdf['id'] = gdf.index
    
    # Calculate areas and sort by area (descending)
    gdf['area'] = gdf.geometry.area
    gdf = gdf.sort_values(by='area', ascending=False)
    gdf = gdf.reset_index(drop=True)
    
    # Create spatial index for efficient querying
    spatial_idx = index.Index()
    for i, geom in enumerate(gdf.geometry):
        if geom.is_valid:
            spatial_idx.insert(i, geom.bounds)
        else:
            # Fix invalid geometries
            fixed_geom = geom.buffer(0)
            if fixed_geom.is_valid:
                spatial_idx.insert(i, fixed_geom.bounds)
    
    # Track ID replacements to avoid repeated processing
    id_mapping = {}
    
    # Process each building (skip the largest one)
    for i in range(1, len(gdf)):
        current_poly = gdf.iloc[i].geometry
        current_area = gdf.iloc[i].area
        current_id = gdf.iloc[i]['id']
        
        # Skip if already mapped
        if current_id in id_mapping:
            continue
        
        # Ensure geometry is valid
        if not current_poly.is_valid:
            current_poly = current_poly.buffer(0)
            if not current_poly.is_valid:
                continue
        
        # Find potential overlaps with larger polygons
        potential_overlaps = [j for j in spatial_idx.intersection(current_poly.bounds) if j < i]
        
        for j in potential_overlaps:
            larger_poly = gdf.iloc[j].geometry
            larger_id = gdf.iloc[j]['id']
            
            # Skip if already processed
            if larger_id in id_mapping:
                larger_id = id_mapping[larger_id]
            
            # Ensure geometry is valid
            if not larger_poly.is_valid:
                larger_poly = larger_poly.buffer(0)
                if not larger_poly.is_valid:
                    continue
            
            try:
                # Calculate overlap
                if current_poly.intersects(larger_poly):
                    overlap = current_poly.intersection(larger_poly)
                    overlap_ratio = overlap.area / current_area
                    
                    # Replace ID if overlap exceeds threshold
                    if overlap_ratio > overlap_threshold:
                        id_mapping[current_id] = larger_id
                        gdf.at[i, 'id'] = larger_id
                        break  # Stop at first significant overlap
            except (GEOSException, ValueError) as e:
                # Handle geometry errors gracefully
                continue
    
    # Propagate ID changes through the original DataFrame
    for i, row in filtered_gdf.iterrows():
        orig_id = row.get('id')
        if orig_id in id_mapping:
            filtered_gdf.at[i, 'id'] = id_mapping[orig_id]
    
    return filtered_gdf