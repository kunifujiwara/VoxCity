"""
Conversion utilities between GeoJSON-like features and GeoPandas GeoDataFrames,
plus helpers to filter and transform geometries for export.
"""

import json
from typing import List, Dict

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, shape


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
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs(epsg=4326)

    gdf['height'] = pd.to_numeric(gdf['height'], downcast='float')
    gdf['confidence'] = -1.0

    rectangle_polygon = Polygon(rectangle_vertices)

    gdf.sindex
    possible_matches_index = list(gdf.sindex.intersection(rectangle_polygon.bounds))
    possible_matches = gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(rectangle_polygon)]
    filtered_gdf = precise_matches.copy()

    del gdf, possible_matches, precise_matches

    features = []
    feature_id = 1
    for _, row in filtered_gdf.iterrows():
        geom = row['geometry'].__geo_interface__
        properties = {
            'height': row['height'],
            'confidence': row['confidence'],
            'id': feature_id
        }

        if geom['type'] == 'MultiPolygon':
            for polygon_coords in geom['coordinates']:
                single_geom = {
                    'type': 'Polygon',
                    'coordinates': polygon_coords
                }
                feature = {
                    'type': 'Feature',
                    'properties': properties.copy(),
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
            pass

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    del filtered_gdf, features

    return geojson["features"]


def geojson_to_gdf(geojson_data, id_col='id'):
    """
    Convert a list of GeoJSON-like dict features into a GeoDataFrame.

    This function takes a list of GeoJSON feature dictionaries (Fiona-like format)
    and converts them into a GeoDataFrame, handling geometry conversion and property
    extraction. It ensures each feature has a unique identifier.
    """
    geometries = []
    all_props = []

    for i, feature in enumerate(geojson_data):
        geom = feature.get('geometry')
        shapely_geom = shape(geom) if geom else None

        props = feature.get('properties', {})
        if id_col not in props:
            props[id_col] = i

        geometries.append(shapely_geom)
        all_props.append(props)

    gdf = gpd.GeoDataFrame(all_props, geometry=geometries, crs="EPSG:4326")
    return gdf


def gdf_to_geojson_dicts(gdf, id_col='id'):
    """
    Convert a GeoDataFrame to a list of dicts similar to GeoJSON features.
    """
    records = gdf.to_dict(orient='records')
    features = []

    for rec in records:
        geom = rec.pop('geometry', None)
        if geom is not None:
            geom = geom.__geo_interface__

        _ = rec.get(id_col, None)
        props = {k: v for k, v in rec.items() if k != id_col}

        feature = {
            'type': 'Feature',
            'properties': props,
            'geometry': geom
        }
        features.append(feature)

    return features


