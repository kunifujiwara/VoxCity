"""
I/O helpers for reading/writing vector data (GPKG, gzipped GeoJSON lines) and
saving FeatureCollections.
"""

import copy
import gzip
import json
from typing import List

import geopandas as gpd

from .conversion import filter_and_convert_gdf_to_geojson


def get_geojson_from_gpkg(gpkg_path, rectangle_vertices):
    """
    Read a GeoPackage file and convert it to GeoJSON features within a bounding rectangle.
    """
    print(f"Opening GPKG file: {gpkg_path}")
    gdf = gpd.read_file(gpkg_path)
    geojson = filter_and_convert_gdf_to_geojson(gdf, rectangle_vertices)
    return geojson


def get_gdf_from_gpkg(gpkg_path, rectangle_vertices):
    """
    Read a GeoPackage file and convert it to a GeoDataFrame with consistent CRS.

    Note: rectangle_vertices is currently unused but kept for signature compatibility.
    """
    print(f"Opening GPKG file: {gpkg_path}")
    gdf = gpd.read_file(gpkg_path)

    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    gdf['id'] = gdf.index
    return gdf


def load_gdf_from_multiple_gz(file_paths):
    """
    Load GeoJSON features from multiple gzipped files into a single GeoDataFrame.
    Each line in each file must be a single GeoJSON Feature.
    """
    geojson_objects = []

    for gz_file_path in file_paths:
        with gzip.open(gz_file_path, 'rt', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line)
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

    gdf = gpd.GeoDataFrame.from_features(geojson_objects)
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf


def swap_coordinates(features):
    """
    Swap coordinate ordering in GeoJSON features from (lat, lon) to (lon, lat).
    Modifies the input features in-place.
    """
    for feature in features:
        if feature['geometry']['type'] == 'Polygon':
            new_coords = [[[lon, lat] for lat, lon in polygon] for polygon in feature['geometry']['coordinates']]
            feature['geometry']['coordinates'] = new_coords
        elif feature['geometry']['type'] == 'MultiPolygon':
            new_coords = [[[[lon, lat] for lat, lon in polygon] for polygon in multipolygon] for multipolygon in feature['geometry']['coordinates']]
            feature['geometry']['coordinates'] = new_coords


def save_geojson(features, save_path):
    """
    Save GeoJSON features to a file with coordinate swapping and pretty printing.
    """
    geojson_features = copy.deepcopy(features)
    swap_coordinates(geojson_features)

    geojson = {
        "type": "FeatureCollection",
        "features": geojson_features
    }

    with open(save_path, 'w') as f:
        json.dump(geojson, f, indent=2)


