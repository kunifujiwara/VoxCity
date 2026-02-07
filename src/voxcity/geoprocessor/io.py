"""
I/O helpers for reading/writing vector data (GPKG, gzipped GeoJSON lines).
"""

import gzip
import json

import geopandas as gpd

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

