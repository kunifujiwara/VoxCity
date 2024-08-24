import os
import math
from math import radians, sin, cos, sqrt, atan2
import numpy as np
from pyproj import Geod, Transformer
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.warp import transform_bounds
from shapely.geometry import Polygon, box, shape
from fiona.crs import from_epsg
import gzip
import json
from rtree import index

def tile_from_lat_lon(lat, lon, level_of_detail):
    sin_lat = math.sin(lat * math.pi / 180)
    x = (lon + 180) / 360
    y = 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)
    map_size = 256 << level_of_detail
    tile_x = int(x * map_size / 256)
    tile_y = int(y * map_size / 256)
    return tile_x, tile_y

def quadkey_to_tile(quadkey):
    tile_x = tile_y = 0
    level_of_detail = len(quadkey)
    for i in range(level_of_detail):
        bit = level_of_detail - i - 1
        mask = 1 << bit
        if quadkey[i] == '1':
            tile_x |= mask
        elif quadkey[i] == '2':
            tile_y |= mask
        elif quadkey[i] == '3':
            tile_x |= mask
            tile_y |= mask
    return tile_x, tile_y, level_of_detail

# def swap_coordinates(features):
#     for feature in features:
#         if feature['geometry']['type'] == 'Polygon':
#             new_coords = []
#             for polygon in feature['geometry']['coordinates']:
#                 new_coords.append([[lat, lon] for lon, lat in polygon])
#             feature['geometry']['coordinates'] = new_coords
#         elif feature['geometry']['type'] == 'MultiPolygon':
#             new_coords = []
#             for multipolygon in feature['geometry']['coordinates']:
#                 new_multipolygon = []
#                 for polygon in multipolygon:
#                     new_multipolygon.append([[lat, lon] for lon, lat in polygon])
#                 new_coords.append(new_multipolygon)
#             feature['geometry']['coordinates'] = new_coords

def swap_coordinates(features):
    for feature in features:
        if feature['geometry']['type'] == 'Polygon':
            new_coords = [[[lat, lon] for lon, lat in polygon] for polygon in feature['geometry']['coordinates']]
            feature['geometry']['coordinates'] = new_coords
        elif feature['geometry']['type'] == 'MultiPolygon':
            new_coords = [[[[lat, lon] for lon, lat in polygon] for polygon in multipolygon] for multipolygon in feature['geometry']['coordinates']]
            feature['geometry']['coordinates'] = new_coords

def initialize_geod():
    return Geod(ellps='WGS84')

def calculate_distance(geod, lon1, lat1, lon2, lat2):
    _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    return dist

def normalize_to_one_meter(vector, distance_in_meters):
    return vector * (1 / distance_in_meters)

def setup_transformer(from_crs, to_crs):
    return Transformer.from_crs(from_crs, to_crs, always_xy=True)

def transform_coords(transformer, lon, lat):
    try:
        x, y = transformer.transform(lon, lat)
        if np.isinf(x) or np.isinf(y):
            print(f"Transformation resulted in inf values for coordinates: {lon}, {lat}")
        return x, y
    except Exception as e:
        print(f"Error transforming coordinates {lon}, {lat}: {e}")
        return None, None

def create_polygon(vertices):
    flipped_vertices = [(lon, lat) for lat, lon in vertices]
    return Polygon(flipped_vertices)

def create_geodataframe(polygon, crs=4326):
    return gpd.GeoDataFrame({'geometry': [polygon]}, crs=from_epsg(crs))

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def get_raster_bbox(raster_path):
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
    return box(bounds.left, bounds.bottom, bounds.right, bounds.top)

def raster_intersects_polygon(raster_path, polygon):
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        if src.crs.to_epsg() != 4326:
            bounds = transform_bounds(src.crs, 'EPSG:4326', *bounds)
        raster_bbox = box(*bounds)
        intersects = raster_bbox.intersects(polygon) or polygon.intersects(raster_bbox)
        # print(f"Raster bounds: {bounds}")
        # print(f"Polygon bounds: {polygon.bounds}")
        # print(f"Intersects: {intersects}")
        return intersects

def save_raster(input_path, output_path):
    import shutil
    shutil.copy(input_path, output_path)
    print(f"Copied original file to: {output_path}")

def merge_geotiffs(geotiff_files, output_dir):
    if not geotiff_files:
        # print("No files intersected with the polygon.")
        return

    src_files_to_mosaic = [rasterio.open(file) for file in geotiff_files if os.path.exists(file)]

    if src_files_to_mosaic:
        try:
            mosaic, out_trans = merge(src_files_to_mosaic)

            out_meta = src_files_to_mosaic[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans
            })

            merged_path = os.path.join(output_dir, "lulc.tif")
            with rasterio.open(merged_path, "w", **out_meta) as dest:
                dest.write(mosaic)

            print(f"Merged output saved to: {merged_path}")
        except Exception as e:
            print(f"Error merging files: {e}")
    else:
        print("No valid files to merge.")

    for src in src_files_to_mosaic:
        src.close()

# Sampling and Classification Functions
def sample_geotiff(geotiff_path, transformed_coords):
    with rasterio.open(geotiff_path) as src:
        sampled_values = np.array(list(src.sample(transformed_coords.reshape(-1, 2))))
    return sampled_values.reshape(transformed_coords.shape[:-1] + (3,))

def get_land_cover_class(rgb, land_cover_classes):
    return land_cover_classes.get(tuple(rgb), 'Unknown')

def find_full_class_name(partial_name, land_cover_classes):
    for full_name in land_cover_classes.values():
        if partial_name.lower() == full_name.lower()[:len(partial_name)]:
            return full_name
    return 'Unknown'

def get_dominant_class(cell_values, land_cover_classes):
    unique, counts = np.unique(cell_values.reshape(-1, 3), axis=0, return_counts=True)
    dominant_rgb = unique[np.argmax(counts)]
    class_name = get_land_cover_class(dominant_rgb, land_cover_classes)
    # if class_name == 'Unknown':
    #     print(f"Unknown RGB value: {dominant_rgb}")
    return class_name

def calculate_dominant_classes(sampled_values, land_cover_classes):
    return np.apply_along_axis(lambda x: get_dominant_class(x, land_cover_classes), axis=2, arr=sampled_values)

def create_grid(dominant_classes, land_cover_classes):
    class_to_index = {cls: idx for idx, cls in enumerate(land_cover_classes.values())}
    return np.array([[class_to_index[find_full_class_name(cls, land_cover_classes)] for cls in row] for row in dominant_classes])

def filter_buildings(geojson_data, plotting_box):
    return [feature for feature in geojson_data if plotting_box.intersects(shape(feature['geometry']))]

def create_building_polygons(filtered_buildings):
    building_polygons = []
    idx = index.Index()
    for i, building in enumerate(filtered_buildings):
        polygon = Polygon(building['geometry']['coordinates'][0])
        height = max(0, building['properties']['height'])
        building_polygons.append((polygon, height))
        idx.insert(i, polygon.bounds)
    return building_polygons, idx

# GeoJSON and Data Loading Functions
def load_geojsons_from_multiple_gz(file_paths):
    geojson_objects = []
    for gz_file_path in file_paths:
        with gzip.open(gz_file_path, 'rt', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    geojson_objects.append(data)
                except json.JSONDecodeError as e:
                    print(f"Skipping line in {gz_file_path} due to JSONDecodeError: {e}")
    return geojson_objects