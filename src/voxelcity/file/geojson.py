import geopandas as gpd
import json
from shapely.geometry import Polygon
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, shape
from shapely.errors import GEOSException, ShapelyError
import gzip
import json
from typing import List, Dict
from pyproj import Transformer, CRS
import rasterio
from rasterio.mask import mask
import copy

from ..geo.utils import validate_polygon_coordinates

def filter_and_convert_gdf_to_geojson(gdf, rectangle_vertices):

    # Reproject to WGS84 if necessary
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs(epsg=4326)

    # Downcast 'height' to save memory
    gdf['height'] = pd.to_numeric(gdf['height'], downcast='float')

    # Add 'confidence' column
    gdf['confidence'] = -1.0

    # Define rectangle polygon
    # rectangle_vertices = [
    #     (56.168518, 14.85961),
    #     (56.172627, 14.85961),
    #     (56.172627, 14.866734),
    #     (56.168518, 14.866734)
    # ]
    rectangle_vertices_lonlat = [(lon, lat) for lat, lon in rectangle_vertices]
    rectangle_polygon = Polygon(rectangle_vertices_lonlat)

    # Use spatial index to filter geometries
    gdf.sindex  # Ensure spatial index is built
    possible_matches_index = list(gdf.sindex.intersection(rectangle_polygon.bounds))
    possible_matches = gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(rectangle_polygon)]
    filtered_gdf = precise_matches.copy()

    # Delete intermediate data to save memory
    del gdf, possible_matches, precise_matches

    # Function to swap coordinates
    def swap_coordinates(coords):
        if isinstance(coords[0][0], (float, int)):
            return [[lat, lon] for lon, lat in coords]
        else:
            return [swap_coordinates(ring) for ring in coords]

    # Create GeoJSON features
    features = []
    for idx, row in filtered_gdf.iterrows():
        geom = row['geometry'].__geo_interface__
        properties = {
            'height': row['height'],
            'confidence': row['confidence']
        }

        if geom['type'] == 'MultiPolygon':
            for polygon_coords in geom['coordinates']:
                single_geom = {
                    'type': 'Polygon',
                    'coordinates': swap_coordinates(polygon_coords)
                }
                feature = {
                    'type': 'Feature',
                    'properties': properties,
                    'geometry': single_geom
                }
                features.append(feature)
        elif geom['type'] == 'Polygon':
            geom['coordinates'] = swap_coordinates(geom['coordinates'])
            feature = {
                'type': 'Feature',
                'properties': properties,
                'geometry': geom
            }
            features.append(feature)
        else:
            pass  # Handle other geometry types if necessary

    # Create a FeatureCollection
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    # # Write the GeoJSON data to a file
    # with open('output.geojson', 'w') as f:
    #     json.dump(geojson, f)

    # Clean up
    del filtered_gdf, features

    # print("Script execution completed.")
    return geojson["features"]

def get_geojson_from_gpkg(gpkg_path, rectangle_vertices):
    # Open and read the GPKG file
    print(f"Opening GPKG file: {gpkg_path}")
    gdf = gpd.read_file(gpkg_path)
    geojson = filter_and_convert_gdf_to_geojson(gdf, rectangle_vertices)
    return geojson

def extract_building_heights_from_geojson(geojson_data_0: List[Dict], geojson_data_1: List[Dict]) -> List[Dict]:
    # Convert geojson_data_1 to Shapely polygons with height information
    reference_buildings = []
    for feature in geojson_data_1:
        geom = shape(feature['geometry'])
        height = feature['properties']['height']
        reference_buildings.append((geom, height))

    count_0 = 0
    count_1 = 0
    count_2 = 0
    # Process geojson_data_0 and update heights where necessary
    updated_geojson_data_0 = []
    for feature in geojson_data_0:
        geom = shape(feature['geometry'])
        height = feature['properties']['height']
        if height == 0:     
            count_0 += 1       
            # Find overlapping buildings in geojson_data_1
            overlapping_height_area = 0
            overlapping_area = 0
            for ref_geom, ref_height in reference_buildings:
                try:
                    if geom.intersects(ref_geom):
                        overlap_area = geom.intersection(ref_geom).area
                        # if overlap_area / geom.area > 0.3:  # More than 50% overlap
                            # overlapping_heights.append(ref_height)
                        overlapping_height_area += ref_height * overlap_area
                        overlapping_area += overlap_area
                except GEOSException as e:
                    print(f"GEOS error at a building polygon {ref_geom}")
                    # Attempt to fix the polygon
                    try:
                        fixed_ref_geom = ref_geom.buffer(0)
                        if geom.intersects(fixed_ref_geom):
                            overlap_area = geom.intersection(ref_geom).area
                            # if overlap_area / geom.area > 0.3:  # More than 50% overlap
                            #     overlapping_heights.append(ref_height)
                            #     break
                            overlapping_height_area += ref_height * overlap_area
                            overlapping_area += overlap_area
                    except Exception as fix_error:
                        print(f"Failed to fix polygon")
                    continue
            
            # Update height if overlapping buildings found
            if overlapping_height_area > 0:
                count_1 += 1
                # new_height = max(overlapping_heights)
                new_height = overlapping_height_area / overlapping_area
                feature['properties']['height'] = new_height
            else:
                count_2 += 1
                feature['properties']['height'] = np.nan
        
        updated_geojson_data_0.append(feature)
    
    if count_0 > 0:
        print(f"{count_0} of the total {len(geojson_data_0)} building footprint from OSM did not have height data.")
        print(f"For {count_1} of these building footprints without height, values from Microsoft Building Footprints were assigned.")
        # print(f"For {count_2} of these building footprints without height, no data exist in Microsoft Building Footprints. Height values of 10m were set instead")

    return updated_geojson_data_0

def load_geojsons_from_multiple_gz(file_paths):
    geojson_objects = []
    for gz_file_path in file_paths:
        with gzip.open(gz_file_path, 'rt', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    # Check and set default height if necessary
                    if 'properties' in data and 'height' in data['properties']:
                        if data['properties']['height'] is None:
                            # print("No building height data was found. A height of 10 meters was set instead.")
                            data['properties']['height'] = 0
                    else:
                        # If 'height' property doesn't exist, add it with default value
                        if 'properties' not in data:
                            data['properties'] = {}
                        # print("No building height data was found. A height of 10 meters was set instead.")
                        data['properties']['height'] = 0
                    geojson_objects.append(data)
                except json.JSONDecodeError as e:
                    print(f"Skipping line in {gz_file_path} due to JSONDecodeError: {e}")
    return geojson_objects

def filter_buildings(geojson_data, plotting_box):
    filtered_features = []
    for feature in geojson_data:
        if not validate_polygon_coordinates(feature['geometry']):
            print("Skipping feature with invalid geometry")
            print(feature['geometry'])
            continue
        try:
            geom = shape(feature['geometry'])
            if not geom.is_valid:
                print("Skipping invalid geometry")
                print(geom)
                continue
            if plotting_box.intersects(geom):
                filtered_features.append(feature)
        except ShapelyError as e:
            print(f"Skipping feature due to geometry error: {e}")
    return filtered_features

def extract_building_heights_from_geotiff(geotiff_path, geojson_data):
    # Check if geojson_data is a string, if so, parse it
    if isinstance(geojson_data, str):
        geojson = json.loads(geojson_data)
        input_was_string = True
    else:
        geojson = geojson_data
        input_was_string = False

    count_0 = 0
    count_1 = 0
    count_2 = 0

    # Open the GeoTIFF file and keep it open for the entire process
    with rasterio.open(geotiff_path) as src:
        # print("Raster CRS:", src.crs)
        # print("Raster Bounds:", src.bounds)
        # print("Raster Affine Transform:", src.transform)

        # Create a transformer for coordinate conversion
        transformer = Transformer.from_crs(CRS.from_epsg(4326), src.crs, always_xy=True)

        # Process each feature in the GeoJSON
        for feature in geojson:
            if (feature['geometry']['type'] == 'Polygon') & (feature['properties']['height']<=0):
                count_0 += 1
                # Transform coordinates from (lat, lon) to the raster's CRS
                coords = feature['geometry']['coordinates'][0]
                transformed_coords = [transformer.transform(lon, lat) for lat, lon in coords]
                
                # Create a shapely polygon from the transformed coordinates
                polygon = shape({"type": "Polygon", "coordinates": [transformed_coords]})
                
                try:
                    # Mask the raster data with the polygon
                    masked, mask_transform = mask(src, [polygon], crop=True, all_touched=True)
                    
                    # Extract valid height values
                    heights = masked[0][masked[0] != src.nodata]
                    
                    # Calculate average height if we have valid samples
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

    if count_0 > 0:
        print(f"{count_0} of the total {len(geojson_data)} building footprint from OSM did not have height data.")
        print(f"For {count_1} of these building footprints without height, values from Open Building 2.5D Temporal were assigned.")
        print(f"For {count_2} of these building footprints without height, no data exist in Open Building 2.5D Temporal. Height values of 10m were set instead")

    # Return the result in the same format as the input
    if input_was_string:
        return json.dumps(geojson, indent=2)
    else:
        return geojson

def get_geojson_from_gpkg(gpkg_path, rectangle_vertices):
    # Open and read the GPKG file
    print(f"Opening GPKG file: {gpkg_path}")
    gdf = gpd.read_file(gpkg_path)
    geojson = filter_and_convert_gdf_to_geojson(gdf, rectangle_vertices)
    return geojson

def swap_coordinates(features):
    for feature in features:
        if feature['geometry']['type'] == 'Polygon':
            new_coords = [[[lat, lon] for lon, lat in polygon] for polygon in feature['geometry']['coordinates']]
            feature['geometry']['coordinates'] = new_coords
        elif feature['geometry']['type'] == 'MultiPolygon':
            new_coords = [[[[lat, lon] for lon, lat in polygon] for polygon in multipolygon] for multipolygon in feature['geometry']['coordinates']]
            feature['geometry']['coordinates'] = new_coords

def save_geojson(features, save_path):
    """
    Save a GeoJSON structure with swapped coordinates.
    """
    geojson_features = copy.deepcopy(features)
    swap_coordinates(geojson_features)
    # print(geojson_features[0])

    geojson = {
        "type": "FeatureCollection",
        "features": geojson_features
    }

    # Write to file
    with open(save_path, 'w') as f:
        json.dump(geojson, f, indent=2)