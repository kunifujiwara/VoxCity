"""
Module for downloading and processing building data from OpenMapTiles vector tiles.

This module provides functionality to download and process building footprint data from
OpenMapTiles vector tile service. It handles downloading PBF tiles, extracting building
geometries, and converting them to GeoJSON format with standardized properties.
"""

import mercantile
import requests
import mapbox_vector_tile
from shapely.geometry import shape, mapping
from shapely.affinity import affine_transform
import shapely.ops
import json
from pyproj import Transformer
import json

def load_geojsons_from_openmaptiles(rectangle_vertices, API_KEY):
    """Download and process building footprint data from OpenMapTiles vector tiles.

    Args:
        rectangle_vertices: List of (lat, lon) coordinates defining the bounding box
        API_KEY: OpenMapTiles API key for authentication

    Returns:
        list: List of GeoJSON features containing building footprints with standardized properties
    """
    # Extract latitudes and longitudes from vertices to find bounding box
    lats = [coord[0] for coord in rectangle_vertices]
    lons = [coord[1] for coord in rectangle_vertices]

    min_lat = min(lats)
    max_lat = max(lats)
    min_lon = min(lons)
    max_lon = max(lons)

    # Use zoom level 15 which provides good detail for buildings while keeping data size manageable
    zoom = 15

    # Get list of tile coordinates that cover the bounding box at specified zoom level
    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zoom))

    building_features = []

    # Set up coordinate transformer to convert from Web Mercator (EPSG:3857) to WGS84 (EPSG:4326)
    # always_xy=True ensures longitude comes before latitude
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    for tile in tiles:
        x, y, z = tile.x, tile.y, tile.z

        # Construct URL for vector tile using MapTiler API
        tile_url = f'https://api.maptiler.com/tiles/v3/{z}/{x}/{y}.pbf?key={API_KEY}'

        print(f'Downloading tile {z}/{x}/{y}')
        response = requests.get(tile_url)

        if response.status_code != 200:
            print(f'Failed to download tile {z}/{x}/{y}')
            continue

        # Decode the Protocol Buffer (PBF) formatted vector tile
        tile_data = mapbox_vector_tile.decode(response.content)

        # Process building layer if it exists in the tile
        if 'building' in tile_data:
            building_layer = tile_data['building']
            for feature in building_layer['features']:
                # Convert feature geometry to shapely object for manipulation
                geometry = shape(feature['geometry'])

                # Vector tiles use local coordinates from 0-4096
                # Need to transform these to real world coordinates
                x_min, y_min = 0, 0
                x_max, y_max = 4096, 4096

                # Get tile bounds in Web Mercator coordinates
                tile_bbox_mercator = mercantile.xy_bounds(x, y, z)

                # Calculate scale factors to transform local tile coordinates to Web Mercator
                scale_x = (tile_bbox_mercator.right - tile_bbox_mercator.left) / (x_max - x_min)
                scale_y = (tile_bbox_mercator.bottom - tile_bbox_mercator.top) / (y_max - y_min)

                # Create affine transformation matrix:
                # [a b xoff]
                # [d e yoff]
                # [0 0 1  ]
                a = scale_x  # x scale
                b = 0        # rotation
                d = 0        # rotation
                e = -scale_y # y scale (negative because y axis is flipped)
                xoff = tile_bbox_mercator.left    # x translation
                yoff = tile_bbox_mercator.bottom  # y translation

                transform_matrix = [a, b, d, e, xoff, yoff]

                # Transform geometry from tile coordinates to Web Mercator
                transformed_geom = affine_transform(geometry, transform_matrix)

                # Transform from Web Mercator to WGS84 geographic coordinates
                transformed_geometry = shapely.ops.transform(transformer.transform, transformed_geom)

                # Create standardized GeoJSON feature
                geojson_feature = {
                    'type': 'Feature',
                    'geometry': mapping(transformed_geometry),
                    'properties': feature['properties']
                }

                building_features.append(geojson_feature)

    # Convert features to standardized format with height information
    converted_geojson_data = convert_geojson_format(building_features)
    return converted_geojson_data

def get_height_from_properties(properties):
    """Extract building height from properties, using levels if height is not available.

    Args:
        properties: Dictionary containing building properties

    Returns:
        float: Building height in meters, defaults to 0 if no valid height found
    """
    # First try explicit render_height property
    height = properties.get('render_height')
    if height is not None:
        try:
            return float(height)
        except ValueError:
            pass
    
    # If no height available, estimate from number of levels
    # OpenMapTiles uses building:levels tag for number of floors
    levels = properties.get('building:levels')
    if levels is not None:
        try:
            return float(levels) * 5.0  # Assume average floor height of 5 meters
        except ValueError:
            pass
    
    return 0  # Default height if no valid data found

def convert_geojson_format(features):
    """Convert building features to standardized format with height information.

    Args:
        features: List of GeoJSON features containing building footprints

    Returns:
        list: List of standardized GeoJSON features with height properties
    """
    new_features = []

    for feature in features:
        geometry = feature['geometry']
        properties = feature['properties']

        # Extract height information
        height = get_height_from_properties(properties)
        min_height = properties.get('render_min_height', 0)
        try:
            min_height = float(min_height)
        except ValueError:
            min_height = 0

        # Create standardized properties dictionary
        new_properties = {
            'height': height,
            'min_height': min_height,
            'confidence': -1.0,  # No confidence score available from OpenMapTiles
            'is_inner': False    # Will be set based on ring position
        }

        # Handle MultiPolygon geometries by splitting into separate Polygon features
        if geometry['type'] == 'MultiPolygon':
            for i, polygon_coords in enumerate(geometry['coordinates']):
                # Process each ring in the polygon (outer ring + inner holes)
                for j, ring in enumerate(polygon_coords):
                    ring_properties = new_properties.copy()
                    # First ring (j=0) is outer boundary, others are inner holes
                    ring_properties['is_inner'] = j > 0
                    ring_properties['role'] = 'inner' if j > 0 else 'outer'

                    # Create new geometry with swapped coordinate order (lon,lat) -> (lat,lon)
                    new_geometry = {
                        'type': 'Polygon',
                        'coordinates': [[(coord[1], coord[0]) for coord in ring]]
                    }

                    new_feature = {
                        'type': 'Feature',
                        'properties': ring_properties,
                        'geometry': new_geometry
                    }
                    new_features.append(new_feature)

        # Handle single Polygon geometries
        elif geometry['type'] == 'Polygon':
            # Process each ring in the polygon
            for i, ring in enumerate(geometry['coordinates']):
                ring_properties = new_properties.copy()
                ring_properties['is_inner'] = i > 0
                ring_properties['role'] = 'inner' if i > 0 else 'outer'

                # Create new geometry with swapped coordinate order
                new_geometry = {
                    'type': 'Polygon',
                    'coordinates': [[(coord[1], coord[0]) for coord in ring]]
                }

                new_feature = {
                    'type': 'Feature',
                    'properties': ring_properties,
                    'geometry': new_geometry
                }
                new_features.append(new_feature)
    
    return new_features
