"""
Module for downloading and processing building data from OpenMapTiles vector tiles.

This module provides functionality to download and process building footprint data from
OpenMapTiles vector tile service. It handles downloading PBF tiles, extracting building
geometries, and converting them to GeoJSON format with standardized properties.

Key Features:
    - Downloads vector tiles from OpenMapTiles API
    - Extracts building footprints and properties
    - Converts coordinates from tile-local to WGS84
    - Standardizes building height information
    - Handles both Polygon and MultiPolygon geometries
    - Separates inner and outer rings of building footprints

Dependencies:
    - mercantile: For tile calculations and coordinate transformations
    - mapbox_vector_tile: For decoding PBF vector tiles
    - shapely: For geometry operations
    - pyproj: For coordinate system transformations
    - geopandas: For working with geospatial data
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
import geopandas as gpd

def load_gdf_from_openmaptiles(rectangle_vertices, API_KEY):
    """Download and process building footprint data from OpenMapTiles vector tiles.

    This function downloads vector tiles covering the specified area, extracts building
    footprints, and converts them to a standardized format in a GeoDataFrame.

    Args:
        rectangle_vertices (list): List of (lon, lat) tuples defining the bounding box corners.
            The coordinates should be in WGS84 (EPSG:4326) format.
        API_KEY (str): OpenMapTiles API key for authentication. Must be valid for the v3 endpoint.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing building footprints with the following columns:
            - geometry: Building footprint geometry in WGS84 coordinates
            - height: Building height in meters
            - min_height: Minimum height (e.g., for elevated structures) in meters
            - confidence: Confidence score (-1.0 for OpenMapTiles data)
            - is_inner: Boolean indicating if the polygon is an inner ring
            - role: String indicating 'inner' or 'outer' ring
            - id: Unique identifier for each building feature

    Notes:
        - Uses zoom level 15 for optimal detail vs data size balance
        - Converts coordinates from Web Mercator (EPSG:3857) to WGS84 (EPSG:4326)
        - Handles both Polygon and MultiPolygon geometries
        - Separates complex building footprints into their constituent parts
    """
    # Extract longitudes and latitudes from vertices to find bounding box
    lons = [coord[0] for coord in rectangle_vertices]
    lats = [coord[1] for coord in rectangle_vertices]

    min_lon = min(lons)
    max_lon = max(lons)
    min_lat = min(lats)
    max_lat = max(lats)

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

    gdf = gpd.GeoDataFrame.from_features(converted_geojson_data)
    gdf.set_crs(epsg=4326, inplace=True)
    
    # Replace id column with index numbers
    gdf['id'] = gdf.index
    
    return gdf

def get_height_from_properties(properties):
    """Extract building height from properties, using levels if height is not available.

    This function implements a fallback strategy for determining building heights:
    1. First tries to use explicit render_height property
    2. If not available, estimates height from number of building levels
    3. Returns 0 if no valid height information is found

    Args:
        properties (dict): Dictionary containing building properties from OpenMapTiles.
            Expected keys:
            - render_height: Direct height specification in meters
            - building:levels: Number of building floors/levels

    Returns:
        float: Building height in meters. Values can be:
            - Explicit height from render_height property
            - Estimated height (levels * 5.0 meters per level)
            - 0.0 if no valid height information is found

    Notes:
        - Assumes average floor height of 5 meters when estimating from levels
        - Handles potential invalid values gracefully by returning 0
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

    This function processes raw OpenMapTiles building features into a standardized format,
    handling complex geometries and adding consistent property attributes.

    Args:
        features (list): List of GeoJSON features containing building footprints.
            Each feature should have:
            - geometry: GeoJSON geometry (Polygon or MultiPolygon)
            - properties: Dictionary of building properties

    Returns:
        list: List of standardized GeoJSON features where:
            - Complex MultiPolygons are split into individual Polygons
            - Each Polygon ring (outer and inner) becomes a separate feature
            - Properties are standardized to include:
                - height: Building height in meters
                - min_height: Minimum height in meters
                - confidence: Set to -1.0 for OpenMapTiles data
                - is_inner: Boolean flag for inner rings
                - role: String indicating 'inner' or 'outer' ring

    Notes:
        - Preserves coordinate order as (longitude, latitude)
        - Maintains topological relationships through is_inner and role properties
        - Splits complex geometries for easier processing downstream
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

                    # Create new geometry keeping coordinate order as (lon,lat)
                    new_geometry = {
                        'type': 'Polygon',
                        'coordinates': [ring]
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

                # Create new geometry keeping coordinate order as (lon,lat)
                new_geometry = {
                    'type': 'Polygon',
                    'coordinates': [ring]
                }

                new_feature = {
                    'type': 'Feature',
                    'properties': ring_properties,
                    'geometry': new_geometry
                }
                new_features.append(new_feature)
    
    return new_features
