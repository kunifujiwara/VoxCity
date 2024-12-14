"""
Utility functions for geographic operations and coordinate transformations.

This module provides various utility functions for working with geographic data,
including coordinate transformations, distance calculations, geocoding, and building
polygon processing.
"""

import os
import math
from math import radians, sin, cos, sqrt, atan2
import numpy as np
from pyproj import Geod, Transformer
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.warp import transform_bounds
from rasterio.mask import mask
from shapely.geometry import Polygon, box
from fiona.crs import from_epsg
from rtree import index
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.extra.rate_limiter import RateLimiter
import warnings
import reverse_geocoder as rg
import pycountry

from timezonefinder import TimezoneFinder
import pytz
from datetime import datetime

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

floor_height = 2.5

def tile_from_lat_lon(lat, lon, level_of_detail):
    """
    Convert latitude/longitude coordinates to tile coordinates at a given zoom level.
    
    Args:
        lat (float): Latitude in degrees
        lon (float): Longitude in degrees
        level_of_detail (int): Zoom level
        
    Returns:
        tuple: (tile_x, tile_y) tile coordinates
    """
    # Convert latitude to radians and calculate sine
    sin_lat = math.sin(lat * math.pi / 180)
    
    # Convert longitude to normalized x coordinate (0-1)
    x = (lon + 180) / 360
    
    # Convert latitude to y coordinate using Mercator projection formula
    y = 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)
    
    # Calculate map size in pixels at this zoom level (256 * 2^zoom)
    map_size = 256 << level_of_detail
    
    # Convert x,y to tile coordinates
    tile_x = int(x * map_size / 256)
    tile_y = int(y * map_size / 256)
    return tile_x, tile_y

def quadkey_to_tile(quadkey):
    """
    Convert a quadkey string to tile coordinates.
    A quadkey is a string of digits (0-3) that identifies a tile at a certain zoom level.
    Each digit in the quadkey represents a tile at a zoom level, with each subsequent digit
    representing a more detailed zoom level.
    
    Args:
        quadkey (str): Quadkey string
        
    Returns:
        tuple: (tile_x, tile_y, level_of_detail) tile coordinates and zoom level
    """
    tile_x = tile_y = 0
    level_of_detail = len(quadkey)
    
    # Process each character in quadkey
    for i in range(level_of_detail):
        bit = level_of_detail - i - 1
        mask = 1 << bit
        
        # Quadkey digit to binary: 
        # 0 = neither x nor y bit set
        # 1 = x bit set
        # 2 = y bit set 
        # 3 = both x and y bits set
        if quadkey[i] == '1':
            tile_x |= mask
        elif quadkey[i] == '2':
            tile_y |= mask
        elif quadkey[i] == '3':
            tile_x |= mask
            tile_y |= mask
    return tile_x, tile_y, level_of_detail

def initialize_geod():
    """
    Initialize a Geod object for geodetic calculations using WGS84 ellipsoid.
    The WGS84 ellipsoid is the standard reference system used by GPS.
    
    Returns:
        Geod: Initialized Geod object
    """
    return Geod(ellps='WGS84')

def calculate_distance(geod, lon1, lat1, lon2, lat2):
    """
    Calculate geodetic distance between two points.
    Uses inverse geodetic computation to find the shortest distance along the ellipsoid.
    
    Args:
        geod (Geod): Geod object for calculations
        lon1, lat1 (float): Coordinates of first point
        lon2, lat2 (float): Coordinates of second point
        
    Returns:
        float: Distance in meters
    """
    # inv() returns forward azimuth, back azimuth, and distance
    _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    return dist

def normalize_to_one_meter(vector, distance_in_meters):
    """
    Normalize a vector to represent one meter.
    Useful for creating unit vectors in geographic calculations.
    
    Args:
        vector (numpy.ndarray): Vector to normalize
        distance_in_meters (float): Current distance in meters
        
    Returns:
        numpy.ndarray: Normalized vector
    """
    return vector * (1 / distance_in_meters)

def setup_transformer(from_crs, to_crs):
    """
    Set up a coordinate transformer between two CRS.
    The always_xy=True parameter ensures consistent handling of coordinate order.
    
    Args:
        from_crs: Source coordinate reference system
        to_crs: Target coordinate reference system
        
    Returns:
        Transformer: Initialized transformer object
    """
    return Transformer.from_crs(from_crs, to_crs, always_xy=True)

def transform_coords(transformer, lon, lat):
    """
    Transform coordinates using provided transformer.
    Includes error handling for invalid transformations and infinite values.
    
    Args:
        transformer (Transformer): Coordinate transformer
        lon, lat (float): Input coordinates
        
    Returns:
        tuple: (x, y) transformed coordinates or (None, None) if transformation fails
    """
    try:
        x, y = transformer.transform(lon, lat)
        if np.isinf(x) or np.isinf(y):
            print(f"Transformation resulted in inf values for coordinates: {lon}, {lat}")
        return x, y
    except Exception as e:
        print(f"Error transforming coordinates {lon}, {lat}: {e}")
        return None, None

def create_polygon(vertices):
    """
    Create a Shapely polygon from vertices.
    Converts from (lat,lon) format to (lon,lat) format required by Shapely.
    
    Args:
        vertices (list): List of (lat, lon) coordinate pairs
        
    Returns:
        Polygon: Shapely polygon object
    """
    flipped_vertices = [(lon, lat) for lat, lon in vertices]
    return Polygon(flipped_vertices)

def create_geodataframe(polygon, crs=4326):
    """
    Create a GeoDataFrame from a polygon.
    Default CRS is WGS84 (EPSG:4326).
    
    Args:
        polygon (Polygon): Shapely polygon object
        crs (int): Coordinate reference system EPSG code
        
    Returns:
        GeoDataFrame: GeoDataFrame containing the polygon
    """
    return gpd.GeoDataFrame({'geometry': [polygon]}, crs=from_epsg(crs))

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points using Haversine formula.
    This is an approximation that treats the Earth as a perfect sphere.
    
    Args:
        lat1, lon1 (float): Coordinates of first point
        lat2, lon2 (float): Coordinates of second point
        
    Returns:
        float: Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    # Convert all coordinates to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Calculate differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def get_raster_bbox(raster_path):
    """
    Get bounding box of a raster file.
    Returns a rectangular polygon representing the spatial extent of the raster.
    
    Args:
        raster_path (str): Path to raster file
        
    Returns:
        box: Shapely box representing the raster bounds
    """
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
    return box(bounds.left, bounds.bottom, bounds.right, bounds.top)

def raster_intersects_polygon(raster_path, polygon):
    """
    Check if a raster intersects with a polygon.
    Transforms coordinates to WGS84 if needed before checking intersection.
    
    Args:
        raster_path (str): Path to raster file
        polygon (Polygon): Shapely polygon to check intersection with
        
    Returns:
        bool: True if raster intersects polygon, False otherwise
    """
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        # Transform bounds to WGS84 if raster is in different CRS
        if src.crs.to_epsg() != 4326:
            bounds = transform_bounds(src.crs, 'EPSG:4326', *bounds)
        raster_bbox = box(*bounds)
        intersects = raster_bbox.intersects(polygon) or polygon.intersects(raster_bbox)
        return intersects

def save_raster(input_path, output_path):
    """
    Save a copy of a raster file.
    Creates a direct copy without any transformation or modification.
    
    Args:
        input_path (str): Source raster file path
        output_path (str): Destination raster file path
    """
    import shutil
    shutil.copy(input_path, output_path)
    print(f"Copied original file to: {output_path}")

def merge_geotiffs(geotiff_files, output_dir):
    """
    Merge multiple GeoTIFF files into a single file.
    
    Args:
        geotiff_files (list): List of GeoTIFF file paths to merge
        output_dir (str): Directory to save merged output
    """
    if not geotiff_files:
        return

    # Open all valid GeoTIFF files
    src_files_to_mosaic = [rasterio.open(file) for file in geotiff_files if os.path.exists(file)]

    if src_files_to_mosaic:
        try:
            # Merge rasters into a single mosaic and get output transform
            mosaic, out_trans = merge(src_files_to_mosaic)

            # Copy metadata from first raster and update for merged output
            out_meta = src_files_to_mosaic[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans
            })

            # Save merged raster to output file
            merged_path = os.path.join(output_dir, "lulc.tif")
            with rasterio.open(merged_path, "w", **out_meta) as dest:
                dest.write(mosaic)

            print(f"Merged output saved to: {merged_path}")
        except Exception as e:
            print(f"Error merging files: {e}")
    else:
        print("No valid files to merge.")

    # Clean up by closing all opened files
    for src in src_files_to_mosaic:
        src.close()

def convert_format_lat_lon(input_coords):
    """
    Convert coordinate format and close polygon.
    
    Args:
        input_coords (list): List of [lon, lat] coordinates
        
    Returns:
        list: List of [lat, lon] coordinates with first point repeated at end
    """
    # Swap lon/lat to lat/lon format and create list
    output_coords = [[coord[1], coord[0]] for coord in input_coords]
    # Close polygon by repeating first point at end
    output_coords.append(output_coords[0])
    return output_coords

def get_coordinates_from_cityname(place_name):
    """
    Get coordinates for a city name using geocoding.
    
    Args:
        place_name (str): Name of city to geocode
        
    Returns:
        tuple: (latitude, longitude) or None if geocoding fails
    """
    # Initialize geocoder with user agent
    geolocator = Nominatim(user_agent="my_geocoding_script")
    
    try:
        # Attempt to geocode the place name
        location = geolocator.geocode(place_name)
        
        if location:
            return (location.latitude, location.longitude)
        else:
            return None
    except (GeocoderTimedOut, GeocoderServiceError):
        print(f"Error: Geocoding service timed out or encountered an error for {place_name}")
        return None

def get_city_country_name_from_rectangle(coordinates):
    """
    Get city and country name from rectangle coordinates.
    
    Args:
        coordinates (list): List of (lat, lon) coordinates defining rectangle
        
    Returns:
        str: String in format "city/ country" or None if lookup fails
    """
    # Calculate center point of rectangle
    latitudes = [coord[0] for coord in coordinates]
    longitudes = [coord[1] for coord in coordinates]
    center_lat = sum(latitudes) / len(latitudes)
    center_lon = sum(longitudes) / len(longitudes)
    center_coord = (center_lat, center_lon)

    # Initialize geocoder with rate limiting to avoid hitting API limits
    geolocator = Nominatim(user_agent="your_app_name (your_email@example.com)")
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=2, error_wait_seconds=5, max_retries=3)

    try:
        # Attempt reverse geocoding of center coordinates
        location = reverse(center_coord, language='en')
        if location:
            address = location.raw['address']
            # Try multiple address fields to find city name, falling back to county if needed
            city = address.get('city', '') or address.get('town', '') or address.get('village', '') or address.get('county', '')
            country = address.get('country', '')
            return f"{city}/ {country}"
        else:
            print("Location not found")
    except Exception as e:
        print(f"Error retrieving location for {center_coord}: {e}")

def get_timezone_info(rectangle_coords):
    """
    Get timezone and central meridian info for a location.
    
    Args:
        rectangle_coords (list): List of (lat, lon) coordinates defining rectangle
        
    Returns:
        tuple: (timezone string, central meridian longitude string)
    """
    # Calculate center point of rectangle
    latitudes = [coord[0] for coord in rectangle_coords]
    longitudes = [coord[1] for coord in rectangle_coords]
    center_lat = sum(latitudes) / len(latitudes)
    center_lon = sum(longitudes) / len(longitudes)
    center_coord = (center_lat, center_lon)
    
    # Find timezone at center coordinates
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lng=center_lon, lat=center_lat)
    
    if timezone_str:
        # Get current time in local timezone to calculate offset
        timezone = pytz.timezone(timezone_str)
        now = datetime.now(timezone)
        offset_seconds = now.utcoffset().total_seconds()
        offset_hours = offset_seconds / 3600

        # Format timezone offset and calculate central meridian
        utc_offset = f"UTC{offset_hours:+.2f}"
        timezone_longitude = offset_hours * 15  # Each hour offset = 15 degrees longitude
        timezone_longitude_str = f"{timezone_longitude:.5f}"

        return utc_offset, timezone_longitude_str
    else:
        raise ValueError("Time zone not found for the given location.")

def validate_polygon_coordinates(geometry):
    """
    Validate and close polygon coordinate rings.
    
    Args:
        geometry (dict): GeoJSON geometry object
        
    Returns:
        bool: True if valid polygon coordinates, False otherwise
    """
    if geometry['type'] == 'Polygon':
        for ring in geometry['coordinates']:
            # Ensure polygon is closed by checking/adding first point at end
            if ring[0] != ring[-1]:
                ring.append(ring[0])  # Close the ring
            # Check minimum points needed for valid polygon (3 points + closing point)
            if len(ring) < 4:
                return False
        return True
    elif geometry['type'] == 'MultiPolygon':
        for polygon in geometry['coordinates']:
            for ring in polygon:
                if ring[0] != ring[-1]:
                    ring.append(ring[0])  # Close the ring
                if len(ring) < 4:
                    return False
        return True
    else:
        return False

def create_building_polygons(filtered_buildings):
    """
    Create building polygons with properties from filtered GeoJSON features.
    
    Args:
        filtered_buildings (list): List of GeoJSON building features
        
    Returns:
        tuple: (list of building polygons with properties, spatial index)
    """
    building_polygons = []
    idx = index.Index()
    count = 0
    
    # Find highest existing ID to avoid duplicates
    id_list = []
    for i, building in enumerate(filtered_buildings):
        if building['properties'].get('id') is not None:
            id_list.append(building['properties']['id'])
    if len(id_list) > 0:
        id_count = max(id_list)+1
    else:
        id_count = 1

    for i, building in enumerate(filtered_buildings):
        # Create polygon from coordinates
        polygon = Polygon(building['geometry']['coordinates'][0])
        
        # Extract height information from various possible property fields
        height = building['properties'].get('height')
        levels = building['properties'].get('levels')
        floors = building['properties'].get('num_floors')
        min_height = building['properties'].get('min_height')
        min_level = building['properties'].get('min_level')    
        min_floor = building['properties'].get('min_floor')        

        # Calculate height if not directly specified
        if (height is None) or (height<=0):
            if levels is not None:
                height = floor_height * levels
            elif floors is not None:
                height = floor_height * floors
            else:
                count += 1
                height = np.nan

        # Calculate minimum height if not directly specified
        if (min_height is None) or (min_height<=0):
            if min_level is not None:
                min_height = floor_height * float(min_level) 
            elif min_floor is not None:
                min_height = floor_height * float(min_floor)
            else:
                min_height = 0

        # Get or assign building ID
        if building['properties'].get('id') is not None:
            feature_id = building['properties']['id']
        else:
            feature_id = id_count
            id_count += 1

        # Check if building is inner part of another building
        if building['properties'].get('is_inner') is not None:
            is_inner = building['properties']['is_inner']
        else:
            is_inner = False

        # Store polygon with all properties and add to spatial index
        building_polygons.append((polygon, height, min_height, is_inner, feature_id))
        idx.insert(i, polygon.bounds)

    return building_polygons, idx

def get_country_name(lat, lon):
    """
    Get country name from coordinates using reverse geocoding.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        str: Country name or None if lookup fails
    """
    # Use reverse geocoder to get country code
    results = rg.search((lat, lon))
    country_code = results[0]['cc']
    
    # Convert country code to full name using pycountry
    country = pycountry.countries.get(alpha_2=country_code)

    if country:
        return country.name
    else:
        return None