"""
Utility functions for geographic operations and coordinate transformations.

This module provides various utility functions for working with geographic data,
including coordinate transformations, distance calculations, geocoding, and building
polygon processing. It supports operations such as:

- Tile coordinate calculations and quadkey conversions
- Geographic distance calculations (Haversine and geodetic)
- Coordinate system transformations
- Polygon and GeoDataFrame operations
- Raster file processing and merging
- Geocoding and reverse geocoding
- Timezone and location information retrieval
- Building polygon validation and processing

The module uses several external libraries for geographic operations:
- pyproj: For coordinate transformations and geodetic calculations
- geopandas: For handling geographic data frames
- rasterio: For raster file operations
- shapely: For geometric operations
- geopy: For geocoding services
- timezonefinder: For timezone lookups
"""

# Standard library imports
import os
import math
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

# Third-party geographic processing libraries
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

# Geocoding and location services
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderInsufficientPrivileges
from geopy.extra.rate_limiter import RateLimiter
import reverse_geocoder as rg
import pycountry

# Timezone handling
from timezonefinder import TimezoneFinder
import pytz

# Suppress rasterio warnings for non-georeferenced files
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Global constants
floor_height = 2.5  # Standard floor height in meters used for building height calculations

# Build a compliant Nominatim user agent once and reuse it
try:
    # Prefer package metadata if available
    from voxcity import __version__ as _vox_version, __email__ as _vox_email
except Exception:
    _vox_version, _vox_email = "dev", "contact@voxcity.local"

_ENV_UA = os.environ.get("VOXCITY_NOMINATIM_UA", "").strip()
_DEFAULT_UA = f"voxcity/{_vox_version} (+https://github.com/kunifujiwara/voxcity; contact: {_vox_email})"
_NOMINATIM_USER_AGENT = _ENV_UA or _DEFAULT_UA

def _create_nominatim_geolocator() -> Nominatim:
    """
    Create a Nominatim geolocator with a compliant identifying user agent.
    The user agent can be overridden via the environment variable
    VOXCITY_NOMINATIM_UA.
    """
    return Nominatim(user_agent=_NOMINATIM_USER_AGENT)

def tile_from_lat_lon(lat, lon, level_of_detail):
    """
    Convert latitude/longitude coordinates to tile coordinates at a given zoom level.
    Uses the Web Mercator projection (EPSG:3857) commonly used in web mapping.
    
    Args:
        lat (float): Latitude in degrees (-90 to 90)
        lon (float): Longitude in degrees (-180 to 180)
        level_of_detail (int): Zoom level (0-23, where 0 is the entire world)
        
    Returns:
        tuple: (tile_x, tile_y) tile coordinates in the global tile grid
        
    Example:
        >>> tile_x, tile_y = tile_from_lat_lon(35.6762, 139.6503, 12)  # Tokyo at zoom 12
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
    
    The quadkey numbering scheme:
        - 0: Top-left quadrant
        - 1: Top-right quadrant
        - 2: Bottom-left quadrant
        - 3: Bottom-right quadrant
    
    Args:
        quadkey (str): Quadkey string (e.g., "120" for zoom level 3)
        
    Returns:
        tuple: (tile_x, tile_y, level_of_detail) tile coordinates and zoom level
        
    Example:
        >>> x, y, zoom = quadkey_to_tile("120")  # Returns coordinates at zoom level 3
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
    The WGS84 ellipsoid (EPSG:4326) is the standard reference system used by GPS
    and most modern mapping applications.
    
    The Geod object provides methods for:
    - Forward geodetic calculations (direct)
    - Inverse geodetic calculations (inverse)
    - Area calculations
    - Line length calculations
    
    Returns:
        Geod: Initialized Geod object for WGS84 calculations
        
    Example:
        >>> geod = initialize_geod()
        >>> fwd_az, back_az, dist = geod.inv(lon1, lat1, lon2, lat2)
    """
    return Geod(ellps='WGS84')

def calculate_distance(geod, lon1, lat1, lon2, lat2):
    """
    Calculate geodetic distance between two points on the Earth's surface.
    Uses inverse geodetic computation to find the shortest distance along the ellipsoid,
    which is more accurate than great circle (spherical) calculations.
    
    Args:
        geod (Geod): Geod object for calculations, initialized with WGS84
        lon1, lat1 (float): Coordinates of first point in decimal degrees
        lon2, lat2 (float): Coordinates of second point in decimal degrees
        
    Returns:
        float: Distance in meters between the two points along the ellipsoid
        
    Example:
        >>> geod = initialize_geod()
        >>> distance = calculate_distance(geod, 139.6503, 35.6762, 
        ...                             -74.0060, 40.7128)  # Tokyo to NYC
    """
    # inv() returns forward azimuth, back azimuth, and distance
    _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    return dist

def normalize_to_one_meter(vector, distance_in_meters):
    """
    Normalize a vector to represent one meter in geographic space.
    Useful for creating unit vectors in geographic calculations, particularly
    when working with distance-based operations or scaling geographic features.
    
    Args:
        vector (numpy.ndarray): Vector to normalize, typically a direction vector
        distance_in_meters (float): Current distance in meters that the vector represents
        
    Returns:
        numpy.ndarray: Normalized vector where magnitude represents 1 meter
        
    Example:
        >>> direction = np.array([3.0, 4.0])  # Vector of length 5
        >>> unit_meter = normalize_to_one_meter(direction, 5.0)
    """
    return vector * (1 / distance_in_meters)

def setup_transformer(from_crs, to_crs):
    """
    Set up a coordinate transformer between two Coordinate Reference Systems (CRS).
    The always_xy=True parameter ensures consistent handling of coordinate order
    by always using (x,y) or (longitude,latitude) order regardless of CRS definition.
    
    Common CRS codes:
    - EPSG:4326 - WGS84 (latitude/longitude)
    - EPSG:3857 - Web Mercator
    - EPSG:2263 - NY State Plane
    
    Args:
        from_crs: Source coordinate reference system (EPSG code, proj4 string, or CRS dict)
        to_crs: Target coordinate reference system (EPSG code, proj4 string, or CRS dict)
        
    Returns:
        Transformer: Initialized transformer object for coordinate conversion
        
    Example:
        >>> transformer = setup_transformer("EPSG:4326", "EPSG:3857")
        >>> x, y = transformer.transform(longitude, latitude)
    """
    return Transformer.from_crs(from_crs, to_crs, always_xy=True)

def transform_coords(transformer, lon, lat):
    """
    Transform coordinates using provided transformer with error handling.
    Includes validation for infinite values that may result from invalid transformations
    or coordinates outside the valid range for the target CRS.
    
    Args:
        transformer (Transformer): Coordinate transformer from setup_transformer()
        lon, lat (float): Input coordinates in the source CRS
        
    Returns:
        tuple: (x, y) transformed coordinates in the target CRS, or (None, None) if transformation fails
        
    Example:
        >>> transformer = setup_transformer("EPSG:4326", "EPSG:3857")
        >>> x, y = transform_coords(transformer, -74.0060, 40.7128)  # NYC coordinates
        >>> if x is not None:
        ...     print(f"Transformed coordinates: ({x}, {y})")
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
    Create a Shapely polygon from a list of vertices.
    Input vertices must be in (longitude, latitude) format as required by Shapely.
    The polygon will be automatically closed if the first and last vertices don't match.
    
    Args:
        vertices (list): List of (longitude, latitude) coordinate pairs forming the polygon.
                        The coordinates should be in counter-clockwise order for exterior rings
                        and clockwise order for interior rings (holes).
        
    Returns:
        Polygon: Shapely polygon object that can be used for spatial operations
        
    Example:
        >>> vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]  # Square
        >>> polygon = create_polygon(vertices)
        >>> print(f"Polygon area: {polygon.area}")
    """
    return Polygon(vertices)

def create_geodataframe(polygon, crs=4326):
    """
    Create a GeoDataFrame from a Shapely polygon.
    Default CRS is WGS84 (EPSG:4326) for geographic coordinates.
    The GeoDataFrame provides additional functionality for spatial operations,
    data analysis, and export to various geographic formats.
    
    Args:
        polygon (Polygon): Shapely polygon object to convert
        crs (int): Coordinate reference system EPSG code (default: 4326 for WGS84)
        
    Returns:
        GeoDataFrame: GeoDataFrame containing the polygon with specified CRS
        
    Example:
        >>> vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> polygon = create_polygon(vertices)
        >>> gdf = create_geodataframe(polygon)
        >>> gdf.to_file("polygon.geojson", driver="GeoJSON")
    """
    return gpd.GeoDataFrame({'geometry': [polygon]}, crs=from_epsg(crs))

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate great-circle distance between two points using Haversine formula.
    This is an approximation that treats the Earth as a perfect sphere.
    
    Args:
        lon1, lat1 (float): Coordinates of first point
        lon2, lat2 (float): Coordinates of second point
        
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
    Get the bounding box of a raster file in its native coordinate system.
    Returns a rectangular polygon representing the spatial extent of the raster,
    which can be used for spatial queries and intersection tests.
    
    Args:
        raster_path (str): Path to the raster file (GeoTIFF, IMG, etc.)
        
    Returns:
        box: Shapely box representing the raster bounds in the raster's CRS
        
    Example:
        >>> bbox = get_raster_bbox("elevation.tif")
        >>> print(f"Raster extent: {bbox.bounds}")  # (minx, miny, maxx, maxy)
    """
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
    return box(bounds.left, bounds.bottom, bounds.right, bounds.top)

def raster_intersects_polygon(raster_path, polygon):
    """
    Check if a raster file's extent intersects with a given polygon.
    Automatically handles coordinate system transformations by converting
    the raster bounds to WGS84 (EPSG:4326) if needed before the intersection test.
    
    Args:
        raster_path (str): Path to the raster file to check
        polygon (Polygon): Shapely polygon to test intersection with (in WGS84)
        
    Returns:
        bool: True if raster intersects or contains the polygon, False otherwise
        
    Example:
        >>> aoi = create_polygon([(lon1, lat1), (lon2, lat2), ...])  # Area of interest
        >>> if raster_intersects_polygon("dem.tif", aoi):
        ...     print("Raster covers the area of interest")
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
    Create a copy of a raster file at a new location.
    Performs a direct file copy without any transformation or modification,
    preserving all metadata, georeferencing, and pixel values.
    
    Args:
        input_path (str): Source raster file path
        output_path (str): Destination path for the copied raster
        
    Example:
        >>> save_raster("original.tif", "backup/copy.tif")
        >>> print("Copied original file to: backup/copy.tif")
    """
    import shutil
    shutil.copy(input_path, output_path)
    print(f"Copied original file to: {output_path}")

def merge_geotiffs(geotiff_files, output_dir):
    """
    Merge multiple GeoTIFF files into a single mosaic.
    Handles edge matching and overlapping areas between adjacent rasters.
    The output will have the same coordinate system and data type as the input files.
    
    Important considerations:
    - All input files should have the same coordinate system
    - All input files should have the same data type
    - Overlapping areas are handled by taking the first value encountered
    
    Args:
        geotiff_files (list): List of paths to GeoTIFF files to merge
        output_dir (str): Directory where the merged output will be saved
        
    Example:
        >>> files = ["tile1.tif", "tile2.tif", "tile3.tif"]
        >>> merge_geotiffs(files, "output_directory")
        >>> print("Merged output saved to: output_directory/lulc.tif")
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
    Input coordinates are already in [lon, lat] format.
    
    Args:
        input_coords (list): List of [lon, lat] coordinates
        
    Returns:
        list: List of [lon, lat] coordinates with first point repeated at end
    """
    # Create list with coordinates in same order
    output_coords = input_coords.copy()
    # Close polygon by repeating first point at end
    output_coords.append(output_coords[0])
    return output_coords

def get_coordinates_from_cityname(place_name):
    """
    Geocode a city name to get its coordinates using OpenStreetMap's Nominatim service.
    Includes rate limiting and error handling to comply with Nominatim's usage policy.
    
    Note:
    - Results may vary based on the specificity of the place name
    - For better results, include country or state information
    - Service has usage limits and may timeout
    
    Args:
        place_name (str): Name of the city to geocode (e.g., "Tokyo, Japan")
        
    Returns:
        tuple: (latitude, longitude) coordinates or None if geocoding fails
        
    Example:
        >>> coords = get_coordinates_from_cityname("Paris, France")
        >>> if coords:
        ...     lat, lon = coords
        ...     print(f"Paris coordinates: {lat}, {lon}")
    """
    # Initialize geocoder with compliant user agent
    geolocator = _create_nominatim_geolocator()
    geocode_once = RateLimiter(geolocator.geocode, min_delay_seconds=1.0, max_retries=0)
    
    try:
        # Attempt to geocode the place name (single try; no retries on 403)
        location = geocode_once(place_name, exactly_one=True, timeout=10)
        
        if location:
            return (location.latitude, location.longitude)
        else:
            return None
    except GeocoderInsufficientPrivileges:
        print("Warning: Nominatim blocked the request (HTTP 403). Please set a proper user agent and avoid bulk requests.")
        return None
    except (GeocoderTimedOut, GeocoderServiceError):
        print(f"Error: Geocoding service timed out or encountered an error for {place_name}")
        return None

def get_city_country_name_from_rectangle(coordinates):
    """
    Get the city and country name for a location defined by a rectangle.
    Uses reverse geocoding to find the nearest named place to the rectangle's center.
    
    The function:
    1. Calculates the center point of the rectangle
    2. Performs reverse geocoding with rate limiting
    3. Extracts city and country information from the result
    
    Args:
        coordinates (list): List of (longitude, latitude) coordinates defining the rectangle
        
    Returns:
        str: String in format "city/ country" or fallback value if lookup fails
        
    Example:
        >>> coords = [(139.65, 35.67), (139.66, 35.67),
        ...           (139.66, 35.68), (139.65, 35.68)]
        >>> location = get_city_country_name_from_rectangle(coords)
        >>> print(f"Location: {location}")  # e.g., "Shibuya/ Japan"
    """
    # Calculate center point of rectangle
    longitudes = [coord[0] for coord in coordinates]
    latitudes = [coord[1] for coord in coordinates]
    center_lon = sum(longitudes) / len(longitudes)
    center_lat = sum(latitudes) / len(latitudes)
    center_coord = (center_lat, center_lon)

    # Initialize geocoder with compliant user agent and conservative rate limit (1 req/sec)
    geolocator = _create_nominatim_geolocator()
    reverse_once = RateLimiter(geolocator.reverse, min_delay_seconds=1.0, max_retries=0)

    try:
        # Attempt reverse geocoding of center coordinates (single try; no retries on 403)
        location = reverse_once(center_coord, language='en', exactly_one=True, timeout=10)
        if location:
            address = location.raw['address']
            # Try multiple address fields to find city name, falling back to county if needed
            city = address.get('city', '') or address.get('town', '') or address.get('village', '') or address.get('county', '')
            country = address.get('country', '')
            return f"{city}/ {country}"
        else:
            print("Location not found")
            return "Unknown Location/ Unknown Country"
    except GeocoderInsufficientPrivileges:
        # Fallback to offline reverse_geocoder at coarse resolution
        try:
            results = rg.search((center_lat, center_lon))
            name = results[0].get('name') or ''
            country = get_country_name(center_lon, center_lat) or ''
            if name or country:
                return f"{name}/ {country}".strip()
        except Exception:
            pass
        print("Warning: Nominatim blocked the request (HTTP 403). Falling back to offline coarse reverse geocoding.")
        return "Unknown Location/ Unknown Country"
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Error retrieving location for {center_coord}: {e}")
        return "Unknown Location/ Unknown Country"

def get_timezone_info(rectangle_coords):
    """
    Get timezone and central meridian information for a location.
    Uses the rectangle's center point to determine the local timezone and
    calculates the central meridian based on the UTC offset.
    
    The function provides:
    1. Local timezone identifier (e.g., "America/New_York")
    2. UTC offset (e.g., "UTC-04:00")
    3. Central meridian longitude for the timezone
    
    Args:
        rectangle_coords (list): List of (longitude, latitude) coordinates defining the area
        
    Returns:
        tuple: (timezone string with UTC offset, central meridian longitude string)
        
    Example:
        >>> coords = [(139.65, 35.67), (139.66, 35.67),
        ...           (139.66, 35.68), (139.65, 35.68)]
        >>> tz, meridian = get_timezone_info(coords)
        >>> print(f"Timezone: {tz}, Meridian: {meridian}")  # e.g., "UTC+09:00, 135.00000"
    """
    # Calculate center point of rectangle
    longitudes = [coord[0] for coord in rectangle_coords]
    latitudes = [coord[1] for coord in rectangle_coords]
    center_lon = sum(longitudes) / len(longitudes)
    center_lat = sum(latitudes) / len(latitudes)
    
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
        # Return fallback values if timezone cannot be determined
        print("Warning: Timezone not found for the given location, using UTC+00:00")
        return "UTC+00:00", "0.00000"

def validate_polygon_coordinates(geometry):
    """
    Validate and ensure proper closure of polygon coordinate rings.
    Performs validation and correction of GeoJSON polygon geometries according to
    the GeoJSON specification requirements.
    
    Validation checks:
    1. Geometry type (Polygon or MultiPolygon)
    2. Ring closure (first point equals last point)
    3. Minimum number of points (4, including closure)
    
    Args:
        geometry (dict): GeoJSON geometry object with 'type' and 'coordinates' properties
        
    Returns:
        bool: True if polygon coordinates are valid or were successfully corrected,
              False if validation failed
        
    Example:
        >>> geom = {
        ...     "type": "Polygon",
        ...     "coordinates": [[[0,0], [1,0], [1,1], [0,1]]]  # Not closed
        ... }
        >>> if validate_polygon_coordinates(geom):
        ...     print("Polygon is valid")  # Will close the ring automatically
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
    Processes a list of GeoJSON building features to create Shapely polygons
    with associated height and other properties, while also building a spatial index.
    
    Processing steps:
    1. Extract and validate coordinates
    2. Create Shapely polygons
    3. Process building properties (height, levels, etc.)
    4. Build spatial index for efficient querying
    
    Height calculation rules:
    - Use explicit height if available
    - Calculate from levels * floor_height if height not available
    - Calculate from floors * floor_height if levels not available
    - Use NaN if no height information available
    
    Args:
        filtered_buildings (list): List of GeoJSON building features with properties
        
    Returns:
        tuple: (
            list of tuples (polygon, height, min_height, is_inner, feature_id),
            rtree spatial index for the polygons
        )
        
    Example:
        >>> buildings = [
        ...     {
        ...         "type": "Feature",
        ...         "geometry": {"type": "Polygon", "coordinates": [...]},
        ...         "properties": {"height": 30, "levels": 10}
        ...     },
        ...     # ... more buildings ...
        ... ]
        >>> polygons, spatial_idx = create_building_polygons(buildings)
    """
    building_polygons = []
    idx = index.Index()
    valid_count = 0
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

    for building in filtered_buildings:
        try:
            # Handle potential nested coordinate tuples
            coords = building['geometry']['coordinates'][0]
            # Flatten coordinates if they're nested tuples
            if isinstance(coords[0], tuple):
                coords = [list(c) for c in coords]
            elif isinstance(coords[0][0], tuple):
                coords = [list(c[0]) for c in coords]
                
            # Create polygon from coordinates
            polygon = Polygon(coords)
            
            # Skip invalid geometries
            if not polygon.is_valid:
                print(f"Warning: Skipping invalid polygon geometry")
                continue
                
            height = building['properties'].get('height')
            levels = building['properties'].get('levels')
            floors = building['properties'].get('num_floors')
            min_height = building['properties'].get('min_height')
            min_level = building['properties'].get('min_level')    
            min_floor = building['properties'].get('min_floor')        

            if (height is None) or (height<=0):
                if levels is not None:
                    height = floor_height * levels
                elif floors is not None:
                    height = floor_height * floors
                else:
                    count += 1
                    height = np.nan

            if (min_height is None) or (min_height<=0):
                if min_level is not None:
                    min_height = floor_height * float(min_level) 
                elif min_floor is not None:
                    min_height = floor_height * float(min_floor)
                else:
                    min_height = 0

            if building['properties'].get('id') is not None:
                feature_id = building['properties']['id']
            else:
                feature_id = id_count
                id_count += 1

            if building['properties'].get('is_inner') is not None:
                is_inner = building['properties']['is_inner']
            else:
                is_inner = False

            building_polygons.append((polygon, height, min_height, is_inner, feature_id))
            idx.insert(valid_count, polygon.bounds)
            valid_count += 1
            
        except Exception as e:
            print(f"Warning: Skipping invalid building geometry: {e}")
            continue

    return building_polygons, idx

def get_country_name(lon, lat):
    """
    Get country name from coordinates using reverse geocoding.
    Uses a local database for fast reverse geocoding to country level,
    then converts the country code to full name using pycountry.
    
    Args:
        lon (float): Longitude in decimal degrees
        lat (float): Latitude in decimal degrees
        
    Returns:
        str: Full country name or None if lookup fails
        
    Example:
        >>> country = get_country_name(139.6503, 35.6762)
        >>> print(f"Country: {country}")  # "Japan"
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