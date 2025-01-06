"""
Module for downloading and processing OpenEarthMap Japan (OEMJ) satellite imagery.

This module provides functionality to download, compose, crop and save satellite imagery tiles
from OpenEarthMap Japan as georeferenced GeoTIFF files. It handles coordinate conversions between
latitude/longitude and tile coordinates, downloads tiles within a polygon region, and saves the
final image with proper geospatial metadata.
"""

import requests
from PIL import Image, ImageDraw
from io import BytesIO
import math
import numpy as np
from osgeo import gdal, osr
import pyproj

def deg2num(lon_deg, lat_deg, zoom):
    """Convert longitude/latitude coordinates to tile coordinates.
    
    Args:
        lon_deg (float): Longitude in degrees
        lat_deg (float): Latitude in degrees
        zoom (int): Zoom level
        
    Returns:
        tuple: (x, y) tile coordinates
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    """Convert tile coordinates to longitude/latitude coordinates.
    
    Args:
        xtile (float): X tile coordinate
        ytile (float): Y tile coordinate
        zoom (int): Zoom level
        
    Returns:
        tuple: (longitude, latitude) in degrees
    """
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lon_deg, lat_deg)

def download_tiles(polygon, zoom):
    """Download satellite imagery tiles covering a polygon region.
    
    Args:
        polygon (list): List of (lon, lat) coordinates defining the region
        zoom (int): Zoom level for tile detail
        
    Returns:
        tuple: (tiles dict mapping (x,y) to Image objects, bounds tuple)
    """
    print(f"Downloading tiles")

    # Find bounding box of polygon
    min_lon = min(p[0] for p in polygon)
    max_lon = max(p[0] for p in polygon)
    min_lat = min(p[1] for p in polygon)
    max_lat = max(p[1] for p in polygon)
    
    # Convert to tile coordinates
    min_x, max_y = map(math.floor, deg2num(min_lon, max_lat, zoom))
    max_x, min_y = map(math.ceil, deg2num(max_lon, min_lat, zoom))
    
    # Download tiles within bounds
    tiles = {}
    for x in range(min(min_x, max_x), max(min_x, max_x) + 1):
        for y in range(min(min_y, max_y), max(min_y, max_y) + 1):
            url = f"https://www.open-earth-map.org/demo/Japan/{zoom}/{x}/{y}.png"
            response = requests.get(url)
            if response.status_code == 200:
                tiles[(x, y)] = Image.open(BytesIO(response.content))
            else:
                print(f"Failed to download tile: {url}")
    
    return tiles, (min(min_x, max_x), min(min_y, max_y), max(min_x, max_x), max(min_y, max_y))

def compose_image(tiles, bounds):
    """Compose downloaded tiles into a single image.
    
    Args:
        tiles (dict): Mapping of (x,y) coordinates to tile Image objects
        bounds (tuple): (min_x, min_y, max_x, max_y) tile bounds
        
    Returns:
        Image: Composed PIL Image
    """
    min_x, min_y, max_x, max_y = bounds
    width = abs(max_x - min_x + 1) * 256
    height = abs(max_y - min_y + 1) * 256
    print(f"Composing image with dimensions: {width}x{height}")
    result = Image.new('RGB', (width, height))
    for (x, y), tile in tiles.items():
        result.paste(tile, ((x - min_x) * 256, (y - min_y) * 256))
    return result

def crop_image(image, polygon, bounds, zoom):
    """Crop composed image to polygon boundary.
    
    Args:
        image (Image): PIL Image to crop
        polygon (list): List of (lon, lat) coordinates
        bounds (tuple): (min_x, min_y, max_x, max_y) tile bounds
        zoom (int): Zoom level
        
    Returns:
        tuple: (cropped Image, bounding box)
    """
    min_x, min_y, max_x, max_y = bounds
    img_width, img_height = image.size
    
    # Convert polygon coordinates to pixel coordinates
    polygon_pixels = []
    for lon, lat in polygon:
        x, y = deg2num(lon, lat, zoom)
        px = (x - min_x) * 256
        py = (y - min_y) * 256
        polygon_pixels.append((px, py))
    
    # Create mask from polygon
    mask = Image.new('L', (img_width, img_height), 0)
    ImageDraw.Draw(mask).polygon(polygon_pixels, outline=255, fill=255)
    
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("The polygon does not intersect with the downloaded tiles.")
    
    # Crop to polygon boundary
    cropped = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), mask)
    return cropped.crop(bbox), bbox

def save_as_geotiff(image, polygon, zoom, bbox, bounds, output_path):
    """Save cropped image as georeferenced GeoTIFF.
    
    Args:
        image (Image): PIL Image to save
        polygon (list): List of (lon, lat) coordinates
        zoom (int): Zoom level
        bbox (tuple): Bounding box of cropped image
        bounds (tuple): (min_x, min_y, max_x, max_y) tile bounds
        output_path (str): Path to save GeoTIFF
    """
    min_x, min_y, max_x, max_y = bounds
    
    # Calculate georeferencing coordinates
    lon_upper_left, lat_upper_left = num2deg(min_x + bbox[0]/256, min_y + bbox[1]/256, zoom)
    lon_lower_right, lat_lower_right = num2deg(min_x + bbox[2]/256, min_y + bbox[3]/256, zoom)
    
    # Create transformation from WGS84 to Web Mercator
    wgs84 = pyproj.CRS('EPSG:4326')
    web_mercator = pyproj.CRS('EPSG:3857')
    transformer = pyproj.Transformer.from_crs(wgs84, web_mercator, always_xy=True)
    
    # Transform coordinates to Web Mercator
    upper_left_x, upper_left_y = transformer.transform(lon_upper_left, lat_upper_left)
    lower_right_x, lower_right_y = transformer.transform(lon_lower_right, lat_lower_right)
    
    # Calculate pixel size
    pixel_size_x = (lower_right_x - upper_left_x) / image.width
    pixel_size_y = (upper_left_y - lower_right_y) / image.height
    
    # Create GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, image.width, image.height, 3, gdal.GDT_Byte)
    
    # Set geotransform and projection
    dataset.SetGeoTransform((upper_left_x, pixel_size_x, 0, upper_left_y, 0, -pixel_size_y))
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)  # Web Mercator
    dataset.SetProjection(srs.ExportToWkt())
    
    # Write image data
    for i in range(3):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(np.array(image)[:,:,i])
    
    dataset = None

def save_oemj_as_geotiff(polygon, filepath, zoom=16):
    """Download and save OpenEarthMap Japan imagery as GeoTIFF.
    
    Args:
        polygon (list): List of (lon, lat) coordinates defining region
        filepath (str): Output path for GeoTIFF
        zoom (int, optional): Zoom level for detail. Defaults to 16.
    """
    try:
        tiles, bounds = download_tiles(polygon, zoom)
        if not tiles:
            raise ValueError("No tiles were downloaded. Please check the polygon coordinates and zoom level.")

        composed_image = compose_image(tiles, bounds)
        cropped_image, bbox = crop_image(composed_image, polygon, bounds, zoom)
        save_as_geotiff(cropped_image, polygon, zoom, bbox, bounds, filepath)
        print(f"GeoTIFF saved as '{filepath}' in Web Mercator projection (EPSG:3857).")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check the polygon coordinates and zoom level, and ensure you have an active internet connection.")