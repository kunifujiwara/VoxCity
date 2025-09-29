"""
Module for downloading and processing OpenEarthMap Japan (OEMJ) satellite imagery.

This module provides functionality to download, compose, crop and save satellite imagery tiles
from OpenEarthMap Japan as georeferenced GeoTIFF files. It handles coordinate conversions between
latitude/longitude and tile coordinates, downloads tiles within a polygon region, and saves the
final image with proper geospatial metadata.

Key Features:
    - Convert between geographic (lat/lon) and tile coordinates
    - Download satellite imagery tiles for a specified region
    - Compose multiple tiles into a single image
    - Crop images to a specified polygon boundary
    - Save results as georeferenced GeoTIFF files

Example Usage:
    polygon = [(139.7, 35.6), (139.8, 35.6), (139.8, 35.7), (139.7, 35.7)]  # Tokyo area
    save_oemj_as_geotiff(polygon, "tokyo_satellite.tiff", zoom=16)
"""

import requests
import os
from PIL import Image, ImageDraw
from io import BytesIO
import math
import numpy as np
from osgeo import gdal, osr
import pyproj

def deg2num(lon_deg, lat_deg, zoom):
    """Convert longitude/latitude coordinates to tile coordinates using Web Mercator projection.
    
    The function converts geographic coordinates to tile coordinates using the standard
    Web Mercator tiling scheme (XYZ). The resulting coordinates can be used to identify
    and download specific map tiles.
    
    Args:
        lon_deg (float): Longitude in degrees (-180 to 180)
        lat_deg (float): Latitude in degrees (-90 to 90)
        zoom (int): Zoom level (0-20, where 0 is most zoomed out)
        
    Returns:
        tuple: (x, y) tile coordinates as floats
        
    Example:
        >>> x, y = deg2num(139.7, 35.6, 16)  # Tokyo coordinates
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    """Convert tile coordinates back to longitude/latitude coordinates.
    
    This is the inverse operation of deg2num(). It converts tile coordinates
    back to geographic coordinates using the Web Mercator projection.
    
    Args:
        xtile (float): X tile coordinate
        ytile (float): Y tile coordinate
        zoom (int): Zoom level (0-20)
        
    Returns:
        tuple: (longitude, latitude) in degrees
        
    Example:
        >>> lon, lat = num2deg(29326, 13249, 15)  # Sample tile coordinates
    """
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lon_deg, lat_deg)

def download_tiles(polygon, zoom, *, ssl_verify=True, allow_insecure_ssl=False, allow_http_fallback=False, timeout_s=30):
    """Download satellite imagery tiles covering a polygon region.
    
    Downloads all tiles that intersect with the given polygon at the specified zoom level
    from the OpenEarthMap Japan server. The function calculates the minimum bounding box
    that contains the polygon and downloads all tiles within that box.
    
    Args:
        polygon (list): List of (lon, lat) tuples defining the region vertices in clockwise
                       or counterclockwise order
        zoom (int): Zoom level for tile detail (recommended range: 14-18)
        
    Returns:
        tuple: (
            tiles: dict mapping (x,y) tile coordinates to PIL Image objects,
            bounds: tuple of (min_x, min_y, max_x, max_y) tile coordinates
        )
        
    Note:
        Higher zoom levels provide more detail but require downloading more tiles.
        The function will print progress messages during download.
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
            # Try secure HTTPS first with user-provided verification option
            content = None
            try:
                resp = requests.get(url, timeout=timeout_s, verify=ssl_verify)
                if resp.status_code == 200:
                    content = resp.content
                else:
                    print(f"Failed to download tile (status {resp.status_code}): {url}")
            except requests.exceptions.SSLError:
                # Optionally retry with certificate verification disabled
                if allow_insecure_ssl:
                    try:
                        resp = requests.get(url, timeout=timeout_s, verify=False)
                        if resp.status_code == 200:
                            content = resp.content
                        else:
                            print(f"Failed to download tile (status {resp.status_code}) with insecure SSL: {url}")
                    except requests.exceptions.RequestException as e:
                        # Optionally try HTTP fallback
                        if allow_http_fallback and url.lower().startswith("https://"):
                            http_url = "http://" + url.split("://", 1)[1]
                            try:
                                resp = requests.get(http_url, timeout=timeout_s)
                                if resp.status_code == 200:
                                    content = resp.content
                                else:
                                    print(f"Failed to download tile over HTTP (status {resp.status_code}): {http_url}")
                            except requests.exceptions.RequestException as e2:
                                print(f"HTTP fallback failed for tile: {http_url} ({e2})")
                        else:
                            print(f"SSL error downloading tile: {url} ({e})")
                else:
                    if allow_http_fallback and url.lower().startswith("https://"):
                        http_url = "http://" + url.split("://", 1)[1]
                        try:
                            resp = requests.get(http_url, timeout=timeout_s)
                            if resp.status_code == 200:
                                content = resp.content
                            else:
                                print(f"Failed to download tile over HTTP (status {resp.status_code}): {http_url}")
                        except requests.exceptions.RequestException as e:
                            print(f"HTTP fallback failed for tile: {http_url} ({e})")
                    else:
                        print(f"SSL error downloading tile: {url}")
            except requests.exceptions.RequestException as e:
                # Network error (timeout/connection). Try HTTP if allowed.
                if allow_http_fallback and url.lower().startswith("https://"):
                    http_url = "http://" + url.split("://", 1)[1]
                    try:
                        resp = requests.get(http_url, timeout=timeout_s)
                        if resp.status_code == 200:
                            content = resp.content
                        else:
                            print(f"Failed to download tile over HTTP (status {resp.status_code}): {http_url}")
                    except requests.exceptions.RequestException as e2:
                        print(f"HTTP fallback failed for tile: {http_url} ({e2})")
                else:
                    print(f"Error downloading tile: {url} ({e})")

            if content is not None:
                try:
                    tiles[(x, y)] = Image.open(BytesIO(content))
                except Exception as e:
                    print(f"Error decoding tile image for {url}: {e}")
    
    return tiles, (min(min_x, max_x), min(min_y, max_y), max(min_x, max_x), max(min_y, max_y))

def compose_image(tiles, bounds):
    """Compose downloaded tiles into a single continuous image.
    
    Takes individual map tiles and combines them into a single large image based on
    their relative positions. The tiles are placed according to their x,y coordinates
    within the bounds.
    
    Args:
        tiles (dict): Mapping of (x,y) coordinates to tile Image objects
        bounds (tuple): (min_x, min_y, max_x, max_y) tile coordinate bounds
        
    Returns:
        Image: Composed PIL Image with dimensions (width x height) where:
              width = (max_x - min_x + 1) * 256
              height = (max_y - min_y + 1) * 256
              
    Note:
        Each tile is assumed to be 256x256 pixels, which is standard for web maps.
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
    """Crop composed image to the exact polygon boundary.
    
    Creates a mask from the polygon coordinates and uses it to crop the image,
    removing areas outside the polygon of interest. The polygon coordinates are
    converted from geographic coordinates to pixel coordinates in the image space.
    
    Args:
        image (Image): PIL Image to crop
        polygon (list): List of (lon, lat) coordinates defining the boundary
        bounds (tuple): (min_x, min_y, max_x, max_y) tile bounds
        zoom (int): Zoom level used for coordinate conversion
        
    Returns:
        tuple: (
            cropped Image: PIL Image cropped to polygon boundary,
            bbox: tuple of (left, upper, right, lower) pixel coordinates of bounding box
        )
        
    Raises:
        ValueError: If the polygon does not intersect with the downloaded tiles
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
    """Save cropped image as a georeferenced GeoTIFF file.
    
    Converts the image to a GeoTIFF format with proper spatial reference information
    using the Web Mercator projection (EPSG:3857). The function handles coordinate
    transformation and sets up the necessary geospatial metadata.
    
    Args:
        image (Image): PIL Image to save
        polygon (list): List of (lon, lat) coordinates
        zoom (int): Zoom level used for coordinate calculations
        bbox (tuple): Bounding box of cropped image in pixels (left, upper, right, lower)
        bounds (tuple): (min_x, min_y, max_x, max_y) tile bounds
        output_path (str): Path where the GeoTIFF will be saved
        
    Note:
        The output GeoTIFF will have 3 bands (RGB) and use the Web Mercator
        projection (EPSG:3857) for compatibility with most GIS software.
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
    
    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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

def save_oemj_as_geotiff(polygon, filepath, zoom=16, *, ssl_verify=True, allow_insecure_ssl=False, allow_http_fallback=False, timeout_s=30):
    """Download and save OpenEarthMap Japan imagery as a georeferenced GeoTIFF file.
    
    This is the main function that orchestrates the entire process of downloading,
    processing, and saving satellite imagery for a specified region.
    
    Args:
        polygon (list): List of (lon, lat) coordinates defining the region to download.
                       Must be in clockwise or counterclockwise order.
        filepath (str): Output path for the GeoTIFF file
        zoom (int, optional): Zoom level for detail. Defaults to 16.
                            - 14: ~9.5m/pixel
                            - 15: ~4.8m/pixel
                            - 16: ~2.4m/pixel
                            - 17: ~1.2m/pixel
                            - 18: ~0.6m/pixel
    
    Example:
        >>> polygon = [
                (139.7, 35.6),  # Bottom-left
                (139.8, 35.6),  # Bottom-right
                (139.8, 35.7),  # Top-right
                (139.7, 35.7)   # Top-left
            ]
        >>> save_oemj_as_geotiff(polygon, "tokyo_area.tiff", zoom=16)
    
    Note:
        - Higher zoom levels provide better resolution but require more storage
        - The polygon should be relatively small to avoid memory issues
        - The output GeoTIFF will be in Web Mercator projection (EPSG:3857)
    """
    try:
        tiles, bounds = download_tiles(
            polygon,
            zoom,
            ssl_verify=ssl_verify,
            allow_insecure_ssl=allow_insecure_ssl,
            allow_http_fallback=allow_http_fallback,
            timeout_s=timeout_s,
        )
        if not tiles:
            raise ValueError("No tiles were downloaded. Please check the polygon coordinates and zoom level.")

        composed_image = compose_image(tiles, bounds)
        cropped_image, bbox = crop_image(composed_image, polygon, bounds, zoom)
        save_as_geotiff(cropped_image, polygon, zoom, bbox, bounds, filepath)
        print(f"GeoTIFF saved as '{filepath}' in Web Mercator projection (EPSG:3857).")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check the polygon coordinates and zoom level, and ensure you have an active internet connection.")