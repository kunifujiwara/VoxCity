import requests
from PIL import Image, ImageDraw
from io import BytesIO
import math
from shapely.geometry import Polygon, box
import numpy as np
from osgeo import gdal, osr
import pyproj

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def download_tiles(polygon, zoom):
    min_lat = min(p[0] for p in polygon)
    max_lat = max(p[0] for p in polygon)
    min_lon = min(p[1] for p in polygon)
    max_lon = max(p[1] for p in polygon)

    min_x, max_y = map(math.floor, deg2num(max_lat, min_lon, zoom))
    max_x, min_y = map(math.ceil, deg2num(min_lat, max_lon, zoom))

    print(f"Tile coordinates: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")

    tiles = {}
    for x in range(min(min_x, max_x), max(min_x, max_x) + 1):
        for y in range(min(min_y, max_y), max(min_y, max_y) + 1):
            url = f"https://www.open-earth-map.org/demo/Japan/{zoom}/{x}/{y}.png"
            print(f"Downloading tile: {url}")
            response = requests.get(url)
            if response.status_code == 200:
                tiles[(x, y)] = Image.open(BytesIO(response.content))
            else:
                print(f"Failed to download tile: {url}")

    return tiles, (min(min_x, max_x), min(min_y, max_y), max(min_x, max_x), max(min_y, max_y))

def compose_image(tiles, bounds):
    min_x, min_y, max_x, max_y = bounds
    width = abs(max_x - min_x + 1) * 256
    height = abs(max_y - min_y + 1) * 256
    print(f"Composing image with dimensions: {width}x{height}")
    result = Image.new('RGB', (width, height))
    for (x, y), tile in tiles.items():
        result.paste(tile, ((x - min_x) * 256, (y - min_y) * 256))
    return result

def crop_image(image, polygon, bounds, zoom):
    min_x, min_y, max_x, max_y = bounds
    img_width, img_height = image.size

    polygon_pixels = []
    for lat, lon in polygon:
        x, y = deg2num(lat, lon, zoom)
        px = (x - min_x) * 256
        py = (y - min_y) * 256
        polygon_pixels.append((px, py))

    mask = Image.new('L', (img_width, img_height), 0)
    ImageDraw.Draw(mask).polygon(polygon_pixels, outline=255, fill=255)

    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("The polygon does not intersect with the downloaded tiles.")

    expanded_bbox = (max(0, bbox[0] - 1),
                     max(0, bbox[1] - 1),
                     min(img_width, bbox[2] + 1),
                     min(img_height, bbox[3] + 1))

    cropped = Image.composite(image, Image.new('RGB', image.size, (255, 255, 255)), mask)
    return cropped.crop(expanded_bbox), expanded_bbox

def save_as_geotiff(image, polygon, zoom, bbox, bounds, output_path):
    min_x, min_y, max_x, max_y = bounds

    upper_left_lat, upper_left_lon = num2deg(min_x + bbox[0]/256, min_y + bbox[1]/256, zoom)
    lower_right_lat, lower_right_lon = num2deg(min_x + bbox[2]/256, min_y + bbox[3]/256, zoom)

    # Create transformation from WGS84 to Web Mercator
    wgs84 = pyproj.CRS('EPSG:4326')
    web_mercator = pyproj.CRS('EPSG:3857')
    transformer = pyproj.Transformer.from_crs(wgs84, web_mercator, always_xy=True)

    # Transform coordinates to Web Mercator
    upper_left_x, upper_left_y = transformer.transform(upper_left_lon, upper_left_lat)
    lower_right_x, lower_right_y = transformer.transform(lower_right_lon, lower_right_lat)

    pixel_size_x = (lower_right_x - upper_left_x) / image.width
    pixel_size_y = (upper_left_y - lower_right_y) / image.height

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, image.width, image.height, 3, gdal.GDT_Byte)

    dataset.SetGeoTransform((upper_left_x, pixel_size_x, 0, upper_left_y, 0, -pixel_size_y))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)  # Web Mercator
    dataset.SetProjection(srs.ExportToWkt())

    for i in range(3):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(np.array(image)[:,:,i])

    dataset = None