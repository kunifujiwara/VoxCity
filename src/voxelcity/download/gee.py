import numpy as np
import rasterio
from affine import Affine
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ee
import geemap
from pyproj import CRS, Transformer
import rasterio
from pyproj.geod import Geod

from ..geo.utils import convert_format_lat_lon

def initialize_earth_engine():
    ee.Initialize()

def get_roi(input_coords):
    coords = convert_format_lat_lon(input_coords)
    return ee.Geometry.Polygon(coords)

def get_center_point(roi):
    center_point = roi.centroid()
    center_coords = center_point.coordinates().getInfo()
    return center_coords[0], center_coords[1]

def get_image_collection(collection_name, roi):
    collection = ee.ImageCollection(collection_name).filterBounds(roi)
    return collection.sort('system:time_start').first().clip(roi).unmask()

def save_geotiff(image, filename, resolution=1, scale=None, region=None):
    if scale and region:
        geemap.ee_export_image(image, filename=filename, scale=scale, region=region)
    else:
        geemap.ee_to_geotiff(image, filename, resolution=resolution, to_cog=True)

def get_dem_image(roi_buffered):
    dem = ee.Image('USGS/SRTMGL1_003')
    return dem.clip(roi_buffered)


# def get_grid_gee(tag, collection_name, coords, mesh_size, land_cover_classes=None, buffer_distance=None):
#     initialize_earth_engine()

#     roi = get_roi(coords)
#     center_lon, center_lat = get_center_point(roi)

#     if buffer_distance:
#         roi_buffered = roi.buffer(buffer_distance)
#         image = get_dem_image(roi_buffered)
#         save_geotiff(image, f"{tag}.tif", scale=30, region=roi_buffered)
#     else:
#         image = get_image_collection(collection_name, roi)
#         save_geotiff(image, f"{tag}.tif")

#     if tag == 'canopy_height':
#         grid = create_canopy_height_grid(f"{tag}.tif", mesh_size)
#         visualize_grid(grid, mesh_size, title=f'{tag.replace("_", " ").title()} Grid')
#     elif tag == 'land_cover':
#         grid = create_land_cover_grid(f"{tag}.tif", mesh_size, land_cover_classes)
#         color_map = {cls: [r/255, g/255, b/255] for (r,g,b), cls in land_cover_classes.items()}
#         # color_map['No Data'] = [0.5, 0.5, 0.5]
#         visualize_land_cover_grid(grid, mesh_size, color_map, land_cover_classes)
#         grid = convert_land_cover_array(grid, land_cover_classes)
#     elif tag == 'nasa_dem':
#         converted_coords = convert_format(coords)
#         roi_shapely = Polygon(converted_coords)
#         grid = create_dem_grid(f"{tag}.tif", mesh_size, roi_shapely)
#         visualize_grid(grid, mesh_size, title='Digital Elevation Model', cmap='terrain', label='Elevation (m)')

#     print(f"Resulting grid shape: {grid.shape}")

#     return grid