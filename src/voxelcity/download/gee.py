import numpy as np
import rasterio
from affine import Affine
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ee
import geemap
from scipy.interpolate import griddata
from pyproj import CRS, Transformer
import rasterio
from pyproj.geod import Geod


def initialize_earth_engine():
    ee.Initialize()

def convert_format(input_coords):
    # Convert input to the desired output format
    output_coords = [[coord[1], coord[0]] for coord in input_coords]

    # Add the first point to the end to close the polygon
    output_coords.append(output_coords[0])

    return output_coords

def get_roi(input_coords):
    coords = convert_format(input_coords)
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
    
def create_dem_grid(tiff_path, mesh_size, roi_shapely):
    with rasterio.open(tiff_path) as src:
        dem = src.read(1)
        transform = src.transform
        src_crs = src.crs

        # Ensure we're working with EPSG:3857
        if src_crs.to_epsg() != 3857:
            transformer_to_3857 = Transformer.from_crs(src_crs, CRS.from_epsg(3857), always_xy=True)
        else:
            transformer_to_3857 = lambda x, y: (x, y)

        # Transform ROI bounds to EPSG:3857
        roi_bounds = roi_shapely.bounds
        roi_left, roi_bottom = transformer_to_3857.transform(roi_bounds[0], roi_bounds[1])
        roi_right, roi_top = transformer_to_3857.transform(roi_bounds[2], roi_bounds[3])

        # Calculate width and height in meters using geodesic methods
        wgs84 = CRS.from_epsg(4326)
        transformer_to_wgs84 = Transformer.from_crs(CRS.from_epsg(3857), wgs84, always_xy=True)
        roi_left_wgs84, roi_bottom_wgs84 = transformer_to_wgs84.transform(roi_left, roi_bottom)
        roi_right_wgs84, roi_top_wgs84 = transformer_to_wgs84.transform(roi_right, roi_top)

        geod = Geod(ellps="WGS84")
        _, _, roi_width_m = geod.inv(roi_left_wgs84, roi_bottom_wgs84, roi_right_wgs84, roi_bottom_wgs84)
        _, _, roi_height_m = geod.inv(roi_left_wgs84, roi_bottom_wgs84, roi_left_wgs84, roi_top_wgs84)

        # Display width and height in meters
        print(f"ROI Width: {roi_width_m:.2f} meters")
        print(f"ROI Height: {roi_height_m:.2f} meters")

        num_cells_x = int(roi_width_m / mesh_size + 0.5)
        num_cells_y = int(roi_height_m / mesh_size + 0.5)

        # # Adjust mesh_size to fit the ROI exactly
        # adjusted_mesh_size_x = roi_width_m / num_cells_x
        # adjusted_mesh_size_y = roi_height_m / num_cells_y

        # Create grid in EPSG:3857
        x = np.linspace(roi_left, roi_right, num_cells_x, endpoint=False)
        y = np.linspace(roi_top, roi_bottom, num_cells_y, endpoint=False)
        xx, yy = np.meshgrid(x, y)

        # Transform original DEM coordinates to EPSG:3857
        rows, cols = np.meshgrid(range(dem.shape[0]), range(dem.shape[1]), indexing='ij')
        orig_x, orig_y = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
        orig_x, orig_y = transformer_to_3857.transform(orig_x, orig_y)

        # Interpolate DEM values onto new grid
        points = np.column_stack((orig_x, orig_y))
        values = dem.ravel()
        grid = griddata(points, values, (xx, yy), method='cubic')

    return np.flipud(grid)

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