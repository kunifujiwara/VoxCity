import numpy as np
import rasterio
from affine import Affine
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ee
import geemap
from collections import Counter
from scipy.interpolate import griddata
from pyproj import CRS, Transformer
from shapely.geometry import Polygon
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

def create_canopy_height_grid(tiff_path, mesh_size):
    with rasterio.open(tiff_path) as src:
        img = src.read(1)
        left, bottom, right, top = src.bounds
        src_crs = src.crs

        # Calculate width and height in meters
        if src_crs.to_epsg() == 3857:  # Web Mercator
            # Convert bounds to WGS84
            wgs84 = CRS.from_epsg(4326)
            transformer = Transformer.from_crs(src_crs, wgs84, always_xy=True)
            left_wgs84, bottom_wgs84 = transformer.transform(left, bottom)
            right_wgs84, top_wgs84 = transformer.transform(right, top)
        
            # Use geodesic calculations for accuracy
            geod = Geod(ellps="WGS84")
            _, _, width = geod.inv(left_wgs84, bottom_wgs84, right_wgs84, bottom_wgs84)
            _, _, height = geod.inv(left_wgs84, bottom_wgs84, left_wgs84, top_wgs84)
        else:
            # For other projections, assume units are already in meters
            width = right - left
            height = top - bottom

        # Display width and height in meters
        print(f"ROI Width: {width:.2f} meters")
        print(f"ROI Height: {height:.2f} meters")

        num_cells_x = int(width / mesh_size + 0.5)
        num_cells_y = int(height / mesh_size + 0.5)

        # Adjust mesh_size to fit the image exactly
        adjusted_mesh_size_x = (right - left) / num_cells_x
        adjusted_mesh_size_y = (top - bottom) / num_cells_y

        # Create a new affine transformation for the new grid
        new_affine = Affine(adjusted_mesh_size_x, 0, left, 0, -adjusted_mesh_size_y, top)

        cols, rows = np.meshgrid(np.arange(num_cells_x), np.arange(num_cells_y))
        xs, ys = new_affine * (cols, rows)
        xs_flat, ys_flat = xs.flatten(), ys.flatten()

        row, col = src.index(xs_flat, ys_flat)
        row, col = np.array(row), np.array(col)

        valid = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)
        row, col = row[valid], col[valid]

        grid = np.full((num_cells_y, num_cells_x), np.nan)
        flat_indices = np.ravel_multi_index((row, col), img.shape)
        np.put(grid, np.ravel_multi_index((rows.flatten()[valid], cols.flatten()[valid]), grid.shape), img.flat[flat_indices])

    return np.flipud(grid)

def rgb_distance(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2))**2))  
      
def get_nearest_class(pixel, land_cover_classes):
    distances = {class_name: rgb_distance(pixel, color) 
                 for color, class_name in land_cover_classes.items()}
    return min(distances, key=distances.get)

def get_dominant_class(cell_data, land_cover_classes):
    if cell_data.size == 0:
        return 'No Data'
    pixel_classes = [get_nearest_class(tuple(pixel), land_cover_classes) 
                     for pixel in cell_data.reshape(-1, 3)]
    class_counts = Counter(pixel_classes)
    return class_counts.most_common(1)[0][0]

def convert_land_cover_array(input_array, land_cover_classes):
    # Create a mapping of class names to integers
    class_to_int = {name: i for i, name in enumerate(land_cover_classes.values())}

    # Create a vectorized function to map string values to integers
    vectorized_map = np.vectorize(lambda x: class_to_int.get(x, -1))

    # Apply the mapping to the input array
    output_array = vectorized_map(input_array)

    return output_array
def create_land_cover_grid(tiff_path, mesh_size, land_cover_classes):
    with rasterio.open(tiff_path) as src:
        img = src.read((1,2,3))
        left, bottom, right, top = src.bounds
        src_crs = src.crs
        
        # Calculate width and height in meters
        if src_crs.to_epsg() == 3857:  # Web Mercator
            # Convert bounds to WGS84
            wgs84 = CRS.from_epsg(4326)
            transformer = Transformer.from_crs(src_crs, wgs84, always_xy=True)
            left_wgs84, bottom_wgs84 = transformer.transform(left, bottom)
            right_wgs84, top_wgs84 = transformer.transform(right, top)
        
            # Use geodesic calculations for accuracy
            geod = Geod(ellps="WGS84")
            _, _, width = geod.inv(left_wgs84, bottom_wgs84, right_wgs84, bottom_wgs84)
            _, _, height = geod.inv(left_wgs84, bottom_wgs84, left_wgs84, top_wgs84)
        else:
            # For other projections, assume units are already in meters
            width = right - left
            height = top - bottom
        
        # Display width and height in meters
        print(f"ROI Width: {width:.2f} meters")
        print(f"ROI Height: {height:.2f} meters")
        
        num_cells_x = int(width / mesh_size + 0.5)
        num_cells_y = int(height / mesh_size + 0.5)
        
        # Adjust mesh_size to fit the image exactly
        adjusted_mesh_size_x = (right - left) / num_cells_x
        adjusted_mesh_size_y = (top - bottom) / num_cells_y
        
        # Create a new affine transformation for the new grid
        new_affine = Affine(adjusted_mesh_size_x, 0, left, 0, -adjusted_mesh_size_y, top)
        
        cols, rows = np.meshgrid(np.arange(num_cells_x), np.arange(num_cells_y))
        xs, ys = new_affine * (cols, rows)
        xs_flat, ys_flat = xs.flatten(), ys.flatten()
        
        row, col = src.index(xs_flat, ys_flat)
        row, col = np.array(row), np.array(col)
        valid = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)
        row, col = row[valid], col[valid]
        
        grid = np.full((num_cells_y, num_cells_x), 'No Data', dtype=object)
        
        for i, (r, c) in enumerate(zip(row, col)):
            cell_data = img[:, r, c]
            dominant_class = get_dominant_class(cell_data, land_cover_classes)
            grid_row, grid_col = np.unravel_index(i, (num_cells_y, num_cells_x))
            grid[grid_row, grid_col] = dominant_class
    
    return np.flipud(grid)

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

def visualize_grid(grid, mesh_size, title, cmap='viridis', label='Value'):
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap=cmap)
    plt.colorbar(label=label)
    plt.title(f'{title} (Mesh Size: {mesh_size}m)')
    plt.xlabel('Grid Cells (X)')
    plt.ylabel('Grid Cells (Y)')
    plt.show()

def visualize_land_cover_grid(grid, mesh_size, color_map, land_cover_classes):
    all_classes = list(land_cover_classes.values()) + ['No Data']
    for cls in all_classes:
        if cls not in color_map:
            color_map[cls] = [0.5, 0.5, 0.5]

    sorted_classes = sorted(all_classes)
    colors = [color_map[cls] for cls in sorted_classes]
    cmap = mcolors.ListedColormap(colors)

    bounds = np.arange(len(sorted_classes) + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    class_to_num = {cls: i for i, cls in enumerate(sorted_classes)}
    numeric_grid = np.vectorize(class_to_num.get)(grid)

    plt.figure(figsize=(12, 12))
    im = plt.imshow(numeric_grid, cmap=cmap, norm=norm, interpolation='nearest')
    cbar = plt.colorbar(im, ticks=bounds[:-1] + 0.5)
    cbar.set_ticklabels(sorted_classes)
    plt.title(f'Land Use/Land Cover Grid (Mesh Size: {mesh_size}m)')
    plt.xlabel('Grid Cells (X)')
    plt.ylabel('Grid Cells (Y)')
    plt.show()

def get_grid_gee(tag, collection_name, coords, mesh_size, land_cover_classes=None, buffer_distance=None):
    initialize_earth_engine()

    roi = get_roi(coords)
    center_lon, center_lat = get_center_point(roi)

    if buffer_distance:
        roi_buffered = roi.buffer(buffer_distance)
        image = get_dem_image(roi_buffered)
        save_geotiff(image, f"{tag}.tif", scale=30, region=roi_buffered)
    else:
        image = get_image_collection(collection_name, roi)
        save_geotiff(image, f"{tag}.tif")

    if tag == 'canopy_height':
        grid = create_canopy_height_grid(f"{tag}.tif", mesh_size)
        visualize_grid(grid, mesh_size, title=f'{tag.replace("_", " ").title()} Grid')
    elif tag == 'land_cover':
        grid = create_land_cover_grid(f"{tag}.tif", mesh_size, land_cover_classes)
        color_map = {cls: [r/255, g/255, b/255] for (r,g,b), cls in land_cover_classes.items()}
        # color_map['No Data'] = [0.5, 0.5, 0.5]
        visualize_land_cover_grid(grid, mesh_size, color_map, land_cover_classes)
        grid = convert_land_cover_array(grid, land_cover_classes)
    elif tag == 'nasa_dem':
        converted_coords = convert_format(coords)
        roi_shapely = Polygon(converted_coords)
        grid = create_dem_grid(f"{tag}.tif", mesh_size, roi_shapely)
        visualize_grid(grid, mesh_size, title='Digital Elevation Model', cmap='terrain', label='Elevation (m)')

    print(f"Resulting grid shape: {grid.shape}")

    return grid