import numpy as np
import rasterio
from affine import Affine
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ee
import geemap
from collections import Counter
from scipy.interpolate import griddata
from pyproj import Transformer
from shapely.geometry import Polygon

def initialize_earth_engine():
    ee.Initialize()

def get_roi(coords):
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

def create_grid(tiff_path, mesh_size):
    with rasterio.open(tiff_path) as src:
        img = src.read(1)
        left, bottom, right, top = src.bounds
        num_cells_x = int((right - left) / mesh_size)
        num_cells_y = int((top - bottom) / mesh_size)
        new_affine = Affine(mesh_size, 0, left, 0, -mesh_size, top)

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

    return grid

def get_dominant_class(cell_data, land_cover_classes):
    if cell_data.size == 0:
        return 'No Data'
    pixel_tuples = [tuple(pixel) for pixel in cell_data.reshape(-1, 3)]
    class_counts = Counter(land_cover_classes.get(t, 'Unknown') for t in pixel_tuples)
    return class_counts.most_common(1)[0][0]

def create_land_cover_grid(tiff_path, mesh_size, land_cover_classes):
    with rasterio.open(tiff_path) as src:
        img = src.read((1,2,3))
        left, bottom, right, top = src.bounds
        num_cells_x = int((right - left) / mesh_size)
        num_cells_y = int((top - bottom) / mesh_size)
        new_affine = Affine(mesh_size, 0, left, 0, -mesh_size, top)

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

    return grid

def get_dem_image(roi_buffered):
    dem = ee.Image('USGS/SRTMGL1_003')
    return dem.clip(roi_buffered)
    
def create_dem_grid(tiff_path, mesh_size, roi_shapely):
    with rasterio.open(tiff_path) as src:
        dem = src.read(1)
        transform = src.transform
        crs = src.crs

        transformer = Transformer.from_crs(crs.to_string(), "EPSG:3857", always_xy=True)

        roi_bounds = roi_shapely.bounds
        roi_left, roi_bottom = transformer.transform(roi_bounds[0], roi_bounds[1])
        roi_right, roi_top = transformer.transform(roi_bounds[2], roi_bounds[3])

        roi_width_m = roi_right - roi_left
        roi_height_m = roi_top - roi_bottom
        num_cells_x = int(roi_width_m / mesh_size)
        num_cells_y = int(roi_height_m / mesh_size)

        x = np.linspace(roi_left, roi_right, num_cells_x, endpoint=False)
        y = np.linspace(roi_top, roi_bottom, num_cells_y, endpoint=False)
        xx, yy = np.meshgrid(x, y)

        rows, cols = np.mgrid[0:dem.shape[0], 0:dem.shape[1]]
        orig_lon, orig_lat = rasterio.transform.xy(transform, rows, cols)
        lons, lats = transformer.transform(xx, yy, direction='INVERSE')
        points = np.column_stack((np.ravel(orig_lon), np.ravel(orig_lat)))
        values = dem.ravel()
        grid = griddata(points, values, (lons, lats), method='linear')

    return grid

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
    im = plt.imshow(numeric_grid, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ticks=bounds[:-1] + 0.5)
    cbar.set_ticklabels(sorted_classes)
    plt.title(f'Land Use/Land Cover Grid (Mesh Size: {mesh_size}m)')
    plt.xlabel('Grid Cells (X)')
    plt.ylabel('Grid Cells (Y)')
    plt.show()

def get_grid(tag, collection_name, coords, mesh_size, land_cover_classes=None, buffer_distance=None):
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
        grid = create_grid(f"{tag}.tif", mesh_size)
        visualize_grid(grid, mesh_size, title=f'{tag.replace("_", " ").title()} Grid')
    elif tag == 'land_cover':
        grid = create_land_cover_grid(f"{tag}.tif", mesh_size, land_cover_classes)
        color_map = {cls: [r/255, g/255, b/255] for (r,g,b), cls in land_cover_classes.items()}
        color_map['No Data'] = [0.5, 0.5, 0.5]
        visualize_land_cover_grid(grid, mesh_size, color_map, land_cover_classes)
    elif tag == 'nasa_dem':
        roi_shapely = Polygon(coords)
        grid = create_dem_grid(f"{tag}.tif", mesh_size, roi_shapely)
        visualize_grid(grid, mesh_size, title='Digital Elevation Model', cmap='terrain', label='Elevation (m)')

    print(f"Resulting grid shape: {grid.shape}")

    return grid