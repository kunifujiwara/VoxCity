import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon
from affine import Affine
from pyproj import Transformer, CRS
import rasterio
from scipy.interpolate import griddata

from ..utils import initialize_geod
from ...utils.orientation import ensure_orientation, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP


def create_height_grid_from_geotiff_polygon(
    tiff_path: str,
    mesh_size: float,
    polygon: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Create a height grid from a GeoTIFF file within a polygon boundary.
    """
    with rasterio.open(tiff_path) as src:
        img = src.read(1)
        left, bottom, right, top = src.bounds

        poly = Polygon(polygon)
        left_wgs84, bottom_wgs84, right_wgs84, top_wgs84 = poly.bounds

        geod = initialize_geod()
        _, _, width = geod.inv(left_wgs84, bottom_wgs84, right_wgs84, bottom_wgs84)
        _, _, height = geod.inv(left_wgs84, bottom_wgs84, left_wgs84, top_wgs84)

        num_cells_x = int(width / mesh_size + 0.5)
        num_cells_y = int(height / mesh_size + 0.5)

        adjusted_mesh_size_x = (right - left) / num_cells_x
        adjusted_mesh_size_y = (top - bottom) / num_cells_y

        new_affine = Affine(adjusted_mesh_size_x, 0, left, 0, -adjusted_mesh_size_y, top)

        cols, rows = np.meshgrid(np.arange(num_cells_x), np.arange(num_cells_y))
        xs, ys = new_affine * (cols, rows)
        xs_flat, ys_flat = xs.flatten(), ys.flatten()

        row, col = rasterio.transform.rowcol(src.transform, xs_flat, ys_flat)
        row, col = np.array(row), np.array(col)

        valid = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)
        row, col = row[valid], col[valid]

        grid = np.full((num_cells_y, num_cells_x), np.nan)
        flat_indices = np.ravel_multi_index((row, col), img.shape)
        np.put(grid, np.ravel_multi_index((rows.flatten()[valid], cols.flatten()[valid]), grid.shape), img.flat[flat_indices])

    return ensure_orientation(grid, ORIENTATION_SOUTH_UP, ORIENTATION_NORTH_UP)


def create_dem_grid_from_geotiff_polygon(tiff_path, mesh_size, rectangle_vertices, dem_interpolation=False):
    """
    Create a Digital Elevation Model (DEM) grid from a GeoTIFF within a polygon boundary.
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from ..utils import convert_format_lat_lon

    converted_coords = convert_format_lat_lon(rectangle_vertices)
    roi_shapely = ShapelyPolygon(converted_coords)

    with rasterio.open(tiff_path) as src:
        dem = src.read(1)
        dem = np.where(dem < -1000, 0, dem)
        transform = src.transform
        src_crs = src.crs

        # Use UTM instead of Web Mercator for metric grid spacing
        # to avoid Mercator distortion at high latitudes.
        centroid = roi_shapely.centroid
        lon_c, lat_c = float(centroid.x), float(centroid.y)
        zone = int((lon_c + 180.0) // 6) + 1
        epsg_metric = 32600 + zone if lat_c >= 0 else 32700 + zone
        crs_metric = CRS.from_epsg(epsg_metric)

        if src_crs.to_epsg() != epsg_metric:
            transformer_to_metric = Transformer.from_crs(src_crs, crs_metric, always_xy=True)
        else:
            transformer_to_metric = lambda x, y: (x, y)

        roi_bounds = roi_shapely.bounds
        roi_left, roi_bottom = transformer_to_metric.transform(roi_bounds[0], roi_bounds[1])
        roi_right, roi_top = transformer_to_metric.transform(roi_bounds[2], roi_bounds[3])

        wgs84 = CRS.from_epsg(4326)
        transformer_to_wgs84 = Transformer.from_crs(crs_metric, wgs84, always_xy=True)
        roi_left_wgs84, roi_bottom_wgs84 = transformer_to_wgs84.transform(roi_left, roi_bottom)
        roi_right_wgs84, roi_top_wgs84 = transformer_to_wgs84.transform(roi_right, roi_top)

        geod = initialize_geod()
        _, _, roi_width_m = geod.inv(roi_left_wgs84, roi_bottom_wgs84, roi_right_wgs84, roi_bottom_wgs84)
        _, _, roi_height_m = geod.inv(roi_left_wgs84, roi_bottom_wgs84, roi_left_wgs84, roi_top_wgs84)

        num_cells_x = int(roi_width_m / mesh_size + 0.5)
        num_cells_y = int(roi_height_m / mesh_size + 0.5)

        x = np.linspace(roi_left, roi_right, num_cells_x, endpoint=False)
        y = np.linspace(roi_top, roi_bottom, num_cells_y, endpoint=False)
        xx, yy = np.meshgrid(x, y)

        rows, cols = np.meshgrid(range(dem.shape[0]), range(dem.shape[1]), indexing='ij')
        orig_x, orig_y = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
        orig_x, orig_y = transformer_to_metric.transform(orig_x, orig_y)

        points = np.column_stack((orig_x, orig_y))
        values = dem.ravel()
        if dem_interpolation:
            grid = griddata(points, values, (xx, yy), method='cubic')
        else:
            grid = griddata(points, values, (xx, yy), method='nearest')

    return np.flipud(grid)




