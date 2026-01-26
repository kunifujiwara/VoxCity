import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon
from affine import Affine
from pyproj import Geod, Transformer, CRS
import rasterio
from scipy.interpolate import griddata


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

        geod = Geod(ellps="WGS84")
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

    return np.flipud(grid)


def create_dem_grid_from_geotiff_polygon(tiff_path, mesh_size, rectangle_vertices, dem_interpolation=False):
    """
    Create a Digital Elevation Model (DEM) grid from a GeoTIFF within a polygon boundary.
    
    Optimized to use windowed reading to avoid loading the entire raster into memory.
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from rasterio.windows import from_bounds
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from ..utils import convert_format_lat_lon

    converted_coords = convert_format_lat_lon(rectangle_vertices)
    roi_shapely = ShapelyPolygon(converted_coords)

    with rasterio.open(tiff_path) as src:
        src_crs = src.crs
        
        # Transform ROI bounds to source CRS for windowed reading
        wgs84 = CRS.from_epsg(4326)
        if src_crs.to_epsg() != 4326:
            transformer_to_src = Transformer.from_crs(wgs84, src_crs, always_xy=True)
        else:
            transformer_to_src = None
        
        roi_bounds = roi_shapely.bounds  # (minx, miny, maxx, maxy) in WGS84
        
        if transformer_to_src:
            # Transform ROI corners to source CRS
            src_left, src_bottom = transformer_to_src.transform(roi_bounds[0], roi_bounds[1])
            src_right, src_top = transformer_to_src.transform(roi_bounds[2], roi_bounds[3])
        else:
            src_left, src_bottom, src_right, src_top = roi_bounds
        
        # Add buffer (10% of extent) to avoid edge effects
        buffer_x = (src_right - src_left) * 0.1
        buffer_y = (src_top - src_bottom) * 0.1
        src_left -= buffer_x
        src_right += buffer_x
        src_bottom -= buffer_y
        src_top += buffer_y
        
        # Clip to raster bounds
        raster_bounds = src.bounds
        src_left = max(src_left, raster_bounds.left)
        src_right = min(src_right, raster_bounds.right)
        src_bottom = max(src_bottom, raster_bounds.bottom)
        src_top = min(src_top, raster_bounds.top)
        
        # Calculate window for reading only the needed portion
        try:
            window = from_bounds(src_left, src_bottom, src_right, src_top, src.transform)
            # Read only the windowed portion
            dem = src.read(1, window=window)
            window_transform = src.window_transform(window)
        except Exception:
            # Fallback to reading entire raster if window calculation fails
            dem = src.read(1)
            window_transform = src.transform
        
        dem = np.where(dem < -1000, 0, dem)
        
        # Calculate output grid size in meters using geodetic distance
        geod = Geod(ellps="WGS84")
        _, _, roi_width_m = geod.inv(roi_bounds[0], roi_bounds[1], roi_bounds[2], roi_bounds[1])
        _, _, roi_height_m = geod.inv(roi_bounds[0], roi_bounds[1], roi_bounds[0], roi_bounds[3])
        
        num_cells_x = int(roi_width_m / mesh_size + 0.5)
        num_cells_y = int(roi_height_m / mesh_size + 0.5)
        
        # Reproject using rasterio's efficient reproject function
        dst_crs = CRS.from_epsg(3857)  # Web Mercator for output
        
        # Transform ROI to 3857 for output bounds
        transformer_to_3857 = Transformer.from_crs(wgs84, dst_crs, always_xy=True)
        roi_left_3857, roi_bottom_3857 = transformer_to_3857.transform(roi_bounds[0], roi_bounds[1])
        roi_right_3857, roi_top_3857 = transformer_to_3857.transform(roi_bounds[2], roi_bounds[3])
        
        # Create output transform
        dst_transform = rasterio.transform.from_bounds(
            roi_left_3857, roi_bottom_3857, roi_right_3857, roi_top_3857,
            num_cells_x, num_cells_y
        )
        
        # Reproject to output grid
        grid = np.zeros((num_cells_y, num_cells_x), dtype=np.float32)
        resampling = Resampling.cubic if dem_interpolation else Resampling.nearest
        
        reproject(
            source=dem,
            destination=grid,
            src_transform=window_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )

    return np.flipud(grid)




