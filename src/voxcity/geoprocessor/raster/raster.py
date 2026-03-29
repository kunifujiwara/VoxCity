import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon
from affine import Affine
from pyproj import Transformer, CRS
import rasterio
from scipy.interpolate import griddata

from ..utils import initialize_geod
from .core import compute_cell_center_coords
from ...utils.orientation import ensure_orientation, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP


def create_height_grid_from_geotiff_polygon(
    tiff_path: str,
    mesh_size: float,
    polygon: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Create a height grid from a GeoTIFF file within a polygon boundary.

    Uses :func:`compute_cell_center_coords` so rotated rectangles are handled
    correctly.
    """
    cc = compute_cell_center_coords(polygon, mesh_size)
    nx, ny = cc["grid_size"]
    center_lons = cc["lons"].ravel()
    center_lats = cc["lats"].ravel()

    with rasterio.open(tiff_path) as src:
        img = src.read(1)

        # Transform cell centres to raster CRS if needed
        if src.crs and src.crs.to_epsg() != 4326:
            tfm = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            sx, sy = tfm.transform(center_lons, center_lats)
        else:
            sx, sy = center_lons.copy(), center_lats.copy()

        row, col = rasterio.transform.rowcol(src.transform, sx, sy)
        row, col = np.asarray(row), np.asarray(col)

        valid = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)

        grid = np.full(nx * ny, np.nan)
        grid[valid] = img[row[valid], col[valid]]
        grid = grid.reshape(nx, ny)

    return grid


def create_dem_grid_from_geotiff_polygon(tiff_path, mesh_size, rectangle_vertices, dem_interpolation=False):
    """
    Create a Digital Elevation Model (DEM) grid from a GeoTIFF within a polygon boundary.

    Uses :func:`compute_cell_center_coords` so rotated rectangles are handled
    correctly.
    """
    cc = compute_cell_center_coords(rectangle_vertices, mesh_size)
    nx, ny = cc["grid_size"]
    center_lons = cc["lons"]
    center_lats = cc["lats"]

    with rasterio.open(tiff_path) as src:
        dem = src.read(1)
        dem = np.where(dem < -1000, 0, dem)
        transform = src.transform
        src_crs = src.crs

        # Determine a UTM zone for metric interpolation
        clon = float(np.mean(center_lons))
        clat = float(np.mean(center_lats))
        zone = int((clon + 180.0) // 6) + 1
        epsg_metric = 32600 + zone if clat >= 0 else 32700 + zone
        crs_metric = CRS.from_epsg(epsg_metric)

        # Transform cell centres to UTM
        wgs84 = CRS.from_epsg(4326)
        to_metric = Transformer.from_crs(wgs84, crs_metric, always_xy=True)
        sample_x, sample_y = to_metric.transform(
            center_lons.ravel(), center_lats.ravel()
        )
        sample_x, sample_y = np.asarray(sample_x), np.asarray(sample_y)

        # Transform source raster pixel centres to the same UTM zone
        from_src_to_metric = Transformer.from_crs(src_crs, crs_metric, always_xy=True)
        rows_src, cols_src = np.meshgrid(
            range(dem.shape[0]), range(dem.shape[1]), indexing="ij"
        )
        orig_x, orig_y = rasterio.transform.xy(
            transform, rows_src.ravel(), cols_src.ravel()
        )
        orig_x, orig_y = from_src_to_metric.transform(orig_x, orig_y)

        points = np.column_stack((np.asarray(orig_x), np.asarray(orig_y)))
        values = dem.ravel()

        method = "cubic" if dem_interpolation else "nearest"
        grid = griddata(
            points,
            values,
            (sample_x.reshape(nx, ny), sample_y.reshape(nx, ny)),
            method=method,
        )

    return grid




