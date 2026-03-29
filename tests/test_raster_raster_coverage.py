"""Tests for geoprocessor/raster/raster.py to improve coverage.

Both functions (create_height_grid_from_geotiff_polygon,
create_dem_grid_from_geotiff_polygon) require a real GeoTIFF on disk.
We create minimal in-memory GeoTIFFs with rasterio so no external data is
needed.
"""

import numpy as np
import os
import tempfile
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_geotiff(path, data, bounds, crs="EPSG:4326"):
    """Write a single-band float32 GeoTIFF covering *bounds* (left,bottom,right,top)."""
    rows, cols = data.shape
    transform = from_bounds(*bounds, cols, rows)
    with rasterio.open(
        path, "w", driver="GTiff", height=rows, width=cols, count=1,
        dtype=data.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def height_tiff(tmp_path):
    """Create a small GeoTIFF with a gradient."""
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    bounds = (139.756, 35.671, 139.758, 35.673)
    path = str(tmp_path / "height.tif")
    _write_geotiff(path, data, bounds)
    return path, bounds


@pytest.fixture
def dem_tiff(tmp_path):
    """Create a small DEM GeoTIFF."""
    data = np.random.default_rng(0).uniform(10, 50, (20, 20)).astype(np.float32)
    bounds = (139.755, 35.670, 139.759, 35.674)
    path = str(tmp_path / "dem.tif")
    _write_geotiff(path, data, bounds)
    return path, bounds


# ---------------------------------------------------------------------------
# create_height_grid_from_geotiff_polygon
# ---------------------------------------------------------------------------

class TestCreateHeightGridFromGeotiff:
    def test_basic(self, height_tiff):
        from voxcity.geoprocessor.raster.raster import create_height_grid_from_geotiff_polygon
        path, bounds = height_tiff
        polygon = [
            (bounds[0], bounds[1]),
            (bounds[0], bounds[3]),
            (bounds[2], bounds[3]),
            (bounds[2], bounds[1]),
        ]
        grid = create_height_grid_from_geotiff_polygon(path, 50, polygon)
        assert grid.ndim == 2
        assert grid.shape[0] > 0 and grid.shape[1] > 0

    def test_subset_polygon(self, height_tiff):
        """Polygon smaller than the raster extent."""
        from voxcity.geoprocessor.raster.raster import create_height_grid_from_geotiff_polygon
        path, bounds = height_tiff
        polygon = [
            (139.7565, 35.6715),
            (139.7565, 35.6725),
            (139.7575, 35.6725),
            (139.7575, 35.6715),
        ]
        grid = create_height_grid_from_geotiff_polygon(path, 30, polygon)
        assert grid.ndim == 2


class TestCreateDemGridFromGeotiff:
    def test_basic(self, dem_tiff):
        from voxcity.geoprocessor.raster.raster import create_dem_grid_from_geotiff_polygon
        path, bounds = dem_tiff
        polygon = [
            (bounds[0] + 0.0005, bounds[1] + 0.0005),
            (bounds[0] + 0.0005, bounds[3] - 0.0005),
            (bounds[2] - 0.0005, bounds[3] - 0.0005),
            (bounds[2] - 0.0005, bounds[1] + 0.0005),
        ]
        grid = create_dem_grid_from_geotiff_polygon(path, 50, polygon)
        assert grid.ndim == 2
        assert not np.all(np.isnan(grid))

    def test_with_interpolation(self, dem_tiff):
        from voxcity.geoprocessor.raster.raster import create_dem_grid_from_geotiff_polygon
        path, bounds = dem_tiff
        polygon = [
            (bounds[0] + 0.0005, bounds[1] + 0.0005),
            (bounds[0] + 0.0005, bounds[3] - 0.0005),
            (bounds[2] - 0.0005, bounds[3] - 0.0005),
            (bounds[2] - 0.0005, bounds[1] + 0.0005),
        ]
        grid = create_dem_grid_from_geotiff_polygon(path, 50, polygon, dem_interpolation=True)
        assert grid.ndim == 2

    def test_projected_crs_tiff(self, tmp_path):
        """GeoTIFF in a projected CRS (Web Mercator — not matching UTM)."""
        from voxcity.geoprocessor.raster.raster import create_dem_grid_from_geotiff_polygon
        data = np.ones((10, 10), dtype=np.float32) * 25.0
        # Web Mercator bounds covering approx Tokyo area
        bounds = (15554000, 4226000, 15556000, 4228000)
        path = str(tmp_path / "dem_webmerc.tif")
        _write_geotiff(path, data, bounds, crs="EPSG:3857")

        # Polygon still in WGS84
        polygon = [
            (139.756, 35.671),
            (139.756, 35.673),
            (139.758, 35.673),
            (139.758, 35.671),
        ]
        grid = create_dem_grid_from_geotiff_polygon(path, 50, polygon)
        assert grid.ndim == 2
