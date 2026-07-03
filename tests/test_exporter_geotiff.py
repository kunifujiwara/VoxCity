import numpy as np
import pytest
from affine import Affine

from voxcity.exporter.geotiff import _north_up_affine_and_array
from voxcity.geoprocessor.raster.core import compute_cell_center_coords

# Axis-aligned rectangle [SW, SE, NE, NW] near Tokyo
RECT = [(139.70, 35.60), (139.71, 35.60), (139.71, 35.61), (139.70, 35.61)]
MESH = 30.0


def _voxcity_index_grid(rect, mesh):
    """Return (grid, cc) where grid[i, j] == i*1000 + j for shape (nx, ny)."""
    cc = compute_cell_center_coords(rect, mesh)
    nx, ny = cc["grid_size"]
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    grid = (ii * 1000 + jj).astype(np.float64)
    return grid, cc


def test_north_up_affine_and_array_orientation():
    grid, cc = _voxcity_index_grid(RECT, MESH)
    nx, ny = cc["grid_size"]
    array, transform = _north_up_affine_and_array(grid, RECT, MESH)

    # Raster shape is (rows=ny, cols=nx); north-up => negative row step
    assert array.shape == (ny, nx)
    assert isinstance(transform, Affine)
    assert transform.e < 0

    # Cell (i, j) lives at raster (row = ny-1-j, col = i)
    for i in (0, nx - 1):
        for j in (0, ny - 1):
            r, c = ny - 1 - j, i
            assert array[r, c] == grid[i, j]
            # affine pixel-center maps back to the true cell center lon/lat
            lon, lat = transform * (c + 0.5, r + 0.5)
            np.testing.assert_allclose(lon, cc["lons"][i, j], atol=1e-9)
            np.testing.assert_allclose(lat, cc["lats"][i, j], atol=1e-9)


def test_north_up_affine_and_array_shape_mismatch():
    with pytest.raises(ValueError):
        _north_up_affine_and_array(np.zeros((3, 3)), RECT, MESH)


def test_north_up_affine_and_array_insufficient_vertices():
    with pytest.raises(ValueError):
        _north_up_affine_and_array(np.zeros((1, 1)), RECT[:2], MESH)
