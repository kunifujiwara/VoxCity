"""Regression tests for voxcity.geoprocessor.draw._common helpers."""
import pytest

from voxcity.geoprocessor.raster.core import compute_grid_geometry
from voxcity.utils.projector import GridProjector
from voxcity.geoprocessor.draw._common import geo_to_cell


# Axis-aligned ~4 km × 4 km rectangle in Tokyo, vertex order SW/NW/NE/SE.
_RECT = [
    [139.680, 35.680],  # 0 SW
    [139.680, 35.716],  # 1 NW
    [139.716, 35.716],  # 2 NE
    [139.716, 35.680],  # 3 SE
]
_MESHSIZE = 50.0


@pytest.fixture(scope="module")
def grid_geom():
    g = compute_grid_geometry(_RECT, _MESHSIZE)
    assert g is not None
    return g


class TestGeoToCell:
    def test_inside_matches_projector(self, grid_geom):
        """A point inside the grid should match GridProjector.lon_lat_to_cell."""
        lon, lat = 139.698, 35.698
        proj = GridProjector(grid_geom)
        expected_i, expected_j = proj.lon_lat_to_cell(lon, lat)
        shape = grid_geom["grid_size"]

        result_i, result_j = geo_to_cell(lon, lat, grid_geom, shape)

        assert result_i == expected_i
        assert result_j == expected_j
        assert result_i is not None
        assert result_j is not None

    def test_none_grid_geom_returns_none(self):
        """None grid_geom should return (None, None) without error."""
        assert geo_to_cell(139.698, 35.698, None, (100, 100)) == (None, None)

    def test_none_array_shape_returns_none(self, grid_geom):
        """None array_shape should return (None, None) without error."""
        assert geo_to_cell(139.698, 35.698, grid_geom, None) == (None, None)

    def test_outside_grid_returns_none(self, grid_geom):
        """A point outside the grid extent should return (None, None)."""
        # Far outside Tokyo grid
        lon, lat = 0.0, 0.0
        shape = grid_geom["grid_size"]
        assert geo_to_cell(lon, lat, grid_geom, shape) == (None, None)
