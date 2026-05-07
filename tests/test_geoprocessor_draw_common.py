"""Regression tests for voxcity.geoprocessor.draw._common helpers."""
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from voxcity.geoprocessor.raster.core import compute_grid_geometry
from voxcity.utils.projector import GridProjector
from voxcity.geoprocessor.draw._common import build_building_geojson, geo_to_cell


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


class TestBuildBuildingGeojson:
    def test_includes_min_height_when_present(self):
        gdf = gpd.GeoDataFrame(
            [{
                "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                "height": 18.5,
                "min_height": 3.0,
                "height_estimated": False,
            }],
            geometry="geometry",
            crs="EPSG:4326",
        )

        fc = build_building_geojson(gdf, include_height=True)

        props = fc["features"][0]["properties"]
        assert props["height"] == 18.5
        assert props["min_height"] == 3.0

    def test_defaults_min_height_to_zero_for_missing_values(self):
        gdf = gpd.GeoDataFrame(
            [
                {
                    "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    "height": 12.0,
                    "min_height": np.nan,
                },
                {
                    "geometry": Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                    "height": 16.0,
                    "min_height": 1.5,
                },
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

        fc = build_building_geojson(gdf, include_height=True)

        assert fc["features"][0]["properties"]["min_height"] == 0.0
        assert fc["features"][1]["properties"]["min_height"] == 1.5

    def test_defaults_min_height_to_zero_when_column_absent(self):
        gdf = gpd.GeoDataFrame(
            [{
                "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                "height": 12.0,
            }],
            geometry="geometry",
            crs="EPSG:4326",
        )

        fc = build_building_geojson(gdf, include_height=True)

        assert fc["features"][0]["properties"]["min_height"] == 0.0
