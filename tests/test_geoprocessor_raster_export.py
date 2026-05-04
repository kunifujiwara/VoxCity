"""Tests for raster grid export helpers."""

import numpy as np
import pytest


@pytest.fixture
def south_up_grid():
    """A uv_m/SOUTH_UP grid where values increase northward."""
    return np.array(
        [
            [1, 1],  # south row
            [2, 2],
            [3, 3],  # north row
        ]
    )


@pytest.fixture
def rectangle_vertices():
    return [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]


def fail_if_orientation_conversion_is_used(*args, **kwargs):
    raise AssertionError("raster export should consume SOUTH_UP grids directly")


def test_grid_to_geodataframe_uses_south_up_coordinates_without_conversion(
    monkeypatch, south_up_grid, rectangle_vertices
):
    from voxcity.geoprocessor.raster import export

    monkeypatch.setattr(export, "ensure_orientation", fail_if_orientation_conversion_is_used, raising=False)

    gdf = export.grid_to_geodataframe(south_up_grid, rectangle_vertices, meshsize=1.0)

    bounds = gdf.geometry.bounds
    center_lats = (bounds["miny"] + bounds["maxy"]) / 2
    south_lat = center_lats[gdf["value"] == 1].mean()
    north_lat = center_lats[gdf["value"] == 3].mean()
    assert north_lat > south_lat


def test_grid_to_point_geodataframe_uses_south_up_coordinates_without_conversion(
    monkeypatch, south_up_grid, rectangle_vertices
):
    from voxcity.geoprocessor.raster import export

    monkeypatch.setattr(export, "ensure_orientation", fail_if_orientation_conversion_is_used, raising=False)

    gdf = export.grid_to_point_geodataframe(south_up_grid, rectangle_vertices, meshsize=1.0)

    south_lat = gdf.loc[gdf["value"] == 1, "geometry"].y.mean()
    north_lat = gdf.loc[gdf["value"] == 3, "geometry"].y.mean()
    assert north_lat > south_lat