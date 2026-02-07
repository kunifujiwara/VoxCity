"""Tests for network.py helper functions: interpolate_points_along_line, compute_slope_for_group, etc."""
import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point


class TestInterpolatePointsAlongLine:
    """Tests for interpolate_points_along_line function."""

    def test_basic_interpolation(self):
        from voxcity.geoprocessor.network import interpolate_points_along_line
        line = LineString([(0.0, 0.0), (0.01, 0.0)])  # ~1.1 km
        pts = interpolate_points_along_line(line, 100.0)
        assert len(pts) > 2
        assert all(isinstance(p, Point) for p in pts)

    def test_short_line_returns_endpoints(self):
        from voxcity.geoprocessor.network import interpolate_points_along_line
        line = LineString([(0.0, 0.0), (0.0001, 0.0)])  # ~11m
        pts = interpolate_points_along_line(line, 1000.0)
        assert len(pts) == 2

    def test_empty_line_returns_empty(self):
        from voxcity.geoprocessor.network import interpolate_points_along_line
        line = LineString()
        pts = interpolate_points_along_line(line, 100.0)
        assert len(pts) == 0

    def test_zero_length_line_returns_single_point(self):
        from voxcity.geoprocessor.network import interpolate_points_along_line
        line = LineString([(5.0, 5.0), (5.0, 5.0)])
        pts = interpolate_points_along_line(line, 100.0)
        assert len(pts) == 1


class TestComputeSlopeForGroup:
    """Tests for compute_slope_for_group function."""

    def test_flat_surface_zero_slope(self):
        from voxcity.geoprocessor.network import compute_slope_for_group
        # 3 points, all same elevation → 0 slope
        df = pd.DataFrame({
            'index_in_edge': [0, 1, 2],
            'elevation': [10.0, 10.0, 10.0],
            'geometry': [Point(0, 0), Point(100, 0), Point(200, 0)],
        })
        df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:3857')
        slope = compute_slope_for_group(df)
        assert slope == pytest.approx(0.0)

    def test_uphill_slope(self):
        from voxcity.geoprocessor.network import compute_slope_for_group
        df = pd.DataFrame({
            'index_in_edge': [0, 1],
            'elevation': [0.0, 10.0],
            'geometry': [Point(0, 0), Point(100, 0)],
        })
        df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:3857')
        slope = compute_slope_for_group(df)
        # 10m rise / 100m run = 10%
        assert slope == pytest.approx(10.0)

    def test_single_point_returns_nan(self):
        from voxcity.geoprocessor.network import compute_slope_for_group
        df = pd.DataFrame({
            'index_in_edge': [0],
            'elevation': [5.0],
            'geometry': [Point(0, 0)],
        })
        df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:3857')
        slope = compute_slope_for_group(df)
        assert np.isnan(slope)

    def test_coincident_points_returns_nan(self):
        from voxcity.geoprocessor.network import compute_slope_for_group
        df = pd.DataFrame({
            'index_in_edge': [0, 1],
            'elevation': [5.0, 10.0],
            'geometry': [Point(0, 0), Point(0, 0)],
        })
        df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:3857')
        slope = compute_slope_for_group(df)
        assert np.isnan(slope)


class TestCalculateEdgeSlopesFromJoin:
    """Tests for calculate_edge_slopes_from_join."""

    def test_basic_slope_dict(self):
        from voxcity.geoprocessor.network import calculate_edge_slopes_from_join
        pts_data = {
            'edge_id': [0, 0, 1, 1],
            'index_in_edge': [0, 1, 0, 1],
            'elevation': [0.0, 10.0, 5.0, 5.0],
            'geometry': [Point(0, 0), Point(100, 0), Point(200, 0), Point(300, 0)],
        }
        gdf = gpd.GeoDataFrame(pts_data, geometry='geometry', crs='EPSG:3857')
        result = calculate_edge_slopes_from_join(gdf, n_edges=3)
        assert isinstance(result, dict)
        assert 0 in result and 1 in result and 2 in result
        # Edge 0: slope = 10%, Edge 1: slope = 0%
        assert result[0] == pytest.approx(10.0)
        assert result[1] == pytest.approx(0.0)
        # Edge 2 has no data → NaN
        assert np.isnan(result[2])


class TestFetchElevationsForPoints:
    """Tests for fetch_elevations_for_points."""

    def test_spatial_join(self):
        from voxcity.geoprocessor.network import fetch_elevations_for_points
        from shapely.geometry import box
        pts = gpd.GeoDataFrame(
            {'a': [1, 2]},
            geometry=[Point(50, 50), Point(150, 50)],
            crs='EPSG:3857',
        )
        dem = gpd.GeoDataFrame(
            {'value': [100.0, 200.0]},
            geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)],
            crs='EPSG:3857',
        )
        joined = fetch_elevations_for_points(pts, dem, elevation_col='value')
        assert 'elevation' in joined.columns
        assert len(joined) == 2
