"""
Comprehensive tests for voxcity.geoprocessor.network helper functions.
Covers: interpolate_points_along_line, compute_slope_for_group, 
calculate_edge_slopes_from_join, fetch_elevations_for_points
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, box
from pyproj import Transformer

from voxcity.geoprocessor.network import (
    interpolate_points_along_line,
    compute_slope_for_group,
    calculate_edge_slopes_from_join,
    fetch_elevations_for_points,
)


class TestInterpolatePointsAlongLine:
    def test_basic_line(self):
        # Create a line about 100m long in EPSG:4326
        line = LineString([(0.0, 0.0), (0.001, 0.0)])
        pts = interpolate_points_along_line(line, interval=20)
        assert len(pts) >= 2

    def test_short_line(self):
        # Very short line (<interval)
        line = LineString([(0.0, 0.0), (0.000001, 0.0)])
        pts = interpolate_points_along_line(line, interval=1000)
        assert len(pts) == 2  # Start + end only

    def test_empty_line(self):
        line = LineString()
        pts = interpolate_points_along_line(line, interval=10)
        assert len(pts) == 0

    def test_long_line(self):
        # ~1km line
        line = LineString([(0.0, 0.0), (0.01, 0.0)])
        pts = interpolate_points_along_line(line, interval=100)
        assert len(pts) >= 5
        # All should be Point objects
        assert all(isinstance(p, Point) for p in pts)

    def test_includes_endpoints(self):
        line = LineString([(139.7, 35.68), (139.71, 35.68)])
        pts = interpolate_points_along_line(line, interval=100)
        # Start point should be close to the beginning
        assert abs(pts[0].x - 139.7) < 0.001
        # End point should be close to the end
        assert abs(pts[-1].x - 139.71) < 0.001


class TestComputeSlopeForGroup:
    def test_flat_surface(self):
        # Points at same elevation
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        points = [Point(transformer.transform(139.7 + i * 0.0001, 35.68)) for i in range(5)]
        df = gpd.GeoDataFrame({
            'geometry': points,
            'elevation': [10.0, 10.0, 10.0, 10.0, 10.0],
            'index_in_edge': [0, 1, 2, 3, 4]
        }, crs="EPSG:3857")
        slope = compute_slope_for_group(df)
        assert slope == pytest.approx(0.0, abs=0.01)

    def test_uphill(self):
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        points = [Point(transformer.transform(139.7 + i * 0.001, 35.68)) for i in range(5)]
        df = gpd.GeoDataFrame({
            'geometry': points,
            'elevation': [0.0, 1.0, 2.0, 3.0, 4.0],
            'index_in_edge': [0, 1, 2, 3, 4]
        }, crs="EPSG:3857")
        slope = compute_slope_for_group(df)
        assert slope > 0

    def test_coincident_points(self):
        pt = Point(15527744, 4257177)  # Some mercator coordinate
        df = gpd.GeoDataFrame({
            'geometry': [pt, pt],
            'elevation': [10.0, 20.0],
            'index_in_edge': [0, 1]
        }, crs="EPSG:3857")
        slope = compute_slope_for_group(df)
        assert np.isnan(slope)

    def test_single_point(self):
        df = gpd.GeoDataFrame({
            'geometry': [Point(0, 0)],
            'elevation': [10.0],
            'index_in_edge': [0]
        }, crs="EPSG:3857")
        slope = compute_slope_for_group(df)
        assert np.isnan(slope)


class TestCalculateEdgeSlopesFromJoin:
    def test_basic(self):
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        pts = []
        elevs = []
        edge_ids = []
        idx_in_edge = []
        for i in range(5):
            p = Point(transformer.transform(139.7 + i * 0.001, 35.68))
            pts.append(p)
            elevs.append(float(i) * 2.0)
            edge_ids.append(0)
            idx_in_edge.append(i)
        gdf = gpd.GeoDataFrame({
            'geometry': pts,
            'elevation': elevs,
            'edge_id': edge_ids,
            'index_in_edge': idx_in_edge
        }, crs="EPSG:3857")
        result = calculate_edge_slopes_from_join(gdf, n_edges=2)
        assert 0 in result
        assert 1 in result  # Filled with NaN
        assert np.isnan(result[1])
        assert result[0] > 0

    def test_multiple_edges(self):
        pts = []
        elevs = []
        edge_ids = []
        idx_in_edge = []
        for edge in range(3):
            for i in range(3):
                pts.append(Point(edge * 1000 + i * 100, 0))
                elevs.append(float(i) * 5.0)
                edge_ids.append(edge)
                idx_in_edge.append(i)
        gdf = gpd.GeoDataFrame({
            'geometry': pts,
            'elevation': elevs,
            'edge_id': edge_ids,
            'index_in_edge': idx_in_edge
        }, crs="EPSG:3857")
        result = calculate_edge_slopes_from_join(gdf, n_edges=3)
        assert len(result) == 3
        for edge_id in range(3):
            assert edge_id in result


class TestFetchElevationsForPoints:
    def test_basic_join(self):
        # Create DEM polygons
        dem_polys = [box(0, 0, 100, 100), box(100, 0, 200, 100)]
        dem_gdf = gpd.GeoDataFrame({
            'value': [10.0, 20.0],
            'geometry': dem_polys
        }, crs="EPSG:3857")
        # Create points
        pts_gdf = gpd.GeoDataFrame({
            'geometry': [Point(50, 50), Point(150, 50)],
            'edge_id': [0, 0],
            'index_in_edge': [0, 1]
        }, crs="EPSG:3857")
        result = fetch_elevations_for_points(pts_gdf, dem_gdf)
        assert 'elevation' in result.columns
        assert len(result) == 2
        assert result['elevation'].iloc[0] == pytest.approx(10.0)
        assert result['elevation'].iloc[1] == pytest.approx(20.0)
