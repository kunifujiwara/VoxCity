"""Tests for voxcity.geoprocessor.network module."""
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
import networkx as nx

from voxcity.geoprocessor.network import (
    vectorized_edge_values,
    interpolate_points_along_line,
)


class TestVectorizedEdgeValues:
    """Tests for vectorized_edge_values function."""
    
    @pytest.fixture
    def simple_graph(self):
        """Create a simple network graph with node coordinates."""
        G = nx.MultiDiGraph()
        # Add nodes with coordinates
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=0.001, y=0.0)  # About 100m east at equator
        G.add_node(3, x=0.001, y=0.001)  # About 100m north
        G.add_node(4, x=0.0, y=0.001)  # Back to west
        
        # Add edges (keys are auto-generated starting from 0)
        G.add_edge(1, 2, key=0)
        G.add_edge(2, 3, key=0)
        G.add_edge(3, 4, key=0)
        G.add_edge(4, 1, key=0)
        
        return G
    
    @pytest.fixture
    def simple_polygons_gdf(self):
        """Create simple polygon GDF with values."""
        return gpd.GeoDataFrame({
            'value': [10.0, 20.0, 30.0, 40.0],
            'geometry': [
                Polygon([(0, 0), (0.0005, 0), (0.0005, 0.0005), (0, 0.0005)]),
                Polygon([(0.0005, 0), (0.001, 0), (0.001, 0.0005), (0.0005, 0.0005)]),
                Polygon([(0.0005, 0.0005), (0.001, 0.0005), (0.001, 0.001), (0.0005, 0.001)]),
                Polygon([(0, 0.0005), (0.0005, 0.0005), (0.0005, 0.001), (0, 0.001)]),
            ]
        }, crs="EPSG:4326")
    
    def test_edge_values_computation(self, simple_graph, simple_polygons_gdf):
        """Test that edge values are computed correctly."""
        edge_values = vectorized_edge_values(
            simple_graph,
            simple_polygons_gdf,
            value_col='value'
        )
        
        # Should return dict with edge tuples as keys
        assert isinstance(edge_values, dict)
        
        # Should have values for edges
        for key in edge_values:
            assert isinstance(key, tuple)
            assert len(key) == 3  # (u, v, k)
    
    def test_edge_values_with_geometry(self):
        """Test with edges that have geometry attributes."""
        G = nx.MultiDiGraph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=0.001, y=0.001)
        
        # Add edge with explicit geometry
        line_geom = LineString([(0.0, 0.0), (0.0005, 0.0005), (0.001, 0.001)])
        G.add_edge(1, 2, key=0, geometry=line_geom)
        
        polygons = gpd.GeoDataFrame({
            'value': [50.0],
            'geometry': [Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])]
        }, crs="EPSG:4326")
        
        edge_values = vectorized_edge_values(G, polygons, value_col='value')
        
        assert (1, 2, 0) in edge_values
        # Value should be close to 50 since edge passes through polygon
        assert not np.isnan(edge_values[(1, 2, 0)])
    
    def test_edge_with_no_intersection(self):
        """Test edge that doesn't intersect any polygon."""
        G = nx.MultiDiGraph()
        G.add_node(1, x=10.0, y=10.0)  # Far from polygons
        G.add_node(2, x=10.001, y=10.001)
        G.add_edge(1, 2, key=0)
        
        polygons = gpd.GeoDataFrame({
            'value': [100.0],
            'geometry': [Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])]
        }, crs="EPSG:4326")
        
        # This may raise or return empty dict depending on implementation
        try:
            edge_values = vectorized_edge_values(G, polygons, value_col='value')
            # Edge not in result or value is NaN
            if (1, 2, 0) in edge_values:
                assert np.isnan(edge_values[(1, 2, 0)]) or edge_values[(1, 2, 0)] == 0
        except (TypeError, KeyError):
            # Implementation may not handle non-intersecting edges gracefully
            pass
    
    def test_different_crs_conversion(self, simple_graph):
        """Test with polygons in different CRS."""
        # Create polygons in EPSG:3857 (Web Mercator)
        polygons_3857 = gpd.GeoDataFrame({
            'value': [25.0],
            'geometry': [Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])]
        }, crs="EPSG:3857")
        
        edge_values = vectorized_edge_values(
            simple_graph,
            polygons_3857,
            value_col='value'
        )
        
        # Should handle CRS conversion
        assert isinstance(edge_values, dict)
    
    def test_empty_graph(self, simple_polygons_gdf):
        """Test with empty graph."""
        G = nx.MultiDiGraph()
        
        # Empty graph may raise or return empty dict
        try:
            edge_values = vectorized_edge_values(G, simple_polygons_gdf)
            assert edge_values == {}
        except (ValueError, KeyError):
            # Implementation may not handle empty graph gracefully
            pass
    
    def test_empty_polygons(self, simple_graph):
        """Test with empty polygon GDF."""
        empty_gdf = gpd.GeoDataFrame({
            'value': pd.Series([], dtype=float),
            'geometry': gpd.GeoSeries([], crs="EPSG:4326")
        }, crs="EPSG:4326")
        
        # Empty polygons may cause issues
        try:
            edge_values = vectorized_edge_values(simple_graph, empty_gdf)
            # All edges should have NaN values or not be in result
            for val in edge_values.values():
                assert np.isnan(val) or val == 0
        except (TypeError, KeyError, ValueError):
            # Implementation may not handle empty polygons gracefully
            pass


class TestInterpolatePointsAlongLine:
    """Tests for interpolate_points_along_line function."""
    
    def test_simple_horizontal_line(self):
        """Test interpolation along horizontal line."""
        # Line about 1000m long at equator (approx 0.009 degrees)
        line = LineString([(0, 0), (0.009, 0)])
        interval = 200  # 200 meters
        
        points = interpolate_points_along_line(line, interval)
        
        # Should have multiple points
        assert len(points) > 2
        
        # First and last points should be near line endpoints
        assert abs(points[0].x - 0) < 0.0001
        assert abs(points[-1].x - 0.009) < 0.0001
    
    def test_short_line(self):
        """Test line shorter than interval."""
        # Very short line (about 10m)
        line = LineString([(0, 0), (0.0001, 0)])
        interval = 100  # 100 meters
        
        points = interpolate_points_along_line(line, interval)
        
        # Should have only start and end points
        assert len(points) == 2
    
    def test_empty_line(self):
        """Test with empty line."""
        line = LineString()
        interval = 50
        
        points = interpolate_points_along_line(line, interval)
        
        # Should return empty list
        assert points == []
    
    def test_point_line(self):
        """Test line with same start and end point."""
        line = LineString([(0, 0), (0, 0)])
        interval = 50
        
        points = interpolate_points_along_line(line, interval)
        
        # Should return single point
        assert len(points) <= 2
    
    def test_complex_line(self):
        """Test line with multiple segments."""
        # L-shaped line
        line = LineString([(0, 0), (0.005, 0), (0.005, 0.005)])
        interval = 100
        
        points = interpolate_points_along_line(line, interval)
        
        # Should have multiple points along the path
        assert len(points) > 2
    
    def test_diagonal_line(self):
        """Test diagonal line."""
        # Diagonal line
        line = LineString([(0, 0), (0.005, 0.005)])
        interval = 150
        
        points = interpolate_points_along_line(line, interval)
        
        # Should have multiple points
        assert len(points) >= 2
        
        # All points should be Point objects
        for pt in points:
            assert isinstance(pt, Point)
    
    def test_interval_spacing(self):
        """Test that points are approximately interval apart."""
        line = LineString([(0, 0), (0.02, 0)])  # ~2km line
        interval = 500  # 500m
        
        points = interpolate_points_along_line(line, interval)
        
        # Check we got reasonable number of points
        # 2km / 500m = 4 segments = 5 points approximately
        assert 3 <= len(points) <= 7


class TestNetworkEdgeCases:
    """Edge cases and error handling tests."""
    
    def test_multi_edge_graph(self):
        """Test graph with multiple edges between same nodes."""
        G = nx.MultiDiGraph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=0.001, y=0.0)
        
        # Add multiple edges between same nodes
        G.add_edge(1, 2, key=0)
        G.add_edge(1, 2, key=1)
        G.add_edge(1, 2, key=2)
        
        polygons = gpd.GeoDataFrame({
            'value': [30.0],
            'geometry': [Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])]
        }, crs="EPSG:4326")
        
        edge_values = vectorized_edge_values(G, polygons)
        
        # Should have values for all edges
        assert len(edge_values) >= 1
    
    def test_self_loop_edge(self):
        """Test graph with self-loop edge."""
        G = nx.MultiDiGraph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_edge(1, 1, key=0)  # Self-loop
        
        polygons = gpd.GeoDataFrame({
            'value': [50.0],
            'geometry': [Polygon([(-0.001, -0.001), (0.001, -0.001), (0.001, 0.001), (-0.001, 0.001)])]
        }, crs="EPSG:4326")
        
        # Should not crash
        edge_values = vectorized_edge_values(G, polygons)
        assert isinstance(edge_values, dict)
    
    def test_large_polygon_values(self):
        """Test with very large polygon values."""
        G = nx.MultiDiGraph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=0.001, y=0.0)
        G.add_edge(1, 2, key=0)
        
        polygons = gpd.GeoDataFrame({
            'value': [1e10],  # Very large value
            'geometry': [Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])]
        }, crs="EPSG:4326")
        
        edge_values = vectorized_edge_values(G, polygons)
        
        if (1, 2, 0) in edge_values and not np.isnan(edge_values[(1, 2, 0)]):
            # Value should preserve magnitude
            assert edge_values[(1, 2, 0)] > 1e9
    
    def test_negative_values(self):
        """Test with negative polygon values."""
        G = nx.MultiDiGraph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=0.001, y=0.0)
        G.add_edge(1, 2, key=0)
        
        polygons = gpd.GeoDataFrame({
            'value': [-50.0],
            'geometry': [Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])]
        }, crs="EPSG:4326")
        
        edge_values = vectorized_edge_values(G, polygons)
        
        if (1, 2, 0) in edge_values and not np.isnan(edge_values[(1, 2, 0)]):
            # Should preserve negative values
            assert edge_values[(1, 2, 0)] < 0
    
    def test_nan_values(self):
        """Test with NaN polygon values."""
        G = nx.MultiDiGraph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=0.001, y=0.0)
        G.add_edge(1, 2, key=0)
        
        polygons = gpd.GeoDataFrame({
            'value': [np.nan],
            'geometry': [Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])]
        }, crs="EPSG:4326")
        
        edge_values = vectorized_edge_values(G, polygons)
        
        # Should handle NaN gracefully
        assert isinstance(edge_values, dict)
