"""
Tests for voxcity.geoprocessor.network: generate_edge_interpolated_points (lines 343-371).
Uses a synthetic NetworkX graph to avoid network calls.
"""

import numpy as np
import pytest
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString

from voxcity.geoprocessor.network import (
    gather_interpolation_points,
    interpolate_points_along_line,
)


class TestGenerateEdgeInterpolatedPoints:
    def _make_graph(self):
        """Create a simple MultiDiGraph with known geometry."""
        G = nx.MultiDiGraph()
        G.add_node(0, x=0.0, y=0.0)
        G.add_node(1, x=1.0, y=0.0)
        G.add_node(2, x=1.0, y=1.0)
        G.add_edge(0, 1, geometry=LineString([(0, 0), (1, 0)]))
        G.add_edge(1, 2, geometry=LineString([(1, 0), (1, 1)]))
        return G

    def test_returns_geodataframe(self):
        G = self._make_graph()
        result = gather_interpolation_points(G, interval=0.5, n_jobs=1)
        assert isinstance(result, gpd.GeoDataFrame)
        assert 'edge_id' in result.columns
        assert 'index_in_edge' in result.columns
        assert 'geometry' in result.columns

    def test_point_count(self):
        G = self._make_graph()
        result = gather_interpolation_points(G, interval=0.3, n_jobs=1)
        assert len(result) > 0

    def test_edge_without_geometry(self):
        """Edges without geometry attr should use node coords."""
        G = nx.MultiDiGraph()
        G.add_node(0, x=0.0, y=0.0)
        G.add_node(1, x=2.0, y=0.0)
        G.add_edge(0, 1)  # No geometry key
        result = gather_interpolation_points(G, interval=0.5, n_jobs=1)
        assert len(result) > 0

    def test_single_edge(self):
        G = nx.MultiDiGraph()
        G.add_node(0, x=0.0, y=0.0)
        G.add_node(1, x=0.0, y=1.0)
        G.add_edge(0, 1, geometry=LineString([(0, 0), (0, 1)]))
        result = gather_interpolation_points(G, interval=0.25, n_jobs=1)
        assert len(result) >= 2  # At least start and end

    def test_crs_is_set(self):
        G = self._make_graph()
        result = gather_interpolation_points(G, interval=0.5, n_jobs=1)
        assert result.crs is not None
        assert result.crs.to_epsg() == 4326
