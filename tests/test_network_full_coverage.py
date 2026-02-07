"""
Tests for geoprocessor/network.py remaining coverage gaps:
  - get_network_values  (lines 100-257)
  - analyze_network_slopes (lines 497-709)
All external I/O (osmnx, contextily) is mocked.
"""
import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import LineString, Point, box
from unittest.mock import patch, MagicMock
import networkx as nx


# ── helpers ──────────────────────────────────────────────────────────────

def _make_simple_graph():
    """Create a simple networkx MultiDiGraph with 3 nodes and 2 edges."""
    G = nx.MultiDiGraph()
    G.add_node(1, x=139.75, y=35.68)
    G.add_node(2, x=139.76, y=35.68)
    G.add_node(3, x=139.76, y=35.69)
    G.add_edge(1, 2, 0, geometry=LineString([(139.75, 35.68), (139.76, 35.68)]))
    G.add_edge(2, 3, 0, geometry=LineString([(139.76, 35.68), (139.76, 35.69)]))
    return G


def _make_grid_gdf():
    """Create a simple grid GeoDataFrame with values."""
    polys = [box(139.75 + i * 0.005, 35.68 + j * 0.005,
                 139.75 + (i + 1) * 0.005, 35.68 + (j + 1) * 0.005)
             for i in range(4) for j in range(4)]
    values = np.random.rand(16) * 100
    return gpd.GeoDataFrame({"value": values, "geometry": polys}, crs="EPSG:4326")


RECT = [(139.75, 35.68), (139.75, 35.69), (139.76, 35.69), (139.76, 35.68)]


# ══════════════════════════════════════════════════════════════════════════
# compute_network_edge_values
# ══════════════════════════════════════════════════════════════════════════

class TestGetNetworkValues:

    @patch("voxcity.geoprocessor.network.plt")
    @patch("voxcity.geoprocessor.network.ctx")
    @patch("voxcity.geoprocessor.network.ox")
    def test_basic_with_gdf_input(self, mock_ox, mock_ctx, mock_plt):
        """Pass a GeoDataFrame as grid directly."""
        G = _make_simple_graph()
        mock_ox.graph.graph_from_bbox.return_value = G

        grid_gdf = _make_grid_gdf()
        from voxcity.geoprocessor.network import get_network_values
        G_out, edge_gdf = get_network_values(
            grid_gdf, value_name="test_val", rectangle_vertices=RECT, meshsize=0.005,
            vis_graph=False
        )
        assert isinstance(G_out, nx.MultiDiGraph)
        assert isinstance(edge_gdf, gpd.GeoDataFrame)
        assert "test_val" in edge_gdf.columns

    @patch("voxcity.geoprocessor.network.plt")
    @patch("voxcity.geoprocessor.network.ctx")
    @patch("voxcity.geoprocessor.network.ox")
    def test_vis_graph_branch(self, mock_ox, mock_ctx, mock_plt):
        """vis_graph=True triggers plotting."""
        G = _make_simple_graph()
        mock_ox.graph.graph_from_bbox.return_value = G
        grid_gdf = _make_grid_gdf()
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        from voxcity.geoprocessor.network import get_network_values
        get_network_values(
            grid_gdf, value_name="val", rectangle_vertices=RECT, meshsize=0.005,
            vis_graph=True
        )
        mock_plt.show.assert_called_once()

    @patch("voxcity.geoprocessor.network.plt")
    @patch("voxcity.geoprocessor.network.ctx")
    @patch("voxcity.geoprocessor.network.ox")
    def test_voxcity_object_input(self, mock_ox, mock_ctx, mock_plt):
        """Derive rectangle_vertices and meshsize from VoxCity-like object."""
        G = _make_simple_graph()
        mock_ox.graph.graph_from_bbox.return_value = G
        grid_gdf = _make_grid_gdf()
        
        # Create VoxCity-like object
        vc = MagicMock()
        vc.extras = {"rectangle_vertices": RECT}
        vc.voxels.meta.meshsize = 0.005
        vc.voxels.meta.bounds = None
        
        from voxcity.geoprocessor.network import get_network_values
        G_out, edge_gdf = get_network_values(
            grid_gdf, value_name="val", voxcity=vc, vis_graph=False
        )
        assert isinstance(edge_gdf, gpd.GeoDataFrame)

    @patch("voxcity.geoprocessor.network.plt")
    @patch("voxcity.geoprocessor.network.ctx")
    @patch("voxcity.geoprocessor.network.ox")
    def test_no_rectangle_raises(self, mock_ox, mock_ctx, mock_plt):
        """No rectangle_vertices and no voxcity -> ValueError."""
        grid_gdf = _make_grid_gdf()
        from voxcity.geoprocessor.network import get_network_values
        with pytest.raises(ValueError, match="rectangle_vertices"):
            get_network_values(grid_gdf, value_name="val", vis_graph=False)

    @patch("voxcity.geoprocessor.network.plt")
    @patch("voxcity.geoprocessor.network.ctx")
    @patch("voxcity.geoprocessor.network.ox")
    def test_save_path(self, mock_ox, mock_ctx, mock_plt, tmp_path):
        """save_path triggers file save."""
        G = _make_simple_graph()
        mock_ox.graph.graph_from_bbox.return_value = G
        grid_gdf = _make_grid_gdf()
        save = str(tmp_path / "edges.gpkg")
        from voxcity.geoprocessor.network import get_network_values
        _, edge_gdf = get_network_values(
            grid_gdf, value_name="val", rectangle_vertices=RECT, meshsize=0.005,
            vis_graph=False, save_path=save
        )
        import os
        assert os.path.exists(save)

    @patch("voxcity.geoprocessor.network.plt")
    @patch("voxcity.geoprocessor.network.ctx")
    @patch("voxcity.geoprocessor.network.ox")
    def test_edge_without_geometry(self, mock_ox, mock_ctx, mock_plt):
        """Edge without geometry key falls back to node coords."""
        G = nx.MultiDiGraph()
        G.add_node(1, x=139.75, y=35.68)
        G.add_node(2, x=139.76, y=35.68)
        G.add_edge(1, 2, 0)  # no geometry attribute
        mock_ox.graph.graph_from_bbox.return_value = G
        grid_gdf = _make_grid_gdf()
        from voxcity.geoprocessor.network import get_network_values
        _, edge_gdf = get_network_values(
            grid_gdf, value_name="val", rectangle_vertices=RECT, meshsize=0.005,
            vis_graph=False
        )
        assert len(edge_gdf) == 1


# ══════════════════════════════════════════════════════════════════════════
# compute_network_slope_analysis
# ══════════════════════════════════════════════════════════════════════════

class TestAnalyzeNetworkSlopes:

    @patch("voxcity.geoprocessor.network.plt")
    @patch("voxcity.geoprocessor.network.ctx")
    @patch("voxcity.geoprocessor.network.ox")
    @patch("voxcity.geoprocessor.network.calculate_edge_slopes_from_join")
    @patch("voxcity.geoprocessor.network.fetch_elevations_for_points")
    @patch("voxcity.geoprocessor.network.gather_interpolation_points")
    def test_basic_slope(self, mock_gather, mock_fetch, mock_slopes, mock_ox, mock_ctx, mock_plt):
        """Basic smoke test for slope analysis."""
        G = _make_simple_graph()
        mock_ox.graph.graph_from_bbox.return_value = G

        # Fake interpolation points
        pts = gpd.GeoDataFrame(
            {"edge_id": [0, 0, 1, 1], "geometry": [Point(0, 0)] * 4},
            crs="EPSG:4326"
        )
        mock_gather.return_value = pts
        mock_fetch.return_value = pts  # doesn't matter for slopes
        mock_slopes.return_value = {0: 5.0, 1: 3.0}

        dem_grid = np.random.rand(4, 4) * 100
        from voxcity.geoprocessor.network import analyze_network_slopes
        G_out, edge_gdf = analyze_network_slopes(
            dem_grid, meshsize=0.005,
            rectangle_vertices=RECT, vis_graph=False
        )
        assert isinstance(edge_gdf, gpd.GeoDataFrame)
        assert "slope" in edge_gdf.columns

    @patch("voxcity.geoprocessor.network.plt")
    @patch("voxcity.geoprocessor.network.ctx")
    @patch("voxcity.geoprocessor.network.ox")
    @patch("voxcity.geoprocessor.network.calculate_edge_slopes_from_join")
    @patch("voxcity.geoprocessor.network.fetch_elevations_for_points")
    @patch("voxcity.geoprocessor.network.gather_interpolation_points")
    def test_vis_graph(self, mock_gather, mock_fetch, mock_slopes, mock_ox, mock_ctx, mock_plt):
        """vis_graph=True triggers plot."""
        G = _make_simple_graph()
        mock_ox.graph.graph_from_bbox.return_value = G
        pts = gpd.GeoDataFrame(
            {"edge_id": [0, 1], "geometry": [Point(0, 0)] * 2}, crs="EPSG:4326"
        )
        mock_gather.return_value = pts
        mock_fetch.return_value = pts
        mock_slopes.return_value = {0: 5.0, 1: 3.0}
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        dem_grid = np.random.rand(4, 4) * 100
        from voxcity.geoprocessor.network import analyze_network_slopes
        analyze_network_slopes(
            dem_grid, meshsize=0.005,
            rectangle_vertices=RECT, vis_graph=True
        )
        mock_plt.show.assert_called()

    @patch("voxcity.geoprocessor.network.plt")
    @patch("voxcity.geoprocessor.network.ctx")
    @patch("voxcity.geoprocessor.network.ox")
    @patch("voxcity.geoprocessor.network.calculate_edge_slopes_from_join")
    @patch("voxcity.geoprocessor.network.fetch_elevations_for_points")
    @patch("voxcity.geoprocessor.network.gather_interpolation_points")
    def test_output_directory(self, mock_gather, mock_fetch, mock_slopes, mock_ox, mock_ctx, mock_plt, tmp_path):
        """output_directory triggers save."""
        G = _make_simple_graph()
        mock_ox.graph.graph_from_bbox.return_value = G
        pts = gpd.GeoDataFrame(
            {"edge_id": [0, 1], "geometry": [Point(0, 0)] * 2}, crs="EPSG:4326"
        )
        mock_gather.return_value = pts
        mock_fetch.return_value = pts
        mock_slopes.return_value = {0: 2.0, 1: 4.0}

        dem_grid = np.random.rand(4, 4) * 100
        from voxcity.geoprocessor.network import analyze_network_slopes
        analyze_network_slopes(
            dem_grid, meshsize=0.005,
            rectangle_vertices=RECT, vis_graph=False,
            output_directory=str(tmp_path)
        )
        import os
        assert any(f.endswith(".gpkg") for f in os.listdir(tmp_path))

    @patch("voxcity.geoprocessor.network.plt")
    @patch("voxcity.geoprocessor.network.ctx")
    @patch("voxcity.geoprocessor.network.ox")
    def test_no_rectangle_raises(self, mock_ox, mock_ctx, mock_plt):
        """Missing rectangle_vertices -> ValueError."""
        dem_grid = np.random.rand(4, 4) * 100
        from voxcity.geoprocessor.network import analyze_network_slopes
        with pytest.raises(ValueError, match="rectangle_vertices"):
            analyze_network_slopes(dem_grid, meshsize=0.005, vis_graph=False)
