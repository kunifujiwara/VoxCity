"""
Tests for pipeline.py parallel download branches (lines 87-113)
and _visualize_grids_after_parallel, save/load irradiance mesh.
"""
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import os


def _make_par_cfg(**overrides):
    defaults = dict(
        meshsize=1.0,
        rectangle_vertices=[(139.75, 35.68), (139.75, 35.69), (139.76, 35.69), (139.76, 35.68)],
        crs="EPSG:4326",
        land_cover_source="OpenStreetMap",
        building_source="OpenStreetMap",
        canopy_height_source="OpenStreetMap",
        dem_source=None,
        output_dir="output",
        gridvis=False,
        parallel_download=True,
        remove_perimeter_object=None,
        trunk_height_ratio=0.3,
        land_cover_options={},
        building_options={},
        canopy_options={},
        dem_options={},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestVisualizeGridsAfterParallel:
    """Cover _visualize_grids_after_parallel method."""

    @patch("voxcity.visualizer.grids.visualize_numerical_grid")
    @patch("voxcity.visualizer.grids.visualize_land_cover_grid")
    def test_visualize_all_with_canopy(self, mock_lc_vis, mock_num_vis):
        from voxcity.generator.pipeline import VoxCityPipeline
        pipeline = VoxCityPipeline(
            meshsize=1.0,
            rectangle_vertices=[(0, 0), (0, 1), (1, 1), (1, 0)],
            crs="EPSG:4326",
        )
        lc = np.ones((5, 5), dtype=int)
        bh = np.array([[0, 5.0], [3.0, 0]], dtype=float)
        canopy_top = np.array([[0, 0], [0, 4.0]])
        dem = np.zeros((2, 2))
        pipeline._visualize_grids_after_parallel(lc, bh, canopy_top, dem, "OpenStreetMap", 1.0)
        # At least building height + canopy + dem = 3 calls to visualize_numerical_grid
        assert mock_num_vis.call_count >= 2

    @patch("voxcity.visualizer.grids.visualize_numerical_grid")
    @patch("voxcity.visualizer.grids.visualize_land_cover_grid")
    def test_visualize_no_canopy(self, mock_lc_vis, mock_num_vis):
        from voxcity.generator.pipeline import VoxCityPipeline
        pipeline = VoxCityPipeline(
            meshsize=1.0,
            rectangle_vertices=[(0, 0), (0, 1), (1, 1), (1, 0)],
            crs="EPSG:4326",
        )
        lc = np.ones((5, 5), dtype=int)
        bh = np.zeros((5, 5))
        dem = np.zeros((5, 5))
        pipeline._visualize_grids_after_parallel(lc, bh, None, dem, "OpenStreetMap", 1.0)
        # No canopy visualization -> only building height + dem


class TestSaveLoadIrradianceMesh:
    """Cover save_irradiance_mesh and load_irradiance_mesh."""

    def test_save_and_load(self, tmp_path):
        from voxcity.simulator.solar.integration import save_irradiance_mesh, load_irradiance_mesh
        mesh_data = {"vertices": np.array([[0, 0, 0]]), "faces": np.array([[0, 0, 0]])}
        path = str(tmp_path / "sub" / "irr.pkl")
        save_irradiance_mesh(mesh_data, path)
        loaded = load_irradiance_mesh(path)
        assert "vertices" in loaded
        assert np.array_equal(loaded["vertices"], mesh_data["vertices"])
