"""Round 6 – cover pipeline.py parallel download helpers and visualization (lines 87-452)."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**overrides):
    """Build a minimal PipelineConfig-like namespace."""
    defaults = dict(
        meshsize=5.0,
        rectangle_vertices=[(0, 0), (0, 1), (1, 1), (1, 0)],
        land_cover_source="OpenStreetMap",
        building_source="OpenStreetMap",
        canopy_height_source="Static",
        dem_source="Flat",
        output_dir="output",
        gridvis=False,
        remove_perimeter_object=None,
        static_tree_height=10.0,
        trunk_height_ratio=None,
        land_cover_options={},
        building_options={},
        canopy_options={},
        dem_options={},
        crs="EPSG:4326",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _dummy_grids(shape=(4, 4)):
    lc = np.ones(shape, dtype=int)
    bh = np.zeros(shape, dtype=float)
    bmin = np.empty(shape, dtype=object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            bmin[i, j] = []
    bid = np.zeros(shape, dtype=int)
    canopy_top = np.zeros(shape, dtype=float)
    canopy_bottom = np.zeros(shape, dtype=float)
    dem = np.zeros(shape, dtype=float)
    return lc, bh, bmin, bid, canopy_top, canopy_bottom, dem


# ===========================================================================
# Tests for _visualize_grids_after_parallel
# ===========================================================================

class TestVisualizeGridsAfterParallel:
    """Cover pipeline.py lines 202-253: _visualize_grids_after_parallel."""

    def _make_pipeline(self):
        from voxcity.generator.pipeline import VoxCityPipeline
        return VoxCityPipeline(
            meshsize=5.0,
            rectangle_vertices=[(0, 0), (0, 1), (1, 1), (1, 0)],
        )

    @patch("voxcity.visualizer.grids.visualize_numerical_grid")
    @patch("voxcity.visualizer.grids.visualize_land_cover_grid")
    @patch("voxcity.utils.lc.get_land_cover_classes")
    def test_basic_visualization(self, mock_classes, mock_vis_lc, mock_vis_num):
        """All four visualizations (LC, building, canopy, DEM) run."""
        mock_classes.return_value = {(0, 255, 0): "Tree", (128, 128, 128): "Road"}
        pipe = self._make_pipeline()
        lc = np.array([[0, 1], [1, 0]], dtype=int)
        bh = np.array([[5.0, 0.0], [0.0, 10.0]])
        canopy = np.array([[3.0, 0.0], [0.0, 4.0]])
        dem = np.array([[1.0, 2.0], [3.0, 4.0]])
        pipe._visualize_grids_after_parallel(lc, bh, canopy, dem, "OpenStreetMap", 5.0)
        assert mock_vis_lc.called
        assert mock_vis_num.call_count >= 2  # building + canopy + dem = 3

    @patch("voxcity.visualizer.grids.visualize_numerical_grid")
    @patch("voxcity.visualizer.grids.visualize_land_cover_grid")
    @patch("voxcity.utils.lc.get_land_cover_classes")
    def test_canopy_none_skipped(self, mock_classes, mock_vis_lc, mock_vis_num):
        """When canopy_top is None, only building + DEM are visualized numerically."""
        mock_classes.return_value = {(0, 255, 0): "Tree"}
        pipe = self._make_pipeline()
        lc = np.zeros((2, 2), dtype=int)
        bh = np.zeros((2, 2))
        dem = np.zeros((2, 2))
        pipe._visualize_grids_after_parallel(lc, bh, None, dem, "OpenStreetMap", 5.0)
        # Should have 2 numerical calls: building height + DEM (canopy skipped)
        assert mock_vis_num.call_count == 2

    @patch("voxcity.visualizer.grids.visualize_numerical_grid", side_effect=RuntimeError("viz fail"))
    @patch("voxcity.visualizer.grids.visualize_land_cover_grid")
    @patch("voxcity.utils.lc.get_land_cover_classes")
    def test_visualization_exception_swallowed(self, mock_classes, mock_vis_lc, mock_vis_num):
        """Exceptions inside individual visualization blocks are logged, not raised."""
        mock_classes.return_value = {(0, 255, 0): "Tree"}
        pipe = self._make_pipeline()
        lc = np.zeros((2, 2), dtype=int)
        bh = np.zeros((2, 2))
        dem = np.zeros((2, 2))
        # Should not raise
        pipe._visualize_grids_after_parallel(lc, bh, None, dem, "OpenStreetMap", 5.0)


# ===========================================================================
# Tests for _run_parallel_downloads
# ===========================================================================

class TestRunParallelDownloads:
    """Cover pipeline.py lines 255-357: _run_parallel_downloads."""

    def _make_pipeline(self):
        from voxcity.generator.pipeline import VoxCityPipeline
        return VoxCityPipeline(
            meshsize=5.0,
            rectangle_vertices=[(0, 0), (0, 1), (1, 1), (1, 0)],
        )

    def test_parallel_downloads_all_succeed(self):
        """All four strategies return results via ThreadPoolExecutor."""
        pipe = self._make_pipeline()
        cfg = _make_cfg(canopy_height_source="GEE")

        lc = np.ones((3, 3), dtype=int)
        bh = np.zeros((3, 3)); bmin = np.empty((3, 3), dtype=object)
        for i in range(3):
            for j in range(3):
                bmin[i, j] = []
        bid = np.zeros((3, 3), dtype=int)
        ct = np.zeros((3, 3)); cb = np.zeros((3, 3))
        dem = np.zeros((3, 3))

        land_strat = MagicMock()
        land_strat.build_grid.return_value = lc
        build_strat = MagicMock()
        build_strat.build_grids.return_value = (bh, bmin, bid, None)
        canopy_strat = MagicMock()
        canopy_strat.build_grids.return_value = (ct, cb)
        dem_strat = MagicMock()
        dem_strat.build_grid.return_value = dem

        with patch("voxcity.generator.grids.get_last_effective_land_cover_source", return_value="OpenStreetMap"):
            result = pipe._run_parallel_downloads(
                cfg, land_strat, build_strat, canopy_strat, dem_strat,
                None, None, {}
            )

        assert len(result) == 9  # 9-tuple
        np.testing.assert_array_equal(result[0], lc)  # land_cover_grid

    def test_parallel_download_failure_raises(self):
        """If one download fails, the exception propagates."""
        pipe = self._make_pipeline()
        cfg = _make_cfg(canopy_height_source="GEE")

        land_strat = MagicMock()
        land_strat.build_grid.side_effect = RuntimeError("network error")
        build_strat = MagicMock()
        build_strat.build_grids.return_value = (np.zeros((2, 2)),) * 4
        canopy_strat = MagicMock()
        canopy_strat.build_grids.return_value = (np.zeros((2, 2)),) * 2
        dem_strat = MagicMock()
        dem_strat.build_grid.return_value = np.zeros((2, 2))

        with pytest.raises(RuntimeError, match="network error"):
            pipe._run_parallel_downloads(
                cfg, land_strat, build_strat, canopy_strat, dem_strat,
                None, None, {}
            )


# ===========================================================================
# Tests for _run_parallel_downloads_static_canopy
# ===========================================================================

class TestRunParallelDownloadsStaticCanopy:
    """Cover pipeline.py lines 361-452: _run_parallel_downloads_static_canopy."""

    def _make_pipeline(self):
        from voxcity.generator.pipeline import VoxCityPipeline
        return VoxCityPipeline(
            meshsize=5.0,
            rectangle_vertices=[(0, 0), (0, 1), (1, 1), (1, 0)],
        )

    def test_static_canopy_parallel(self):
        """3-parallel downloads + deferred canopy result."""
        pipe = self._make_pipeline()
        cfg = _make_cfg(canopy_height_source="Static")

        lc = np.ones((3, 3), dtype=int)
        bh = np.zeros((3, 3)); bmin = np.empty((3, 3), dtype=object)
        for i in range(3):
            for j in range(3):
                bmin[i, j] = []
        bid = np.zeros((3, 3), dtype=int)
        dem = np.zeros((3, 3))

        land_strat = MagicMock()
        land_strat.build_grid.return_value = lc
        build_strat = MagicMock()
        build_strat.build_grids.return_value = (bh, bmin, bid, None)
        dem_strat = MagicMock()
        dem_strat.build_grid.return_value = dem

        with patch("voxcity.generator.grids.get_last_effective_land_cover_source", return_value="OpenStreetMap"):
            result = pipe._run_parallel_downloads_static_canopy(
                cfg, land_strat, build_strat, dem_strat,
                None, None, {}
            )

        assert len(result) == 7  # 7-tuple (no canopy)
        np.testing.assert_array_equal(result[0], lc)

    def test_static_canopy_parallel_failure(self):
        """Failure in one thread propagates."""
        pipe = self._make_pipeline()
        cfg = _make_cfg(canopy_height_source="Static")

        land_strat = MagicMock()
        land_strat.build_grid.return_value = np.ones((2, 2), dtype=int)
        build_strat = MagicMock()
        build_strat.build_grids.side_effect = RuntimeError("build fail")
        dem_strat = MagicMock()
        dem_strat.build_grid.return_value = np.zeros((2, 2))

        with pytest.raises(RuntimeError, match="build fail"):
            pipe._run_parallel_downloads_static_canopy(
                cfg, land_strat, build_strat, dem_strat,
                None, None, {}
            )
