"""
Tests for voxelizer.py Python fallback loop (lines 251-278)
when NUMBA_AVAILABLE=False.
"""
import numpy as np
import pytest
from unittest.mock import patch


class TestVoxelizerPythonFallback:
    """Cover the pure-Python fallback when NUMBA_AVAILABLE=False."""

    @patch("voxcity.generator.voxelizer.NUMBA_AVAILABLE", False)
    def test_fallback_loop_basic(self):
        from voxcity.generator.voxelizer import Voxelizer
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="OpenStreetMap", trunk_height_ratio=0.3)
        building_height = np.array([[0, 5.0], [3.0, 0]])
        # building_min_height: each cell is a list of (min_h, max_h) tuples
        building_min_height = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                building_min_height[i, j] = []
        building_min_height[0, 1] = [(0, 5.0)]
        building_min_height[1, 0] = [(0, 3.0)]
        building_id = np.array([[0, 1], [2, 0]], dtype=int)
        land_cover = np.array([[1, 1], [1, 1]], dtype=np.int8)
        dem = np.zeros((2, 2))
        tree_grid = np.array([[0, 0], [0, 4.0]])

        vox = voxelizer.generate_combined(
            building_height_grid_ori=building_height,
            building_min_height_grid_ori=building_min_height,
            building_id_grid_ori=building_id,
            land_cover_grid_ori=land_cover,
            dem_grid_ori=dem,
            tree_grid_ori=tree_grid,
        )
        assert vox.ndim == 3
        assert vox.shape[0] == 2 and vox.shape[1] == 2

    @patch("voxcity.generator.voxelizer.NUMBA_AVAILABLE", False)
    def test_fallback_with_canopy_bottom(self):
        from voxcity.generator.voxelizer import Voxelizer
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="OpenStreetMap", trunk_height_ratio=0.3)
        building_height = np.zeros((3, 3))
        building_min_height = np.empty((3, 3), dtype=object)
        for i in range(3):
            for j in range(3):
                building_min_height[i, j] = []
        building_id = np.zeros((3, 3), dtype=int)
        land_cover = np.ones((3, 3), dtype=np.int8)
        dem = np.zeros((3, 3))
        tree_grid = np.array([[0, 0, 0], [0, 8.0, 0], [0, 0, 0]])
        canopy_bottom = np.array([[0, 0, 0], [0, 3.0, 0], [0, 0, 0]])

        vox = voxelizer.generate_combined(
            building_height_grid_ori=building_height,
            building_min_height_grid_ori=building_min_height,
            building_id_grid_ori=building_id,
            land_cover_grid_ori=land_cover,
            dem_grid_ori=dem,
            tree_grid_ori=tree_grid,
            canopy_bottom_height_grid_ori=canopy_bottom,
        )
        assert vox.ndim == 3
        # tree voxels should be present
        assert np.any(vox == -2)  # TREE_CODE


class TestReplaceNanInNested:
    """Cover replace_nan_in_nested function."""

    def test_replace_nan_values_in_lists(self):
        from voxcity.generator.voxelizer import replace_nan_in_nested
        arr = np.empty((2, 2), dtype=object)
        arr[0, 0] = [[0, 5.0]]
        arr[0, 1] = [[float('nan'), 3.0]]  # list segment, NaN should be replaced
        arr[1, 0] = []
        arr[1, 1] = [[1.0, float('nan')]]
        result = replace_nan_in_nested(arr)
        # NaN values in list segments should be replaced with 10.0
        assert result[0, 1][0][0] == 10.0
        assert result[1, 1][0][1] == 10.0

    def test_replace_nan_values_in_arrays(self):
        from voxcity.generator.voxelizer import replace_nan_in_nested
        arr = np.empty((1, 1), dtype=object)
        arr[0, 0] = [np.array([float('nan'), 3.0])]
        result = replace_nan_in_nested(arr)
        assert result[0, 0][0][0] == 10.0

    def test_none_and_empty(self):
        from voxcity.generator.voxelizer import replace_nan_in_nested
        arr = np.empty((2, 1), dtype=object)
        arr[0, 0] = None
        arr[1, 0] = []
        result = replace_nan_in_nested(arr)
        assert result[0, 0] == []
        assert result[1, 0] == []

    def test_non_array_input(self):
        from voxcity.generator.voxelizer import replace_nan_in_nested
        result = replace_nan_in_nested("not_an_array")
        assert result == "not_an_array"
