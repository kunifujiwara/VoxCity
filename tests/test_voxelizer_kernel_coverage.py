"""
Tests for voxcity.generator.voxelizer _voxelize_kernel to improve coverage.
"""

import numpy as np
import pytest

from voxcity.generator.voxelizer import (
    _voxelize_kernel,
    _flatten_building_segments,
    GROUND_CODE,
    TREE_CODE,
    BUILDING_CODE,
)


class TestVoxelizeGrid:
    def _run_voxelize(self, lc, dem, tree, canopy_bottom, building_grid, voxel_size=1.0, nz=20, trunk_ratio=0.5):
        """Helper to run _voxelize_kernel with proper setup."""
        rows, cols = lc.shape
        voxel_grid = np.zeros((rows, cols, nz), dtype=np.int8)

        has_canopy_bottom = canopy_bottom is not None
        if canopy_bottom is None:
            canopy_bottom = np.zeros_like(tree)

        seg_starts, seg_ends, seg_offsets, seg_counts = _flatten_building_segments(building_grid, voxel_size)

        _voxelize_kernel(
            voxel_grid, lc, dem, tree, canopy_bottom,
            has_canopy_bottom, seg_starts, seg_ends, seg_offsets, seg_counts,
            trunk_ratio, voxel_size
        )
        return voxel_grid

    def test_ground_layer(self):
        """Ground should be filled below surface."""
        lc = np.ones((3, 3), dtype=np.int32) * 2  # Some land cover code
        dem = np.zeros((3, 3))
        tree = np.zeros((3, 3))
        building = np.empty((3, 3), dtype=object)
        for i in range(3):
            for j in range(3):
                building[i, j] = []

        vg = self._run_voxelize(lc, dem, tree, None, building)
        # Ground level = 0/voxel_size + 0.5 + 1 = 1
        # z=0 should be land cover (2)
        assert vg[0, 0, 0] == 2  # land cover at surface

    def test_dem_elevation(self):
        """DEM raises the ground level."""
        lc = np.ones((3, 3), dtype=np.int32) * 3
        dem = np.ones((3, 3)) * 3.0  # 3m elevation
        tree = np.zeros((3, 3))
        building = np.empty((3, 3), dtype=object)
        for i in range(3):
            for j in range(3):
                building[i, j] = []

        vg = self._run_voxelize(lc, dem, tree, None, building)
        # ground_level = int(3.0/1.0 + 0.5) + 1 = 4
        # z=0..3 should be GROUND_CODE, z=3 should be land cover
        assert vg[0, 0, 0] == GROUND_CODE
        assert vg[0, 0, 3] == 3  # land cover at top of ground

    def test_trees_without_canopy_bottom(self):
        """Trees with default trunk ratio."""
        lc = np.ones((3, 3), dtype=np.int32)
        dem = np.zeros((3, 3))
        tree = np.zeros((3, 3))
        tree[1, 1] = 10.0  # 10m tree
        building = np.empty((3, 3), dtype=object)
        for i in range(3):
            for j in range(3):
                building[i, j] = []

        vg = self._run_voxelize(lc, dem, tree, None, building, trunk_ratio=0.5)
        # ground_level = 1, trunk ratio 0.5 => crown base = 5m (level 5), top = 10m (level 10)
        # Tree voxels at z = 1+5 to 1+10 = z6..z10
        assert np.any(vg[1, 1, :] == TREE_CODE)

    def test_trees_with_canopy_bottom(self):
        """Trees with explicit canopy bottom."""
        lc = np.ones((3, 3), dtype=np.int32)
        dem = np.zeros((3, 3))
        tree = np.zeros((3, 3))
        tree[1, 1] = 10.0
        canopy_bottom = np.zeros((3, 3))
        canopy_bottom[1, 1] = 3.0  # Crown starts at 3m
        building = np.empty((3, 3), dtype=object)
        for i in range(3):
            for j in range(3):
                building[i, j] = []

        vg = self._run_voxelize(lc, dem, tree, canopy_bottom, building)
        assert np.any(vg[1, 1, :] == TREE_CODE)

    def test_buildings(self):
        """Buildings from segment grid."""
        lc = np.ones((3, 3), dtype=np.int32)
        dem = np.zeros((3, 3))
        tree = np.zeros((3, 3))
        building = np.empty((3, 3), dtype=object)
        for i in range(3):
            for j in range(3):
                building[i, j] = []
        building[1, 1] = [[0.0, 15.0]]  # Building from 0m to 15m

        vg = self._run_voxelize(lc, dem, tree, None, building)
        assert np.any(vg[1, 1, :] == BUILDING_CODE)

    def test_building_with_min_height(self):
        """Building with elevated base (e.g., piloti)."""
        lc = np.ones((3, 3), dtype=np.int32)
        dem = np.zeros((3, 3))
        tree = np.zeros((3, 3))
        building = np.empty((3, 3), dtype=object)
        for i in range(3):
            for j in range(3):
                building[i, j] = []
        building[1, 1] = [[5.0, 20.0]]  # Building from 5m to 20m

        vg = self._run_voxelize(lc, dem, tree, None, building)
        # Building should not be at low z, but should be higher up
        assert np.any(vg[1, 1, :] == BUILDING_CODE)

    def test_mixed_scene(self):
        """Combined buildings and trees."""
        lc = np.ones((5, 5), dtype=np.int32) * 2
        dem = np.zeros((5, 5))
        tree = np.zeros((5, 5))
        tree[1, 1] = 8.0
        building = np.empty((5, 5), dtype=object)
        for i in range(5):
            for j in range(5):
                building[i, j] = []
        building[3, 3] = [[0.0, 20.0]]

        vg = self._run_voxelize(lc, dem, tree, None, building, nz=30)
        assert np.any(vg[1, 1, :] == TREE_CODE)
        assert np.any(vg[3, 3, :] == BUILDING_CODE)
        # Empty cells should still have ground
        assert vg[0, 0, 0] != 0

    def test_multiple_building_segments(self):
        """Building with multiple height segments."""
        lc = np.ones((3, 3), dtype=np.int32)
        dem = np.zeros((3, 3))
        tree = np.zeros((3, 3))
        building = np.empty((3, 3), dtype=object)
        for i in range(3):
            for j in range(3):
                building[i, j] = []
        building[1, 1] = [[0.0, 5.0], [8.0, 15.0]]  # Two segments

        vg = self._run_voxelize(lc, dem, tree, None, building)
        assert np.any(vg[1, 1, :] == BUILDING_CODE)
