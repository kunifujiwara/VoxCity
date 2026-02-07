"""
Comprehensive tests for voxcity.generator.voxelizer to improve coverage.
"""

import numpy as np
import pytest

from voxcity.generator.voxelizer import (
    Voxelizer,
    replace_nan_in_nested,
    _flatten_building_segments,
    GROUND_CODE,
    TREE_CODE,
    BUILDING_CODE,
)


class TestReplaceNanInNested:
    def test_basic_replacement(self):
        arr = np.empty((2, 2), dtype=object)
        arr[0, 0] = [[0.0, float('nan')]]
        arr[0, 1] = [[1.0, 2.0]]
        arr[1, 0] = []
        arr[1, 1] = None

        result = replace_nan_in_nested(arr)
        # NaN should be replaced with 10.0
        assert result[0, 0][0][1] == 10.0
        assert result[0, 1][0] == [1.0, 2.0]
        assert result[1, 0] == []
        assert result[1, 1] == []

    def test_custom_replace_value(self):
        arr = np.empty((1, 1), dtype=object)
        arr[0, 0] = [[float('nan'), 5.0]]

        result = replace_nan_in_nested(arr, replace_value=99.0)
        assert result[0, 0][0][0] == 99.0
        assert result[0, 0][0][1] == 5.0

    def test_non_array_passthrough(self):
        result = replace_nan_in_nested("not an array")
        assert result == "not an array"

    def test_numpy_segment(self):
        arr = np.empty((1, 1), dtype=object)
        arr[0, 0] = [np.array([float('nan'), 3.0])]

        result = replace_nan_in_nested(arr)
        assert result[0, 0][0][0] == 10.0
        assert result[0, 0][0][1] == 3.0

    def test_empty_cells(self):
        arr = np.empty((2, 2), dtype=object)
        arr[0, 0] = []
        arr[0, 1] = []
        arr[1, 0] = None
        arr[1, 1] = []

        result = replace_nan_in_nested(arr)
        assert result[0, 0] == []
        assert result[1, 0] == []

    def test_multiple_segments(self):
        arr = np.empty((1, 1), dtype=object)
        arr[0, 0] = [[0.0, 5.0], [10.0, float('nan')]]

        result = replace_nan_in_nested(arr)
        assert result[0, 0][0] == [0.0, 5.0]
        assert result[0, 0][1][1] == 10.0

    def test_no_nan_values(self):
        arr = np.empty((1, 1), dtype=object)
        arr[0, 0] = [[1.0, 2.0], [3.0, 4.0]]

        result = replace_nan_in_nested(arr)
        assert result[0, 0][0] == [1.0, 2.0]
        assert result[0, 0][1] == [3.0, 4.0]


class TestFlattenBuildingSegments:
    def test_basic(self):
        grid = np.empty((2, 2), dtype=object)
        grid[0, 0] = [[0.0, 5.0]]
        grid[0, 1] = [[2.0, 8.0]]
        grid[1, 0] = []
        grid[1, 1] = [[0.0, 3.0], [5.0, 10.0]]

        seg_starts, seg_ends, offsets, counts = _flatten_building_segments(grid, 1.0)

        assert counts[0, 0] == 1
        assert counts[0, 1] == 1
        assert counts[1, 0] == 0
        assert counts[1, 1] == 2
        assert len(seg_starts) == 4  # total segments

    def test_empty_grid(self):
        grid = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                grid[i, j] = []

        seg_starts, seg_ends, offsets, counts = _flatten_building_segments(grid, 1.0)
        assert np.all(counts == 0)
        assert len(seg_starts) == 0

    def test_voxel_size_scaling(self):
        grid = np.empty((1, 1), dtype=object)
        grid[0, 0] = [[0.0, 10.0]]

        seg_starts, seg_ends, offsets, counts = _flatten_building_segments(grid, 2.0)
        assert counts[0, 0] == 1
        # 10.0 / 2.0 + 0.5 = 5.5 -> 5
        assert seg_ends[0] == 5


class TestVoxelizerInit:
    def test_basic_init(self):
        v = Voxelizer(voxel_size=1.0, land_cover_source='OpenStreetMap')
        assert v.voxel_size == 1.0
        assert v.land_cover_source == 'OpenStreetMap'
        assert v.trunk_height_ratio == pytest.approx(11.76 / 19.98)

    def test_custom_trunk_ratio(self):
        v = Voxelizer(voxel_size=2.0, land_cover_source='ESA WorldCover',
                      trunk_height_ratio=0.5)
        assert v.trunk_height_ratio == 0.5

    def test_memory_limit(self):
        v = Voxelizer(voxel_size=1.0, land_cover_source='OpenStreetMap',
                      max_voxel_ram_mb=100.0)
        assert v.max_voxel_ram_mb == 100.0


class TestVoxelizerEstimateAndAllocate:
    def test_basic_allocation(self):
        v = Voxelizer(voxel_size=1.0, land_cover_source='OpenStreetMap')
        grid = v._estimate_and_allocate(10, 10, 20)
        assert grid.shape == (10, 10, 20)
        assert np.all(grid == 0)

    def test_prints_memory_info(self, capsys):
        v = Voxelizer(voxel_size=1.0, land_cover_source='OpenStreetMap')
        grid = v._estimate_and_allocate(5, 5, 5)
        captured = capsys.readouterr()
        assert 'Voxel grid shape' in captured.out
        assert grid.shape == (5, 5, 5)


class TestVoxelizerConvertLandCover:
    def test_osm_source(self):
        v = Voxelizer(voxel_size=1.0, land_cover_source='OpenStreetMap')
        grid = np.array([[0, 1, 2], [3, 4, 5]])
        result = v._convert_land_cover(grid)
        expected = grid + 1
        np.testing.assert_array_equal(result, expected)


class TestVoxelizerGenerateCombined:
    def test_simple_voxelization(self):
        v = Voxelizer(voxel_size=1.0, land_cover_source='OpenStreetMap')

        size = (5, 5)
        building_height = np.zeros(size)
        building_height[2, 2] = 3.0

        building_min_height = np.empty(size, dtype=object)
        for i in range(size[0]):
            for j in range(size[1]):
                building_min_height[i, j] = []
        building_min_height[2, 2] = [[0.0, 3.0]]

        building_id = np.zeros(size)
        building_id[2, 2] = 1

        land_cover = np.ones(size, dtype=int) * 2  # Rangeland index
        dem = np.zeros(size)
        tree = np.zeros(size)

        result = v.generate_combined(
            building_height, building_min_height, building_id,
            land_cover, dem, tree,
            print_class_info=False,
        )

        assert result.ndim == 3
        assert result.shape[0] == size[0]
        assert result.shape[1] == size[1]
        # Ground level should have land cover or ground code
        assert np.any(result != 0)

    def test_with_trees(self):
        v = Voxelizer(voxel_size=1.0, land_cover_source='OpenStreetMap')

        size = (4, 4)
        building_height = np.zeros(size)
        building_min_height = np.empty(size, dtype=object)
        for i in range(size[0]):
            for j in range(size[1]):
                building_min_height[i, j] = []

        building_id = np.zeros(size)
        land_cover = np.ones(size, dtype=int)
        dem = np.zeros(size)
        tree = np.zeros(size)
        tree[1, 1] = 5.0

        result = v.generate_combined(
            building_height, building_min_height, building_id,
            land_cover, dem, tree,
            print_class_info=False,
        )

        assert result.ndim == 3
        # There should be tree voxels somewhere
        assert np.any(result == TREE_CODE)


class TestVoxelizerConstants:
    def test_ground_code(self):
        assert GROUND_CODE == -1

    def test_tree_code(self):
        assert TREE_CODE == -2

    def test_building_code(self):
        assert BUILDING_CODE == -3
