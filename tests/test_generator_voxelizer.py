"""Tests for voxcity.generator.voxelizer module."""
import pytest
import numpy as np

from voxcity.generator.voxelizer import (
    Voxelizer,
    _flatten_building_segments,
    replace_nan_in_nested,
    GROUND_CODE,
    TREE_CODE,
    BUILDING_CODE,
)


class TestVoxelCodes:
    def test_ground_code_value(self):
        assert GROUND_CODE == -1

    def test_tree_code_value(self):
        assert TREE_CODE == -2

    def test_building_code_value(self):
        assert BUILDING_CODE == -3

    def test_codes_are_negative(self):
        """Voxel codes should be negative to distinguish from land cover."""
        assert GROUND_CODE < 0
        assert TREE_CODE < 0
        assert BUILDING_CODE < 0

    def test_codes_are_unique(self):
        codes = [GROUND_CODE, TREE_CODE, BUILDING_CODE]
        assert len(codes) == len(set(codes))


class TestFlattenBuildingSegments:
    def test_empty_grid(self):
        """Test with no buildings."""
        # Must be 2D array with object dtype containing lists
        grid = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                grid[i, j] = []
        starts, ends, offsets, counts = _flatten_building_segments(grid, 1.0)
        
        assert len(starts) == 0
        assert len(ends) == 0
        assert counts.sum() == 0

    def test_single_building(self):
        """Test with one building segment."""
        grid = np.array([
            [[(0, 10)], []],
            [[], []]
        ], dtype=object)
        
        starts, ends, offsets, counts = _flatten_building_segments(grid, 1.0)
        
        assert counts[0, 0] == 1
        assert counts[0, 1] == 0
        assert starts[0] == 0  # min_height/voxel_size
        assert ends[0] == 10   # max_height/voxel_size

    def test_multiple_segments_same_cell(self):
        """Test with multiple building segments in one cell (stacked buildings)."""
        grid = np.array([
            [[(0, 5), (10, 15)], []],
            [[], []]
        ], dtype=object)
        
        starts, ends, offsets, counts = _flatten_building_segments(grid, 1.0)
        
        assert counts[0, 0] == 2
        assert starts[0] == 0
        assert ends[0] == 5
        assert starts[1] == 10
        assert ends[1] == 15

    def test_voxel_size_scaling(self):
        """Test that voxel size correctly scales segment heights."""
        grid = np.empty((1, 1), dtype=object)
        grid[0, 0] = [(0, 10)]
        
        starts, ends, offsets, counts = _flatten_building_segments(grid, 2.0)
        
        # With voxel_size=2, heights should be halved
        assert starts[0] == 0
        assert ends[0] == 5  # 10/2


class TestVoxelizer:
    @pytest.fixture
    def voxelizer(self):
        return Voxelizer(voxel_size=1.0, land_cover_source="Urbanwatch")

    def test_initialization(self, voxelizer):
        assert voxelizer.voxel_size == 1.0
        assert voxelizer.land_cover_source == "Urbanwatch"

    def test_default_trunk_height_ratio(self, voxelizer):
        # Default ratio is 11.76/19.98
        expected = 11.76 / 19.98
        assert voxelizer.trunk_height_ratio == pytest.approx(expected)

    def test_custom_trunk_height_ratio(self):
        voxelizer = Voxelizer(
            voxel_size=1.0,
            land_cover_source="OpenStreetMap",
            trunk_height_ratio=0.5
        )
        assert voxelizer.trunk_height_ratio == 0.5

    def test_voxel_dtype(self):
        voxelizer = Voxelizer(
            voxel_size=1.0,
            land_cover_source="Urbanwatch",
            voxel_dtype=np.int16
        )
        assert voxelizer.voxel_dtype == np.int16

    def test_estimate_and_allocate(self, voxelizer):
        grid = voxelizer._estimate_and_allocate(10, 10, 20)
        assert grid.shape == (10, 10, 20)
        assert grid.dtype == np.int8

    def test_convert_land_cover_osm(self):
        """OpenStreetMap should just add 1 to shift to 1-based indices."""
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="OpenStreetMap")
        arr = np.array([[0, 1, 2]])
        result = voxelizer._convert_land_cover(arr)
        assert result.tolist() == [[1, 2, 3]]

    def test_convert_land_cover_urbanwatch(self, voxelizer):
        """Urbanwatch should use the convert_land_cover function."""
        arr = np.array([[0, 1, 2]], dtype=np.uint8)
        result = voxelizer._convert_land_cover(arr)
        # Should be mapped: 0->13, 1->12, 2->11
        assert result.tolist() == [[13, 12, 11]]


class TestVoxelizerGenerateCombined:
    @pytest.fixture
    def simple_inputs(self):
        """Create minimal input grids for testing."""
        shape = (3, 3)
        
        # Building heights (10m building at center)
        building_heights = np.zeros(shape)
        building_heights[1, 1] = 10.0
        
        # Building min heights (simple list structure)
        building_min_heights = np.empty(shape, dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                building_min_heights[i, j] = []
        building_min_heights[1, 1] = [(0, 10)]
        
        # Building IDs
        building_ids = np.zeros(shape, dtype=int)
        building_ids[1, 1] = 1
        
        # Land cover (all grass = 2)
        land_cover = np.full(shape, 2, dtype=np.uint8)
        
        # DEM (flat terrain)
        dem = np.zeros(shape)
        
        # Tree heights (5m tree at corner)
        tree_heights = np.zeros(shape)
        tree_heights[0, 0] = 5.0
        
        return {
            "building_height_grid_ori": building_heights,
            "building_min_height_grid_ori": building_min_heights,
            "building_id_grid_ori": building_ids,
            "land_cover_grid_ori": land_cover,
            "dem_grid_ori": dem,
            "tree_grid_ori": tree_heights,
        }

    def test_generate_combined_shape(self, simple_inputs):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="Urbanwatch")
        result = voxelizer.generate_combined(**simple_inputs, print_class_info=False)
        
        # Should have correct x,y dimensions
        assert result.shape[0] == 3
        assert result.shape[1] == 3
        # Z dimension should be > 0
        assert result.shape[2] > 0

    def test_generate_combined_has_building(self, simple_inputs):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="Urbanwatch")
        result = voxelizer.generate_combined(**simple_inputs, print_class_info=False)
        
        # Should have building code somewhere
        assert BUILDING_CODE in result

    def test_generate_combined_has_tree(self, simple_inputs):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="Urbanwatch")
        result = voxelizer.generate_combined(**simple_inputs, print_class_info=False)
        
        # Should have tree code somewhere
        assert TREE_CODE in result

    def test_generate_combined_has_ground(self, simple_inputs):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="Urbanwatch")
        result = voxelizer.generate_combined(**simple_inputs, print_class_info=False)
        
        # The voxelizer puts land cover class (positive int) at z=0 layer
        # Land cover values are positive (e.g., 11 for developed space from Urbanwatch)
        # Check that there are positive values at z=0 (land cover layer)
        assert np.any(result[:, :, 0] > 0)


class TestReplaceNanInNested:
    """Tests for replace_nan_in_nested function."""

    def test_non_array_returns_unchanged(self):
        """Non-array input should be returned unchanged."""
        result = replace_nan_in_nested("not an array")
        assert result == "not an array"
        
        result = replace_nan_in_nested(42)
        assert result == 42
        
        result = replace_nan_in_nested(None)
        assert result is None

    def test_empty_cells_stay_empty(self):
        """Empty list cells should remain empty lists."""
        arr = np.empty((2, 2), dtype=object)
        arr[0, 0] = []
        arr[0, 1] = []
        arr[1, 0] = []
        arr[1, 1] = []
        
        result = replace_nan_in_nested(arr)
        
        assert result[0, 0] == []
        assert result[0, 1] == []

    def test_none_cells_become_empty(self):
        """None cells should become empty lists."""
        arr = np.empty((1, 1), dtype=object)
        arr[0, 0] = None
        
        result = replace_nan_in_nested(arr)
        
        assert result[0, 0] == []

    def test_replaces_nan_in_list_segments(self):
        """NaN values in list segments should be replaced."""
        arr = np.empty((1, 1), dtype=object)
        arr[0, 0] = [[np.nan, 10.0], [5.0, np.nan]]
        
        result = replace_nan_in_nested(arr, replace_value=99.0)
        
        assert result[0, 0][0][0] == 99.0
        assert result[0, 0][0][1] == 10.0
        assert result[0, 0][1][0] == 5.0
        assert result[0, 0][1][1] == 99.0

    def test_replaces_nan_in_numpy_segments(self):
        """NaN values in numpy array segments should be replaced."""
        arr = np.empty((1, 1), dtype=object)
        arr[0, 0] = [np.array([np.nan, 10.0])]
        
        result = replace_nan_in_nested(arr, replace_value=7.0)
        
        assert result[0, 0][0][0] == 7.0
        assert result[0, 0][0][1] == 10.0

    def test_preserves_non_nan_values(self):
        """Non-NaN values should be preserved."""
        arr = np.empty((1, 1), dtype=object)
        arr[0, 0] = [[1.0, 2.0], [3.0, 4.0]]
        
        result = replace_nan_in_nested(arr, replace_value=99.0)
        
        assert result[0, 0][0][0] == 1.0
        assert result[0, 0][0][1] == 2.0
        assert result[0, 0][1][0] == 3.0
        assert result[0, 0][1][1] == 4.0

    def test_default_replace_value(self):
        """Default replace value should be 10.0."""
        arr = np.empty((1, 1), dtype=object)
        arr[0, 0] = [[np.nan]]
        
        result = replace_nan_in_nested(arr)
        
        assert result[0, 0][0][0] == 10.0

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        arr = np.empty((3, 4), dtype=object)
        for i in range(3):
            for j in range(4):
                arr[i, j] = []
        
        result = replace_nan_in_nested(arr)
        
        assert result.shape == (3, 4)
