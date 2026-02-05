"""Tests for voxcity.utils.shape helper functions."""
import pytest
import numpy as np

from voxcity.utils.shape import (
    _compute_center_crop_indices,
    _pad_split,
    _pad_crop_2d,
    _pad_crop_3d_zbottom,
    normalize_voxcity_shape,
)
from voxcity.models import VoxCity, VoxelGrid, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid


class TestComputeCenterCropIndices:
    """Tests for _compute_center_crop_indices function."""

    def test_no_crop_needed(self):
        """Test when size <= target."""
        start, end = _compute_center_crop_indices(5, 10)
        assert start == 0
        assert end == 5

    def test_crop_even_excess(self):
        """Test cropping with even excess."""
        start, end = _compute_center_crop_indices(10, 6)
        assert end - start == 6
        assert start == 2

    def test_crop_odd_excess(self):
        """Test cropping with odd excess."""
        start, end = _compute_center_crop_indices(10, 5)
        assert end - start == 5
        # start = max(0, (10 - 5) // 2) = 2
        assert start == 2

    def test_size_equals_target(self):
        """Test when size equals target."""
        start, end = _compute_center_crop_indices(8, 8)
        assert start == 0
        assert end == 8

    def test_crop_to_one(self):
        """Test cropping to size 1."""
        start, end = _compute_center_crop_indices(10, 1)
        assert end - start == 1


class TestPadSplit:
    """Tests for _pad_split function."""

    def test_even_padding(self):
        """Test splitting even padding."""
        a, b = _pad_split(10)
        assert a + b == 10
        assert a == 5
        assert b == 5

    def test_odd_padding(self):
        """Test splitting odd padding."""
        a, b = _pad_split(9)
        assert a + b == 9
        # a = 9 // 2 = 4, b = 9 - 4 = 5
        assert a == 4
        assert b == 5

    def test_zero_padding(self):
        """Test zero padding."""
        a, b = _pad_split(0)
        assert a == 0
        assert b == 0

    def test_one_padding(self):
        """Test padding of 1."""
        a, b = _pad_split(1)
        assert a == 0
        assert b == 1


class TestPadCrop2d:
    """Tests for _pad_crop_2d function."""

    def test_no_change_needed(self):
        """Test when array already matches target."""
        arr = np.ones((5, 5))
        result = _pad_crop_2d(arr, (5, 5), pad_value=0)
        np.testing.assert_array_equal(result, arr)

    def test_padding_only(self):
        """Test padding when array is smaller."""
        arr = np.ones((3, 3))
        result = _pad_crop_2d(arr, (5, 5), pad_value=0, align_xy="center")
        assert result.shape == (5, 5)
        # Center should have ones
        assert result[1, 1] == 1
        # Edges should be padded with zeros
        assert result[0, 0] == 0

    def test_cropping_only(self):
        """Test cropping when array is larger."""
        arr = np.arange(25).reshape(5, 5)
        result = _pad_crop_2d(arr, (3, 3), pad_value=0, align_xy="center")
        assert result.shape == (3, 3)

    def test_pad_and_crop_different_dimensions(self):
        """Test when one dimension needs padding and other needs cropping."""
        arr = np.ones((3, 7))
        result = _pad_crop_2d(arr, (5, 5), pad_value=0, align_xy="center")
        assert result.shape == (5, 5)

    def test_top_left_alignment_padding(self):
        """Test top-left alignment for padding."""
        arr = np.ones((3, 3))
        result = _pad_crop_2d(arr, (5, 5), pad_value=0, align_xy="top-left")
        assert result.shape == (5, 5)
        # Original array should be at top-left
        assert result[0, 0] == 1
        assert result[2, 2] == 1
        # Padding should be on bottom/right
        assert result[4, 4] == 0

    def test_top_left_alignment_cropping(self):
        """Test top-left alignment for cropping."""
        arr = np.arange(25).reshape(5, 5)
        result = _pad_crop_2d(arr, (3, 3), pad_value=0, align_xy="top-left")
        assert result.shape == (3, 3)
        # Should keep top-left portion
        assert result[0, 0] == 0
        assert result[2, 2] == 12

    def test_no_crop_when_not_allowed(self):
        """Test that cropping is skipped when not allowed."""
        arr = np.ones((5, 5))
        result = _pad_crop_2d(arr, (3, 3), pad_value=0, allow_crop_xy=False)
        # Should return original since cropping not allowed
        assert result.shape == (5, 5)

    def test_3d_array_padding(self):
        """Test padding with 3D array (preserves trailing dims)."""
        arr = np.ones((3, 3, 2))
        result = _pad_crop_2d(arr, (5, 5), pad_value=0, align_xy="center")
        assert result.shape == (5, 5, 2)


class TestPadCrop3dZbottom:
    """Tests for _pad_crop_3d_zbottom function."""

    def test_no_change_needed(self):
        """Test when array already matches target."""
        arr = np.ones((5, 5, 5))
        result = _pad_crop_3d_zbottom(arr, (5, 5, 5), pad_value=0)
        np.testing.assert_array_equal(result, arr)

    def test_z_padding_at_top(self):
        """Test that Z padding is added at top (preserves ground at z=0)."""
        arr = np.ones((3, 3, 3))
        result = _pad_crop_3d_zbottom(arr, (3, 3, 5), pad_value=0)
        assert result.shape == (3, 3, 5)
        # Original data should be at bottom (z=0,1,2)
        assert result[1, 1, 0] == 1
        assert result[1, 1, 2] == 1
        # Padding at top (z=3,4)
        assert result[1, 1, 3] == 0
        assert result[1, 1, 4] == 0

    def test_z_no_crop_by_default(self):
        """Test that Z is not cropped when allow_crop_z=False."""
        arr = np.ones((3, 3, 10))
        result = _pad_crop_3d_zbottom(arr, (3, 3, 5), pad_value=0, allow_crop_z=False)
        # Should keep original Z since cropping not allowed
        assert result.shape == (3, 3, 10)

    def test_z_crop_when_allowed(self):
        """Test that Z is cropped when allow_crop_z=True."""
        arr = np.arange(27).reshape(3, 3, 3)
        result = _pad_crop_3d_zbottom(arr, (3, 3, 2), pad_value=0, allow_crop_z=True)
        assert result.shape == (3, 3, 2)
        # Should keep z=0,1 (ground level preserved)
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 1

    def test_xy_padding(self):
        """Test XY padding combined with Z."""
        arr = np.ones((3, 3, 3))
        result = _pad_crop_3d_zbottom(arr, (5, 5, 5), pad_value=0)
        assert result.shape == (5, 5, 5)

    def test_combined_xy_crop_z_pad(self):
        """Test XY cropping with Z padding."""
        arr = np.ones((10, 10, 3))
        result = _pad_crop_3d_zbottom(arr, (5, 5, 5), pad_value=0)
        assert result.shape == (5, 5, 5)


class TestNormalizeVoxcityShape:
    """Tests for normalize_voxcity_shape function."""

    @pytest.fixture
    def simple_voxcity(self):
        """Create a simple VoxCity for testing."""
        meta = {"meshsize": 1.0, "bounds": (0, 0, 10, 10)}
        
        voxels = VoxelGrid(
            classes=np.zeros((5, 5, 5), dtype=np.int32),
            meta=meta
        )
        
        # Create object dtype array for min_heights
        min_heights = np.empty((5, 5), dtype=object)
        for i in range(5):
            for j in range(5):
                min_heights[i, j] = []
        
        buildings = BuildingGrid(
            heights=np.zeros((5, 5), dtype=np.float32),
            min_heights=min_heights,
            ids=np.zeros((5, 5), dtype=np.int32),
            meta=meta
        )
        
        land_cover = LandCoverGrid(
            classes=np.zeros((5, 5), dtype=np.int32),
            meta=meta
        )
        
        dem = DemGrid(
            elevation=np.zeros((5, 5), dtype=np.float32),
            meta=meta
        )
        
        canopy = CanopyGrid(
            top=np.zeros((5, 5), dtype=np.float32),
            bottom=np.zeros((5, 5), dtype=np.float32),
            meta=meta
        )
        
        return VoxCity(
            voxels=voxels,
            buildings=buildings,
            land_cover=land_cover,
            dem=dem,
            tree_canopy=canopy,
        )

    def test_same_shape_no_change(self, simple_voxcity):
        """Test that same target shape returns equivalent data."""
        result = normalize_voxcity_shape(simple_voxcity, (5, 5, 5))
        assert result.voxels.classes.shape == (5, 5, 5)

    def test_expand_shape(self, simple_voxcity):
        """Test expanding the shape."""
        result = normalize_voxcity_shape(simple_voxcity, (8, 8, 8))
        assert result.voxels.classes.shape == (8, 8, 8)
        assert result.buildings.heights.shape == (8, 8)
        assert result.land_cover.classes.shape == (8, 8)
        assert result.dem.elevation.shape == (8, 8)

    def test_shrink_xy_expand_z(self, simple_voxcity):
        """Test shrinking XY while expanding Z."""
        result = normalize_voxcity_shape(simple_voxcity, (3, 3, 8))
        assert result.voxels.classes.shape == (3, 3, 8)

    def test_custom_pad_values(self, simple_voxcity):
        """Test with custom pad values."""
        pad_values = {
            "voxels": -1,
            "dem": 100.0,
        }
        result = normalize_voxcity_shape(
            simple_voxcity, 
            (8, 8, 8), 
            pad_values=pad_values
        )
        # Check that padding was applied
        assert result.voxels.classes.shape == (8, 8, 8)

    def test_top_left_alignment(self, simple_voxcity):
        """Test with top-left alignment."""
        result = normalize_voxcity_shape(
            simple_voxcity, 
            (8, 8, 8),
            align_xy="top-left"
        )
        assert result.voxels.classes.shape == (8, 8, 8)

    def test_preserves_metadata(self, simple_voxcity):
        """Test that metadata is preserved."""
        result = normalize_voxcity_shape(simple_voxcity, (8, 8, 8))
        assert result.voxels.meta == simple_voxcity.voxels.meta

    def test_min_heights_preserved(self, simple_voxcity):
        """Test that min_heights array type is preserved."""
        result = normalize_voxcity_shape(simple_voxcity, (8, 8, 8))
        assert result.buildings.min_heights.dtype == object

    def test_canopy_bottom_none(self):
        """Test handling when canopy bottom is None."""
        meta = {"meshsize": 1.0, "bounds": (0, 0, 10, 10)}
        
        voxels = VoxelGrid(
            classes=np.zeros((5, 5, 5), dtype=np.int32),
            meta=meta
        )
        
        min_heights = np.empty((5, 5), dtype=object)
        for i in range(5):
            for j in range(5):
                min_heights[i, j] = []
        
        buildings = BuildingGrid(
            heights=np.zeros((5, 5), dtype=np.float32),
            min_heights=min_heights,
            ids=np.zeros((5, 5), dtype=np.int32),
            meta=meta
        )
        
        land_cover = LandCoverGrid(
            classes=np.zeros((5, 5), dtype=np.int32),
            meta=meta
        )
        
        dem = DemGrid(
            elevation=np.zeros((5, 5), dtype=np.float32),
            meta=meta
        )
        
        canopy = CanopyGrid(
            top=np.zeros((5, 5), dtype=np.float32),
            bottom=None,
            meta=meta
        )
        
        city = VoxCity(
            voxels=voxels,
            buildings=buildings,
            land_cover=land_cover,
            dem=dem,
            tree_canopy=canopy,
        )
        
        result = normalize_voxcity_shape(city, (8, 8, 8))
        assert result.tree_canopy.bottom is None
