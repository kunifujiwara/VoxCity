"""Tests for voxcity.utils.shape module - shape normalization utilities."""
import pytest
import numpy as np

from voxcity.utils.shape import (
    _compute_center_crop_indices,
    _pad_split,
    _pad_crop_2d,
    _pad_crop_3d_zbottom,
)


class TestComputeCenterCropIndices:
    def test_size_less_than_target(self):
        start, end = _compute_center_crop_indices(5, 10)
        assert start == 0
        assert end == 5

    def test_size_equal_to_target(self):
        start, end = _compute_center_crop_indices(10, 10)
        assert start == 0
        assert end == 10

    def test_size_greater_than_target(self):
        start, end = _compute_center_crop_indices(10, 6)
        assert start == 2
        assert end == 8
        assert end - start == 6

    def test_odd_difference(self):
        # size=11, target=8 -> crop 3 total, start = (11-8)//2 = 1
        start, end = _compute_center_crop_indices(11, 8)
        assert start == 1
        assert end == 9
        assert end - start == 8


class TestPadSplit:
    def test_even_padding(self):
        a, b = _pad_split(10)
        assert a == 5
        assert b == 5
        assert a + b == 10

    def test_odd_padding(self):
        a, b = _pad_split(7)
        assert a == 3
        assert b == 4
        assert a + b == 7

    def test_zero_padding(self):
        a, b = _pad_split(0)
        assert a == 0
        assert b == 0


class TestPadCrop2D:
    def test_no_change_needed(self):
        arr = np.array([[1, 2], [3, 4]])
        result = _pad_crop_2d(arr, (2, 2), pad_value=0)
        assert np.array_equal(result, arr)

    def test_padding_center(self):
        arr = np.array([[1, 2], [3, 4]])
        result = _pad_crop_2d(arr, (4, 4), pad_value=0, align_xy="center")
        assert result.shape == (4, 4)
        # Center should contain original
        assert result[1, 1] == 1
        assert result[1, 2] == 2
        assert result[2, 1] == 3
        assert result[2, 2] == 4
        # Edges should be padded
        assert result[0, 0] == 0
        assert result[3, 3] == 0

    def test_cropping_center(self):
        arr = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ])
        result = _pad_crop_2d(arr, (2, 2), pad_value=0, align_xy="center", allow_crop_xy=True)
        assert result.shape == (2, 2)
        # Should be center 2x2
        assert result[0, 0] == 6
        assert result[0, 1] == 7
        assert result[1, 0] == 10
        assert result[1, 1] == 11

    def test_top_left_alignment_padding(self):
        arr = np.array([[1, 2], [3, 4]])
        result = _pad_crop_2d(arr, (4, 4), pad_value=0, align_xy="top-left")
        assert result.shape == (4, 4)
        # Original at top-left
        assert result[0, 0] == 1
        assert result[0, 1] == 2
        assert result[1, 0] == 3
        assert result[1, 1] == 4
        # Padding at bottom-right
        assert result[2, 2] == 0
        assert result[3, 3] == 0

    def test_crop_disabled(self):
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = _pad_crop_2d(arr, (2, 2), pad_value=0, allow_crop_xy=False)
        # Should not crop, return original shape
        assert result.shape == (3, 3)


class TestPadCrop3DZBottom:
    def test_no_change_needed(self):
        arr = np.ones((2, 2, 5))
        result = _pad_crop_3d_zbottom(arr, (2, 2, 5), pad_value=0)
        assert result.shape == (2, 2, 5)
        assert np.all(result == 1)

    def test_z_padding_at_top(self):
        arr = np.ones((2, 2, 3))
        result = _pad_crop_3d_zbottom(arr, (2, 2, 6), pad_value=0)
        assert result.shape == (2, 2, 6)
        # Original values preserved at bottom
        assert np.all(result[:, :, :3] == 1)
        # Padding at top (z=3,4,5)
        assert np.all(result[:, :, 3:] == 0)

    def test_z_expand_when_crop_not_allowed(self):
        arr = np.ones((2, 2, 10))
        # Target z is 5, but allow_crop_z=False, so should expand to 10
        result = _pad_crop_3d_zbottom(arr, (2, 2, 5), pad_value=0, allow_crop_z=False)
        assert result.shape == (2, 2, 10)

    def test_z_crop_when_allowed(self):
        arr = np.arange(20).reshape((2, 2, 5))
        result = _pad_crop_3d_zbottom(arr, (2, 2, 3), pad_value=0, allow_crop_z=True)
        assert result.shape == (2, 2, 3)
        # Should keep bottom z layers
        assert np.array_equal(result, arr[:, :, :3])

    def test_xy_and_z_combined(self):
        arr = np.ones((3, 3, 4))
        result = _pad_crop_3d_zbottom(arr, (5, 5, 6), pad_value=0, align_xy="center")
        assert result.shape == (5, 5, 6)
