"""Tests for voxcity.utils.orientation module."""
import pytest
import numpy as np

from voxcity.utils.orientation import (
    ORIENTATION_NORTH_UP,
    ORIENTATION_SOUTH_UP,
    ensure_orientation,
)


class TestOrientationConstants:
    def test_north_up_value(self):
        assert ORIENTATION_NORTH_UP == "north_up"

    def test_south_up_value(self):
        assert ORIENTATION_SOUTH_UP == "south_up"


class TestEnsureOrientation:
    @pytest.fixture
    def sample_grid(self):
        """A simple 3x3 grid with distinct values to track flipping."""
        return np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])

    def test_same_orientation_returns_unchanged(self, sample_grid):
        result = ensure_orientation(sample_grid, "north_up", "north_up")
        assert np.array_equal(result, sample_grid)

    def test_same_orientation_south_up(self, sample_grid):
        result = ensure_orientation(sample_grid, "south_up", "south_up")
        assert np.array_equal(result, sample_grid)

    def test_north_up_to_south_up_flips_vertically(self, sample_grid):
        result = ensure_orientation(sample_grid, "north_up", "south_up")
        expected = np.array([
            [7, 8, 9],
            [4, 5, 6],
            [1, 2, 3],
        ])
        assert np.array_equal(result, expected)

    def test_south_up_to_north_up_flips_vertically(self, sample_grid):
        result = ensure_orientation(sample_grid, "south_up", "north_up")
        expected = np.array([
            [7, 8, 9],
            [4, 5, 6],
            [1, 2, 3],
        ])
        assert np.array_equal(result, expected)

    def test_double_flip_restores_original(self, sample_grid):
        flipped_once = ensure_orientation(sample_grid, "north_up", "south_up")
        flipped_twice = ensure_orientation(flipped_once, "south_up", "north_up")
        assert np.array_equal(flipped_twice, sample_grid)

    def test_no_copy_when_no_change_needed(self, sample_grid):
        result = ensure_orientation(sample_grid, "north_up", "north_up")
        # Should return same object (no copy)
        assert result is sample_grid

    def test_works_with_3d_array(self):
        grid_3d = np.arange(24).reshape((2, 3, 4))
        result = ensure_orientation(grid_3d, "north_up", "south_up")
        # flipud flips along axis 0
        expected = np.flipud(grid_3d)
        assert np.array_equal(result, expected)

    def test_works_with_float_array(self):
        grid = np.array([[1.5, 2.5], [3.5, 4.5]])
        result = ensure_orientation(grid, "north_up", "south_up")
        expected = np.array([[3.5, 4.5], [1.5, 2.5]])
        assert np.allclose(result, expected)

    def test_default_output_orientation(self, sample_grid):
        # Default orientation_out should be north_up
        result = ensure_orientation(sample_grid, "south_up")
        expected = np.flipud(sample_grid)
        assert np.array_equal(result, expected)


import numpy as np

from voxcity.utils.orientation import (
    ensure_orientation,
    to_rasterio_layout,
    from_rasterio_layout,
    grid_to_rotated_raster,
    voxels_to_magicavoxel_axes,
    voxels_to_kji,
)


def test_to_rasterio_layout_matches_inline_transpose():
    g = np.arange(12).reshape(3, 4)
    out = to_rasterio_layout(g)
    assert np.array_equal(out, g.T)
    assert out.flags["C_CONTIGUOUS"]


def test_rasterio_layout_round_trip():
    g = np.arange(12).reshape(3, 4)
    assert np.array_equal(from_rasterio_layout(to_rasterio_layout(g)), g)


def test_from_rasterio_layout_matches_inline():
    arr = np.arange(12).reshape(4, 3)
    out = from_rasterio_layout(arr)
    assert np.array_equal(out, arr.T)
    assert out.flags["C_CONTIGUOUS"]


def test_grid_to_rotated_raster_matches_inline():
    g = np.arange(12).reshape(3, 4)
    assert np.array_equal(grid_to_rotated_raster(g), np.flipud(g.T))


def test_voxels_to_magicavoxel_axes_matches_inline():
    a = np.arange(24).reshape(2, 3, 4)
    expected = np.transpose(np.flip(a, axis=2), (0, 2, 1))
    assert np.array_equal(voxels_to_magicavoxel_axes(a), expected)


def test_voxels_to_kji_matches_inline():
    a = np.arange(24).reshape(2, 3, 4)
    assert np.array_equal(voxels_to_kji(a), a.transpose(2, 0, 1))


def test_ensure_orientation_involution():
    g = np.arange(12).reshape(3, 4)
    flipped = ensure_orientation(g, "south_up", "north_up")
    assert np.array_equal(ensure_orientation(flipped, "north_up", "south_up"), g)
