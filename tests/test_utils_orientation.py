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
