"""Tests for voxcity.utils.classes module - voxel and land cover class definitions."""
import pytest
import numpy as np
from io import StringIO
import sys

from voxcity.utils.classes import (
    VOXEL_CODES,
    VOXEL_CODE_DESCRIPTIONS,
    LAND_COVER_CLASSES,
    LAND_COVER_DESCRIPTIONS,
    print_voxel_codes,
    print_land_cover_classes,
    print_class_definitions,
    get_land_cover_name,
    get_voxel_code_name,
    summarize_voxel_grid,
    summarize_land_cover_grid,
)


class TestVoxelCodes:
    def test_voxel_codes_dict(self):
        assert VOXEL_CODES[-3] == "Building"
        assert VOXEL_CODES[-2] == "Tree Canopy"
        assert VOXEL_CODES[-1] == "Ground/Subsurface"

    def test_voxel_code_descriptions_exists(self):
        assert "Building" in VOXEL_CODE_DESCRIPTIONS
        assert "Tree canopy" in VOXEL_CODE_DESCRIPTIONS


class TestLandCoverClasses:
    def test_land_cover_classes_dict(self):
        assert LAND_COVER_CLASSES[1] == "Bareland"
        assert LAND_COVER_CLASSES[9] == "Water"
        assert LAND_COVER_CLASSES[13] == "Building"
        assert LAND_COVER_CLASSES[14] == "No Data"

    def test_land_cover_classes_complete(self):
        # Should have classes 1-14
        for i in range(1, 15):
            assert i in LAND_COVER_CLASSES

    def test_land_cover_descriptions_exists(self):
        assert "Bareland" in LAND_COVER_DESCRIPTIONS
        assert "Water" in LAND_COVER_DESCRIPTIONS


class TestPrintFunctions:
    def test_print_voxel_codes(self, capsys):
        print_voxel_codes()
        captured = capsys.readouterr()
        assert "Building" in captured.out
        assert "-3" in captured.out

    def test_print_land_cover_classes(self, capsys):
        print_land_cover_classes()
        captured = capsys.readouterr()
        assert "Bareland" in captured.out
        assert "Water" in captured.out

    def test_print_class_definitions(self, capsys):
        print_class_definitions()
        captured = capsys.readouterr()
        assert "VoxCity Class Definitions" in captured.out
        assert "Building" in captured.out
        assert "Bareland" in captured.out


class TestGetLandCoverName:
    def test_valid_indices(self):
        assert get_land_cover_name(1) == "Bareland"
        assert get_land_cover_name(9) == "Water"
        assert get_land_cover_name(13) == "Building"

    def test_invalid_index(self):
        assert get_land_cover_name(0) == "Unknown"
        assert get_land_cover_name(100) == "Unknown"
        assert get_land_cover_name(-5) == "Unknown"


class TestGetVoxelCodeName:
    def test_negative_codes(self):
        assert get_voxel_code_name(-3) == "Building"
        assert get_voxel_code_name(-2) == "Tree Canopy"
        assert get_voxel_code_name(-1) == "Ground/Subsurface"

    def test_zero_code(self):
        assert get_voxel_code_name(0) == "Empty/Air"

    def test_positive_codes_land_cover(self):
        result = get_voxel_code_name(1)
        assert "Land Cover" in result
        assert "Bareland" in result

    def test_unknown_negative_code(self):
        assert get_voxel_code_name(-100) == "Unknown"


class TestSummarizeVoxelGrid:
    def test_simple_grid(self):
        grid = np.array([
            [[0, 0, -3], [0, -1, -1]],
            [[-2, -2, 0], [1, 1, 0]]
        ])
        summary = summarize_voxel_grid(grid, print_output=False)
        assert isinstance(summary, dict)
        assert 0 in summary  # air
        assert -3 in summary  # building
        assert -2 in summary  # tree
        assert -1 in summary  # ground

    def test_with_print(self, capsys):
        grid = np.array([[[0, -3], [-1, -2]]])
        summarize_voxel_grid(grid, print_output=True)
        captured = capsys.readouterr()
        assert "Voxel Grid Summary" in captured.out

    def test_count_accuracy(self):
        grid = np.zeros((2, 2, 2), dtype=int)
        grid[0, 0, 0] = -3  # one building voxel
        grid[0, 0, 1] = -3  # another building voxel
        summary = summarize_voxel_grid(grid, print_output=False)
        assert summary[-3] == 2
        assert summary[0] == 6  # remaining are air


class TestSummarizeLandCoverGrid:
    def test_simple_grid(self):
        grid = np.array([[1, 1, 9], [9, 13, 13]])
        summary = summarize_land_cover_grid(grid, print_output=False)
        assert summary[1] == 2   # Bareland
        assert summary[9] == 2   # Water
        assert summary[13] == 2  # Building

    def test_with_print(self, capsys):
        grid = np.array([[1, 9], [13, 14]])
        summarize_land_cover_grid(grid, print_output=True)
        captured = capsys.readouterr()
        assert "Land Cover Grid Summary" in captured.out

    def test_handles_source_specific_codes(self, capsys):
        """Test that non-standard codes (like 0 or negative) are labeled."""
        grid = np.array([[0, 1], [2, 3]])
        summarize_land_cover_grid(grid, print_output=True)
        captured = capsys.readouterr()
        assert "Source-specific" in captured.out or "Bareland" in captured.out
