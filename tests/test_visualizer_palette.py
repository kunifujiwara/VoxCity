"""Tests for voxcity.visualizer.palette module."""
import pytest

from voxcity.visualizer.palette import get_voxel_color_map


class TestGetVoxelColorMap:
    """Tests for get_voxel_color_map function."""

    def test_default_returns_dict(self):
        """Test default scheme returns dictionary."""
        result = get_voxel_color_map('default')
        assert isinstance(result, dict)

    def test_default_has_building_code(self):
        """Test default scheme has building code -3."""
        result = get_voxel_color_map('default')
        assert -3 in result

    def test_default_has_tree_code(self):
        """Test default scheme has tree code -2."""
        result = get_voxel_color_map('default')
        assert -2 in result

    def test_default_has_ground_code(self):
        """Test default scheme has ground code -1."""
        result = get_voxel_color_map('default')
        assert -1 in result

    def test_rgb_values_are_lists(self):
        """Test that all RGB values are lists."""
        result = get_voxel_color_map('default')
        for code, rgb in result.items():
            assert isinstance(rgb, list), f"Code {code} RGB is not a list"
            assert len(rgb) == 3, f"Code {code} RGB does not have 3 values"

    def test_rgb_values_in_range(self):
        """Test RGB values are in 0-255 range."""
        result = get_voxel_color_map('default')
        for code, rgb in result.items():
            for channel in rgb:
                assert 0 <= channel <= 255, f"Code {code} has invalid RGB value"

    def test_high_contrast_scheme(self):
        """Test high_contrast scheme."""
        result = get_voxel_color_map('high_contrast')
        assert isinstance(result, dict)
        assert -3 in result

    def test_monochrome_scheme(self):
        """Test monochrome scheme."""
        result = get_voxel_color_map('monochrome')
        assert isinstance(result, dict)
        assert -3 in result

    def test_pastel_scheme(self):
        """Test pastel scheme."""
        result = get_voxel_color_map('pastel')
        assert isinstance(result, dict)
        assert -3 in result

    def test_dark_mode_scheme(self):
        """Test dark_mode scheme."""
        result = get_voxel_color_map('dark_mode')
        assert isinstance(result, dict)
        assert -3 in result

    def test_grayscale_scheme(self):
        """Test grayscale scheme."""
        result = get_voxel_color_map('grayscale')
        assert isinstance(result, dict)
        assert -3 in result

    def test_white_mode_scheme(self):
        """Test white_mode scheme."""
        result = get_voxel_color_map('white_mode')
        assert isinstance(result, dict)
        assert -3 in result

    def test_unknown_scheme_returns_default(self):
        """Test unknown scheme returns default."""
        result = get_voxel_color_map('nonexistent_scheme')
        default = get_voxel_color_map('default')
        assert result == default

    def test_all_schemes_have_same_keys(self):
        """Test all schemes have same voxel codes."""
        default_keys = set(get_voxel_color_map('default').keys())
        schemes = ['high_contrast', 'monochrome', 'pastel', 'dark_mode', 'grayscale', 'white_mode']
        
        for scheme in schemes:
            scheme_keys = set(get_voxel_color_map(scheme).keys())
            assert scheme_keys == default_keys, f"{scheme} has different keys than default"

    def test_negative_codes_for_built_environment(self):
        """Test that negative codes exist for built environment."""
        result = get_voxel_color_map('default')
        # Built environment codes
        expected_negative = [-3, -2, -1]  # Building, Tree, Ground
        for code in expected_negative:
            assert code in result

    def test_positive_codes_for_land_cover(self):
        """Test that positive codes exist for land cover."""
        result = get_voxel_color_map('default')
        # Land cover classes 1-14
        for code in range(1, 15):
            assert code in result
