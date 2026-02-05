"""Tests for voxcity.downloader.gba module."""
import pytest
import math

from voxcity.downloader.gba import (
    _bbox_from_rectangle_vertices,
    _pad_lon,
    _pad_lat,
    _lon_tag,
    _lat_tag,
    _snap_down,
    _snap_up,
    _generate_tile_bounds_for_bbox,
    _tile_filename,
)


class TestBboxFromRectangleVertices:
    """Tests for _bbox_from_rectangle_vertices function."""
    
    def test_basic_bbox(self):
        """Test basic bbox calculation."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        bbox = _bbox_from_rectangle_vertices(vertices)
        
        assert bbox == (0, 0, 1, 1)
    
    def test_negative_coordinates(self):
        """Test bbox with negative coordinates."""
        vertices = [(-10, -20), (-5, -20), (-5, -15), (-10, -15)]
        bbox = _bbox_from_rectangle_vertices(vertices)
        
        assert bbox == (-10, -20, -5, -15)
    
    def test_mixed_coordinates(self):
        """Test bbox with mixed positive/negative coordinates."""
        vertices = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
        bbox = _bbox_from_rectangle_vertices(vertices)
        
        assert bbox == (-5, -5, 5, 5)
    
    def test_real_coordinates(self):
        """Test bbox with real world coordinates (Tokyo)."""
        vertices = [
            (139.7564, 35.6713),
            (139.7564, 35.6758),
            (139.7619, 35.6758),
            (139.7619, 35.6713)
        ]
        bbox = _bbox_from_rectangle_vertices(vertices)
        
        min_lon, min_lat, max_lon, max_lat = bbox
        assert min_lon == pytest.approx(139.7564, rel=1e-4)
        assert max_lon == pytest.approx(139.7619, rel=1e-4)
    
    def test_empty_vertices_raises(self):
        """Test that empty vertices raises ValueError."""
        with pytest.raises(ValueError):
            _bbox_from_rectangle_vertices([])


class TestPaddingFunctions:
    """Tests for _pad_lon and _pad_lat functions."""
    
    def test_pad_lon_zero(self):
        """Test longitude padding for zero."""
        assert _pad_lon(0) == "000"
    
    def test_pad_lon_small(self):
        """Test longitude padding for small values."""
        assert _pad_lon(5) == "005"
        assert _pad_lon(10) == "010"
    
    def test_pad_lon_large(self):
        """Test longitude padding for large values."""
        assert _pad_lon(100) == "100"
        assert _pad_lon(180) == "180"
    
    def test_pad_lon_negative(self):
        """Test longitude padding for negative values (uses abs)."""
        assert _pad_lon(-60) == "060"
    
    def test_pad_lat_zero(self):
        """Test latitude padding for zero."""
        assert _pad_lat(0) == "00"
    
    def test_pad_lat_small(self):
        """Test latitude padding for small values."""
        assert _pad_lat(5) == "05"
        assert _pad_lat(10) == "10"
    
    def test_pad_lat_large(self):
        """Test latitude padding for large values."""
        assert _pad_lat(90) == "90"
    
    def test_pad_lat_negative(self):
        """Test latitude padding for negative values."""
        assert _pad_lat(-45) == "45"


class TestTagFunctions:
    """Tests for _lon_tag and _lat_tag functions."""
    
    def test_lon_tag_positive(self):
        """Test longitude tag for positive values (east)."""
        assert _lon_tag(0) == "e000"
        assert _lon_tag(10) == "e010"
        assert _lon_tag(140) == "e140"
    
    def test_lon_tag_negative(self):
        """Test longitude tag for negative values (west)."""
        assert _lon_tag(-60) == "w060"
        assert _lon_tag(-180) == "w180"
    
    def test_lat_tag_positive(self):
        """Test latitude tag for positive values (north)."""
        assert _lat_tag(0) == "n00"
        assert _lat_tag(35) == "n35"
        assert _lat_tag(90) == "n90"
    
    def test_lat_tag_negative(self):
        """Test latitude tag for negative values (south)."""
        assert _lat_tag(-25) == "s25"
        assert _lat_tag(-90) == "s90"


class TestSnapFunctions:
    """Tests for _snap_down and _snap_up functions."""
    
    def test_snap_down_exact(self):
        """Test snap down for exact multiples."""
        assert _snap_down(10, 5) == 10
        assert _snap_down(15, 5) == 15
    
    def test_snap_down_partial(self):
        """Test snap down for partial values."""
        assert _snap_down(12, 5) == 10
        assert _snap_down(14.9, 5) == 10
        assert _snap_down(17.5, 5) == 15
    
    def test_snap_down_negative(self):
        """Test snap down for negative values."""
        assert _snap_down(-7, 5) == -10
        assert _snap_down(-10, 5) == -10
    
    def test_snap_up_exact(self):
        """Test snap up for exact multiples."""
        assert _snap_up(10, 5) == 10
        assert _snap_up(15, 5) == 15
    
    def test_snap_up_partial(self):
        """Test snap up for partial values."""
        assert _snap_up(11, 5) == 15
        assert _snap_up(10.1, 5) == 15
    
    def test_snap_up_negative(self):
        """Test snap up for negative values."""
        assert _snap_up(-7, 5) == -5
        assert _snap_up(-10, 5) == -10


class TestGenerateTileBounds:
    """Tests for _generate_tile_bounds_for_bbox function."""
    
    def test_single_tile(self):
        """Test generation for area within single tile."""
        bounds = list(_generate_tile_bounds_for_bbox(1, 1, 4, 4, tile_size_deg=5))
        
        assert len(bounds) == 1
        assert bounds[0] == (0, 0, 5, 5)
    
    def test_multiple_tiles(self):
        """Test generation spanning multiple tiles."""
        bounds = list(_generate_tile_bounds_for_bbox(0, 0, 10, 10, tile_size_deg=5))
        
        # Should cover 2x2 = 4 tiles
        assert len(bounds) == 4
    
    def test_crossing_prime_meridian(self):
        """Test generation crossing prime meridian."""
        bounds = list(_generate_tile_bounds_for_bbox(-3, 50, 3, 52, tile_size_deg=5))
        
        # Should cover 2 tiles (west and east of prime meridian)
        assert len(bounds) >= 2
    
    def test_negative_area(self):
        """Test generation in negative coordinate area."""
        bounds = list(_generate_tile_bounds_for_bbox(-60, -30, -55, -25, tile_size_deg=5))
        
        assert len(bounds) >= 1
        # All bounds should be in expected range
        for west, south, east, north in bounds:
            assert west <= -55
            assert east >= -60


class TestTileFilename:
    """Tests for _tile_filename function."""
    
    def test_positive_coordinates(self):
        """Test filename for positive coordinates."""
        filename = _tile_filename(10, 45, 15, 50)
        
        assert filename == "e010_n50_e015_n45.parquet"
    
    def test_negative_longitude(self):
        """Test filename for negative longitude (west)."""
        filename = _tile_filename(-60, -30, -55, -25)
        
        assert filename.startswith("w060_")
        assert ".parquet" in filename
    
    def test_negative_latitude(self):
        """Test filename for negative latitude (south)."""
        filename = _tile_filename(140, -25, 145, -30)
        
        assert "s25" in filename
        assert "s30" in filename
    
    def test_filename_format(self):
        """Test that filename follows expected format."""
        filename = _tile_filename(0, 0, 5, 5)
        
        # Should end with .parquet
        assert filename.endswith(".parquet")
        
        # Should have 4 coordinate parts separated by underscores
        parts = filename.replace(".parquet", "").split("_")
        assert len(parts) == 4


class TestGbaFunctionSignatures:
    """Tests for function availability."""
    
    def test_load_gdf_function_exists(self):
        """Test that load_gdf_from_gba exists."""
        from voxcity.downloader.gba import load_gdf_from_gba
        assert callable(load_gdf_from_gba)
    
    def test_tile_url_function_exists(self):
        """Test that _tile_url exists."""
        from voxcity.downloader.gba import _tile_url
        assert callable(_tile_url)
