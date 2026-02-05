"""Tests for voxcity.downloader.ocean module."""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from voxcity.downloader.ocean import (
    get_cache_path,
    CACHE_DIR,
)


class TestOceanCachePath:
    """Tests for cache path generation."""
    
    @pytest.fixture
    def sample_vertices(self):
        """Sample rectangle vertices."""
        return [
            (139.7, 35.6),
            (139.7, 35.61),
            (139.71, 35.61),
            (139.71, 35.6)
        ]
    
    def test_cache_path_returns_path(self, sample_vertices):
        """Test that cache path returns a Path object."""
        path = get_cache_path(sample_vertices, (100, 100))
        
        assert isinstance(path, Path)
    
    def test_cache_path_consistent(self, sample_vertices):
        """Test that same inputs produce same cache path."""
        path1 = get_cache_path(sample_vertices, (100, 100))
        path2 = get_cache_path(sample_vertices, (100, 100))
        
        assert path1 == path2
    
    def test_cache_path_different_for_different_vertices(self, sample_vertices):
        """Test that different vertices produce different paths."""
        different_vertices = [
            (140.7, 36.6),
            (140.7, 36.61),
            (140.71, 36.61),
            (140.71, 36.6)
        ]
        
        path1 = get_cache_path(sample_vertices, (100, 100))
        path2 = get_cache_path(different_vertices, (100, 100))
        
        assert path1 != path2
    
    def test_cache_path_different_for_different_shapes(self, sample_vertices):
        """Test that different grid shapes produce different paths."""
        path1 = get_cache_path(sample_vertices, (100, 100))
        path2 = get_cache_path(sample_vertices, (200, 200))
        
        assert path1 != path2
    
    def test_cache_path_filename_format(self, sample_vertices):
        """Test cache path filename format."""
        path = get_cache_path(sample_vertices, (100, 100))
        
        assert path.name.startswith("ocean_mask_")
        assert path.name.endswith(".npy")
    
    def test_cache_dir_constant(self):
        """Test CACHE_DIR is a Path object."""
        assert isinstance(CACHE_DIR, Path)
        assert "voxcity_ocean_cache" in str(CACHE_DIR)


class TestOceanFunctionSignatures:
    """Tests for ocean detection function availability."""
    
    def test_get_land_polygon_exists(self):
        """Test get_land_polygon_for_area function exists."""
        from voxcity.downloader.ocean import get_land_polygon_for_area
        assert callable(get_land_polygon_for_area)
    
    def test_query_coastlines_exists(self):
        """Test query_coastlines_from_overpass function exists."""
        from voxcity.downloader.ocean import query_coastlines_from_overpass
        assert callable(query_coastlines_from_overpass)


class TestOceanEdgeCases:
    """Edge case tests for ocean detection."""
    
    def test_cache_path_extreme_coordinates(self):
        """Test cache path with extreme coordinates."""
        extreme_vertices = [
            (-180, -90),
            (-180, 90),
            (180, 90),
            (180, -90)
        ]
        
        path = get_cache_path(extreme_vertices, (10, 10))
        assert isinstance(path, Path)
    
    def test_cache_path_small_area(self):
        """Test cache path with very small area."""
        small_vertices = [
            (0, 0),
            (0, 0.0001),
            (0.0001, 0.0001),
            (0.0001, 0)
        ]
        
        path = get_cache_path(small_vertices, (1, 1))
        assert isinstance(path, Path)
    
    def test_cache_path_large_grid(self):
        """Test cache path with large grid shape."""
        vertices = [(0, 0), (0, 1), (1, 1), (1, 0)]
        
        path = get_cache_path(vertices, (10000, 10000))
        assert isinstance(path, Path)
