import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from voxcity.simulator.solar import (
    compute_direct_solar_irradiance_map_binary,
    get_direct_solar_irradiance_map,
    get_diffuse_solar_irradiance_map,
    get_global_solar_irradiance_map,
    get_solar_positions_astral,
    _configure_num_threads,
    _auto_time_batch_size
)

from voxcity.simulator.view import (
    get_view_index,
    get_sky_view_factor_map,
    get_surface_view_factor
)

@pytest.fixture
def sample_voxel_data():
    """Sample voxel data for testing"""
    return np.random.randint(0, 2, (10, 10, 10), dtype=bool)

@pytest.fixture
def sample_sun_direction():
    """Sample sun direction vector"""
    return np.array([0.0, 0.0, 1.0])

@pytest.fixture
def sample_view_point():
    """Sample view point coordinates"""
    return np.array([5.0, 5.0, 5.0])

class TestSolarSimulation:
    """Tests for solar simulation functions"""
    
    def test_compute_direct_solar_irradiance_map_binary(self, sample_voxel_data, sample_sun_direction):
        """Test direct solar irradiance map computation (binary)"""
        view_point_height = 5.0
        hit_values = [1, 2, 3]
        meshsize = 1.0
        tree_k = 0.5
        tree_lad = 0.3
        inclusion_mode = 'binary'
        
        result = compute_direct_solar_irradiance_map_binary(
            sample_voxel_data,
            sample_sun_direction,
            view_point_height,
            hit_values,
            meshsize,
            tree_k,
            tree_lad,
            inclusion_mode
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_voxel_data.shape[:2]
    
    def test_get_direct_solar_irradiance_map(self, sample_voxel_data):
        """Test direct solar irradiance map generation"""
        meshsize = 1.0
        azimuth_degrees = 180.0
        elevation_degrees = 45.0
        view_point_height = 5.0
        hit_values = [1, 2, 3]
        tree_k = 0.5
        tree_lad = 0.3
        
        result = get_direct_solar_irradiance_map(
            sample_voxel_data,
            meshsize,
            azimuth_degrees,
            elevation_degrees,
            view_point_height,
            hit_values,
            tree_k,
            tree_lad
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_voxel_data.shape[:2]
    
    def test_get_diffuse_solar_irradiance_map(self, sample_voxel_data):
        """Test diffuse solar irradiance map generation"""
        meshsize = 1.0
        diffuse_irradiance = 1.0
        
        result = get_diffuse_solar_irradiance_map(
            sample_voxel_data,
            meshsize,
            diffuse_irradiance
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_voxel_data.shape[:2]
    
    def test_get_global_solar_irradiance_map(self, sample_voxel_data):
        """Test global solar irradiance map generation"""
        meshsize = 1.0
        azimuth_degrees = 180.0
        elevation_degrees = 45.0
        view_point_height = 5.0
        hit_values = [1, 2, 3]
        tree_k = 0.5
        tree_lad = 0.3
        diffuse_irradiance = 1.0
        
        result = get_global_solar_irradiance_map(
            sample_voxel_data,
            meshsize,
            azimuth_degrees,
            elevation_degrees,
            view_point_height,
            hit_values,
            tree_k,
            tree_lad,
            diffuse_irradiance
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_voxel_data.shape[:2]
    
    def test_get_solar_positions_astral(self):
        """Test solar position calculation using astral"""
        times = ['2024-01-01 12:00:00', '2024-01-01 18:00:00']
        lon = 139.7564
        lat = 35.6713
        
        result = get_solar_positions_astral(times, lon, lat)
        
        assert isinstance(result, dict)
        assert 'azimuth' in result
        assert 'elevation' in result
    
    def test_configure_num_threads(self):
        """Test thread configuration"""
        result = _configure_num_threads(desired_threads=4)
        assert isinstance(result, int)
        assert result > 0
    
    def test_auto_time_batch_size(self):
        """Test automatic time batch size calculation"""
        n_faces = 1000
        total_steps = 100
        
        result = _auto_time_batch_size(n_faces, total_steps)
        assert isinstance(result, int)
        assert result > 0

class TestViewSimulation:
    """Tests for view simulation functions"""
    
    def test_get_view_index(self, sample_voxel_data):
        """Test view index calculation"""
        meshsize = 1.0
        mode = 'default'
        hit_values = [1, 2, 3]
        
        result = get_view_index(
            sample_voxel_data,
            meshsize,
            mode,
            hit_values
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_voxel_data.shape[:2]
    
    def test_get_sky_view_factor_map(self, sample_voxel_data):
        """Test sky view factor map generation"""
        meshsize = 1.0
        
        result = get_sky_view_factor_map(
            sample_voxel_data,
            meshsize
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_voxel_data.shape[:2]
    
    def test_get_surface_view_factor(self, sample_voxel_data):
        """Test surface view factor calculation"""
        meshsize = 1.0
        
        result = get_surface_view_factor(
            sample_voxel_data,
            meshsize
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_voxel_data.shape[:2]
