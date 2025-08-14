import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from voxcity.geoprocessor.mesh import (
    create_voxel_mesh,
    create_sim_surface_mesh,
    create_city_meshes,
    export_meshes
)

@pytest.fixture
def sample_voxel_grid():
    """Sample voxel grid for testing"""
    return np.random.randint(0, 5, (10, 10, 10), dtype=int)

@pytest.fixture
def sample_building_height_grid():
    """Sample building height grid for testing"""
    return np.random.uniform(0, 50, (10, 10))

@pytest.fixture
def sample_land_cover_grid():
    """Sample land cover grid for testing"""
    return np.random.randint(1, 5, (10, 10), dtype=int)

@pytest.fixture
def sample_dem_grid():
    """Sample DEM grid for testing"""
    return np.random.uniform(0, 100, (10, 10))

class TestMeshGeneration:
    """Tests for mesh generation functions"""
    
    def test_create_voxel_mesh(self, sample_voxel_grid):
        """Test voxel mesh creation"""
        class_id = 1
        meshsize = 1.0
        
        result = create_voxel_mesh(
            sample_voxel_grid,
            class_id,
            meshsize
        )
        
        assert result is not None
        # Check if the result has the expected attributes
        assert hasattr(result, 'vertices')
        assert hasattr(result, 'faces')
    
    def test_create_sim_surface_mesh(self, sample_land_cover_grid, sample_dem_grid):
        """Test simulation surface mesh creation"""
        meshsize = 1.0
        land_cover_source = 'Urbanwatch'
        
        result = create_sim_surface_mesh(
            sample_land_cover_grid,
            sample_dem_grid,
            meshsize,
            land_cover_source
        )
        
        assert result is not None
        assert hasattr(result, 'vertices')
        assert hasattr(result, 'faces')
    
    def test_create_city_meshes(self, sample_voxel_grid, sample_building_height_grid, sample_dem_grid):
        """Test city meshes creation"""
        meshsize = 1.0
        land_cover_source = 'Urbanwatch'
        
        result = create_city_meshes(
            sample_voxel_grid,
            sample_building_height_grid,
            sample_dem_grid,
            meshsize,
            land_cover_source
        )
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_export_meshes(self, temp_output_dir):
        """Test mesh export functionality"""
        meshes = [MagicMock(), MagicMock()]
        base_filename = "test_mesh"
        
        result = export_meshes(
            meshes,
            temp_output_dir,
            base_filename
        )
        
        assert result is not None
