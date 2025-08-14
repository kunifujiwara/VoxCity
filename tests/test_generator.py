import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from voxcity.generator import (
    get_land_cover_grid,
    get_building_height_grid,
    get_canopy_height_grid,
    get_dem_grid,
    create_3d_voxel,
    get_voxcity
)

@pytest.fixture
def sample_rectangle_vertices():
    """Sample rectangle vertices for testing"""
    return [
        (139.7564216011559, 35.671290792464255),
        (139.7564216011559, 35.67579720669077),
        (139.76194439884412, 35.67579720669077),
        (139.76194439884412, 35.671290792464255)
    ]

@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)

@pytest.fixture
def sample_building_gdf():
    """Sample building GeoDataFrame for testing"""
    import geopandas as gpd
    from shapely.geometry import Polygon
    
    buildings = [
        {
            'geometry': Polygon([(139.7564, 35.6713), (139.7564, 35.6758), 
                               (139.7619, 35.6758), (139.7619, 35.6713)]),
            'height': 25.0,
            'min_height': 0.0,
            'id': 1
        }
    ]
    return gpd.GeoDataFrame(buildings, crs='EPSG:4326')

class TestGeneratorFunctions:
    """Tests for main generator functions"""
    
    @patch('voxcity.generator.save_geotiff_esa_land_cover')
    def test_get_land_cover_grid(self, mock_save, sample_rectangle_vertices, temp_output_dir):
        """Test land cover grid generation"""
        meshsize = 100
        source = 'ESA_WorldCover'
        
        with patch('voxcity.generator.initialize_earth_engine'):
            result = get_land_cover_grid(
                sample_rectangle_vertices, 
                meshsize, 
                source, 
                temp_output_dir
            )
        
        assert result is not None
        mock_save.assert_called_once()
    
    @patch('voxcity.generator.create_building_height_grid_from_gdf_polygon')
    def test_get_building_height_grid_with_gdf(self, mock_create, sample_rectangle_vertices, 
                                             temp_output_dir, sample_building_gdf):
        """Test building height grid generation with GeoDataFrame input"""
        meshsize = 100
        source = 'GeoDataFrame'
        
        result = get_building_height_grid(
            sample_rectangle_vertices,
            meshsize,
            source,
            temp_output_dir,
            building_gdf=sample_building_gdf
        )
        
        assert result is not None
        mock_create.assert_called_once()
    
    @patch('voxcity.generator.save_geotiff_dsm_minus_dtm')
    def test_get_canopy_height_grid(self, mock_save, sample_rectangle_vertices, temp_output_dir):
        """Test canopy height grid generation"""
        meshsize = 100
        source = 'GEDI'
        
        with patch('voxcity.generator.initialize_earth_engine'):
            result = get_canopy_height_grid(
                sample_rectangle_vertices,
                meshsize,
                source,
                temp_output_dir
            )
        
        assert result is not None
        mock_save.assert_called_once()
    
    @patch('voxcity.generator.get_dem_image')
    def test_get_dem_grid(self, mock_dem, sample_rectangle_vertices, temp_output_dir):
        """Test DEM grid generation"""
        meshsize = 100
        source = 'SRTM'
        
        with patch('voxcity.generator.initialize_earth_engine'):
            result = get_dem_grid(
                sample_rectangle_vertices,
                meshsize,
                source,
                temp_output_dir
            )
        
        assert result is not None
        mock_dem.assert_called_once()
    
    def test_create_3d_voxel(self):
        """Test 3D voxel creation"""
        building_height_grid = np.array([[10, 20], [15, 25]])
        building_min_height_grid = np.array([[0, 0], [0, 0]])
        land_cover_grid = np.array([[1, 2], [1, 2]])
        dem_grid = np.array([[5, 5], [5, 5]])
        tree_grid = np.array([[0, 0], [0, 0]])
        voxel_size = 10
        
        result = create_3d_voxel(
            building_height_grid,
            building_min_height_grid,
            land_cover_grid,
            dem_grid,
            tree_grid,
            voxel_size
        )
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
    
    @patch('voxcity.generator.get_land_cover_grid')
    @patch('voxcity.generator.get_building_height_grid')
    @patch('voxcity.generator.get_canopy_height_grid')
    @patch('voxcity.generator.get_dem_grid')
    @patch('voxcity.generator.create_3d_voxel')
    def test_get_voxcity(self, mock_voxel, mock_dem, mock_canopy, mock_building, 
                        mock_landcover, sample_rectangle_vertices, temp_output_dir):
        """Test complete voxcity generation"""
        meshsize = 100
        building_source = 'GeoDataFrame'
        land_cover_source = 'ESA_WorldCover'
        canopy_height_source = 'GEDI'
        dem_source = 'SRTM'
        
        # Mock return values
        mock_landcover.return_value = np.array([[1, 2], [1, 2]])
        mock_building.return_value = (np.array([[10, 20], [15, 25]]), 
                                    np.array([[0, 0], [0, 0]]))
        mock_canopy.return_value = np.array([[0, 0], [0, 0]])
        mock_dem.return_value = np.array([[5, 5], [5, 5]])
        mock_voxel.return_value = np.zeros((2, 2, 3))
        
        result = get_voxcity(
            sample_rectangle_vertices,
            building_source,
            land_cover_source,
            canopy_height_source,
            dem_source,
            meshsize,
            output_dir=temp_output_dir
        )
        
        assert result is not None
        mock_landcover.assert_called_once()
        mock_building.assert_called_once()
        mock_canopy.assert_called_once()
        mock_dem.assert_called_once()
        mock_voxel.assert_called_once()
