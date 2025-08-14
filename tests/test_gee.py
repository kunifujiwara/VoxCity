import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from voxcity.downloader.gee import (
    initialize_earth_engine,
    get_roi,
    get_ee_image_collection,
    get_ee_image,
    save_geotiff,
    get_dem_image,
    save_geotiff_esa_land_cover,
    save_geotiff_esri_landcover,
    save_geotiff_dynamic_world_v1,
    save_geotiff_open_buildings_temporal,
    save_geotiff_dsm_minus_dtm
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

class TestGoogleEarthEngine:
    """Tests for Google Earth Engine functions"""
    
    @patch('voxcity.downloader.gee.ee')
    def test_initialize_earth_engine(self, mock_ee):
        """Test Earth Engine initialization"""
        mock_ee.Initialize.return_value = None
        
        result = initialize_earth_engine()
        
        assert result is not None
        mock_ee.Initialize.assert_called_once()
    
    @patch('voxcity.downloader.gee.ee')
    def test_get_roi(self, mock_ee, sample_rectangle_vertices):
        """Test ROI creation from rectangle vertices"""
        mock_geometry = MagicMock()
        mock_ee.Geometry.Rectangle.return_value = mock_geometry
        
        result = get_roi(sample_rectangle_vertices)
        
        assert result is not None
        mock_ee.Geometry.Rectangle.assert_called_once()
    
    @patch('voxcity.downloader.gee.ee')
    def test_get_ee_image_collection(self, mock_ee):
        """Test Earth Engine image collection retrieval"""
        collection_name = 'LANDSAT/LC08/C02/T1_L2'
        start_date = '2020-01-01'
        end_date = '2020-12-31'
        
        mock_collection = MagicMock()
        mock_ee.ImageCollection.return_value = mock_collection
        
        result = get_ee_image_collection(collection_name, start_date, end_date)
        
        assert result is not None
        mock_ee.ImageCollection.assert_called_once()
    
    @patch('voxcity.downloader.gee.ee')
    def test_get_ee_image(self, mock_ee):
        """Test Earth Engine image retrieval"""
        image_id = 'LANDSAT/LC08/C02/T1_L2/LC08_123032_20200101'
        
        mock_image = MagicMock()
        mock_ee.Image.return_value = mock_image
        
        result = get_ee_image(image_id)
        
        assert result is not None
        mock_ee.Image.assert_called_once_with(image_id)
    
    @patch('voxcity.downloader.gee.ee')
    def test_save_geotiff(self, mock_ee, sample_rectangle_vertices, temp_output_dir):
        """Test GeoTIFF saving from Earth Engine image"""
        mock_image = MagicMock()
        output_path = temp_output_dir / "test.tif"
        
        result = save_geotiff(
            mock_image,
            sample_rectangle_vertices,
            str(output_path)
        )
        
        assert result is not None
        # Note: In a real test, we'd check if the file was created
        # but here we're just testing the function call
    
    @patch('voxcity.downloader.gee.ee')
    def test_get_dem_image(self, mock_ee, sample_rectangle_vertices):
        """Test DEM image retrieval"""
        source = 'SRTM'
        
        mock_image = MagicMock()
        mock_ee.Image.return_value = mock_image
        
        result = get_dem_image(sample_rectangle_vertices, source)
        
        assert result is not None
        mock_ee.Image.assert_called_once()
    
    @patch('voxcity.downloader.gee.ee')
    def test_save_geotiff_esa_land_cover(self, mock_ee, sample_rectangle_vertices, temp_output_dir):
        """Test ESA WorldCover land cover GeoTIFF saving"""
        output_path = temp_output_dir / "esa_landcover.tif"
        
        mock_image = MagicMock()
        mock_ee.Image.return_value = mock_image
        
        result = save_geotiff_esa_land_cover(
            sample_rectangle_vertices,
            str(output_path)
        )
        
        assert result is not None
        mock_ee.Image.assert_called_once()
    
    @patch('voxcity.downloader.gee.ee')
    def test_save_geotiff_esri_landcover(self, mock_ee, sample_rectangle_vertices, temp_output_dir):
        """Test ESRI land cover GeoTIFF saving"""
        output_path = temp_output_dir / "esri_landcover.tif"
        
        mock_image = MagicMock()
        mock_ee.Image.return_value = mock_image
        
        result = save_geotiff_esri_landcover(
            sample_rectangle_vertices,
            str(output_path)
        )
        
        assert result is not None
        mock_ee.Image.assert_called_once()
    
    @patch('voxcity.downloader.gee.ee')
    def test_save_geotiff_dynamic_world_v1(self, mock_ee, sample_rectangle_vertices, temp_output_dir):
        """Test Dynamic World V1 GeoTIFF saving"""
        output_path = temp_output_dir / "dynamic_world.tif"
        start_date = '2020-01-01'
        end_date = '2020-12-31'
        
        mock_image = MagicMock()
        mock_ee.Image.return_value = mock_image
        
        result = save_geotiff_dynamic_world_v1(
            sample_rectangle_vertices,
            str(output_path),
            start_date,
            end_date
        )
        
        assert result is not None
        mock_ee.Image.assert_called_once()
    
    @patch('voxcity.downloader.gee.ee')
    def test_save_geotiff_open_buildings_temporal(self, mock_ee, sample_rectangle_vertices, temp_output_dir):
        """Test Open Buildings Temporal GeoTIFF saving"""
        output_path = temp_output_dir / "open_buildings.tif"
        start_date = '2020-01-01'
        end_date = '2020-12-31'
        
        mock_image = MagicMock()
        mock_ee.Image.return_value = mock_image
        
        result = save_geotiff_open_buildings_temporal(
            sample_rectangle_vertices,
            str(output_path),
            start_date,
            end_date
        )
        
        assert result is not None
        mock_ee.Image.assert_called_once()
    
    @patch('voxcity.downloader.gee.ee')
    def test_save_geotiff_dsm_minus_dtm(self, mock_ee, sample_rectangle_vertices, temp_output_dir):
        """Test DSM minus DTM GeoTIFF saving"""
        output_path = temp_output_dir / "dsm_minus_dtm.tif"
        
        mock_image = MagicMock()
        mock_ee.Image.return_value = mock_image
        
        result = save_geotiff_dsm_minus_dtm(
            sample_rectangle_vertices,
            str(output_path)
        )
        
        assert result is not None
        mock_ee.Image.assert_called_once()
