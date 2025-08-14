import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from voxcity.downloader.utils import (
    download_file,
    download_file_google_drive
)

from voxcity.downloader.osm import (
    load_gdf_from_openstreetmap,
    load_land_cover_gdf_from_osm,
    osm_json_to_geojson,
    is_way_polygon,
    get_way_coords
)

from voxcity.downloader.mbfp import get_mbfp_gdf

from voxcity.downloader.overture import (
    load_gdf_from_overture,
    convert_gdf_to_geojson,
    rectangle_to_bbox
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
def sample_osm_data():
    """Sample OSM data for testing"""
    return {
        "elements": [
            {
                "type": "way",
                "id": 123,
                "nodes": [1, 2, 3, 4],
                "tags": {"building": "yes", "height": "25"}
            },
            {
                "type": "node",
                "id": 1,
                "lat": 35.6713,
                "lon": 139.7564
            }
        ]
    }

class TestDownloaderUtils:
    """Tests for downloader utility functions"""
    
    def test_download_file(self, tmp_path):
        """Test file download functionality"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'test content'
        
        with patch('requests.get', return_value=mock_response):
            filepath = tmp_path / "test.txt"
            download_file("http://test.com/file", str(filepath))
            assert filepath.exists()
            assert filepath.read_bytes() == b'test content'
    
    @patch('gdown.download')
    def test_download_file_google_drive(self, mock_gdown, tmp_path):
        """Test Google Drive file download"""
        mock_gdown.return_value = True
        result = download_file_google_drive("test_id", str(tmp_path / "test.txt"))
        assert result is True
        mock_gdown.assert_called_once()

class TestOSMDownloader:
    """Tests for OpenStreetMap downloader functions"""
    
    @patch('voxcity.downloader.osm.overpass_query')
    def test_load_gdf_from_openstreetmap(self, mock_overpass, sample_rectangle_vertices):
        """Test OSM building data loading"""
        mock_overpass.return_value = sample_osm_data
        
        result = load_gdf_from_openstreetmap(sample_rectangle_vertices)
        assert result is not None
        mock_overpass.assert_called_once()
    
    @patch('voxcity.downloader.osm.overpass_query')
    def test_load_land_cover_gdf_from_osm(self, mock_overpass, sample_rectangle_vertices):
        """Test OSM land cover data loading"""
        mock_overpass.return_value = sample_osm_data
        
        result = load_land_cover_gdf_from_osm(sample_rectangle_vertices)
        assert result is not None
        mock_overpass.assert_called_once()
    
    def test_osm_json_to_geojson(self, sample_osm_data):
        """Test OSM to GeoJSON conversion"""
        result = osm_json_to_geojson(sample_osm_data)
        assert isinstance(result, dict)
        assert "features" in result
    
    def test_is_way_polygon(self):
        """Test way polygon detection"""
        way = {"tags": {"area": "yes"}}
        assert is_way_polygon(way) is True
        
        way = {"tags": {"building": "yes"}}
        assert is_way_polygon(way) is True
        
        way = {"tags": {"highway": "residential"}}
        assert is_way_polygon(way) is False
    
    def test_get_way_coords(self):
        """Test way coordinate extraction"""
        way = {"nodes": [1, 2, 3]}
        nodes = {
            1: {"lat": 35.6713, "lon": 139.7564},
            2: {"lat": 35.6758, "lon": 139.7564},
            3: {"lat": 35.6758, "lon": 139.7619}
        }
        
        coords = get_way_coords(way, nodes)
        assert len(coords) == 3
        assert all(len(coord) == 2 for coord in coords)

class TestMBFPDownloader:
    """Tests for MBFP downloader functions"""
    
    @patch('voxcity.downloader.mbfp.gpd.read_file')
    def test_get_mbfp_gdf(self, mock_read, sample_rectangle_vertices):
        """Test MBFP data loading"""
        mock_gdf = MagicMock()
        mock_read.return_value = mock_gdf
        
        result = get_mbfp_gdf(sample_rectangle_vertices)
        assert result is not None
        mock_read.assert_called_once()

class TestOvertureDownloader:
    """Tests for Overture downloader functions"""
    
    @patch('voxcity.downloader.overture.requests.get')
    def test_load_gdf_from_overture(self, mock_get, sample_rectangle_vertices):
        """Test Overture data loading"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"features": []}
        mock_get.return_value = mock_response
        
        result = load_gdf_from_overture(sample_rectangle_vertices)
        assert result is not None
        mock_get.assert_called_once()
    
    def test_convert_gdf_to_geojson(self):
        """Test GeoDataFrame to GeoJSON conversion"""
        import geopandas as gpd
        from shapely.geometry import Polygon
        
        gdf = gpd.GeoDataFrame(
            [{"geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])}],
            crs='EPSG:4326'
        )
        
        result = convert_gdf_to_geojson(gdf)
        assert isinstance(result, dict)
        assert "features" in result
    
    def test_rectangle_to_bbox(self, sample_rectangle_vertices):
        """Test rectangle to bbox conversion"""
        bbox = rectangle_to_bbox(sample_rectangle_vertices)
        assert len(bbox) == 4
        assert all(isinstance(coord, float) for coord in bbox)
