import pytest
import numpy as np
from shapely.geometry import Polygon, Point
from unittest.mock import patch, MagicMock

from voxcity.geoprocessor.draw import (
    rotate_rectangle,
    draw_rectangle_map,
    get_polygon_vertices
)

from voxcity.geoprocessor.polygon import (
    get_gdf_from_gpkg,
    save_geojson
)

@pytest.fixture
def sample_polygon():
    """Sample polygon for testing"""
    return Polygon([
        (0, 0), (1, 0), (1, 1), (0, 1)
    ])

@pytest.fixture
def sample_rectangle_vertices():
    """Sample rectangle vertices for testing"""
    return [(0, 0), (1, 0), (1, 1), (0, 1)]

class TestDrawingFunctions:
    """Tests for drawing functions"""
    
    def test_rotate_rectangle(self, sample_rectangle_vertices):
        """Test rectangle rotation"""
        angle = 90  # degrees
        
        result = rotate_rectangle(None, sample_rectangle_vertices, angle)
        
        assert result is not None
        assert len(result) == len(sample_rectangle_vertices)
    
    def test_draw_rectangle_map(self):
        """Test rectangle map drawing"""
        center = (40, -100)
        zoom = 4
        
        result = draw_rectangle_map(center, zoom)
        
        assert result is not None
    
    def test_get_polygon_vertices(self):
        """Test polygon vertices extraction"""
        drawn_polygons = [{"geometry": {"coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}}]
        
        result = get_polygon_vertices(drawn_polygons)
        
        assert result is not None
        assert len(result) > 0

class TestPolygonOperations:
    """Tests for polygon operation functions"""
    
    @patch('voxcity.geoprocessor.polygon.gpd.read_file')
    def test_get_gdf_from_gpkg(self, mock_read):
        """Test reading GeoPackage file"""
        mock_gdf = MagicMock()
        mock_read.return_value = mock_gdf
        
        result = get_gdf_from_gpkg("test.gpkg")
        
        assert result is not None
        mock_read.assert_called_once_with("test.gpkg")
    
    def test_save_geojson(self, sample_polygon, tmp_path):
        """Test saving GeoJSON file"""
        gdf = MagicMock()
        gdf.__iter__ = lambda self: iter([sample_polygon])
        gdf.crs = 'EPSG:4326'
        
        output_path = tmp_path / "test.geojson"
        
        result = save_geojson(gdf, output_path)
        
        assert result is not None
        assert output_path.exists()
