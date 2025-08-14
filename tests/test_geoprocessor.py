import pytest
import numpy as np
from shapely.geometry import Polygon, Point
from unittest.mock import patch, MagicMock

from voxcity.geoprocessor.utils import (
    initialize_geod,
    calculate_distance,
    normalize_to_one_meter,
    get_timezone_info,
    get_city_country_name_from_rectangle,
    get_coordinates_from_cityname,
    get_country_name,
    create_polygon,
    create_geodataframe,
    haversine_distance
)

from voxcity.geoprocessor.grid import (
    group_and_label_cells,
    process_grid,
    create_land_cover_grid_from_gdf_polygon,
    create_building_height_grid_from_gdf_polygon,
    create_dem_grid_from_gdf_polygon
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
def sample_polygon():
    """Sample polygon for testing"""
    return Polygon([
        (139.7564, 35.6713),
        (139.7564, 35.6758),
        (139.7619, 35.6758),
        (139.7619, 35.6713)
    ])

class TestGeoUtils:
    """Tests for geographic utility functions"""
    
    def test_initialize_geod(self):
        """Test geodetic system initialization"""
        geod = initialize_geod()
        assert geod.sphere is False
        assert geod.a > 0
        assert geod.f != 0
    
    def test_calculate_distance(self):
        """Test distance calculation"""
        geod = initialize_geod()
        dist = calculate_distance(geod, 139.7564, 35.6713, 139.7619, 35.6758)
        assert dist > 0
        assert isinstance(dist, float)
    
    def test_normalize_to_one_meter(self):
        """Test vector normalization"""
        vector = np.array([3.0, 4.0])
        distance = 5.0
        result = normalize_to_one_meter(vector, distance)
        assert np.allclose(np.linalg.norm(result), 1/distance)
    
    def test_get_timezone_info(self, sample_rectangle_vertices):
        """Test timezone information retrieval"""
        timezone, longitude = get_timezone_info(sample_rectangle_vertices)
        assert timezone.startswith("UTC+")
        assert isinstance(float(longitude), float)
    
    def test_get_city_country_name_from_rectangle(self, sample_rectangle_vertices):
        """Test city/country name retrieval from rectangle"""
        location = get_city_country_name_from_rectangle(sample_rectangle_vertices)
        assert isinstance(location, str)
        assert "/" in location
    
    @patch('voxcity.geoprocessor.utils.Nominatim')
    def test_get_coordinates_from_cityname(self, mock_nominatim):
        """Test coordinate retrieval from city name"""
        mock_geocoder = MagicMock()
        mock_geocoder.geocode.return_value.latitude = 35.6762
        mock_geocoder.geocode.return_value.longitude = 139.6503
        mock_nominatim.return_value = mock_geocoder
        
        coords = get_coordinates_from_cityname("tokyo")
        assert coords is not None
        assert len(coords) == 2
        assert isinstance(coords[0], float)
        assert isinstance(coords[1], float)
    
    def test_get_country_name(self):
        """Test country name retrieval"""
        country = get_country_name(139.6503, 35.6762)
        assert country == "Japan"
    
    def test_create_polygon(self, sample_rectangle_vertices):
        """Test polygon creation from vertices"""
        polygon = create_polygon(sample_rectangle_vertices)
        assert isinstance(polygon, Polygon)
        assert polygon.is_valid
    
    def test_create_geodataframe(self, sample_polygon):
        """Test GeoDataFrame creation"""
        gdf = create_geodataframe(sample_polygon)
        assert gdf.crs == 'EPSG:4326'
        assert len(gdf) == 1
    
    def test_haversine_distance(self):
        """Test haversine distance calculation"""
        dist = haversine_distance(139.7564, 35.6713, 139.7619, 35.6758)
        assert dist > 0
        assert isinstance(dist, float)

class TestGridOperations:
    """Tests for grid processing functions"""
    
    def test_group_and_label_cells(self):
        """Test cell grouping and labeling"""
        input_array = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        result = group_and_label_cells(input_array)
        assert result.shape == input_array.shape
        assert np.max(result) <= np.sum(input_array > 0)
    
    def test_process_grid(self):
        """Test grid processing"""
        grid_bi = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        dem_grid = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
        result = process_grid(grid_bi, dem_grid)
        assert isinstance(result, np.ndarray)
        assert result.shape == grid_bi.shape
    
    @patch('voxcity.geoprocessor.grid.rasterize')
    def test_create_land_cover_grid_from_gdf_polygon(self, mock_rasterize, sample_polygon):
        """Test land cover grid creation from GeoDataFrame"""
        mock_rasterize.return_value = np.array([[1, 2], [1, 2]])
        
        result = create_land_cover_grid_from_gdf_polygon(
            sample_polygon, 
            meshsize=100, 
            land_cover_gdf=None
        )
        
        assert isinstance(result, np.ndarray)
        mock_rasterize.assert_called_once()
    
    @patch('voxcity.geoprocessor.grid.rasterize')
    def test_create_building_height_grid_from_gdf_polygon(self, mock_rasterize, sample_polygon):
        """Test building height grid creation from GeoDataFrame"""
        mock_rasterize.return_value = np.array([[10, 20], [15, 25]])
        
        result = create_building_height_grid_from_gdf_polygon(
            sample_polygon, 
            meshsize=100, 
            building_gdf=None
        )
        
        assert isinstance(result, np.ndarray)
        mock_rasterize.assert_called_once()
    
    @patch('voxcity.geoprocessor.grid.rasterize')
    def test_create_dem_grid_from_gdf_polygon(self, mock_rasterize, sample_polygon):
        """Test DEM grid creation from GeoDataFrame"""
        mock_rasterize.return_value = np.array([[5, 5], [5, 5]])
        
        result = create_dem_grid_from_gdf_polygon(
            sample_polygon, 
            meshsize=100, 
            dem_gdf=None
        )
        
        assert isinstance(result, np.ndarray)
        mock_rasterize.assert_called_once()
