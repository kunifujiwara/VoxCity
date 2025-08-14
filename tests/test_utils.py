import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from voxcity.utils.lc import (
    rgb_distance,
    get_land_cover_classes,
    convert_land_cover,
    convert_land_cover_array,
    get_class_priority,
    get_nearest_class,
    get_dominant_class
)

from voxcity.utils.material import (
    get_material_dict,
    get_modulo_numbers,
    set_building_material_by_id,
    set_building_material_by_gdf
)

from voxcity.utils.weather import (
    safe_rename,
    safe_extract,
    parse_coordinates,
    get_nearest_epw_from_climate_onebuilding
)

@pytest.fixture
def sample_land_cover_array():
    """Sample land cover array for testing"""
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

@pytest.fixture
def sample_voxel_grid():
    """Sample voxel grid for testing"""
    return np.zeros((10, 10, 10), dtype=int)

@pytest.fixture
def sample_building_gdf():
    """Sample building GeoDataFrame for testing"""
    import geopandas as gpd
    from shapely.geometry import Polygon
    
    buildings = [
        {
            'geometry': Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            'height': 25.0,
            'id': 1
        }
    ]
    return gpd.GeoDataFrame(buildings, crs='EPSG:4326')

class TestLandCoverUtils:
    """Tests for land cover utility functions"""
    
    def test_rgb_distance(self):
        """Test RGB color distance calculation"""
        color1 = (255, 0, 0)  # Red
        color2 = (0, 0, 255)  # Blue
        distance = rgb_distance(color1, color2)
        assert distance > 0
        assert isinstance(distance, float)
    
    def test_get_land_cover_classes(self):
        """Test land cover classes retrieval"""
        classes = get_land_cover_classes('Urbanwatch')
        assert isinstance(classes, dict)
        assert len(classes) > 0
    
    def test_convert_land_cover(self):
        """Test land cover conversion"""
        input_array = np.array([[1, 2], [3, 4]])
        result = convert_land_cover(input_array, 'Urbanwatch')
        assert isinstance(result, np.ndarray)
        assert result.shape == input_array.shape
    
    def test_convert_land_cover_array(self, sample_land_cover_array):
        """Test land cover array conversion"""
        land_cover_classes = {1: 'Building', 2: 'Road', 3: 'Vegetation'}
        result = convert_land_cover_array(sample_land_cover_array, land_cover_classes)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_land_cover_array.shape
    
    def test_get_class_priority(self):
        """Test class priority retrieval"""
        priority = get_class_priority('Urbanwatch')
        assert isinstance(priority, dict)
        assert len(priority) > 0
    
    def test_get_nearest_class(self):
        """Test nearest class finding"""
        pixel = (255, 0, 0)  # Red
        land_cover_classes = {
            (255, 0, 0): 'Building',
            (0, 255, 0): 'Vegetation',
            (0, 0, 255): 'Water'
        }
        result = get_nearest_class(pixel, land_cover_classes)
        assert result == 'Building'
    
    def test_get_dominant_class(self):
        """Test dominant class finding"""
        cell_data = np.array([[1, 1, 2], [1, 2, 2], [2, 2, 2]])
        land_cover_classes = {1: 'Building', 2: 'Road'}
        result = get_dominant_class(cell_data, land_cover_classes)
        assert result == 'Road'

class TestMaterialUtils:
    """Tests for material utility functions"""
    
    def test_get_material_dict(self):
        """Test material dictionary retrieval"""
        materials = get_material_dict()
        assert isinstance(materials, dict)
        assert len(materials) > 0
    
    def test_get_modulo_numbers(self):
        """Test modulo numbers calculation"""
        window_ratio = 0.125
        result = get_modulo_numbers(window_ratio)
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_set_building_material_by_id(self, sample_voxel_grid):
        """Test building material setting by ID"""
        building_id_grid = np.ones((10, 10, 10), dtype=int)
        ids = [1, 2, 3]
        mark = 100
        
        result = set_building_material_by_id(
            sample_voxel_grid, 
            building_id_grid, 
            ids, 
            mark
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_voxel_grid.shape
    
    def test_set_building_material_by_gdf(self, sample_voxel_grid, sample_building_gdf):
        """Test building material setting by GeoDataFrame"""
        building_id_grid = np.ones((10, 10, 10), dtype=int)
        
        result = set_building_material_by_gdf(
            sample_voxel_grid,
            building_id_grid,
            sample_building_gdf
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_voxel_grid.shape

class TestWeatherUtils:
    """Tests for weather utility functions"""
    
    def test_safe_rename(self, tmp_path):
        """Test safe file renaming"""
        src_file = tmp_path / "source.txt"
        dst_file = tmp_path / "destination.txt"
        
        src_file.write_text("test content")
        result = safe_rename(src_file, dst_file)
        
        assert result == dst_file
        assert dst_file.exists()
        assert not src_file.exists()
    
    def test_safe_extract(self, tmp_path):
        """Test safe archive extraction"""
        import zipfile
        
        # Create a test zip file
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.writestr("test.txt", "test content")
        
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        
        with zipfile.ZipFile(zip_path) as zip_ref:
            result = safe_extract(zip_ref, "test.txt", extract_dir)
        
        assert result.exists()
        assert result.read_text() == "test content"
    
    def test_parse_coordinates(self):
        """Test coordinate parsing"""
        coords = parse_coordinates("139.7564, 35.6713, 25.0")
        assert len(coords) == 3
        assert all(isinstance(coord, float) for coord in coords)
    
    @patch('voxcity.utils.weather.requests.get')
    def test_get_nearest_epw_from_climate_onebuilding(self, mock_get, tmp_path):
        """Test nearest EPW file retrieval"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'<kml>test</kml>'
        mock_get.return_value = mock_response
        
        result = get_nearest_epw_from_climate_onebuilding(
            139.7564, 
            35.6713, 
            str(tmp_path)
        )
        
        assert result is not None
        mock_get.assert_called_once()
