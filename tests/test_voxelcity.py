import pytest
import numpy as np
from shapely.geometry import Polygon
import os
import tempfile
import json
from pathlib import Path

from voxcity.geo.utils import (
    get_coordinates_from_cityname,
    get_country_name,
    calculate_distance,
    initialize_geod,
    get_timezone_info
)
from voxcity.download.omt import load_geojsons_from_openmaptiles
from voxcity.download.osm import load_geojsons_from_openstreetmap
from voxcity.download.overture import load_geojsons_from_overture
from voxcity.download.eubucco import load_geojson_from_eubucco
from voxcity.geo.grid import (
    create_building_height_grid_from_geojson_polygon,
    create_land_cover_grid_from_geojson_polygon,
    apply_operation,
    translate_array,
    group_and_label_cells
)

# Test fixtures
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
def sample_building_data():
    """Sample building GeoJSON data for testing"""
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[139.7564, 35.6713], [139.7564, 35.6758], 
                           [139.7619, 35.6758], [139.7619, 35.6713], 
                           [139.7564, 35.6713]]]
        },
        "properties": {
            "height": 25.0,
            "min_height": 0.0,
            "id": 1
        }
    }

class TestGeocoding:
    """Tests for geocoding functionality"""
    
    def test_get_coordinates_from_cityname(self):
        coords = get_coordinates_from_cityname("tokyo")
        assert coords is not None
        assert len(coords) == 2
        assert isinstance(coords[0], float)
        assert isinstance(coords[1], float)
        assert 35 < coords[0] < 36
        assert 139 < coords[1] < 140

    def test_get_country_name(self):
        country = get_country_name(35.6762, 139.6503)
        assert country == "Japan"

    def test_get_timezone_info(self, sample_rectangle_vertices):
        timezone, longitude = get_timezone_info(sample_rectangle_vertices)
        assert timezone.startswith("UTC+")
        assert isinstance(float(longitude), float)

class TestGridOperations:
    """Tests for grid operations"""

    def test_apply_operation(self):
        test_array = np.array([[1.2, 2.7], [3.4, 4.8]])
        meshsize = 1.0
        result = apply_operation(test_array, meshsize)
        assert isinstance(result, np.ndarray)
        assert result.shape == test_array.shape

    def test_translate_array(self):
        input_array = np.array([[1, 2], [3, 4]])
        translation_dict = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        result = translate_array(input_array, translation_dict)
        assert result.shape == input_array.shape
        assert result[0, 0] == 'A'

    def test_group_and_label_cells(self):
        input_array = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        result = group_and_label_cells(input_array)
        assert result.shape == input_array.shape
        assert np.max(result) <= np.sum(input_array > 0)

class TestDataLoading:
    """Tests for data loading functionality"""

    def test_load_geojsons_from_openstreetmap(self, sample_rectangle_vertices):
        result = load_geojsons_from_openstreetmap(sample_rectangle_vertices)
        assert isinstance(result, list)
        assert len(result) > 0
        for feature in result:
            assert 'type' in feature
            assert 'geometry' in feature
            assert 'properties' in feature

    @pytest.mark.skip(reason="Requires API key")
    def test_load_geojsons_from_openmaptiles(self, sample_rectangle_vertices, api_keys):
        result = load_geojsons_from_openmaptiles(
            sample_rectangle_vertices, 
            api_keys['maptiler']
        )
        assert isinstance(result, list)

class TestFileOperations:
    """Tests for file operations"""

    def test_file_operations(self, temp_output_dir):
        test_file = temp_output_dir / "test.txt"
        test_file.write_text("test")
        assert test_file.exists()
        assert test_file.read_text() == "test"

    def test_json_operations(self, temp_output_dir, sample_building_data):
        json_file = temp_output_dir / "test.json"
        with json_file.open('w') as f:
            json.dump(sample_building_data, f)
        assert json_file.exists()
        
        with json_file.open('r') as f:
            loaded_data = json.load(f)
        assert loaded_data == sample_building_data

# @pytest.mark.integration
# class TestIntegration:
#     """Integration tests"""

#     def test_full_workflow(self, sample_rectangle_vertices, temp_output_dir):
#         meshsize = 5.0
        
#         # Load building data
#         buildings = load_geojsons_from_openstreetmap(sample_rectangle_vertices)
#         assert len(buildings) > 0
        
#         # Create height grid
#         height_grid, min_height_grid, id_grid, _ = create_building_height_grid_from_geojson_polygon(
#             buildings, meshsize, sample_rectangle_vertices
#         )
        
#         assert height_grid.shape[0] > 0
#         assert height_grid.shape[1] > 0
#         # TODO: Fix this test once we resolve the coordinate system issues
#         # assert np.any(height_grid > 0)

#         # Save outputs
#         np.save(temp_output_dir / "height_grid.npy", height_grid)
#         np.save(temp_output_dir / "min_height_grid.npy", min_height_grid)
#         np.save(temp_output_dir / "id_grid.npy", id_grid)

#         # Verify saved files
#         assert (temp_output_dir / "height_grid.npy").exists()
#         assert (temp_output_dir / "min_height_grid.npy").exists()
#         assert (temp_output_dir / "id_grid.npy").exists()

if __name__ == '__main__':
    pytest.main([__file__]) 