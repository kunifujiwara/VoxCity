import pytest
import numpy as np
from shapely.geometry import Polygon, Point
from voxelcity.geo.utils import (
    initialize_geod,
    calculate_distance,
    normalize_to_one_meter,
    get_timezone_info,
    get_city_country_name_from_rectangle
)
from voxelcity.geo.grid import (
    apply_operation,
    translate_array,
    group_and_label_cells,
    process_grid
)

class TestGridOperations:
    def test_apply_operation_basic(self):
        test_array = np.array([[1.2, 2.7], [3.4, 4.8]])
        meshsize = 1.0
        result = apply_operation(test_array, meshsize)
        assert isinstance(result, np.ndarray)
        assert result.shape == test_array.shape

    def test_apply_operation_zero(self):
        test_array = np.zeros((3, 3))
        meshsize = 1.0
        result = apply_operation(test_array, meshsize)
        assert np.array_equal(result, test_array)

    def test_translate_array_empty_dict(self):
        input_array = np.array([[1, 2], [3, 4]])
        translation_dict = {}
        result = translate_array(input_array, translation_dict)
        assert result.shape == input_array.shape
        assert np.all(result == '')

    def test_group_and_label_cells_empty(self):
        input_array = np.zeros((3, 3))
        result = group_and_label_cells(input_array)
        assert np.array_equal(result, input_array)

    def test_process_grid(self):
        grid_bi = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        dem_grid = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
        result = process_grid(grid_bi, dem_grid)
        assert isinstance(result, np.ndarray)
        assert result.shape == grid_bi.shape

class TestGeoUtils:
    def test_initialize_geod(self):
        geod = initialize_geod()
        assert geod.sphere is False
        assert geod.a > 0
        assert geod.f != 0

    def test_calculate_distance(self):
        geod = initialize_geod()
        dist = calculate_distance(geod, 139.7564, 35.6713, 139.7619, 35.6758)
        assert dist > 0
        assert isinstance(dist, float)

    def test_normalize_to_one_meter(self):
        vector = np.array([3.0, 4.0])
        distance = 5.0
        result = normalize_to_one_meter(vector, distance)
        assert np.allclose(np.linalg.norm(result), 1/distance)

    def test_get_timezone_info(self, sample_rectangle_vertices):
        timezone, longitude = get_timezone_info(sample_rectangle_vertices)
        assert timezone.startswith("UTC+")
        assert isinstance(float(longitude), float)

    def test_get_city_country_name(self, sample_rectangle_vertices):
        location = get_city_country_name_from_rectangle(sample_rectangle_vertices)
        assert isinstance(location, str)
        assert "/" in location 