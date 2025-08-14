import pytest
import numpy as np

from voxcity.geoprocessor.utils import (
    initialize_geod,
    calculate_distance,
    normalize_to_one_meter,
    get_timezone_info,
    get_city_country_name_from_rectangle,
)


def test_initialize_geod():
    geod = initialize_geod()
    assert geod.sphere is False
    assert geod.a > 0
    assert geod.f != 0


def test_calculate_distance():
    geod = initialize_geod()
    dist = calculate_distance(geod, 139.7564, 35.6713, 139.7619, 35.6758)
    assert dist > 0
    assert isinstance(dist, float)


def test_normalize_to_one_meter():
    vector = np.array([3.0, 4.0])
    distance = 5.0
    result = normalize_to_one_meter(vector, distance)
    # Function returns a unit-length direction vector scaled to 1 meter
    assert np.allclose(np.linalg.norm(result), 1.0)


def test_get_timezone_info(sample_rectangle_vertices):
    timezone, longitude = get_timezone_info(sample_rectangle_vertices)
    assert isinstance(timezone, str)
    assert isinstance(float(longitude), float)


def test_get_city_country_name(sample_rectangle_vertices):
    # Should return a string even if network-based reverse geocoding fails (offline fallback)
    location = get_city_country_name_from_rectangle(sample_rectangle_vertices)
    assert isinstance(location, str)

