import pytest
import numpy as np
from voxelcity.geo.utils import (
    initialize_geod,
    calculate_distance,
    normalize_to_one_meter,
    get_timezone_info
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
    assert np.allclose(np.linalg.norm(result), 1/distance)

def test_get_timezone_info(sample_rectangle_vertices):
    timezone, longitude = get_timezone_info(sample_rectangle_vertices)
    assert timezone.startswith("UTC+")
    assert isinstance(float(longitude), float) 