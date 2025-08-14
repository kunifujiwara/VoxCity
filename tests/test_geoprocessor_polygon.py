import pytest
from shapely.geometry import box

from voxcity.geoprocessor.polygon import (
    filter_buildings,
    swap_coordinates,
)


@pytest.fixture
def sample_geojson_feature():
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
            "min_height": 0.0
        }
    }


def test_swap_coordinates(sample_geojson_feature):
    features = [sample_geojson_feature]
    swap_coordinates(features)
    coords = features[0]['geometry']['coordinates'][0]
    # Expect order preserved as (lon, lat); first tuple lon < lat for these numbers
    assert coords[0][0] < coords[0][1]


def test_filter_buildings(sample_geojson_feature):
    features = [sample_geojson_feature]
    bbox = box(139.75, 35.67, 139.77, 35.68)
    result = filter_buildings(features, bbox)
    assert len(result) > 0


def test_invalid_geometry():
    invalid_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[1, 1]]]  # Invalid polygon (less than 4 points including closure)
        },
        "properties": {}
    }
    bbox = box(0, 0, 2, 2)
    result = filter_buildings([invalid_feature], bbox)
    assert len(result) == 0

