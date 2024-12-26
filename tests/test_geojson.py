import pytest
from shapely.geometry import box
from voxelcity.file.geojson import (
    filter_buildings,
    swap_coordinates
)

def test_swap_coordinates(sample_geojson):
    features = [sample_geojson]
    swap_coordinates(features)
    coords = features[0]['geometry']['coordinates'][0]
    assert coords[0][0] < coords[0][1]  # Latitude should be greater than longitude

def test_filter_buildings(sample_geojson):
    features = [sample_geojson]
    bbox = box(139.75, 35.67, 139.77, 35.68)
    result = filter_buildings(features, bbox)
    assert len(result) > 0

def test_invalid_geometry():
    invalid_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[1, 1]]]  # Invalid polygon
        },
        "properties": {}
    }
    bbox = box(0, 0, 2, 2)
    result = filter_buildings([invalid_feature], bbox)
    assert len(result) == 0 