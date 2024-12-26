import pytest
import numpy as np
import json
from pathlib import Path
from shapely.geometry import box
from voxelcity.file.geojson import (
    filter_and_convert_gdf_to_geojson,
    load_geojsons_from_multiple_gz,
    filter_buildings,
    swap_coordinates
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

class TestGeojsonOperations:
    def test_swap_coordinates(self, sample_geojson_feature):
        features = [sample_geojson_feature]
        swap_coordinates(features)
        coords = features[0]['geometry']['coordinates'][0]
        assert coords[0][0] < coords[0][1]  # Latitude should be greater than longitude

    def test_filter_buildings(self, sample_geojson_feature):
        from shapely.geometry import box
        features = [sample_geojson_feature]
        bbox = box(139.75, 35.67, 139.77, 35.68)
        result = filter_buildings(features, bbox)
        assert len(result) > 0

    def test_invalid_geometry(self):
        invalid_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[1, 1]]]  # Invalid polygon (less than 3 points)
            },
            "properties": {}
        }
        bbox = box(0, 0, 2, 2)
        result = filter_buildings([invalid_feature], bbox)
        assert len(result) == 0 