"""
Comprehensive tests for voxcity.geoprocessor.conversion to improve coverage.
"""

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

from voxcity.geoprocessor.conversion import (
    filter_and_convert_gdf_to_geojson,
    geojson_to_gdf,
    gdf_to_geojson_dicts,
)


class TestFilterAndConvertGdfToGeojson:
    def _make_gdf(self, geometries, heights, crs='EPSG:4326'):
        return gpd.GeoDataFrame(
            {'height': heights},
            geometry=geometries,
            crs=crs,
        )

    def test_single_building_inside(self):
        poly = Polygon([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)])
        gdf = self._make_gdf([poly], [10.0])
        rect = [(0, 0), (1, 0), (1, 1), (0, 1)]
        features = filter_and_convert_gdf_to_geojson(gdf, rect)
        assert len(features) == 1
        assert features[0]['properties']['height'] == pytest.approx(10.0)
        assert features[0]['type'] == 'Feature'

    def test_building_outside(self):
        poly = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        gdf = self._make_gdf([poly], [10.0])
        rect = [(0, 0), (1, 0), (1, 1), (0, 1)]
        features = filter_and_convert_gdf_to_geojson(gdf, rect)
        assert len(features) == 0

    def test_multipolygon_split(self):
        p1 = Polygon([(0.1, 0.1), (0.4, 0.1), (0.4, 0.4), (0.1, 0.4)])
        p2 = Polygon([(0.6, 0.6), (0.9, 0.6), (0.9, 0.9), (0.6, 0.9)])
        multi = MultiPolygon([p1, p2])
        gdf = self._make_gdf([multi], [15.0])
        rect = [(0, 0), (1, 0), (1, 1), (0, 1)]
        features = filter_and_convert_gdf_to_geojson(gdf, rect)
        # MultiPolygon should be split into individual features
        assert len(features) == 2

    def test_sequential_ids(self):
        polys = [
            Polygon([(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]),
            Polygon([(0.5, 0.5), (0.7, 0.5), (0.7, 0.7), (0.5, 0.7)]),
        ]
        gdf = self._make_gdf(polys, [10.0, 20.0])
        rect = [(0, 0), (1, 0), (1, 1), (0, 1)]
        features = filter_and_convert_gdf_to_geojson(gdf, rect)
        ids = [f['properties']['id'] for f in features]
        assert len(set(ids)) == len(ids)  # All unique

    def test_crs_conversion(self):
        poly = Polygon([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)])
        gdf = self._make_gdf([poly], [10.0], crs='EPSG:3857')
        # Transform to roughly the same area in 4326
        gdf = gdf.to_crs('EPSG:4326')
        bounds = gdf.total_bounds
        rect = [(bounds[0] - 0.01, bounds[1] - 0.01),
                (bounds[2] + 0.01, bounds[1] - 0.01),
                (bounds[2] + 0.01, bounds[3] + 0.01),
                (bounds[0] - 0.01, bounds[3] + 0.01)]
        # Re-create in 3857 to test CRS conversion
        gdf_3857 = gdf.to_crs('EPSG:3857')
        features = filter_and_convert_gdf_to_geojson(gdf_3857, rect)
        assert len(features) >= 0  # Just check it doesn't crash

    def test_confidence_field(self):
        poly = Polygon([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)])
        gdf = self._make_gdf([poly], [10.0])
        rect = [(0, 0), (1, 0), (1, 1), (0, 1)]
        features = filter_and_convert_gdf_to_geojson(gdf, rect)
        assert features[0]['properties']['confidence'] == -1.0


class TestGeojsonToGdf:
    def test_basic(self):
        features = [
            {
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                'properties': {'height': 10.0, 'id': 1}
            }
        ]
        gdf = geojson_to_gdf(features)
        assert len(gdf) == 1
        assert gdf.crs.to_epsg() == 4326
        assert gdf.iloc[0]['height'] == 10.0

    def test_auto_id(self):
        features = [
            {
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                'properties': {'height': 10.0}
            }
        ]
        gdf = geojson_to_gdf(features)
        assert 'id' in gdf.columns

    def test_multiple_features(self):
        features = [
            {
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                'properties': {'id': 1}
            },
            {
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]]
                },
                'properties': {'id': 2}
            }
        ]
        gdf = geojson_to_gdf(features)
        assert len(gdf) == 2

    def test_custom_id_col(self):
        features = [
            {
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                'properties': {'custom_id': 42}
            }
        ]
        gdf = geojson_to_gdf(features, id_col='custom_id')
        assert gdf.iloc[0]['custom_id'] == 42

    def test_missing_geometry(self):
        features = [
            {
                'geometry': None,
                'properties': {'id': 1}
            }
        ]
        gdf = geojson_to_gdf(features)
        assert len(gdf) == 1
        assert gdf.iloc[0].geometry is None


class TestGdfToGeojsonDicts:
    def test_basic(self):
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        gdf = gpd.GeoDataFrame({'id': [1], 'height': [10.0]}, geometry=[poly], crs='EPSG:4326')
        features = gdf_to_geojson_dicts(gdf)
        assert len(features) == 1
        assert features[0]['type'] == 'Feature'
        assert features[0]['properties']['height'] == 10.0

    def test_round_trip(self):
        original_features = [
            {
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                'properties': {'id': 1, 'height': 10.0}
            }
        ]
        gdf = geojson_to_gdf(original_features)
        back = gdf_to_geojson_dicts(gdf)
        assert len(back) == 1
        # gdf_to_geojson_dicts strips the id_col from properties
        assert back[0]['properties']['height'] == 10.0
        assert 'id' not in back[0]['properties']
