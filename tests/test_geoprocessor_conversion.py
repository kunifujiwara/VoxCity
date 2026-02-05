"""Tests for voxcity.geoprocessor.conversion module."""
import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

from voxcity.geoprocessor.conversion import (
    geojson_to_gdf,
    gdf_to_geojson_dicts,
)


class TestGeojsonToGdf:
    def test_simple_conversion(self):
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                "properties": {"height": 10.0, "id": 1}
            }
        ]
        gdf = geojson_to_gdf(features)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1
        assert gdf.iloc[0]["height"] == 10.0

    def test_auto_generates_id(self):
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                "properties": {"height": 10.0}  # no id
            }
        ]
        gdf = geojson_to_gdf(features)
        assert "id" in gdf.columns
        assert gdf.iloc[0]["id"] == 0

    def test_multiple_features(self):
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                "properties": {"height": 10.0}
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]]
                },
                "properties": {"height": 20.0}
            }
        ]
        gdf = geojson_to_gdf(features)
        assert len(gdf) == 2
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326

    def test_handles_missing_geometry(self):
        features = [
            {
                "type": "Feature",
                "properties": {"height": 10.0}
            }
        ]
        gdf = geojson_to_gdf(features)
        assert len(gdf) == 1
        assert gdf.iloc[0].geometry is None

    def test_custom_id_column(self):
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                "properties": {"custom_id": 42}
            }
        ]
        gdf = geojson_to_gdf(features, id_col="custom_id")
        assert gdf.iloc[0]["custom_id"] == 42


class TestGdfToGeojsonDicts:
    def test_simple_conversion(self):
        gdf = gpd.GeoDataFrame(
            {"height": [10.0], "id": [1]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326"
        )
        features = gdf_to_geojson_dicts(gdf)
        assert len(features) == 1
        assert features[0]["type"] == "Feature"
        assert features[0]["properties"]["height"] == 10.0

    def test_geometry_converted_to_geojson(self):
        gdf = gpd.GeoDataFrame(
            {"height": [10.0]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326"
        )
        features = gdf_to_geojson_dicts(gdf)
        geom = features[0]["geometry"]
        assert geom["type"] == "Polygon"
        assert "coordinates" in geom

    def test_multiple_features(self):
        gdf = gpd.GeoDataFrame(
            {"height": [10.0, 20.0], "name": ["A", "B"]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
            ],
            crs="EPSG:4326"
        )
        features = gdf_to_geojson_dicts(gdf)
        assert len(features) == 2
        assert features[0]["properties"]["name"] == "A"
        assert features[1]["properties"]["name"] == "B"


class TestRoundTrip:
    def test_geojson_to_gdf_to_geojson(self):
        """Test that conversion is reversible (approximately)."""
        original = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                "properties": {"height": 15.5}
            }
        ]
        gdf = geojson_to_gdf(original)
        result = gdf_to_geojson_dicts(gdf)
        
        assert len(result) == 1
        assert result[0]["type"] == "Feature"
        assert result[0]["properties"]["height"] == 15.5
