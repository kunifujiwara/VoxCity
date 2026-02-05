"""Tests for voxcity.geoprocessor.io module."""
import pytest
import json
import gzip
import numpy as np
import geopandas as gpd

from voxcity.geoprocessor.io import swap_coordinates, save_geojson, load_gdf_from_multiple_gz


class TestSwapCoordinates:
    """Tests for swap_coordinates function."""

    def test_swap_polygon_coordinates(self):
        """Test swapping coordinates for Polygon geometry."""
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[35.0, 139.0], [36.0, 140.0], [37.0, 141.0], [35.0, 139.0]]]
                },
                "properties": {}
            }
        ]
        
        swap_coordinates(features)
        
        # Check coordinates are swapped from (lat, lon) to (lon, lat)
        coords = features[0]["geometry"]["coordinates"][0]
        assert coords[0] == [139.0, 35.0]
        assert coords[1] == [140.0, 36.0]
        assert coords[2] == [141.0, 37.0]

    def test_swap_multipolygon_coordinates(self):
        """Test swapping coordinates for MultiPolygon geometry."""
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[[[35.0, 139.0], [36.0, 140.0], [35.0, 139.0]]]]
                },
                "properties": {}
            }
        ]
        
        swap_coordinates(features)
        
        coords = features[0]["geometry"]["coordinates"][0][0]
        assert coords[0] == [139.0, 35.0]
        assert coords[1] == [140.0, 36.0]

    def test_swap_multiple_features(self):
        """Test swapping coordinates for multiple features."""
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]]]},
                "properties": {}
            },
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[5.0, 6.0], [7.0, 8.0], [5.0, 6.0]]]},
                "properties": {}
            }
        ]
        
        swap_coordinates(features)
        
        assert features[0]["geometry"]["coordinates"][0][0] == [2.0, 1.0]
        assert features[1]["geometry"]["coordinates"][0][0] == [6.0, 5.0]


class TestSaveGeojson:
    """Tests for save_geojson function."""

    def test_save_creates_valid_geojson(self, tmp_path):
        """Test that save_geojson creates valid GeoJSON file."""
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[35.0, 139.0], [36.0, 140.0], [35.0, 139.0]]]},
                "properties": {"name": "test"}
            }
        ]
        
        output_file = tmp_path / "test.geojson"
        save_geojson(features, str(output_file))
        
        # Read and verify
        assert output_file.exists()
        with open(output_file) as f:
            result = json.load(f)
        
        assert result["type"] == "FeatureCollection"
        assert len(result["features"]) == 1
        assert result["features"][0]["properties"]["name"] == "test"

    def test_save_swaps_coordinates(self, tmp_path):
        """Test that save_geojson swaps coordinates."""
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[35.0, 139.0], [36.0, 140.0], [35.0, 139.0]]]},
                "properties": {}
            }
        ]
        
        output_file = tmp_path / "test.geojson"
        save_geojson(features, str(output_file))
        
        with open(output_file) as f:
            result = json.load(f)
        
        # Coordinates should be swapped
        coords = result["features"][0]["geometry"]["coordinates"][0][0]
        assert coords == [139.0, 35.0]

    def test_save_does_not_modify_original(self, tmp_path):
        """Test that save_geojson does not modify original features."""
        original_coords = [[[35.0, 139.0], [36.0, 140.0], [35.0, 139.0]]]
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": original_coords},
                "properties": {}
            }
        ]
        
        output_file = tmp_path / "test.geojson"
        save_geojson(features, str(output_file))
        
        # Original should be unchanged (deep copy is used)
        assert features[0]["geometry"]["coordinates"][0][0] == [35.0, 139.0]

    def test_save_empty_features(self, tmp_path):
        """Test saving empty feature list."""
        output_file = tmp_path / "empty.geojson"
        save_geojson([], str(output_file))
        
        with open(output_file) as f:
            result = json.load(f)
        
        assert result["type"] == "FeatureCollection"
        assert result["features"] == []

    def test_save_pretty_printed(self, tmp_path):
        """Test that output is pretty printed (indented)."""
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]]]},
                "properties": {}
            }
        ]
        
        output_file = tmp_path / "test.geojson"
        save_geojson(features, str(output_file))
        
        with open(output_file) as f:
            content = f.read()
        
        # Should have newlines (indented)
        assert "\n" in content


class TestLoadGdfFromMultipleGz:
    """Tests for load_gdf_from_multiple_gz function."""

    def test_load_single_gz_file(self, tmp_path):
        """Test loading features from a single gzipped file."""
        # Create a gzipped file with GeoJSON lines
        gz_file = tmp_path / "test.gz"
        features = [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}, "properties": {"height": 10}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 1]}, "properties": {"height": 20}},
        ]
        with gzip.open(gz_file, 'wt', encoding='utf-8') as f:
            for feature in features:
                f.write(json.dumps(feature) + "\n")
        
        gdf = load_gdf_from_multiple_gz([str(gz_file)])
        
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 2
        assert gdf.iloc[0]["height"] == 10
        assert gdf.iloc[1]["height"] == 20
        assert gdf.crs.to_epsg() == 4326

    def test_load_multiple_gz_files(self, tmp_path):
        """Test loading features from multiple gzipped files."""
        # Create first gz file
        gz_file1 = tmp_path / "test1.gz"
        with gzip.open(gz_file1, 'wt', encoding='utf-8') as f:
            f.write(json.dumps({"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}, "properties": {"height": 10}}) + "\n")
        
        # Create second gz file
        gz_file2 = tmp_path / "test2.gz"
        with gzip.open(gz_file2, 'wt', encoding='utf-8') as f:
            f.write(json.dumps({"type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 1]}, "properties": {"height": 20}}) + "\n")
        
        gdf = load_gdf_from_multiple_gz([str(gz_file1), str(gz_file2)])
        
        assert len(gdf) == 2

    def test_null_height_becomes_zero(self, tmp_path):
        """Test that null height values are converted to 0."""
        gz_file = tmp_path / "test.gz"
        with gzip.open(gz_file, 'wt', encoding='utf-8') as f:
            f.write(json.dumps({"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}, "properties": {"height": None}}) + "\n")
        
        gdf = load_gdf_from_multiple_gz([str(gz_file)])
        
        assert gdf.iloc[0]["height"] == 0

    def test_missing_height_becomes_zero(self, tmp_path):
        """Test that missing height property is set to 0."""
        gz_file = tmp_path / "test.gz"
        with gzip.open(gz_file, 'wt', encoding='utf-8') as f:
            # No height property
            f.write(json.dumps({"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}, "properties": {"name": "test"}}) + "\n")
        
        gdf = load_gdf_from_multiple_gz([str(gz_file)])
        
        assert gdf.iloc[0]["height"] == 0

    def test_missing_properties_creates_height(self, tmp_path):
        """Test that missing properties dict gets height=0."""
        gz_file = tmp_path / "test.gz"
        with gzip.open(gz_file, 'wt', encoding='utf-8') as f:
            # No properties at all
            f.write(json.dumps({"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}}) + "\n")
        
        gdf = load_gdf_from_multiple_gz([str(gz_file)])
        
        assert gdf.iloc[0]["height"] == 0

    def test_skips_invalid_json_lines(self, tmp_path, capsys):
        """Test that invalid JSON lines are skipped with warning."""
        gz_file = tmp_path / "test.gz"
        with gzip.open(gz_file, 'wt', encoding='utf-8') as f:
            f.write(json.dumps({"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}, "properties": {"height": 10}}) + "\n")
            f.write("invalid json\n")  # Bad line
            f.write(json.dumps({"type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 1]}, "properties": {"height": 20}}) + "\n")
        
        gdf = load_gdf_from_multiple_gz([str(gz_file)])
        
        # Should have 2 valid features
        assert len(gdf) == 2
        # Should have printed warning
        captured = capsys.readouterr()
        assert "Skipping line" in captured.out or "JSONDecodeError" in captured.out
