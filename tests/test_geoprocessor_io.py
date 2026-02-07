"""Tests for voxcity.geoprocessor.io module."""
import pytest
import json
import gzip
import numpy as np
import geopandas as gpd

from voxcity.geoprocessor.io import load_gdf_from_multiple_gz


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
