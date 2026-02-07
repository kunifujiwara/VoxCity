"""
Additional tests for voxcity.geoprocessor.io module.
"""

import gzip
import json
import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, MultiPolygon


class TestLoadGdfFromMultipleGz:
    """Tests for load_gdf_from_multiple_gz function."""

    def test_load_single_gz_file(self):
        """Test loading from single gzipped file."""
        from voxcity.geoprocessor.io import load_gdf_from_multiple_gz
        
        features = [
            {
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [139.0, 35.0]},
                'properties': {'height': 10.0}
            },
            {
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [139.1, 35.1]},
                'properties': {'height': 20.0}
            }
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
            gz_path = f.name
        
        try:
            with gzip.open(gz_path, 'wt', encoding='utf-8') as f:
                for feat in features:
                    f.write(json.dumps(feat) + '\n')
            
            gdf = load_gdf_from_multiple_gz([gz_path])
            
            assert len(gdf) == 2
            assert gdf.crs.to_epsg() == 4326
            assert 'height' in gdf.columns
        finally:
            Path(gz_path).unlink(missing_ok=True)

    def test_load_multiple_gz_files(self):
        """Test loading from multiple gzipped files."""
        from voxcity.geoprocessor.io import load_gdf_from_multiple_gz
        
        features1 = [
            {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [139.0, 35.0]}, 'properties': {'height': 10.0}}
        ]
        features2 = [
            {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [139.1, 35.1]}, 'properties': {'height': 20.0}}
        ]
        
        paths = []
        for features in [features1, features2]:
            with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
                paths.append(f.name)
            with gzip.open(paths[-1], 'wt', encoding='utf-8') as f:
                for feat in features:
                    f.write(json.dumps(feat) + '\n')
        
        try:
            gdf = load_gdf_from_multiple_gz(paths)
            assert len(gdf) == 2
        finally:
            for p in paths:
                Path(p).unlink(missing_ok=True)

    def test_load_gz_with_null_height(self):
        """Test that null height is converted to 0."""
        from voxcity.geoprocessor.io import load_gdf_from_multiple_gz
        
        features = [
            {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [139.0, 35.0]}, 'properties': {'height': None}}
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
            gz_path = f.name
        
        try:
            with gzip.open(gz_path, 'wt', encoding='utf-8') as f:
                for feat in features:
                    f.write(json.dumps(feat) + '\n')
            
            gdf = load_gdf_from_multiple_gz([gz_path])
            assert gdf.iloc[0]['height'] == 0
        finally:
            Path(gz_path).unlink(missing_ok=True)

    def test_load_gz_without_height_property(self):
        """Test that missing height property is added with value 0."""
        from voxcity.geoprocessor.io import load_gdf_from_multiple_gz
        
        features = [
            {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [139.0, 35.0]}, 'properties': {'name': 'test'}}
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
            gz_path = f.name
        
        try:
            with gzip.open(gz_path, 'wt', encoding='utf-8') as f:
                for feat in features:
                    f.write(json.dumps(feat) + '\n')
            
            gdf = load_gdf_from_multiple_gz([gz_path])
            assert gdf.iloc[0]['height'] == 0
        finally:
            Path(gz_path).unlink(missing_ok=True)

    def test_load_gz_without_properties(self):
        """Test that missing properties dict is handled."""
        from voxcity.geoprocessor.io import load_gdf_from_multiple_gz
        
        features = [
            {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [139.0, 35.0]}}
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
            gz_path = f.name
        
        try:
            with gzip.open(gz_path, 'wt', encoding='utf-8') as f:
                for feat in features:
                    f.write(json.dumps(feat) + '\n')
            
            gdf = load_gdf_from_multiple_gz([gz_path])
            assert gdf.iloc[0]['height'] == 0
        finally:
            Path(gz_path).unlink(missing_ok=True)

    def test_load_gz_with_invalid_json_line(self, capsys):
        """Test that invalid JSON lines are skipped with warning."""
        from voxcity.geoprocessor.io import load_gdf_from_multiple_gz
        
        with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
            gz_path = f.name
        
        try:
            with gzip.open(gz_path, 'wt', encoding='utf-8') as f:
                f.write('{"type": "Feature", "geometry": {"type": "Point", "coordinates": [139.0, 35.0]}, "properties": {"height": 10}}\n')
                f.write('invalid json line\n')
                f.write('{"type": "Feature", "geometry": {"type": "Point", "coordinates": [139.1, 35.1]}, "properties": {"height": 20}}\n')
            
            gdf = load_gdf_from_multiple_gz([gz_path])
            
            assert len(gdf) == 2  # Invalid line skipped
            captured = capsys.readouterr()
            assert 'Skipping line' in captured.out
        finally:
            Path(gz_path).unlink(missing_ok=True)


class TestGetGdfFromGpkg:
    """Tests for get_gdf_from_gpkg function."""

    def test_get_gdf_basic(self):
        """Test reading GeoDataFrame from GPKG."""
        from voxcity.geoprocessor.io import get_gdf_from_gpkg
        
        # Create test GPKG
        gdf = gpd.GeoDataFrame(
            {'name': ['A', 'B']},
            geometry=[
                Polygon([(139.0, 35.0), (139.1, 35.0), (139.1, 35.1), (139.0, 35.0)]),
                Polygon([(139.1, 35.1), (139.2, 35.1), (139.2, 35.2), (139.1, 35.1)])
            ],
            crs='EPSG:4326'
        )
        
        with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as f:
            gpkg_path = f.name
        
        try:
            gdf.to_file(gpkg_path, driver='GPKG')
            
            result = get_gdf_from_gpkg(gpkg_path, None)
            
            assert len(result) == 2
            assert result.crs.to_epsg() == 4326
            assert 'id' in result.columns
        finally:
            Path(gpkg_path).unlink(missing_ok=True)

    def test_get_gdf_converts_crs(self):
        """Test that non-4326 CRS is converted."""
        from voxcity.geoprocessor.io import get_gdf_from_gpkg
        
        # Create test GPKG with different CRS
        gdf = gpd.GeoDataFrame(
            {'name': ['A']},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])],
            crs='EPSG:32654'  # UTM zone 54N
        )
        
        with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as f:
            gpkg_path = f.name
        
        try:
            gdf.to_file(gpkg_path, driver='GPKG')
            
            result = get_gdf_from_gpkg(gpkg_path, None)
            
            assert result.crs.to_epsg() == 4326
        finally:
            Path(gpkg_path).unlink(missing_ok=True)

    def test_get_gdf_sets_crs_when_none(self):
        """Test that CRS is set to 4326 when None."""
        from voxcity.geoprocessor.io import get_gdf_from_gpkg
        
        # Create test GeoJSON without CRS (will be read as None)
        gdf = gpd.GeoDataFrame(
            {'name': ['A']},
            geometry=[Polygon([(139.0, 35.0), (139.1, 35.0), (139.1, 35.1), (139.0, 35.0)])]
        )
        gdf.crs = None  # Explicitly set to None
        
        with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as f:
            gpkg_path = f.name
        
        try:
            # Save without CRS information
            gdf.to_file(gpkg_path, driver='GPKG')
            
            result = get_gdf_from_gpkg(gpkg_path, None)
            
            assert result.crs is not None
        finally:
            Path(gpkg_path).unlink(missing_ok=True)
