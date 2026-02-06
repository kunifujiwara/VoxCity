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


class TestSwapCoordinates:
    """Tests for swap_coordinates function."""

    def test_swap_coordinates_polygon(self):
        """Test swapping coordinates for Polygon geometry."""
        from voxcity.geoprocessor.io import swap_coordinates
        
        # Lat/lon order (lat, lon)
        features = [{
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[35.0, 139.0], [35.1, 139.0], [35.1, 139.1], [35.0, 139.1], [35.0, 139.0]]]
            },
            'properties': {}
        }]
        
        swap_coordinates(features)
        
        # Should now be (lon, lat)
        coords = features[0]['geometry']['coordinates'][0]
        assert coords[0] == [139.0, 35.0]
        assert coords[1] == [139.0, 35.1]

    def test_swap_coordinates_multipolygon(self):
        """Test swapping coordinates for MultiPolygon geometry."""
        from voxcity.geoprocessor.io import swap_coordinates
        
        features = [{
            'type': 'Feature',
            'geometry': {
                'type': 'MultiPolygon',
                'coordinates': [[[[35.0, 139.0], [35.1, 139.0], [35.1, 139.1], [35.0, 139.0]]]]
            },
            'properties': {}
        }]
        
        swap_coordinates(features)
        
        coords = features[0]['geometry']['coordinates'][0][0]
        assert coords[0] == [139.0, 35.0]

    def test_swap_coordinates_point_ignored(self):
        """Test that Point geometry is not modified (no swap logic for it)."""
        from voxcity.geoprocessor.io import swap_coordinates
        
        features = [{
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [35.0, 139.0]
            },
            'properties': {}
        }]
        
        # Should not raise and not modify (no branch for Point)
        swap_coordinates(features)
        
        assert features[0]['geometry']['coordinates'] == [35.0, 139.0]


class TestSaveGeojson:
    """Tests for save_geojson function."""

    def test_save_geojson_basic(self):
        """Test saving GeoJSON to file."""
        from voxcity.geoprocessor.io import save_geojson
        
        features = [{
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[35.0, 139.0], [35.1, 139.0], [35.1, 139.1], [35.0, 139.0]]]
            },
            'properties': {'name': 'test'}
        }]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            save_path = f.name
        
        try:
            save_geojson(features, save_path)
            
            with open(save_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded['type'] == 'FeatureCollection'
            assert len(loaded['features']) == 1
            assert loaded['features'][0]['properties']['name'] == 'test'
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_save_geojson_preserves_original(self):
        """Test that original features are not modified."""
        from voxcity.geoprocessor.io import save_geojson
        
        original_coords = [[[35.0, 139.0], [35.1, 139.0], [35.0, 139.0]]]
        features = [{
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': original_coords
            },
            'properties': {}
        }]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            save_path = f.name
        
        try:
            save_geojson(features, save_path)
            
            # Original should be unchanged
            assert features[0]['geometry']['coordinates'][0][0] == [35.0, 139.0]
        finally:
            Path(save_path).unlink(missing_ok=True)


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
