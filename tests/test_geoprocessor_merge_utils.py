"""Tests for voxcity.geoprocessor.merge_utils module."""
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point

from voxcity.geoprocessor.merge_utils import (
    _merge_gdfs_with_missing_columns,
    merge_gdfs_with_id_conflict_resolution,
)


class TestMergeGdfsWithMissingColumns:
    """Tests for _merge_gdfs_with_missing_columns helper function."""
    
    @pytest.fixture
    def simple_gdf1(self):
        """Simple GeoDataFrame with columns A, B, geometry."""
        return gpd.GeoDataFrame({
            'A': [1, 2],
            'B': ['x', 'y'],
            'geometry': [Point(0, 0), Point(1, 1)]
        }, crs="EPSG:4326")
    
    @pytest.fixture
    def simple_gdf2(self):
        """Simple GeoDataFrame with columns B, C, geometry."""
        return gpd.GeoDataFrame({
            'B': ['z'],
            'C': [100],
            'geometry': [Point(2, 2)]
        }, crs="EPSG:4326")
    
    def test_merge_with_missing_columns(self, simple_gdf1, simple_gdf2):
        """Test that missing columns are filled with None."""
        result = _merge_gdfs_with_missing_columns(simple_gdf1.copy(), simple_gdf2.copy())
        
        # Should have all columns from both
        assert 'A' in result.columns
        assert 'B' in result.columns
        assert 'C' in result.columns
        assert 'geometry' in result.columns
        
        # Should have 3 rows total
        assert len(result) == 3
        
        # Check that missing values are None
        assert pd.isna(result.iloc[2]['A'])  # From gdf2, doesn't have A
        assert pd.isna(result.iloc[0]['C'])  # From gdf1, doesn't have C
        assert pd.isna(result.iloc[1]['C'])
    
    def test_merge_identical_columns(self):
        """Test merging GDFs with identical columns."""
        gdf1 = gpd.GeoDataFrame({
            'A': [1, 2],
            'geometry': [Point(0, 0), Point(1, 1)]
        }, crs="EPSG:4326")
        
        gdf2 = gpd.GeoDataFrame({
            'A': [3],
            'geometry': [Point(2, 2)]
        }, crs="EPSG:4326")
        
        result = _merge_gdfs_with_missing_columns(gdf1.copy(), gdf2.copy())
        
        assert len(result) == 3
        assert list(result['A']) == [1, 2, 3]
    
    def test_merge_preserves_geometry(self, simple_gdf1, simple_gdf2):
        """Test that geometry is preserved correctly."""
        result = _merge_gdfs_with_missing_columns(simple_gdf1.copy(), simple_gdf2.copy())
        
        assert result.iloc[0].geometry.equals(Point(0, 0))
        assert result.iloc[1].geometry.equals(Point(1, 1))
        assert result.iloc[2].geometry.equals(Point(2, 2))


class TestMergeGdfsWithIdConflictResolution:
    """Tests for merge_gdfs_with_id_conflict_resolution function."""
    
    @pytest.fixture
    def buildings_gdf1(self):
        """Primary GeoDataFrame with building data."""
        return gpd.GeoDataFrame({
            'id': [1, 2, 3],
            'building_id': [101, 102, 103],
            'height': [10.0, 20.0, 30.0],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                Polygon([(4, 0), (5, 0), (5, 1), (4, 1)]),
            ]
        }, crs="EPSG:4326")
    
    @pytest.fixture
    def buildings_gdf2_no_conflict(self):
        """Secondary GeoDataFrame without ID conflicts."""
        return gpd.GeoDataFrame({
            'id': [4, 5],
            'building_id': [104, 105],
            'height': [40.0, 50.0],
            'geometry': [
                Polygon([(6, 0), (7, 0), (7, 1), (6, 1)]),
                Polygon([(8, 0), (9, 0), (9, 1), (8, 1)]),
            ]
        }, crs="EPSG:4326")
    
    @pytest.fixture
    def buildings_gdf2_with_conflict(self):
        """Secondary GeoDataFrame with ID conflicts."""
        return gpd.GeoDataFrame({
            'id': [2, 5],  # id=2 conflicts
            'building_id': [102, 105],  # building_id=102 conflicts
            'height': [40.0, 50.0],
            'geometry': [
                Polygon([(6, 0), (7, 0), (7, 1), (6, 1)]),
                Polygon([(8, 0), (9, 0), (9, 1), (8, 1)]),
            ]
        }, crs="EPSG:4326")
    
    def test_merge_no_conflicts(self, buildings_gdf1, buildings_gdf2_no_conflict, capsys):
        """Test merging without ID conflicts."""
        result = merge_gdfs_with_id_conflict_resolution(
            buildings_gdf1.copy(),
            buildings_gdf2_no_conflict.copy()
        )
        
        assert len(result) == 5
        # All original IDs should be present
        assert set(result['id']) == {1, 2, 3, 4, 5}
        assert set(result['building_id']) == {101, 102, 103, 104, 105}
        
        # Check output message
        captured = capsys.readouterr()
        assert "Merged 3 buildings" in captured.out
        assert "Total buildings in merged dataset: 5" in captured.out
    
    def test_merge_with_conflicts(self, buildings_gdf1, buildings_gdf2_with_conflict, capsys):
        """Test merging with ID conflicts - IDs should be modified."""
        result = merge_gdfs_with_id_conflict_resolution(
            buildings_gdf1.copy(),
            buildings_gdf2_with_conflict.copy()
        )
        
        assert len(result) == 5
        # IDs should be unique
        assert len(result['id'].unique()) == 5
        assert len(result['building_id'].unique()) == 5
        
        # Original IDs from primary should be preserved
        assert 1 in result['id'].values
        assert 2 in result['id'].values
        assert 3 in result['id'].values
        
        # Check output message mentions modified IDs
        captured = capsys.readouterr()
        assert "Modified IDs" in captured.out
    
    def test_merge_missing_id_columns(self, capsys):
        """Test merging when ID columns are missing."""
        gdf1 = gpd.GeoDataFrame({
            'other_col': [1, 2],
            'geometry': [Point(0, 0), Point(1, 1)]
        }, crs="EPSG:4326")
        
        gdf2 = gpd.GeoDataFrame({
            'other_col': [3],
            'geometry': [Point(2, 2)]
        }, crs="EPSG:4326")
        
        result = merge_gdfs_with_id_conflict_resolution(gdf1.copy(), gdf2.copy())
        
        assert len(result) == 3
        
        # Check warning about missing columns
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "missing" in captured.out.lower()
    
    def test_merge_partial_id_columns(self, capsys):
        """Test merging when some ID columns are missing from one GDF."""
        gdf1 = gpd.GeoDataFrame({
            'id': [1, 2],
            'geometry': [Point(0, 0), Point(1, 1)]
        }, crs="EPSG:4326")
        
        gdf2 = gpd.GeoDataFrame({
            'id': [3],
            'geometry': [Point(2, 2)]
        }, crs="EPSG:4326")
        
        result = merge_gdfs_with_id_conflict_resolution(
            gdf1.copy(),
            gdf2.copy(),
            id_columns=['id', 'building_id']  # building_id missing from both
        )
        
        assert len(result) == 3
        
        # Should still work with available ID columns
        captured = capsys.readouterr()
        assert "Warning" in captured.out
    
    def test_preserve_heights_after_merge(self, buildings_gdf1, buildings_gdf2_no_conflict):
        """Test that height data is preserved after merge."""
        result = merge_gdfs_with_id_conflict_resolution(
            buildings_gdf1.copy(),
            buildings_gdf2_no_conflict.copy()
        )
        
        # Check all heights are present
        expected_heights = {10.0, 20.0, 30.0, 40.0, 50.0}
        assert set(result['height']) == expected_heights
    
    def test_custom_id_columns(self):
        """Test with custom ID column names."""
        gdf1 = gpd.GeoDataFrame({
            'custom_id': [1, 2],
            'geometry': [Point(0, 0), Point(1, 1)]
        }, crs="EPSG:4326")
        
        gdf2 = gpd.GeoDataFrame({
            'custom_id': [2, 3],  # 2 conflicts
            'geometry': [Point(2, 2), Point(3, 3)]
        }, crs="EPSG:4326")
        
        result = merge_gdfs_with_id_conflict_resolution(
            gdf1.copy(),
            gdf2.copy(),
            id_columns=['custom_id']
        )
        
        assert len(result) == 4
        # Conflicting ID should be modified
        # Original IDs 1 and 2 from primary preserved, ID 2 in secondary modified
        assert 1 in result['custom_id'].values
        assert 2 in result['custom_id'].values
    
    def test_empty_secondary_gdf(self, buildings_gdf1, capsys):
        """Test merging with empty secondary GDF."""
        empty_gdf = gpd.GeoDataFrame({
            'id': [],
            'building_id': [],
            'height': [],
            'geometry': []
        }, crs="EPSG:4326")
        
        result = merge_gdfs_with_id_conflict_resolution(
            buildings_gdf1.copy(),
            empty_gdf
        )
        
        assert len(result) == 3  # Only primary data
        
        captured = capsys.readouterr()
        assert "Merged 3 buildings" in captured.out
        assert "0 buildings from secondary" in captured.out


class TestEdgeCases:
    """Edge cases and error handling tests."""
    
    def test_merge_with_nan_ids(self):
        """Test merging when some IDs are NaN."""
        gdf1 = gpd.GeoDataFrame({
            'id': [1, np.nan, 3],
            'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)]
        }, crs="EPSG:4326")
        
        gdf2 = gpd.GeoDataFrame({
            'id': [4, np.nan],
            'geometry': [Point(3, 3), Point(4, 4)]
        }, crs="EPSG:4326")
        
        # Should not raise an error
        result = merge_gdfs_with_id_conflict_resolution(
            gdf1.copy(),
            gdf2.copy(),
            id_columns=['id']
        )
        
        assert len(result) == 5
    
    def test_merge_with_string_ids(self):
        """Test merging when IDs are strings."""
        gdf1 = gpd.GeoDataFrame({
            'id': ['a', 'b', 'c'],
            'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)]
        }, crs="EPSG:4326")
        
        gdf2 = gpd.GeoDataFrame({
            'id': ['b', 'd'],  # 'b' conflicts
            'geometry': [Point(3, 3), Point(4, 4)]
        }, crs="EPSG:4326")
        
        result = merge_gdfs_with_id_conflict_resolution(
            gdf1.copy(),
            gdf2.copy(),
            id_columns=['id']
        )
        
        assert len(result) == 5
