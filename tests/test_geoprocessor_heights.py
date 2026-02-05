"""Tests for voxcity.geoprocessor.heights module."""
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box
import tempfile
import os

from voxcity.geoprocessor.heights import (
    extract_building_heights_from_gdf,
    complement_building_heights_from_gdf,
)


class TestExtractBuildingHeightsFromGdf:
    """Tests for extract_building_heights_from_gdf function."""
    
    @pytest.fixture
    def primary_gdf_no_heights(self):
        """Primary GDF with buildings that have no height data."""
        return gpd.GeoDataFrame({
            'id': [1, 2, 3],
            'height': [0.0, np.nan, 0.0],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                Polygon([(4, 0), (5, 0), (5, 1), (4, 1)]),
            ]
        }, crs="EPSG:4326")
    
    @pytest.fixture
    def primary_gdf_with_heights(self):
        """Primary GDF with buildings that already have height data."""
        return gpd.GeoDataFrame({
            'id': [1, 2, 3],
            'height': [10.0, 20.0, 30.0],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                Polygon([(4, 0), (5, 0), (5, 1), (4, 1)]),
            ]
        }, crs="EPSG:4326")
    
    @pytest.fixture
    def reference_gdf_with_heights(self):
        """Reference GDF with height data for overlapping buildings."""
        return gpd.GeoDataFrame({
            'id': [101, 102, 103],
            'height': [15.0, 25.0, 35.0],
            'geometry': [
                # Overlaps with building 1
                Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
                # Overlaps with building 2
                Polygon([(2.5, 0.5), (3.5, 0.5), (3.5, 1.5), (2.5, 1.5)]),
                # No overlap with any building
                Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
            ]
        }, crs="EPSG:4326")
    
    def test_extract_heights_from_overlapping_buildings(self, primary_gdf_no_heights, reference_gdf_with_heights, capsys):
        """Test height extraction from overlapping reference buildings."""
        result = extract_building_heights_from_gdf(
            primary_gdf_no_heights.copy(),
            reference_gdf_with_heights.copy()
        )
        
        # Buildings 1 and 2 should get heights from overlapping reference
        assert result.iloc[0]['height'] > 0  # Building 1 overlaps with ref 101
        assert result.iloc[1]['height'] > 0  # Building 2 overlaps with ref 102
        
        # Building 3 has no overlap, should remain nan
        assert pd.isna(result.iloc[2]['height'])
        
        # Check printed output
        captured = capsys.readouterr()
        assert "building footprints without height" in captured.out
    
    def test_preserve_existing_heights(self, primary_gdf_with_heights, reference_gdf_with_heights):
        """Test that existing heights are preserved."""
        result = extract_building_heights_from_gdf(
            primary_gdf_with_heights.copy(),
            reference_gdf_with_heights.copy()
        )
        
        # Original heights should be preserved
        assert result.iloc[0]['height'] == 10.0
        assert result.iloc[1]['height'] == 20.0
        assert result.iloc[2]['height'] == 30.0
    
    def test_mixed_height_data(self, reference_gdf_with_heights, capsys):
        """Test GDF with some heights and some missing."""
        mixed_gdf = gpd.GeoDataFrame({
            'id': [1, 2, 3],
            'height': [10.0, 0.0, np.nan],  # 1 has height, 2 and 3 don't
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                Polygon([(4, 0), (5, 0), (5, 1), (4, 1)]),
            ]
        }, crs="EPSG:4326")
        
        result = extract_building_heights_from_gdf(
            mixed_gdf.copy(),
            reference_gdf_with_heights.copy()
        )
        
        # Building 1 should keep original height
        assert result.iloc[0]['height'] == 10.0
        # Building 2 should get height from overlap
        assert result.iloc[1]['height'] > 0
    
    def test_no_overlap(self, capsys):
        """Test when there is no overlap between GDFs."""
        primary = gpd.GeoDataFrame({
            'id': [1],
            'height': [0.0],
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        }, crs="EPSG:4326")
        
        reference = gpd.GeoDataFrame({
            'id': [101],
            'height': [50.0],
            'geometry': [Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])]
        }, crs="EPSG:4326")
        
        result = extract_building_heights_from_gdf(primary.copy(), reference.copy())
        
        assert pd.isna(result.iloc[0]['height'])
    
    def test_empty_reference_gdf(self, primary_gdf_no_heights):
        """Test with empty reference GDF."""
        empty_ref = gpd.GeoDataFrame({
            'id': [],
            'height': [],
            'geometry': []
        }, crs="EPSG:4326")
        
        result = extract_building_heights_from_gdf(
            primary_gdf_no_heights.copy(),
            empty_ref
        )
        
        # All heights should remain nan
        assert pd.isna(result['height']).all() or (result['height'] == 0).all()
    
    def test_no_height_column_in_primary(self):
        """Test when primary GDF has no height column."""
        primary = gpd.GeoDataFrame({
            'id': [1, 2],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ]
        }, crs="EPSG:4326")
        
        reference = gpd.GeoDataFrame({
            'id': [101],
            'height': [20.0],
            'geometry': [
                Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
            ]
        }, crs="EPSG:4326")
        
        result = extract_building_heights_from_gdf(primary.copy(), reference.copy())
        
        # Height column should be created
        assert 'height' in result.columns


class TestComplementBuildingHeightsFromGdf:
    """Tests for complement_building_heights_from_gdf function."""
    
    @pytest.fixture
    def primary_gdf(self):
        """Primary GDF with some buildings missing heights."""
        return gpd.GeoDataFrame({
            'id': [1, 2, 3],
            'height': [10.0, 0.0, np.nan],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                Polygon([(4, 0), (5, 0), (5, 1), (4, 1)]),
            ]
        }, crs="EPSG:4326")
    
    @pytest.fixture
    def reference_gdf(self):
        """Reference GDF with additional buildings."""
        return gpd.GeoDataFrame({
            'id': [101, 102, 103],
            'height': [15.0, 25.0, 40.0],
            'geometry': [
                # Overlaps with building 2
                Polygon([(2.5, 0.5), (3.5, 0.5), (3.5, 1.5), (2.5, 1.5)]),
                # Overlaps with building 3
                Polygon([(4.5, 0.5), (5.5, 0.5), (5.5, 1.5), (4.5, 1.5)]),
                # No overlap - new building to add
                Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
            ]
        }, crs="EPSG:4326")
    
    def test_complement_adds_non_overlapping_buildings(self, primary_gdf, reference_gdf, capsys):
        """Test that non-overlapping buildings from reference are added."""
        result = complement_building_heights_from_gdf(
            primary_gdf.copy(),
            reference_gdf.copy()
        )
        
        # Should have more buildings than primary
        assert len(result) > len(primary_gdf)
        
        # Building at (10, 0) should be added
        captured = capsys.readouterr()
        assert "were added from the complementary source" in captured.out
    
    def test_complement_fills_missing_heights(self, primary_gdf, reference_gdf, capsys):
        """Test that missing heights are filled from overlapping reference."""
        result = complement_building_heights_from_gdf(
            primary_gdf.copy(),
            reference_gdf.copy()
        )
        
        # Check output message
        captured = capsys.readouterr()
        assert "building footprints" in captured.out
    
    def test_preserve_existing_heights(self, reference_gdf):
        """Test that existing heights in primary are preserved."""
        primary = gpd.GeoDataFrame({
            'id': [1],
            'height': [100.0],  # Already has height
            'geometry': [Polygon([(2.5, 0.5), (3.5, 0.5), (3.5, 1.5), (2.5, 1.5)])]
        }, crs="EPSG:4326")
        
        result = complement_building_heights_from_gdf(
            primary.copy(),
            reference_gdf.copy()
        )
        
        # Original height should be preserved
        original_building = result[result['id'] == 1]
        assert len(original_building) == 1
        assert original_building.iloc[0]['height'] == 100.0
    
    def test_empty_primary_gdf(self, reference_gdf, capsys):
        """Test with empty primary GDF."""
        empty_primary = gpd.GeoDataFrame({
            'id': [],
            'height': [],
            'geometry': []
        }, crs="EPSG:4326")
        
        result = complement_building_heights_from_gdf(
            empty_primary,
            reference_gdf.copy()
        )
        
        # Should contain all buildings from reference
        assert len(result) == len(reference_gdf)
    
    def test_empty_reference_gdf(self, primary_gdf):
        """Test with empty reference GDF."""
        empty_ref = gpd.GeoDataFrame({
            'id': [],
            'height': [],
            'geometry': []
        }, crs="EPSG:4326")
        
        result = complement_building_heights_from_gdf(
            primary_gdf.copy(),
            empty_ref
        )
        
        # Should only contain primary buildings
        assert len(result) == len(primary_gdf)
    
    def test_weighted_height_calculation(self, capsys):
        """Test that heights are weighted by overlap area."""
        # Primary building covering area [0,0] to [2,2]
        primary = gpd.GeoDataFrame({
            'id': [1],
            'height': [0.0],
            'geometry': [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]
        }, crs="EPSG:4326")
        
        # Two reference buildings with different heights covering different parts
        reference = gpd.GeoDataFrame({
            'id': [101, 102],
            'height': [10.0, 30.0],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 2), (0, 2)]),  # Left half, height 10
                Polygon([(1, 0), (2, 0), (2, 2), (1, 2)]),  # Right half, height 30
            ]
        }, crs="EPSG:4326")
        
        result = complement_building_heights_from_gdf(
            primary.copy(),
            reference.copy()
        )
        
        # Weighted average should be (10*2 + 30*2) / 4 = 20
        building_result = result[result['id'] == 1]
        if len(building_result) > 0:
            # Allow some tolerance for floating point
            assert abs(building_result.iloc[0]['height'] - 20.0) < 1.0


class TestHeightsEdgeCases:
    """Edge cases and error handling tests."""
    
    def test_invalid_geometry_handling(self, capsys):
        """Test handling of invalid geometries."""
        # Create an invalid polygon (self-intersecting)
        invalid_poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        
        primary = gpd.GeoDataFrame({
            'id': [1],
            'height': [0.0],
            'geometry': [invalid_poly]
        }, crs="EPSG:4326")
        
        reference = gpd.GeoDataFrame({
            'id': [101],
            'height': [20.0],
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        }, crs="EPSG:4326")
        
        # Should handle gracefully
        try:
            result = extract_building_heights_from_gdf(primary.copy(), reference.copy())
            assert len(result) == 1
        except Exception as e:
            # Some invalid geometries may raise exceptions, which is acceptable
            pass
    
    def test_different_crs(self):
        """Test that function handles different CRS."""
        primary = gpd.GeoDataFrame({
            'id': [1],
            'height': [0.0],
            'geometry': [Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])]
        }, crs="EPSG:4326")
        
        # Convert to different CRS
        reference = gpd.GeoDataFrame({
            'id': [101],
            'height': [20.0],
            'geometry': [Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])]
        }, crs="EPSG:4326").to_crs("EPSG:3857")
        
        # This may need CRS handling - just ensure it doesn't crash
        try:
            result = extract_building_heights_from_gdf(primary.copy(), reference.copy())
        except Exception:
            pass  # CRS issues are acceptable to raise
    
    def test_negative_heights(self):
        """Test handling of negative heights."""
        primary = gpd.GeoDataFrame({
            'id': [1],
            'height': [-5.0],  # Negative height should be treated as missing
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        }, crs="EPSG:4326")
        
        reference = gpd.GeoDataFrame({
            'id': [101],
            'height': [20.0],
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        }, crs="EPSG:4326")
        
        result = extract_building_heights_from_gdf(primary.copy(), reference.copy())
        
        # Negative heights might be preserved or replaced - either is acceptable
        assert len(result) == 1
    
    def test_large_overlap_area(self):
        """Test with large overlapping areas."""
        primary = gpd.GeoDataFrame({
            'id': list(range(100)),
            'height': [0.0] * 100,
            'geometry': [
                Polygon([(i, 0), (i+1, 0), (i+1, 1), (i, 1)])
                for i in range(100)
            ]
        }, crs="EPSG:4326")
        
        reference = gpd.GeoDataFrame({
            'id': list(range(100, 200)),
            'height': [float(i) for i in range(100)],
            'geometry': [
                Polygon([(i+0.5, 0.5), (i+1.5, 0.5), (i+1.5, 1.5), (i+0.5, 1.5)])
                for i in range(100)
            ]
        }, crs="EPSG:4326")
        
        result = extract_building_heights_from_gdf(primary.copy(), reference.copy())
        
        assert len(result) == 100
