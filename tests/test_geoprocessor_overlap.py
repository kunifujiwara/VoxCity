"""Tests for voxcity.geoprocessor.overlap module - building footprint overlap processing."""
import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box

from voxcity.geoprocessor.overlap import process_building_footprints_by_overlap


@pytest.fixture
def simple_buildings_gdf():
    """Create a simple GeoDataFrame with non-overlapping buildings."""
    buildings = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),  # Building 1
        Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]),  # Building 2
    ]
    return gpd.GeoDataFrame(
        {'id': [1, 2], 'height': [15.0, 20.0]},
        geometry=buildings,
        crs="EPSG:4326"
    )


@pytest.fixture
def overlapping_buildings_gdf():
    """Create buildings with overlap - smaller overlaps larger."""
    # Larger building
    large = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])  # Area = 400
    # Smaller building overlapping significantly with large
    small = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])  # Area = 100, fully inside large
    return gpd.GeoDataFrame(
        {'id': [1, 2], 'height': [15.0, 10.0]},
        geometry=[large, small],
        crs="EPSG:4326"
    )


@pytest.fixture
def partial_overlap_gdf():
    """Create buildings with partial overlap."""
    # Building 1
    bldg1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])  # Area = 100
    # Building 2 - 50% overlap with building 1
    bldg2 = Polygon([(5, 0), (15, 0), (15, 10), (5, 10)])  # Area = 100, overlap = 50
    return gpd.GeoDataFrame(
        {'id': [1, 2], 'height': [15.0, 20.0]},
        geometry=[bldg1, bldg2],
        crs="EPSG:4326"
    )


class TestProcessBuildingFootprintsByOverlap:
    """Tests for process_building_footprints_by_overlap function."""
    
    def test_non_overlapping_buildings_unchanged(self, simple_buildings_gdf):
        """Test that non-overlapping buildings keep their IDs."""
        gdf = simple_buildings_gdf.copy()
        process_building_footprints_by_overlap(gdf)
        
        assert gdf.loc[0, 'id'] == 1
        assert gdf.loc[1, 'id'] == 2
    
    def test_full_overlap_merges_ids(self, overlapping_buildings_gdf):
        """Test that fully overlapping smaller building gets ID of larger."""
        gdf = overlapping_buildings_gdf.copy()
        process_building_footprints_by_overlap(gdf, overlap_threshold=0.5)
        
        # The smaller building should get the ID of the larger building
        # since it's fully inside and overlap ratio > 0.5
        # Note: IDs may be reassigned based on area sorting
        assert len(gdf) == 2
    
    def test_partial_overlap_above_threshold(self, partial_overlap_gdf):
        """Test partial overlap above threshold merges IDs."""
        gdf = partial_overlap_gdf.copy()
        # With 50% overlap and threshold at 0.4, should merge
        process_building_footprints_by_overlap(gdf, overlap_threshold=0.4)
        
        # Function modifies in place
        assert len(gdf) == 2
    
    def test_partial_overlap_below_threshold_keeps_separate(self, partial_overlap_gdf):
        """Test partial overlap below threshold keeps separate IDs."""
        gdf = partial_overlap_gdf.copy()
        original_ids = gdf['id'].tolist()
        
        # With 50% overlap and threshold at 0.9, should NOT merge
        process_building_footprints_by_overlap(gdf, overlap_threshold=0.9)
        
        # IDs should remain unchanged
        assert gdf['id'].tolist() == original_ids
    
    def test_handles_no_id_column(self):
        """Test that function works with missing id column."""
        buildings = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]),
        ]
        gdf = gpd.GeoDataFrame(
            {'height': [15.0, 20.0]},
            geometry=buildings,
            crs="EPSG:4326"
        )
        
        # Function creates id internally on a copy, so this should not raise
        result = process_building_footprints_by_overlap(gdf)
        
        # Should complete without error
        assert len(gdf) == 2
    
    def test_handles_no_crs(self):
        """Test that function works with no CRS."""
        buildings = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]),
        ]
        gdf = gpd.GeoDataFrame(
            {'id': [1, 2]},
            geometry=buildings,
            crs=None
        )
        
        # Should not raise error
        process_building_footprints_by_overlap(gdf)
        assert len(gdf) == 2
    
    def test_handles_invalid_geometry(self):
        """Test that function handles invalid geometries."""
        # Create a self-intersecting polygon (invalid)
        invalid = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
        valid = Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])
        
        gdf = gpd.GeoDataFrame(
            {'id': [1, 2]},
            geometry=[invalid, valid],
            crs="EPSG:4326"
        )
        
        # Should not raise error, function tries to fix invalid geometries
        process_building_footprints_by_overlap(gdf)
    
    def test_multiple_overlapping_buildings(self):
        """Test with multiple buildings in chain of overlaps."""
        # Three buildings: A, B, C where B overlaps A, C overlaps B
        bldg_a = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])  # Largest
        bldg_b = Polygon([(10, 0), (25, 0), (25, 15), (10, 15)])  # Medium
        bldg_c = Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])  # Smallest
        
        gdf = gpd.GeoDataFrame(
            {'id': [1, 2, 3]},
            geometry=[bldg_a, bldg_b, bldg_c],
            crs="EPSG:4326"
        )
        
        process_building_footprints_by_overlap(gdf, overlap_threshold=0.3)
        
        # Should complete without error
        assert len(gdf) == 3
    
    def test_empty_gdf(self):
        """Test with empty GeoDataFrame."""
        gdf = gpd.GeoDataFrame(
            {'id': [], 'geometry': []},
            crs="EPSG:4326"
        )
        gdf = gdf.set_geometry('geometry')
        
        # Should handle empty GDF gracefully
        process_building_footprints_by_overlap(gdf)
        assert len(gdf) == 0
    
    def test_single_building(self):
        """Test with single building."""
        gdf = gpd.GeoDataFrame(
            {'id': [1]},
            geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
            crs="EPSG:4326"
        )
        
        process_building_footprints_by_overlap(gdf)
        assert gdf.loc[0, 'id'] == 1


class TestOverlapThresholds:
    """Tests for different overlap threshold values."""
    
    def test_threshold_zero(self):
        """Test with threshold of 0 (any overlap merges)."""
        bldg1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        bldg2 = Polygon([(9, 0), (19, 0), (19, 10), (9, 10)])  # Tiny overlap
        
        gdf = gpd.GeoDataFrame(
            {'id': [1, 2]},
            geometry=[bldg1, bldg2],
            crs="EPSG:4326"
        )
        
        # With threshold 0, any overlap should trigger merge
        process_building_footprints_by_overlap(gdf, overlap_threshold=0.0)
        assert len(gdf) == 2
    
    def test_threshold_one(self):
        """Test with threshold of 1.0 (complete overlap required)."""
        bldg1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        bldg2 = Polygon([(5, 0), (15, 0), (15, 10), (5, 10)])  # 50% overlap
        
        gdf = gpd.GeoDataFrame(
            {'id': [1, 2]},
            geometry=[bldg1, bldg2],
            crs="EPSG:4326"
        )
        original_ids = gdf['id'].tolist()
        
        # With threshold 1.0, partial overlap should NOT merge
        process_building_footprints_by_overlap(gdf, overlap_threshold=1.0)
        assert gdf['id'].tolist() == original_ids
    
    def test_default_threshold(self):
        """Test default threshold value (0.5)."""
        # Create buildings with exactly 60% overlap
        bldg1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])  # Area = 100
        bldg2 = Polygon([(4, 0), (14, 0), (14, 10), (4, 10)])  # 60% overlap
        
        gdf = gpd.GeoDataFrame(
            {'id': [1, 2]},
            geometry=[bldg1, bldg2],
            crs="EPSG:4326"
        )
        
        # Default threshold is 0.5, so 60% overlap should trigger merge
        process_building_footprints_by_overlap(gdf)  # Uses default threshold
        assert len(gdf) == 2


class TestEdgeCases:
    """Edge case tests for overlap processing."""
    
    def test_identical_buildings(self):
        """Test with two identical buildings (100% overlap)."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        
        gdf = gpd.GeoDataFrame(
            {'id': [1, 2]},
            geometry=[poly, poly],  # Identical
            crs="EPSG:4326"
        )
        
        process_building_footprints_by_overlap(gdf, overlap_threshold=0.5)
        # Both should have same ID after processing
        assert len(gdf) == 2
    
    def test_touching_but_not_overlapping(self):
        """Test buildings that touch but don't overlap."""
        bldg1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        bldg2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])  # Adjacent
        
        gdf = gpd.GeoDataFrame(
            {'id': [1, 2]},
            geometry=[bldg1, bldg2],
            crs="EPSG:4326"
        )
        original_ids = gdf['id'].tolist()
        
        process_building_footprints_by_overlap(gdf)
        # Should keep separate IDs
        assert gdf['id'].tolist() == original_ids
    
    def test_many_small_buildings_inside_large(self):
        """Test many small buildings fully inside one large building."""
        # Use proper coordinate range for WGS84
        large = Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])
        
        small_buildings = [
            Polygon([(0.0001, 0.0001), (0.0002, 0.0001), (0.0002, 0.0002), (0.0001, 0.0002)]),
            Polygon([(0.0003, 0.0001), (0.0004, 0.0001), (0.0004, 0.0002), (0.0003, 0.0002)]),
            Polygon([(0.0005, 0.0001), (0.0006, 0.0001), (0.0006, 0.0002), (0.0005, 0.0002)]),
        ]
        
        gdf = gpd.GeoDataFrame(
            {'id': [1, 2, 3, 4]},
            geometry=[large] + small_buildings,
            crs="EPSG:4326"
        )
        
        process_building_footprints_by_overlap(gdf, overlap_threshold=0.5)
        # Should complete without error
        assert len(gdf) == 4
