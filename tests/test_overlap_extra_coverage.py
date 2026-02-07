"""
Tests for overlap.py additional coverage:
  - Invalid geometry that gets fixed with buffer(0) 
  - Transitive ID re-mapping
  - GEOS exception handling
"""
import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.errors import GEOSException
from unittest.mock import patch, PropertyMock


class TestProcessBuildingFootprintsOverlap:
    """Cover edge-case branches in process_building_footprints_by_overlap."""

    def test_basic_overlap_merges(self):
        """Two overlapping buildings - smaller one gets larger's ID."""
        from voxcity.geoprocessor.overlap import process_building_footprints_by_overlap

        # Large building
        big = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        # Small building overlapping with big
        small = Polygon([(5, 5), (12, 5), (12, 12), (5, 12)])

        gdf = gpd.GeoDataFrame({
            "id": [1, 2],
            "geometry": [big, small],
        })

        result = process_building_footprints_by_overlap(gdf, overlap_threshold=0.3)
        # Small building should be remapped to big building's id
        assert result.iloc[1]["id"] == result.iloc[0]["id"]

    def test_no_overlap_no_merge(self):
        """Non-overlapping buildings remain separate."""
        from voxcity.geoprocessor.overlap import process_building_footprints_by_overlap

        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])

        gdf = gpd.GeoDataFrame({
            "id": [1, 2],
            "geometry": [p1, p2],
        })

        result = process_building_footprints_by_overlap(gdf, overlap_threshold=0.5)
        # IDs should remain unchanged
        ids = result["id"].tolist()
        assert 1 in ids
        assert 2 in ids

    def test_invalid_geometry_gets_buffered(self):
        """Invalid geometry (bowtie) is fixed with buffer(0)."""
        from voxcity.geoprocessor.overlap import process_building_footprints_by_overlap

        # Valid large polygon
        big = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        # Self-intersecting (bowtie) polygon that overlaps with big
        bowtie = Polygon([(3, 3), (7, 7), (7, 3), (3, 7)])

        gdf = gpd.GeoDataFrame({
            "id": [1, 2],
            "geometry": [big, bowtie],
        })

        # Should not raise - invalid geom gets buffer(0) fix
        result = process_building_footprints_by_overlap(gdf, overlap_threshold=0.1)
        assert len(result) == 2

    def test_transitive_id_remapping(self):
        """Larger building already remapped -> transitive lookup."""
        from voxcity.geoprocessor.overlap import process_building_footprints_by_overlap

        # Three buildings: big > medium > small, all overlapping
        big = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        medium = Polygon([(2, 2), (9, 2), (9, 9), (2, 9)])
        small = Polygon([(3, 3), (8, 3), (8, 8), (3, 8)])

        gdf = gpd.GeoDataFrame({
            "id": [1, 2, 3],
            "geometry": [big, medium, small],
        })

        result = process_building_footprints_by_overlap(gdf, overlap_threshold=0.3)
        # All should be remapped to the largest building's id
        final_ids = result["id"].tolist()
        # The biggest polygon's id should be the final id for all
        assert len(set(final_ids)) == 1  # all same id

    def test_geos_exception_handled(self):
        """GEOSException during intersection is caught and skipped."""
        from voxcity.geoprocessor.overlap import process_building_footprints_by_overlap

        big = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        small = Polygon([(5, 5), (8, 5), (8, 8), (5, 8)])

        gdf = gpd.GeoDataFrame({
            "id": [1, 2],
            "geometry": [big, small],
        })

        # Mock intersects to raise GEOSException
        with patch.object(Polygon, "intersects", side_effect=GEOSException("test error")):
            result = process_building_footprints_by_overlap(gdf, overlap_threshold=0.3)
            # Should not crash, small stays with its own id
            assert len(result) == 2

    def test_no_crs(self):
        """GeoDataFrame without CRS should work without projection."""
        from voxcity.geoprocessor.overlap import process_building_footprints_by_overlap

        p1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        p2 = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])

        gdf = gpd.GeoDataFrame({
            "id": [1, 2],
            "geometry": [p1, p2],
        }, crs=None)

        result = process_building_footprints_by_overlap(gdf, overlap_threshold=0.3)
        assert len(result) == 2
