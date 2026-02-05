"""Tests for voxcity.geoprocessor.selection module."""
import pytest
import geopandas as gpd
from shapely.geometry import Polygon, box

from voxcity.geoprocessor.selection import (
    filter_buildings,
    find_building_containing_point,
    get_buildings_in_drawn_polygon,
)


class TestFilterBuildings:
    """Tests for filter_buildings function."""

    def test_filters_by_intersection(self):
        """Should return only buildings that intersect the bounding box."""
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                "properties": {"id": 1}
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[10, 10], [11, 10], [11, 11], [10, 11], [10, 10]]]
                },
                "properties": {"id": 2}
            }
        ]
        
        plotting_box = box(0, 0, 5, 5)  # Covers only first building
        
        result = filter_buildings(features, plotting_box)
        
        assert len(result) == 1
        assert result[0]["properties"]["id"] == 1

    def test_returns_empty_for_no_intersection(self):
        """Should return empty list when no buildings intersect."""
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                "properties": {"id": 1}
            }
        ]
        
        plotting_box = box(100, 100, 101, 101)  # Far away
        
        result = filter_buildings(features, plotting_box)
        
        assert len(result) == 0

    def test_partial_intersection_included(self):
        """Buildings with partial intersection should be included."""
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]
                },
                "properties": {"id": 1}
            }
        ]
        
        plotting_box = box(1, 1, 10, 10)  # Partially overlaps
        
        result = filter_buildings(features, plotting_box)
        
        assert len(result) == 1


class TestFindBuildingContainingPoint:
    """Tests for find_building_containing_point function."""

    def test_finds_containing_building(self):
        """Should find building that contains the point."""
        gdf = gpd.GeoDataFrame({
            'id': [1, 2],
            'height': [10.0, 20.0],
        }, geometry=[
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
        ], crs="EPSG:4326")
        
        point = (1.0, 1.0)  # Inside building 1
        
        result = find_building_containing_point(gdf, point)
        
        assert 1 in result
        assert 2 not in result

    def test_point_outside_all_buildings(self):
        """Should return empty when point is outside all buildings."""
        gdf = gpd.GeoDataFrame({
            'id': [1, 2],
        }, geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
        ], crs="EPSG:4326")
        
        point = (3.0, 3.0)  # Outside both buildings
        
        result = find_building_containing_point(gdf, point)
        
        assert len(result) == 0

    def test_point_on_edge(self):
        """Point on edge may or may not be contained (implementation detail)."""
        gdf = gpd.GeoDataFrame({
            'id': [1],
        }, geometry=[
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        ], crs="EPSG:4326")
        
        point = (0.0, 1.0)  # On edge
        
        # Just verify it doesn't crash - contains behavior on edge is implementation-specific
        result = find_building_containing_point(gdf, point)
        assert isinstance(result, list)


class TestGetBuildingsInDrawnPolygon:
    """Tests for get_buildings_in_drawn_polygon function."""

    @pytest.fixture
    def building_gdf(self):
        """Create a GeoDataFrame with test buildings."""
        return gpd.GeoDataFrame({
            'id': [1, 2, 3],
            'height': [10.0, 20.0, 30.0],
        }, geometry=[
            Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),  # Small building at (1,1)
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),  # Small building at (5,5)
            Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]),  # Building at (10,10)
        ], crs="EPSG:4326")

    def test_within_operation(self, building_gdf):
        """Buildings fully within drawn polygon should be found."""
        drawn_polygons = [
            {'vertices': [(0, 0), (3, 0), (3, 3), (0, 3)]}  # Covers building 1
        ]
        
        result = get_buildings_in_drawn_polygon(building_gdf, drawn_polygons, operation='within')
        
        assert 1 in result
        assert 2 not in result
        assert 3 not in result

    def test_intersect_operation(self, building_gdf):
        """Buildings intersecting drawn polygon should be found."""
        drawn_polygons = [
            {'vertices': [(1.5, 1.5), (5.5, 1.5), (5.5, 5.5), (1.5, 5.5)]}  # Intersects 1 and 2
        ]
        
        result = get_buildings_in_drawn_polygon(building_gdf, drawn_polygons, operation='intersect')
        
        assert 1 in result
        assert 2 in result
        assert 3 not in result

    def test_empty_drawn_polygons(self, building_gdf):
        """Should return empty list when no drawn polygons."""
        result = get_buildings_in_drawn_polygon(building_gdf, [], operation='within')
        
        assert result == []

    def test_invalid_operation_raises(self, building_gdf):
        """Invalid operation should raise ValueError."""
        drawn_polygons = [
            {'vertices': [(0, 0), (10, 0), (10, 10), (0, 10)]}
        ]
        
        with pytest.raises(ValueError, match="operation must be"):
            get_buildings_in_drawn_polygon(building_gdf, drawn_polygons, operation='invalid')

    def test_multiple_drawn_polygons(self, building_gdf):
        """Should check all drawn polygons."""
        drawn_polygons = [
            {'vertices': [(0, 0), (3, 0), (3, 3), (0, 3)]},  # Covers building 1
            {'vertices': [(4, 4), (7, 4), (7, 7), (4, 7)]},  # Covers building 2
        ]
        
        result = get_buildings_in_drawn_polygon(building_gdf, drawn_polygons, operation='within')
        
        assert 1 in result
        assert 2 in result
        assert 3 not in result
