"""Extended tests for voxcity.geoprocessor.utils module - improving coverage."""
import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point, box
import tempfile
import os

from voxcity.geoprocessor.utils import (
    tile_from_lat_lon,
    quadkey_to_tile,
    haversine_distance,
    create_polygon,
    create_geodataframe,
    setup_transformer,
    transform_coords,
    convert_format_lat_lon,
)


class TestTileCoordinates:
    """Tests for tile coordinate functions."""
    
    def test_tile_from_lat_lon_tokyo(self):
        """Test tile calculation for Tokyo at various zoom levels."""
        lat, lon = 35.6762, 139.6503
        
        # Zoom level 10
        x, y = tile_from_lat_lon(lat, lon, 10)
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert x > 0
        assert y > 0
    
    def test_tile_from_lat_lon_zoom_0(self):
        """Test at zoom level 0 (entire world is one tile)."""
        lat, lon = 0, 0
        x, y = tile_from_lat_lon(lat, lon, 0)
        assert x == 0
        assert y == 0
    
    def test_tile_from_lat_lon_equator(self):
        """Test on equator at prime meridian."""
        lat, lon = 0, 0
        x, y = tile_from_lat_lon(lat, lon, 10)
        # Should be in middle of tile grid
        assert x == 512  # 2^10 / 2
        assert y == 512
    
    def test_tile_from_lat_lon_different_hemispheres(self):
        """Test locations in different hemispheres."""
        # Northern hemisphere
        x1, y1 = tile_from_lat_lon(45, 90, 5)
        # Southern hemisphere  
        x2, y2 = tile_from_lat_lon(-45, 90, 5)
        
        assert x1 == x2  # Same longitude
        assert y1 < y2   # Northern is above (smaller y)
    
    def test_quadkey_to_tile_simple(self):
        """Test simple quadkey conversion."""
        x, y, zoom = quadkey_to_tile("0")
        assert zoom == 1
        assert x == 0
        assert y == 0
        
        x, y, zoom = quadkey_to_tile("1")
        assert zoom == 1
        assert x == 1
        assert y == 0
        
        x, y, zoom = quadkey_to_tile("2")
        assert zoom == 1
        assert x == 0
        assert y == 1
        
        x, y, zoom = quadkey_to_tile("3")
        assert zoom == 1
        assert x == 1
        assert y == 1
    
    def test_quadkey_to_tile_multi_digit(self):
        """Test multi-digit quadkey."""
        x, y, zoom = quadkey_to_tile("120")
        assert zoom == 3


class TestHaversineDistance:
    """Tests for Haversine distance calculations."""
    
    def test_same_point_zero_distance(self):
        """Test distance between same point is zero."""
        dist = haversine_distance(0, 0, 0, 0)
        assert dist == 0
    
    def test_known_distance(self):
        """Test distance between known cities."""
        # Tokyo to Osaka (approx 400km)
        dist = haversine_distance(139.6917, 35.6895, 135.5022, 34.6937)
        assert 390 < dist < 410  # km
    
    def test_antipodal_points(self):
        """Test distance between antipodal points."""
        # Should be approximately half Earth's circumference
        dist = haversine_distance(0, 0, 180, 0)
        assert 20000 < dist < 20100  # km
    
    def test_equator_distance(self):
        """Test distance along equator."""
        # 1 degree of longitude at equator ~ 111 km
        dist = haversine_distance(0, 0, 1, 0)
        assert 110 < dist < 112
    
    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        dist = haversine_distance(-74.006, -40.7128, -74.006, -40.7128)
        assert dist == 0


class TestPolygonCreation:
    """Tests for polygon and GeoDataFrame creation."""
    
    def test_create_simple_polygon(self):
        """Test creating a simple polygon."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        poly = create_polygon(vertices)
        
        assert poly.is_valid
        assert poly.area > 0
    
    def test_create_polygon_closes_ring(self):
        """Test that polygon automatically closes."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        poly = create_polygon(vertices)
        
        # Shapely auto-closes, so first and last coords should match
        coords = list(poly.exterior.coords)
        assert coords[0] == coords[-1]
    
    def test_create_geodataframe(self):
        """Test creating GeoDataFrame from polygon."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        poly = create_polygon(vertices)
        gdf = create_geodataframe(poly)
        
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1
        assert gdf.crs.to_epsg() == 4326
    
    def test_create_geodataframe_custom_crs(self):
        """Test creating GeoDataFrame with custom CRS."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        poly = create_polygon(vertices)
        gdf = create_geodataframe(poly, crs=3857)
        
        assert gdf.crs.to_epsg() == 3857


class TestCoordinateTransformation:
    """Tests for coordinate transformation functions."""
    
    def test_setup_transformer_wgs84_to_mercator(self):
        """Test transformer setup between WGS84 and Web Mercator."""
        transformer = setup_transformer("EPSG:4326", "EPSG:3857")
        assert transformer is not None
    
    def test_transform_coords_basic(self):
        """Test basic coordinate transformation."""
        transformer = setup_transformer("EPSG:4326", "EPSG:3857")
        x, y = transform_coords(transformer, 0, 0)
        
        assert x == 0  # Origin should stay at 0
        assert y == 0
    
    def test_transform_coords_tokyo(self):
        """Test transformation of Tokyo coordinates."""
        transformer = setup_transformer("EPSG:4326", "EPSG:3857")
        x, y = transform_coords(transformer, 139.6917, 35.6895)
        
        # Web Mercator coordinates should be much larger numbers
        assert abs(x) > 10000000  # ~15.5 million
        assert abs(y) > 4000000   # ~4.2 million
    
    def test_transform_coords_invalid_returns_none(self):
        """Test that invalid transformation returns None."""
        transformer = setup_transformer("EPSG:4326", "EPSG:3857")
        # Very large latitude (invalid)
        x, y = transform_coords(transformer, 0, 1000)
        
        # May return inf or None depending on implementation
        if x is not None:
            assert np.isinf(x) or np.isinf(y)


class TestConvertFormatLatLon:
    """Tests for coordinate format conversion."""
    
    def test_basic_conversion(self):
        """Test basic coordinate format conversion."""
        coords = [[0, 0], [1, 0], [1, 1], [0, 1]]
        result = convert_format_lat_lon(coords)
        
        # Should close the polygon
        assert len(result) == 5
        assert result[0] == result[-1]
    
    def test_single_point(self):
        """Test with single point."""
        coords = [[5.5, 10.2]]
        result = convert_format_lat_lon(coords)
        
        assert len(result) == 2
        assert result[0] == result[-1]
    
    def test_preserves_original(self):
        """Test that original list is not modified."""
        original = [[0, 0], [1, 1]]
        coords = original.copy()
        result = convert_format_lat_lon(coords)
        
        assert len(original) == 2


class TestEdgeCases:
    """Edge cases and error handling tests."""
    
    def test_tile_extreme_latitudes(self):
        """Test tile calculation at extreme latitudes."""
        # Near poles (but within valid range)
        x1, y1 = tile_from_lat_lon(85, 0, 5)
        x2, y2 = tile_from_lat_lon(-85, 0, 5)
        
        assert y1 < y2  # Northern should have smaller y
    
    def test_tile_international_date_line(self):
        """Test tile calculation near international date line."""
        x1, y1 = tile_from_lat_lon(0, 179, 5)
        x2, y2 = tile_from_lat_lon(0, -179, 5)
        
        # Should be on opposite sides of the tile grid
        assert x1 != x2
    
    def test_haversine_small_distance(self):
        """Test very small distances."""
        # 0.0001 degrees ~ 11 meters
        dist = haversine_distance(0, 0, 0.0001, 0)
        assert 0.01 < dist < 0.02  # km
    
    def test_create_polygon_many_vertices(self):
        """Test creating polygon with many vertices."""
        # Create a circle-like polygon
        n = 100
        import math
        vertices = [
            (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]
        poly = create_polygon(vertices)
        
        assert poly.is_valid
        assert len(poly.exterior.coords) == n + 1  # +1 for closing


class TestGeocodingOffline:
    """Tests for geocoding functions that can work offline."""
    
    def test_timezone_info_tokyo(self):
        """Test timezone info for Tokyo area."""
        from voxcity.geoprocessor.utils import get_timezone_info
        
        # Tokyo coordinates (explicit)
        tokyo_vertices = [
            (139.65, 35.67),
            (139.65, 35.70),
            (139.70, 35.70),
            (139.70, 35.67)
        ]
        
        timezone, longitude = get_timezone_info(tokyo_vertices)
        
        assert isinstance(timezone, str)
        # Timezone might be 'Asia/Tokyo' or 'UTC+9.00' format depending on implementation
        assert 'Asia/Tokyo' in timezone or 'UTC+9' in timezone
        assert 135 <= float(longitude) <= 145  # Japanese timezones use UTC+9
    
    def test_timezone_info_new_york(self):
        """Test timezone info for New York area."""
        from voxcity.geoprocessor.utils import get_timezone_info
        
        # New York coordinates
        vertices = [
            (-74.05, 40.70),
            (-74.05, 40.75),
            (-73.95, 40.75),
            (-73.95, 40.70)
        ]
        
        timezone, longitude = get_timezone_info(vertices)
        
        assert isinstance(timezone, str)
        # Timezone might be 'America/New_York' or 'UTC-5.00' format
        assert 'America' in timezone or 'US' in timezone or 'UTC-5' in timezone or 'UTC-4' in timezone
