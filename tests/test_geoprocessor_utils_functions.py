"""Extended tests for voxcity.geoprocessor.utils module - targeting uncovered functions."""
import pytest
import numpy as np
import os
import tempfile
from shapely.geometry import Polygon, box, Point
import geopandas as gpd

from voxcity.geoprocessor.utils import (
    tile_from_lat_lon,
    quadkey_to_tile,
    initialize_geod,
    calculate_distance,
    normalize_to_one_meter,
    setup_transformer,
    transform_coords,
    create_polygon,
    create_geodataframe,
    haversine_distance,
    convert_format_lat_lon,
)


class TestTileCoordinates:
    """Tests for tile coordinate functions."""
    
    def test_tile_from_lat_lon_basic(self):
        """Test basic tile coordinate calculation."""
        # Tokyo coordinates
        tile_x, tile_y = tile_from_lat_lon(35.6762, 139.6503, 12)
        
        assert isinstance(tile_x, int)
        assert isinstance(tile_y, int)
        assert tile_x >= 0
        assert tile_y >= 0
    
    def test_tile_from_lat_lon_different_zooms(self):
        """Test tile coordinates at different zoom levels."""
        lat, lon = 40.7128, -74.0060  # New York
        
        # Higher zoom = more tiles
        tile_x_low, tile_y_low = tile_from_lat_lon(lat, lon, 5)
        tile_x_high, tile_y_high = tile_from_lat_lon(lat, lon, 15)
        
        # At higher zoom, tile numbers should be larger
        assert tile_x_high > tile_x_low
        assert tile_y_high > tile_y_low
    
    def test_tile_from_lat_lon_equator_prime_meridian(self):
        """Test tile at equator and prime meridian."""
        tile_x, tile_y = tile_from_lat_lon(0, 0, 10)
        
        # At equator and prime meridian, tiles should be near center
        assert tile_x >= 0
        assert tile_y >= 0
    
    def test_tile_from_lat_lon_extreme_coordinates(self):
        """Test tile calculation at extreme coordinates."""
        # Near south pole (limited by Web Mercator)
        tile_x, tile_y = tile_from_lat_lon(-85, 0, 8)
        assert tile_y >= 0
        
        # Far east
        tile_x, tile_y = tile_from_lat_lon(0, 179, 8)
        assert tile_x >= 0


class TestQuadkeyToTile:
    """Tests for quadkey conversion."""
    
    def test_quadkey_to_tile_basic(self):
        """Test basic quadkey conversion."""
        tile_x, tile_y, lod = quadkey_to_tile("0")
        
        assert tile_x == 0
        assert tile_y == 0
        assert lod == 1
    
    def test_quadkey_to_tile_zoom_level(self):
        """Test that quadkey length equals zoom level."""
        quadkey = "12021"
        tile_x, tile_y, lod = quadkey_to_tile(quadkey)
        
        assert lod == len(quadkey)
        assert lod == 5
    
    def test_quadkey_to_tile_digit_1(self):
        """Test quadkey with digit 1 (x bit set)."""
        tile_x, tile_y, _ = quadkey_to_tile("1")
        
        assert tile_x == 1
        assert tile_y == 0
    
    def test_quadkey_to_tile_digit_2(self):
        """Test quadkey with digit 2 (y bit set)."""
        tile_x, tile_y, _ = quadkey_to_tile("2")
        
        assert tile_x == 0
        assert tile_y == 1
    
    def test_quadkey_to_tile_digit_3(self):
        """Test quadkey with digit 3 (both bits set)."""
        tile_x, tile_y, _ = quadkey_to_tile("3")
        
        assert tile_x == 1
        assert tile_y == 1
    
    def test_quadkey_to_tile_complex(self):
        """Test complex quadkey."""
        tile_x, tile_y, lod = quadkey_to_tile("123")
        
        assert lod == 3
        assert tile_x >= 0
        assert tile_y >= 0


class TestGeodAndDistance:
    """Tests for geodetic operations."""
    
    def test_initialize_geod(self):
        """Test Geod initialization."""
        geod = initialize_geod()
        
        # Should be WGS84 ellipsoid
        assert geod is not None
        assert hasattr(geod, 'a')  # Semi-major axis
        assert geod.a > 6000000  # Earth's radius in meters
    
    def test_calculate_distance_same_point(self):
        """Test distance calculation for same point."""
        geod = initialize_geod()
        dist = calculate_distance(geod, 0, 0, 0, 0)
        
        assert dist == 0.0
    
    def test_calculate_distance_known_values(self):
        """Test distance calculation with known values."""
        geod = initialize_geod()
        
        # Distance from Tokyo to Osaka (approximately 400 km)
        dist = calculate_distance(geod, 139.6917, 35.6895, 135.5022, 34.6937)
        
        # Should be approximately 400 km (with some tolerance)
        assert 350000 < dist < 450000
    
    def test_calculate_distance_short(self):
        """Test short distance calculation."""
        geod = initialize_geod()
        
        # Two nearby points in Tokyo
        dist = calculate_distance(geod, 139.7564, 35.6713, 139.7565, 35.6714)
        
        # Should be a small distance (tens of meters)
        assert 0 < dist < 1000
    
    def test_haversine_distance_basic(self):
        """Test haversine distance calculation."""
        # Same as geodetic test but using haversine
        dist_km = haversine_distance(139.6917, 35.6895, 135.5022, 34.6937)
        
        # Convert to meters for comparison
        dist_m = dist_km * 1000
        
        # Should be similar to geodetic distance (with some tolerance)
        assert 350000 < dist_m < 450000
    
    def test_haversine_distance_same_point(self):
        """Test haversine distance for same point."""
        dist = haversine_distance(0, 0, 0, 0)
        
        assert dist == 0.0


class TestNormalization:
    """Tests for vector normalization."""
    
    def test_normalize_to_one_meter_basic(self):
        """Test basic normalization."""
        vector = np.array([3.0, 4.0])  # Length 5
        distance = 5.0
        
        result = normalize_to_one_meter(vector, distance)
        
        # Result should have magnitude of 1
        assert np.isclose(np.linalg.norm(result), 1.0)
    
    def test_normalize_to_one_meter_direction_preserved(self):
        """Test that direction is preserved."""
        vector = np.array([10.0, 0.0])
        distance = 10.0
        
        result = normalize_to_one_meter(vector, distance)
        
        # Should point in same direction (positive x)
        assert result[0] > 0
        assert np.isclose(result[1], 0)
    
    def test_normalize_to_one_meter_large_distance(self):
        """Test normalization with large distance."""
        vector = np.array([1.0, 1.0])
        distance = 1000.0
        
        result = normalize_to_one_meter(vector, distance)
        
        # Magnitude should be very small
        assert np.linalg.norm(result) < 0.01


class TestCoordinateTransformation:
    """Tests for coordinate transformation functions."""
    
    def test_setup_transformer_wgs84_to_webmercator(self):
        """Test transformer setup for common CRS conversion."""
        transformer = setup_transformer("EPSG:4326", "EPSG:3857")
        
        assert transformer is not None
    
    def test_transform_coords_basic(self):
        """Test basic coordinate transformation."""
        transformer = setup_transformer("EPSG:4326", "EPSG:3857")
        
        # Transform Tokyo coordinates
        x, y = transform_coords(transformer, 139.7564, 35.6713)
        
        assert x is not None
        assert y is not None
        # Web Mercator coordinates should be much larger
        assert abs(x) > 1000000
    
    def test_transform_coords_origin(self):
        """Test transformation at origin."""
        transformer = setup_transformer("EPSG:4326", "EPSG:3857")
        
        x, y = transform_coords(transformer, 0, 0)
        
        # Origin should transform to near (0, 0)
        assert x is not None
        assert y is not None
        assert abs(x) < 1  # Very close to 0
        assert abs(y) < 1
    
    def test_transform_coords_round_trip(self):
        """Test round-trip transformation."""
        transformer_to = setup_transformer("EPSG:4326", "EPSG:3857")
        transformer_from = setup_transformer("EPSG:3857", "EPSG:4326")
        
        original_lon, original_lat = 139.7564, 35.6713
        
        # Transform to Web Mercator and back
        x, y = transform_coords(transformer_to, original_lon, original_lat)
        lon_back, lat_back = transform_coords(transformer_from, x, y)
        
        assert np.isclose(original_lon, lon_back, atol=0.0001)
        assert np.isclose(original_lat, lat_back, atol=0.0001)


class TestPolygonCreation:
    """Tests for polygon creation functions."""
    
    def test_create_polygon_square(self):
        """Test creating a square polygon."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        polygon = create_polygon(vertices)
        
        assert polygon.is_valid
        assert polygon.area > 0
    
    def test_create_polygon_triangle(self):
        """Test creating a triangular polygon."""
        vertices = [(0, 0), (1, 0), (0.5, 1)]
        polygon = create_polygon(vertices)
        
        assert polygon.is_valid
        assert polygon.area > 0
    
    def test_create_polygon_complex(self):
        """Test creating a complex polygon."""
        vertices = [
            (0, 0), (2, 0), (2, 1), (1, 1),
            (1, 2), (2, 2), (2, 3), (0, 3)
        ]
        polygon = create_polygon(vertices)
        
        assert polygon.is_valid
    
    def test_create_geodataframe(self):
        """Test creating GeoDataFrame from polygon."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        polygon = create_polygon(vertices)
        gdf = create_geodataframe(polygon)
        
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1
        assert gdf.crs is not None
    
    def test_create_geodataframe_custom_crs(self):
        """Test creating GeoDataFrame with custom CRS."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        polygon = create_polygon(vertices)
        gdf = create_geodataframe(polygon, crs=3857)
        
        assert gdf.crs is not None


class TestConvertFormatLatLon:
    """Tests for coordinate format conversion."""
    
    def test_convert_format_closes_polygon(self):
        """Test that function closes polygon."""
        input_coords = [[0, 0], [1, 0], [1, 1], [0, 1]]
        result = convert_format_lat_lon(input_coords)
        
        # First and last should be equal
        assert result[0] == result[-1]
        
        # Should have one more element
        assert len(result) == len(input_coords) + 1
    
    def test_convert_format_already_closed(self):
        """Test with already closed polygon."""
        input_coords = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
        result = convert_format_lat_lon(input_coords)
        
        # Should have first point repeated again
        assert len(result) == len(input_coords) + 1
        assert result[0] == result[-1]
    
    def test_convert_format_single_point(self):
        """Test with single point."""
        input_coords = [[5, 10]]
        result = convert_format_lat_lon(input_coords)
        
        assert len(result) == 2
        assert result[0] == result[1]
    
    def test_convert_format_preserves_values(self):
        """Test that coordinate values are preserved."""
        input_coords = [[139.7564, 35.6713], [139.7619, 35.6758]]
        result = convert_format_lat_lon(input_coords)
        
        # Original values should be preserved
        assert result[0] == [139.7564, 35.6713]
        assert result[1] == [139.7619, 35.6758]


class TestEdgeCases:
    """Edge case tests."""
    
    def test_tile_from_lat_lon_boundary_longitude(self):
        """Test tile at dateline boundary."""
        # At 180 degrees (dateline)
        tile_x, tile_y = tile_from_lat_lon(0, 180, 10)
        assert tile_x >= 0
        
        # At -180 degrees
        tile_x, tile_y = tile_from_lat_lon(0, -180, 10)
        assert tile_x >= 0
    
    def test_calculate_distance_antipodal(self):
        """Test distance between antipodal points."""
        geod = initialize_geod()
        
        # Points on opposite sides of Earth
        dist = calculate_distance(geod, 0, 0, 180, 0)
        
        # Should be approximately half Earth's circumference (~20000 km)
        assert 19000000 < dist < 21000000
    
    def test_normalize_very_small_distance(self):
        """Test normalization with very small distance."""
        vector = np.array([1e-10, 1e-10])
        distance = 1e-10
        
        # Should not crash
        result = normalize_to_one_meter(vector, distance)
        assert result is not None
    
    def test_empty_quadkey(self):
        """Test empty quadkey."""
        tile_x, tile_y, lod = quadkey_to_tile("")
        
        assert lod == 0
        assert tile_x == 0
        assert tile_y == 0
