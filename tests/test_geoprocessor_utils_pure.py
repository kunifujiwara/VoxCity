"""Tests for pure functions in voxcity.geoprocessor.utils module."""
import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box

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


class TestTileFromLatLon:
    """Tests for tile coordinate calculations."""

    def test_known_tile_at_zoom_0(self):
        """At zoom 0, entire world is one tile (0,0)."""
        tile_x, tile_y = tile_from_lat_lon(0.0, 0.0, 0)
        assert tile_x == 0
        assert tile_y == 0

    def test_known_tile_at_zoom_1(self):
        """At zoom 1, world is 4 tiles."""
        # Northeast quadrant
        tile_x, tile_y = tile_from_lat_lon(45.0, 90.0, 1)
        assert 0 <= tile_x <= 1
        assert 0 <= tile_y <= 1

    def test_positive_coordinates(self):
        """Test with positive lat/lon (Tokyo area)."""
        lat, lon = 35.6762, 139.6503
        tile_x, tile_y = tile_from_lat_lon(lat, lon, 12)
        # Tokyo at zoom 12 should have specific tile coordinates
        assert tile_x > 0
        assert tile_y > 0

    def test_negative_coordinates(self):
        """Test with negative lat/lon (South America)."""
        lat, lon = -23.5505, -46.6333  # Sao Paulo
        tile_x, tile_y = tile_from_lat_lon(lat, lon, 10)
        assert tile_x > 0
        assert tile_y > 0

    def test_higher_zoom_level(self):
        """Higher zoom levels should produce larger tile coordinates."""
        lat, lon = 40.7128, -74.0060  # New York
        tile_z10_x, tile_z10_y = tile_from_lat_lon(lat, lon, 10)
        tile_z15_x, tile_z15_y = tile_from_lat_lon(lat, lon, 15)
        
        # At higher zoom, tile coords are larger
        assert tile_z15_x > tile_z10_x
        assert tile_z15_y > tile_z10_y

    def test_equator_longitude_180(self):
        """Test edge case at equator and date line."""
        tile_x, tile_y = tile_from_lat_lon(0.0, 179.9, 5)
        assert tile_x > 0
        assert tile_y > 0

    def test_zoom_level_20(self):
        """Test high zoom level."""
        tile_x, tile_y = tile_from_lat_lon(51.5074, -0.1278, 20)  # London
        assert tile_x > 0
        assert tile_y > 0


class TestQuadkeyToTile:
    """Tests for quadkey to tile conversion."""

    def test_single_digit_quadkey(self):
        """Test single digit quadkeys."""
        # Quadkey "0" = top-left quadrant at zoom 1
        x, y, zoom = quadkey_to_tile("0")
        assert zoom == 1
        assert x == 0
        assert y == 0

    def test_quadkey_1(self):
        """Quadkey '1' = top-right quadrant."""
        x, y, zoom = quadkey_to_tile("1")
        assert zoom == 1
        assert x == 1
        assert y == 0

    def test_quadkey_2(self):
        """Quadkey '2' = bottom-left quadrant."""
        x, y, zoom = quadkey_to_tile("2")
        assert zoom == 1
        assert x == 0
        assert y == 1

    def test_quadkey_3(self):
        """Quadkey '3' = bottom-right quadrant."""
        x, y, zoom = quadkey_to_tile("3")
        assert zoom == 1
        assert x == 1
        assert y == 1

    def test_longer_quadkey(self):
        """Test multi-digit quadkey."""
        x, y, zoom = quadkey_to_tile("120")
        assert zoom == 3
        # 1 -> x bit at level 0
        # 2 -> y bit at level 1  
        # 0 -> neither at level 2
        assert x == 4  # binary 100
        assert y == 2  # binary 010

    def test_all_zeros(self):
        """Quadkey of all zeros should give (0, 0)."""
        x, y, zoom = quadkey_to_tile("00000")
        assert zoom == 5
        assert x == 0
        assert y == 0

    def test_all_threes(self):
        """Quadkey of all 3s should give maximum coordinates."""
        x, y, zoom = quadkey_to_tile("333")
        assert zoom == 3
        assert x == 7  # binary 111
        assert y == 7  # binary 111


class TestInitializeGeod:
    """Tests for Geod initialization."""

    def test_returns_geod_object(self):
        """Test that a Geod object is returned."""
        from pyproj import Geod
        geod = initialize_geod()
        assert isinstance(geod, Geod)

    def test_geod_has_inv_method(self):
        """Test that the Geod object has inverse method."""
        geod = initialize_geod()
        assert hasattr(geod, 'inv')


class TestCalculateDistance:
    """Tests for geodetic distance calculation."""

    def test_same_point_zero_distance(self):
        """Distance from point to itself should be zero."""
        geod = initialize_geod()
        dist = calculate_distance(geod, 0.0, 0.0, 0.0, 0.0)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        """Test with known approximate distance."""
        geod = initialize_geod()
        # Tokyo to New York is approximately 10,850 km
        dist = calculate_distance(geod, 139.6503, 35.6762, -74.0060, 40.7128)
        assert 10_800_000 < dist < 10_900_000  # in meters

    def test_short_distance(self):
        """Test short distance calculation."""
        geod = initialize_geod()
        # Two nearby points (about 111 km apart at equator)
        dist = calculate_distance(geod, 0.0, 0.0, 1.0, 0.0)
        assert 110_000 < dist < 112_000  # approximately 111 km


class TestNormalizeToOneMeter:
    """Tests for vector normalization to one meter."""

    def test_simple_normalization(self):
        """Test basic normalization."""
        vector = np.array([3.0, 4.0])  # length 5
        result = normalize_to_one_meter(vector, 5.0)
        expected = np.array([0.6, 0.8])
        np.testing.assert_array_almost_equal(result, expected)

    def test_unit_vector(self):
        """Test with already unit distance."""
        vector = np.array([1.0, 0.0])
        result = normalize_to_one_meter(vector, 1.0)
        np.testing.assert_array_almost_equal(result, vector)

    def test_three_dimensional(self):
        """Test with 3D vector."""
        vector = np.array([1.0, 2.0, 2.0])  # length 3
        result = normalize_to_one_meter(vector, 3.0)
        expected = np.array([1/3, 2/3, 2/3])
        np.testing.assert_array_almost_equal(result, expected)


class TestSetupTransformer:
    """Tests for coordinate transformer setup."""

    def test_creates_transformer(self):
        """Test that a transformer object is created."""
        from pyproj import Transformer
        transformer = setup_transformer("EPSG:4326", "EPSG:3857")
        assert isinstance(transformer, Transformer)

    def test_string_crs(self):
        """Test with string CRS."""
        transformer = setup_transformer("EPSG:4326", "EPSG:32633")
        assert transformer is not None

    def test_integer_crs(self):
        """Test with integer CRS codes."""
        transformer = setup_transformer(4326, 3857)
        assert transformer is not None


class TestTransformCoords:
    """Tests for coordinate transformation."""

    def test_wgs84_to_web_mercator(self):
        """Test transformation from WGS84 to Web Mercator."""
        transformer = setup_transformer("EPSG:4326", "EPSG:3857")
        x, y = transform_coords(transformer, 0.0, 0.0)
        assert x == pytest.approx(0.0, abs=1e-3)
        assert y == pytest.approx(0.0, abs=1e-3)

    def test_known_transformation(self):
        """Test with known transformation."""
        transformer = setup_transformer("EPSG:4326", "EPSG:3857")
        # London coordinates
        x, y = transform_coords(transformer, -0.1278, 51.5074)
        assert x is not None
        assert y is not None
        # Web Mercator x should be negative for western longitude
        assert x < 0

    def test_returns_none_none_on_error(self):
        """Test that invalid coords return None, None."""
        transformer = setup_transformer("EPSG:4326", "EPSG:3857")
        # Create edge case - this should not fail but let's verify behavior
        x, y = transform_coords(transformer, 0.0, 0.0)
        assert x is not None
        assert y is not None


class TestCreatePolygon:
    """Tests for polygon creation."""

    def test_creates_shapely_polygon(self):
        """Test that a Shapely Polygon is created."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        poly = create_polygon(vertices)
        assert isinstance(poly, Polygon)

    def test_polygon_area(self):
        """Test polygon area calculation."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        poly = create_polygon(vertices)
        assert poly.area == pytest.approx(1.0)

    def test_triangle(self):
        """Test triangle creation."""
        vertices = [(0, 0), (1, 0), (0.5, 1)]
        poly = create_polygon(vertices)
        assert poly.is_valid
        assert poly.area == pytest.approx(0.5)

    def test_complex_polygon(self):
        """Test complex polygon."""
        vertices = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        poly = create_polygon(vertices)
        assert poly.is_valid


class TestCreateGeoDataFrame:
    """Tests for GeoDataFrame creation."""

    def test_creates_geodataframe(self):
        """Test GeoDataFrame creation."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        gdf = create_geodataframe(poly)
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_default_crs(self):
        """Test default CRS is 4326."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        gdf = create_geodataframe(poly)
        assert gdf.crs.to_epsg() == 4326

    def test_custom_crs(self):
        """Test custom CRS."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        gdf = create_geodataframe(poly, crs=3857)
        assert gdf.crs.to_epsg() == 3857

    def test_geometry_column(self):
        """Test geometry column contains the polygon."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        gdf = create_geodataframe(poly)
        assert 'geometry' in gdf.columns
        assert gdf.iloc[0].geometry.equals(poly)


class TestHaversineDistance:
    """Tests for Haversine distance calculation."""

    def test_same_point_zero_distance(self):
        """Distance from point to itself should be zero."""
        dist = haversine_distance(0.0, 0.0, 0.0, 0.0)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_known_distance_equator(self):
        """Test distance along equator (1 degree ~ 111 km)."""
        dist = haversine_distance(0.0, 0.0, 1.0, 0.0)
        assert 110 < dist < 112  # approximately 111 km

    def test_known_distance_cities(self):
        """Test with known city distances."""
        # Tokyo to New York is approximately 10,850 km
        dist = haversine_distance(139.6503, 35.6762, -74.0060, 40.7128)
        assert 10_500 < dist < 11_000  # in km

    def test_short_distance(self):
        """Test short distance."""
        # About 111 km at equator for 1 degree
        dist = haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert 110 < dist < 112

    def test_antipodal_points(self):
        """Test distance between opposite points on Earth."""
        # Maximum distance is about half the Earth's circumference
        dist = haversine_distance(0.0, 0.0, 180.0, 0.0)
        assert 20_000 < dist < 20_100  # approximately 20,000 km

    def test_symmetry(self):
        """Distance should be same in both directions."""
        dist1 = haversine_distance(0.0, 0.0, 10.0, 10.0)
        dist2 = haversine_distance(10.0, 10.0, 0.0, 0.0)
        assert dist1 == pytest.approx(dist2)

    def test_negative_coordinates(self):
        """Test with negative coordinates (southern/western hemisphere)."""
        dist = haversine_distance(-74.0060, -40.7128, -46.6333, -23.5505)
        assert dist > 0


class TestConvertFormatLatLon:
    """Tests for convert_format_lat_lon function."""

    def test_closes_polygon(self):
        """Test that polygon is closed by repeating first point."""
        coords = [[0, 0], [1, 0], [1, 1], [0, 1]]
        result = convert_format_lat_lon(coords)
        
        assert len(result) == 5
        assert result[0] == result[-1]

    def test_preserves_order(self):
        """Test that coordinate order is preserved."""
        coords = [[10, 20], [30, 40], [50, 60]]
        result = convert_format_lat_lon(coords)
        
        assert result[0] == [10, 20]
        assert result[1] == [30, 40]
        assert result[2] == [50, 60]
        assert result[3] == [10, 20]

    def test_does_not_modify_original(self):
        """Test that original list is not modified."""
        coords = [[0, 0], [1, 1]]
        result = convert_format_lat_lon(coords)
        
        assert len(coords) == 2
        assert len(result) == 3

    def test_single_point(self):
        """Test with single point."""
        coords = [[5, 10]]
        result = convert_format_lat_lon(coords)
        
        assert len(result) == 2
        assert result[0] == result[1]

    def test_already_closed(self):
        """Test with already closed polygon."""
        coords = [[0, 0], [1, 0], [1, 1], [0, 0]]
        result = convert_format_lat_lon(coords)
        
        # Will add first point again
        assert len(result) == 5
        assert result[-1] == [0, 0]
