"""Extended tests for voxcity.geoprocessor.utils module."""
import pytest
import numpy as np
from shapely.geometry import Polygon, box

from voxcity.geoprocessor.utils import (
    tile_from_lat_lon,
    quadkey_to_tile,
    initialize_geod,
    calculate_distance,
    normalize_to_one_meter,
    setup_transformer,
    create_polygon,
    create_geodataframe,
    haversine_distance,
    get_timezone_info,
    validate_polygon_coordinates,
    convert_format_lat_lon,
)


class TestTileFromLatLon:
    def test_tokyo_at_zoom_12(self):
        # Tokyo coordinates
        tile_x, tile_y = tile_from_lat_lon(35.6762, 139.6503, 12)
        assert tile_x > 0
        assert tile_y > 0
        assert isinstance(tile_x, int)
        assert isinstance(tile_y, int)

    def test_equator_prime_meridian(self):
        # (0, 0) should give known tile
        tile_x, tile_y = tile_from_lat_lon(0, 0, 1)
        assert tile_x == 1
        assert tile_y == 1

    def test_zoom_level_affects_tile_count(self):
        # Higher zoom should give larger tile coordinates
        x1, y1 = tile_from_lat_lon(35.6762, 139.6503, 10)
        x2, y2 = tile_from_lat_lon(35.6762, 139.6503, 15)
        assert x2 > x1
        assert y2 > y1


class TestQuadkeyToTile:
    def test_simple_quadkey(self):
        x, y, zoom = quadkey_to_tile("0")
        assert zoom == 1
        assert x == 0
        assert y == 0

    def test_quadkey_1(self):
        x, y, zoom = quadkey_to_tile("1")
        assert zoom == 1
        assert x == 1
        assert y == 0

    def test_quadkey_2(self):
        x, y, zoom = quadkey_to_tile("2")
        assert zoom == 1
        assert x == 0
        assert y == 1

    def test_quadkey_3(self):
        x, y, zoom = quadkey_to_tile("3")
        assert zoom == 1
        assert x == 1
        assert y == 1

    def test_longer_quadkey(self):
        x, y, zoom = quadkey_to_tile("120")
        assert zoom == 3


class TestInitializeGeod:
    def test_returns_geod_object(self):
        geod = initialize_geod()
        assert geod is not None
        assert hasattr(geod, 'inv')
        assert hasattr(geod, 'a')  # semi-major axis

    def test_wgs84_ellipsoid(self):
        geod = initialize_geod()
        # WGS84 semi-major axis is approximately 6378137 meters
        assert abs(geod.a - 6378137) < 1


class TestCalculateDistance:
    def test_same_point_zero_distance(self):
        geod = initialize_geod()
        dist = calculate_distance(geod, 139.65, 35.67, 139.65, 35.67)
        assert dist == pytest.approx(0.0, abs=0.01)

    def test_tokyo_to_osaka(self):
        geod = initialize_geod()
        # Approximate coordinates
        dist = calculate_distance(geod, 139.6917, 35.6895, 135.5022, 34.6937)
        # Distance is approximately 400 km
        assert 350000 < dist < 450000

    def test_returns_float(self):
        geod = initialize_geod()
        dist = calculate_distance(geod, 0, 0, 1, 1)
        assert isinstance(dist, float)


class TestNormalizeToOneMeter:
    def test_unit_vector_5_meters(self):
        vector = np.array([3.0, 4.0])  # length 5
        result = normalize_to_one_meter(vector, 5.0)
        # Should scale to 1/5 of original
        expected = vector / 5.0
        assert np.allclose(result, expected)

    def test_preserves_direction(self):
        vector = np.array([1.0, 0.0])
        result = normalize_to_one_meter(vector, 10.0)
        # Direction should be same (positive x)
        assert result[0] > 0
        assert result[1] == 0


class TestSetupTransformer:
    def test_wgs84_to_mercator(self):
        transformer = setup_transformer("EPSG:4326", "EPSG:3857")
        assert transformer is not None

    def test_transformer_can_transform(self):
        transformer = setup_transformer("EPSG:4326", "EPSG:3857")
        x, y = transformer.transform(139.6917, 35.6895)  # Tokyo
        assert x != 139.6917  # Should be different in Mercator
        assert y != 35.6895


class TestCreatePolygon:
    def test_square_polygon(self):
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        polygon = create_polygon(vertices)
        assert isinstance(polygon, Polygon)
        assert polygon.is_valid

    def test_polygon_area(self):
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        polygon = create_polygon(vertices)
        assert polygon.area == pytest.approx(1.0)


class TestCreateGeoDataFrame:
    def test_creates_gdf(self):
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        polygon = create_polygon(vertices)
        gdf = create_geodataframe(polygon)
        assert gdf is not None
        assert len(gdf) == 1

    def test_default_crs_wgs84(self):
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        polygon = create_polygon(vertices)
        gdf = create_geodataframe(polygon)
        assert gdf.crs is not None
        # CRS should be EPSG:4326
        assert gdf.crs.to_epsg() == 4326

    def test_custom_crs(self):
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        polygon = create_polygon(vertices)
        gdf = create_geodataframe(polygon, crs=3857)
        assert gdf.crs.to_epsg() == 3857


class TestHaversineDistance:
    def test_same_point_zero_distance(self):
        dist = haversine_distance(139.65, 35.67, 139.65, 35.67)
        assert dist == pytest.approx(0.0, abs=0.001)

    def test_equator_one_degree_longitude(self):
        # At equator, 1 degree longitude ≈ 111 km
        dist = haversine_distance(0, 0, 1, 0)
        assert 110 < dist < 112

    def test_returns_kilometers(self):
        # Tokyo to Osaka is about 400 km
        dist = haversine_distance(139.6917, 35.6895, 135.5022, 34.6937)
        assert 350 < dist < 450


class TestGetTimezoneInfo:
    def test_tokyo_timezone(self):
        coords = [
            (139.7564, 35.6713),
            (139.7564, 35.6758),
            (139.7619, 35.6758),
            (139.7619, 35.6713),
        ]
        tz, meridian = get_timezone_info(coords)
        assert "UTC" in tz
        # Tokyo is UTC+9
        assert "+9" in tz or "+09" in tz

    def test_returns_tuple(self):
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = get_timezone_info(coords)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestValidatePolygonCoordinates:
    def test_valid_polygon(self):
        geometry = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        }
        # Should not raise
        validate_polygon_coordinates(geometry)

    def test_closes_unclosed_polygon(self):
        geometry = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]  # not closed
        }
        validate_polygon_coordinates(geometry)
        # Should now be closed
        ring = geometry["coordinates"][0]
        assert ring[0] == ring[-1]


class TestConvertFormatLatLon:
    def test_closes_polygon(self):
        coords = [[0, 0], [1, 0], [1, 1], [0, 1]]
        result = convert_format_lat_lon(coords)
        assert len(result) == 5
        assert result[0] == result[-1]

    def test_preserves_coordinates(self):
        coords = [[139.65, 35.67], [139.66, 35.68]]
        result = convert_format_lat_lon(coords)
        assert result[0] == [139.65, 35.67]
        assert result[1] == [139.66, 35.68]
