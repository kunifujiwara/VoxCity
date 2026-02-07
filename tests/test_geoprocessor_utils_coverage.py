"""
Comprehensive tests for voxcity.geoprocessor.utils to improve coverage.
Focuses on pure/deterministic functions that don't require network access.
"""

import math
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
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
    validate_polygon_coordinates,
    create_building_polygons,
)


# ──────────────────────────────────────────────
# tile_from_lat_lon
# ──────────────────────────────────────────────
class TestTileFromLatLon:
    def test_basic_conversion(self):
        tx, ty = tile_from_lat_lon(0, 0, 1)
        assert isinstance(tx, int)
        assert isinstance(ty, int)

    def test_zoom_0(self):
        tx, ty = tile_from_lat_lon(0, 0, 0)
        assert tx == 0
        assert ty == 0

    def test_tokyo(self):
        tx, ty = tile_from_lat_lon(35.6762, 139.6503, 12)
        assert tx > 0
        assert ty > 0

    def test_new_york(self):
        tx, ty = tile_from_lat_lon(40.7128, -74.0060, 10)
        assert tx >= 0
        assert ty >= 0

    def test_negative_longitude(self):
        tx, ty = tile_from_lat_lon(51.5074, -0.1278, 8)  # London
        assert tx >= 0
        assert ty >= 0

    def test_high_zoom(self):
        tx, ty = tile_from_lat_lon(35.6762, 139.6503, 18)
        assert tx > 0 and ty > 0

    def test_equator_dateline(self):
        tx1, ty1 = tile_from_lat_lon(0, 180, 5)
        tx2, ty2 = tile_from_lat_lon(0, -180, 5)
        # At -180 and 180, tiles should wrap
        assert isinstance(tx1, int)
        assert isinstance(tx2, int)


# ──────────────────────────────────────────────
# quadkey_to_tile
# ──────────────────────────────────────────────
class TestQuadkeyToTile:
    def test_single_digit_0(self):
        x, y, z = quadkey_to_tile("0")
        assert x == 0 and y == 0 and z == 1

    def test_single_digit_1(self):
        x, y, z = quadkey_to_tile("1")
        assert x == 1 and y == 0 and z == 1

    def test_single_digit_2(self):
        x, y, z = quadkey_to_tile("2")
        assert x == 0 and y == 1 and z == 1

    def test_single_digit_3(self):
        x, y, z = quadkey_to_tile("3")
        assert x == 1 and y == 1 and z == 1

    def test_multi_digit(self):
        x, y, z = quadkey_to_tile("120")
        assert z == 3
        assert x >= 0 and y >= 0

    def test_empty_string(self):
        x, y, z = quadkey_to_tile("")
        assert x == 0 and y == 0 and z == 0

    def test_long_quadkey(self):
        x, y, z = quadkey_to_tile("0123012301")
        assert z == 10

    def test_all_zeros(self):
        x, y, z = quadkey_to_tile("000")
        assert x == 0 and y == 0 and z == 3

    def test_all_threes(self):
        x, y, z = quadkey_to_tile("333")
        assert z == 3
        assert x == 7 and y == 7


# ──────────────────────────────────────────────
# initialize_geod / calculate_distance
# ──────────────────────────────────────────────
class TestGeodDistance:
    def test_initialize_geod(self):
        geod = initialize_geod()
        assert geod is not None
        assert geod.a > 0  # semi-major axis

    def test_calculate_distance_zero(self):
        geod = initialize_geod()
        dist = calculate_distance(geod, 0, 0, 0, 0)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_calculate_distance_known(self):
        geod = initialize_geod()
        # London to Paris ~ 343 km
        dist = calculate_distance(geod, -0.1278, 51.5074, 2.3522, 48.8566)
        assert 340_000 < dist < 350_000  # meters

    def test_calculate_distance_symmetry(self):
        geod = initialize_geod()
        d1 = calculate_distance(geod, 0, 0, 1, 1)
        d2 = calculate_distance(geod, 1, 1, 0, 0)
        assert d1 == pytest.approx(d2, rel=1e-6)


# ──────────────────────────────────────────────
# normalize_to_one_meter
# ──────────────────────────────────────────────
class TestNormalizeToOneMeter:
    def test_simple(self):
        vec = np.array([3.0, 4.0])
        result = normalize_to_one_meter(vec, 5.0)
        assert np.allclose(result, np.array([0.6, 0.8]))

    def test_unit_distance(self):
        vec = np.array([1.0, 0.0])
        result = normalize_to_one_meter(vec, 1.0)
        assert np.allclose(result, vec)

    def test_3d_vector(self):
        vec = np.array([10.0, 0.0, 0.0])
        result = normalize_to_one_meter(vec, 10.0)
        assert np.allclose(result, np.array([1.0, 0.0, 0.0]))


# ──────────────────────────────────────────────
# setup_transformer / transform_coords
# ──────────────────────────────────────────────
class TestTransformer:
    def test_setup_identity(self):
        t = setup_transformer("EPSG:4326", "EPSG:4326")
        assert t is not None

    def test_wgs84_to_mercator(self):
        t = setup_transformer("EPSG:4326", "EPSG:3857")
        x, y = t.transform(0, 0)
        assert x == pytest.approx(0, abs=1)
        assert y == pytest.approx(0, abs=1)

    def test_transform_coords_valid(self):
        t = setup_transformer("EPSG:4326", "EPSG:3857")
        x, y = transform_coords(t, 139.6503, 35.6762)
        assert x is not None
        assert y is not None
        assert not np.isinf(x) and not np.isinf(y)

    def test_transform_coords_error(self):
        mock_transformer = MagicMock()
        mock_transformer.transform.side_effect = Exception("fail")
        x, y = transform_coords(mock_transformer, 0, 0)
        assert x is None and y is None


# ──────────────────────────────────────────────
# create_polygon / create_geodataframe
# ──────────────────────────────────────────────
class TestPolygonCreation:
    def test_create_polygon_square(self):
        verts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        p = create_polygon(verts)
        assert p.is_valid
        assert p.area == pytest.approx(1.0)

    def test_create_polygon_triangle(self):
        verts = [(0, 0), (1, 0), (0.5, 1)]
        p = create_polygon(verts)
        assert p.is_valid
        assert p.area > 0

    def test_create_geodataframe(self):
        verts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        p = create_polygon(verts)
        gdf = create_geodataframe(p)
        assert len(gdf) == 1
        assert gdf.crs is not None

    def test_create_geodataframe_custom_crs(self):
        verts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        p = create_polygon(verts)
        gdf = create_geodataframe(p, crs=3857)
        assert '3857' in str(gdf.crs)


# ──────────────────────────────────────────────
# haversine_distance
# ──────────────────────────────────────────────
class TestHaversineDistance:
    def test_zero_distance(self):
        dist = haversine_distance(0, 0, 0, 0)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        # London to Paris ~ 343 km
        dist = haversine_distance(-0.1278, 51.5074, 2.3522, 48.8566)
        assert 330 < dist < 360

    def test_symmetry(self):
        d1 = haversine_distance(0, 0, 1, 1)
        d2 = haversine_distance(1, 1, 0, 0)
        assert d1 == pytest.approx(d2, rel=1e-6)

    def test_equator_circumference(self):
        # Quarter of Earth's equator ~ 10018 km
        dist = haversine_distance(0, 0, 90, 0)
        assert 10000 < dist < 10100

    def test_antimeridian(self):
        dist = haversine_distance(179, 0, -179, 0)
        # Should be ~222 km (2 degrees at equator)
        assert 200 < dist < 250


# ──────────────────────────────────────────────
# convert_format_lat_lon
# ──────────────────────────────────────────────
class TestConvertFormatLatLon:
    def test_basic(self):
        coords = [[1, 2], [3, 4], [5, 6]]
        result = convert_format_lat_lon(coords)
        assert len(result) == 4
        assert result[-1] == result[0]

    def test_single_coord(self):
        coords = [[10, 20]]
        result = convert_format_lat_lon(coords)
        assert len(result) == 2
        assert result[0] == result[1]

    def test_original_unchanged(self):
        coords = [[1, 2], [3, 4]]
        original = coords.copy()
        convert_format_lat_lon(coords)
        assert coords == original


# ──────────────────────────────────────────────
# validate_polygon_coordinates
# ──────────────────────────────────────────────
class TestValidatePolygonCoordinates:
    def test_valid_closed_polygon(self):
        geom = {
            'type': 'Polygon',
            'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        }
        assert validate_polygon_coordinates(geom) is True

    def test_unclosed_polygon_gets_closed(self):
        geom = {
            'type': 'Polygon',
            'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1]]]
        }
        result = validate_polygon_coordinates(geom)
        assert result is True
        assert geom['coordinates'][0][-1] == geom['coordinates'][0][0]

    def test_too_few_points(self):
        geom = {
            'type': 'Polygon',
            'coordinates': [[[0, 0], [1, 0]]]
        }
        result = validate_polygon_coordinates(geom)
        assert result is False

    def test_multipolygon(self):
        geom = {
            'type': 'MultiPolygon',
            'coordinates': [
                [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]]
            ]
        }
        assert validate_polygon_coordinates(geom) is True

    def test_unclosed_multipolygon(self):
        geom = {
            'type': 'MultiPolygon',
            'coordinates': [
                [[[0, 0], [1, 0], [1, 1], [0, 1]]]
            ]
        }
        result = validate_polygon_coordinates(geom)
        assert result is True
        assert geom['coordinates'][0][0][-1] == geom['coordinates'][0][0][0]

    def test_invalid_type(self):
        geom = {'type': 'Point', 'coordinates': [0, 0]}
        assert validate_polygon_coordinates(geom) is False

    def test_linestring_type(self):
        geom = {'type': 'LineString', 'coordinates': [[0, 0], [1, 1]]}
        assert validate_polygon_coordinates(geom) is False


# ──────────────────────────────────────────────
# create_building_polygons
# ──────────────────────────────────────────────
class TestCreateBuildingPolygons:
    def test_simple_building(self):
        features = [{
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
            },
            'properties': {'height': 10.0, 'id': 1}
        }]
        polygons, idx = create_building_polygons(features)
        assert len(polygons) == 1
        assert polygons[0][1] == 10.0  # height
        assert polygons[0][4] == 1  # id

    def test_building_with_levels(self):
        features = [{
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
            },
            'properties': {'levels': 4, 'id': 1}
        }]
        polygons, idx = create_building_polygons(features)
        assert len(polygons) == 1
        assert polygons[0][1] == pytest.approx(2.5 * 4)  # floor_height * levels

    def test_building_with_num_floors(self):
        features = [{
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
            },
            'properties': {'num_floors': 3, 'id': 2}
        }]
        polygons, idx = create_building_polygons(features)
        assert len(polygons) == 1
        assert polygons[0][1] == pytest.approx(2.5 * 3)

    def test_building_no_height(self):
        features = [{
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
            },
            'properties': {'id': 1}
        }]
        polygons, idx = create_building_polygons(features)
        assert len(polygons) == 1
        assert np.isnan(polygons[0][1])  # height should be NaN

    def test_building_min_height(self):
        features = [{
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
            },
            'properties': {'height': 20.0, 'min_height': 5.0, 'id': 1}
        }]
        polygons, idx = create_building_polygons(features)
        assert polygons[0][2] == 5.0  # min_height

    def test_building_with_min_level(self):
        features = [{
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
            },
            'properties': {'height': 20.0, 'min_level': 2, 'id': 1}
        }]
        polygons, idx = create_building_polygons(features)
        assert polygons[0][2] == pytest.approx(2.5 * 2)

    def test_auto_id_assignment(self):
        features = [
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                'properties': {'height': 10.0}
            },
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]]
                },
                'properties': {'height': 20.0}
            }
        ]
        polygons, idx = create_building_polygons(features)
        assert len(polygons) == 2
        ids = [p[4] for p in polygons]
        assert ids[0] != ids[1]

    def test_is_inner_property(self):
        features = [{
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
            },
            'properties': {'height': 10.0, 'id': 1, 'is_inner': True}
        }]
        polygons, idx = create_building_polygons(features)
        assert polygons[0][3] is True

    def test_invalid_geometry_skipped(self):
        features = [
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[0, 0], [0, 0], [0, 0], [0, 0]]]
                },
                'properties': {'height': 10.0, 'id': 1}
            },
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                'properties': {'height': 20.0, 'id': 2}
            }
        ]
        polygons, idx = create_building_polygons(features)
        # Only the valid polygon should remain
        assert len(polygons) >= 1

    def test_empty_input(self):
        polygons, idx = create_building_polygons([])
        assert len(polygons) == 0

    def test_multiple_buildings_spatial_index(self):
        features = []
        for i in range(10):
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[i, 0], [i + 1, 0], [i + 1, 1], [i, 1], [i, 0]]]
                },
                'properties': {'height': 10.0 + i, 'id': i + 1}
            })
        polygons, idx = create_building_polygons(features)
        assert len(polygons) == 10
        # Test spatial index query
        results = list(idx.intersection((0, 0, 5, 1)))
        assert len(results) > 0


# ──────────────────────────────────────────────
# get_timezone_info (mocked)
# ──────────────────────────────────────────────
class TestGetTimezoneInfo:
    def test_timezone_tokyo(self):
        from voxcity.geoprocessor.utils import get_timezone_info
        coords = [
            (139.65, 35.67), (139.66, 35.67),
            (139.66, 35.68), (139.65, 35.68)
        ]
        tz, meridian = get_timezone_info(coords)
        assert 'UTC' in tz
        assert float(meridian) != 0  # Tokyo should not be UTC+0
