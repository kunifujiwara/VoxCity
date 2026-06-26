"""Tests for voxcity.downloader.gsi module."""
import math
import numpy as np
import pytest

from voxcity.downloader.gsi import (
    latlon_to_tile,
    tile_bounds_mercator,
    _bbox_from_rectangle_vertices,
    tile_range_for_bbox,
    _MERC_MAX,
)
from voxcity.downloader.gsi import parse_dem_tile_text, GSI_NODATA


class TestLatlonToTile:
    def test_known_tile_zoom15(self):
        # Tsukuba area center
        x, y = latlon_to_tile(36.225, 140.105, 15)
        assert isinstance(x, int) and isinstance(y, int)
        # Sanity: lon 140.105 at z15 -> x within world range
        assert 0 <= x < 2 ** 15
        assert 0 <= y < 2 ** 15

    def test_origin_corner(self):
        # lon -180, lat ~85.05 maps to tile (0, 0)
        x, y = latlon_to_tile(85.0511, -180.0, 0)
        assert (x, y) == (0, 0)


class TestTileBoundsMercator:
    def test_full_world_at_zoom0(self):
        minx, miny, maxx, maxy = tile_bounds_mercator(0, 0, 0)
        assert minx == pytest.approx(-_MERC_MAX)
        assert maxy == pytest.approx(_MERC_MAX)
        assert maxx == pytest.approx(_MERC_MAX)
        assert miny == pytest.approx(-_MERC_MAX)

    def test_pixel_extent_zoom15(self):
        minx, miny, maxx, maxy = tile_bounds_mercator(100, 200, 15)
        tile = (2 * _MERC_MAX) / (2 ** 15)
        assert (maxx - minx) == pytest.approx(tile)
        assert (maxy - miny) == pytest.approx(tile)


class TestBboxAndRange:
    def test_bbox(self):
        verts = [(140.09, 36.21), (140.12, 36.21), (140.12, 36.24), (140.09, 36.24)]
        assert _bbox_from_rectangle_vertices(verts) == (140.09, 36.21, 140.12, 36.24)

    def test_tile_range_orders_min_max(self):
        bbox = (140.09, 36.21, 140.12, 36.24)
        x_min, y_min, x_max, y_max = tile_range_for_bbox(bbox, 15)
        assert x_min <= x_max
        assert y_min <= y_max


class TestParseDemTileText:
    def test_full_grid(self):
        # 256 rows x 256 cols of "1.5"
        line = ",".join(["1.5"] * 256)
        text = "\n".join([line] * 256)
        arr = parse_dem_tile_text(text)
        assert arr.shape == (256, 256)
        assert arr.dtype == np.float32
        assert np.allclose(arr, 1.5)

    def test_nodata_token(self):
        line = ",".join(["e"] * 256)
        text = "\n".join([line] * 256)
        arr = parse_dem_tile_text(text, nodata=-9999.0)
        assert np.allclose(arr, -9999.0)

    def test_mixed_and_ragged(self):
        # First cell real, rest no-data; short rows; missing rows -> nodata
        text = "12.25,e,e\ne,3.0"
        arr = parse_dem_tile_text(text, nodata=-1.0)
        assert arr.shape == (256, 256)
        assert arr[0, 0] == pytest.approx(12.25)
        assert arr[0, 1] == pytest.approx(-1.0)
        assert arr[1, 0] == pytest.approx(-1.0)
        assert arr[1, 1] == pytest.approx(3.0)
        # Untouched cell stays nodata
        assert arr[5, 5] == pytest.approx(-1.0)

    def test_malformed_token_defaults_to_nodata(self):
        # A garbage/truncated token (e.g. from a mid-stream cut response)
        # must not crash — it should degrade to nodata like other bad input.
        text = "1.0,xyz\n2.0,3.0"
        arr = parse_dem_tile_text(text, nodata=-1.0)
        assert arr[0, 0] == pytest.approx(1.0)
        assert arr[0, 1] == pytest.approx(-1.0)
        assert arr[1, 0] == pytest.approx(2.0)
        assert arr[1, 1] == pytest.approx(3.0)


from unittest.mock import patch, MagicMock
from voxcity.downloader.gsi import check_dem_availability


def _resp(status):
    m = MagicMock()
    m.status_code = status
    return m


class TestCheckDemAvailability:
    def test_picks_finest_available(self):
        # dem5a (first probe) returns 200 -> chosen immediately
        with patch("voxcity.downloader.gsi.requests.get", return_value=_resp(200)) as g:
            dem_type, zoom = check_dem_availability(36.225, 140.105, sleep=0)
        assert (dem_type, zoom) == ("dem5a", 15)
        assert g.call_count == 1

    def test_falls_through_to_dem10b(self):
        # dem5a, dem5b -> 404; dem10b -> 200
        responses = [_resp(404), _resp(404), _resp(200)]
        with patch("voxcity.downloader.gsi.requests.get", side_effect=responses):
            dem_type, zoom = check_dem_availability(36.225, 140.105, sleep=0)
        assert (dem_type, zoom) == ("dem10b", 14)

    def test_all_fail_defaults_to_dem10b(self):
        with patch("voxcity.downloader.gsi.requests.get", return_value=_resp(404)):
            dem_type, zoom = check_dem_availability(36.225, 140.105, sleep=0)
        assert (dem_type, zoom) == ("dem10b", 14)

    def test_network_error_is_skipped(self):
        import requests as _rq
        with patch("voxcity.downloader.gsi.requests.get",
                   side_effect=_rq.exceptions.ConnectTimeout()):
            dem_type, zoom = check_dem_availability(36.225, 140.105, sleep=0)
        assert (dem_type, zoom) == ("dem10b", 14)
