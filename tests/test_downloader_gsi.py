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


from voxcity.downloader.gsi import download_dem_tiles, compose_dem_array


def _txt_resp(value):
    m = MagicMock()
    m.status_code = 200
    line = ",".join([str(value)] * 256)
    m.text = "\n".join([line] * 256)
    return m


def _txt_resp_factory():
    """Infinite generator of valid 200 tile responses (value 3.0)."""
    def gen():
        while True:
            yield _txt_resp(3.0)
    return gen()


class TestDownloadAndCompose:
    def test_download_fills_missing_with_nodata(self):
        # 1x2 tile range; first tile 200, second 404
        tile_range = (10, 20, 10, 21)  # x_min,y_min,x_max,y_max
        responses = [_txt_resp(5.0), _resp(404)]
        with patch("voxcity.downloader.gsi.requests.get", side_effect=responses):
            tiles = download_dem_tiles(tile_range, "dem5a", 15, sleep=0, nodata=-9999.0)
        assert set(tiles.keys()) == {(10, 20), (10, 21)}
        assert np.allclose(tiles[(10, 20)], 5.0)
        assert np.allclose(tiles[(10, 21)], -9999.0)

    def test_download_all_fail_raises(self):
        tile_range = (10, 20, 10, 20)
        with patch("voxcity.downloader.gsi.requests.get", return_value=_resp(404)):
            with pytest.raises(ValueError):
                download_dem_tiles(tile_range, "dem5a", 15, sleep=0)

    def test_compose_places_blocks(self):
        tile_range = (10, 20, 11, 20)  # 2 wide, 1 tall
        tiles = {
            (10, 20): np.full((256, 256), 1.0, dtype=np.float32),
            (11, 20): np.full((256, 256), 2.0, dtype=np.float32),
        }
        mosaic = compose_dem_array(tiles, tile_range, nodata=-9999.0)
        assert mosaic.shape == (256, 512)
        assert np.allclose(mosaic[:, :256], 1.0)
        assert np.allclose(mosaic[:, 256:], 2.0)


from voxcity.downloader.gsi import save_dem_as_geotiff, tile_bounds_mercator


class TestSaveDemAsGeotiff:
    def test_roundtrip_crs_transform_nodata(self, tmp_path):
        import rasterio
        tile_range = (29000, 12900, 29000, 12900)  # single tile
        zoom = 15
        array = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
        out = tmp_path / "dem.tif"
        save_dem_as_geotiff(array, tile_range, zoom, str(out), nodata=-9999.0)
        assert out.exists()
        with rasterio.open(str(out)) as src:
            assert src.crs.to_epsg() == 3857
            assert src.nodata == -9999.0
            data = src.read(1)
            assert np.allclose(data, array)
            # Origin matches top-left tile bounds
            minx, miny, maxx, maxy = tile_bounds_mercator(29000, 12900, zoom)
            assert src.transform.c == pytest.approx(minx, rel=1e-9)
            assert src.transform.f == pytest.approx(maxy, rel=1e-9)
            # Pixel size = tile / 256
            tile = (2 * 20037508.342789244) / (2 ** zoom)
            assert src.transform.a == pytest.approx(tile / 256, rel=1e-9)
            assert src.transform.e == pytest.approx(-tile / 256, rel=1e-9)


from voxcity.downloader.gsi import save_gsi_dem_as_geotiff


class TestSaveGsiDemAsGeotiff:
    def _verts(self):
        return [(140.09, 36.21), (140.12, 36.21), (140.12, 36.24), (140.09, 36.24)]

    def test_auto_detect_then_write(self, tmp_path):
        out = tmp_path / "dem.tif"
        # All dem5a tiles return valid data -> no dem5b/dem10b fetched, and the
        # legacy center probe is no longer used by the auto path.
        with patch("voxcity.downloader.gsi.check_dem_availability") as chk, \
             patch("voxcity.downloader.gsi.requests.get", side_effect=_txt_resp_factory()):
            path = save_gsi_dem_as_geotiff(self._verts(), str(out), sleep=0)
        assert path == str(out)
        assert out.exists()
        chk.assert_not_called()

    def test_forced_type_skips_probe(self, tmp_path):
        out = tmp_path / "dem.tif"
        with patch("voxcity.downloader.gsi.check_dem_availability") as chk, \
             patch("voxcity.downloader.gsi.requests.get", side_effect=_txt_resp_factory()):
            save_gsi_dem_as_geotiff(self._verts(), str(out), dem_type="dem10b", sleep=0)
        chk.assert_not_called()

    def test_invalid_type_raises(self, tmp_path):
        out = tmp_path / "dem.tif"
        with pytest.raises(ValueError):
            save_gsi_dem_as_geotiff(self._verts(), str(out), dem_type="bogus", sleep=0)


from voxcity.downloader.gsi import _download_fine_merged, _backfill_from_coarser


def _parse_url(url):
    """Extract (dem_type, zoom, x, y) from a GSI XYZ tile URL."""
    tail = url.split("/xyz/")[1]            # e.g. dem5a/15/29139/12925.txt
    dem_type, zoom, x, ynxt = tail.split("/")
    return dem_type, int(zoom), int(x), int(ynxt.split(".")[0])


def _dispatch(responder):
    """Build a requests.get side_effect routed by the URL's dem_type/x/y."""
    def _side(url, **kwargs):
        return responder(*_parse_url(url))
    return _side


def _half_nodata_tile(value):
    """200 response whose left half is ``value`` and right half is 'e' nodata."""
    rows = [",".join(["%s" % value] * 128 + ["e"] * 128) for _ in range(256)]
    m = MagicMock()
    m.status_code = 200
    m.text = "\n".join(rows)
    return m


class TestFineMerge:
    def test_dem5b_not_fetched_when_dem5a_complete(self):
        calls = {"dem5a": 0, "dem5b": 0}

        def responder(dem_type, zoom, x, y):
            calls[dem_type] = calls.get(dem_type, 0) + 1
            return _txt_resp(7.0)

        with patch("voxcity.downloader.gsi.requests.get",
                   side_effect=_dispatch(responder)):
            tiles, any_ok = _download_fine_merged(
                (10, 20, 10, 20), sleep=0, nodata=-9999.0
            )
        assert any_ok
        assert calls["dem5a"] == 1
        assert calls["dem5b"] == 0
        assert np.allclose(tiles[(10, 20)], 7.0)

    def test_dem5b_fills_dem5a_partial_holes(self):
        def responder(dem_type, zoom, x, y):
            if dem_type == "dem5a":
                return _half_nodata_tile(5.0)
            if dem_type == "dem5b":
                return _txt_resp(9.0)
            return _resp(404)

        with patch("voxcity.downloader.gsi.requests.get",
                   side_effect=_dispatch(responder)):
            tiles, any_ok = _download_fine_merged(
                (10, 20, 10, 20), sleep=0, nodata=-9999.0
            )
        block = tiles[(10, 20)]
        assert np.allclose(block[:, :128], 5.0)   # dem5a kept
        assert np.allclose(block[:, 128:], 9.0)   # dem5b filled the 'e' holes

    def test_missing_dem5a_tile_filled_by_dem5b(self):
        def responder(dem_type, zoom, x, y):
            if dem_type == "dem5b":
                return _txt_resp(3.0)
            return _resp(404)                     # dem5a 404 everywhere

        with patch("voxcity.downloader.gsi.requests.get",
                   side_effect=_dispatch(responder)):
            tiles, any_ok = _download_fine_merged(
                (10, 20, 10, 20), sleep=0, nodata=-9999.0
            )
        assert any_ok
        assert np.allclose(tiles[(10, 20)], 3.0)


class TestBackfillFromCoarser:
    def test_integer_division_mapping_no_resample(self):
        nodata = -9999.0
        fine = np.full((256, 256), nodata, dtype=np.float32)
        coarse = np.fromfunction(
            lambda i, j: i * 1000.0 + j, (256, 256), dtype=np.float32
        )
        # Aligned origins: fine z15 tile (2,2), coarse z14 tile (1,1).
        out = _backfill_from_coarser(
            fine, (2, 2, 2, 2), coarse, (1, 1, 1, 1), 15, 14, nodata=nodata
        )
        for r, c in [(0, 0), (1, 1), (2, 3), (255, 254)]:
            assert out[r, c] == pytest.approx((r // 2) * 1000.0 + (c // 2))

    def test_only_holes_filled_and_coarse_nodata_preserved(self):
        nodata = -9999.0
        fine = np.full((256, 256), 5.0, dtype=np.float32)
        fine[0, 0] = nodata   # maps to coarse (0, 0) -> filled
        fine[0, 2] = nodata   # maps to coarse (0, 1) -> coarse is nodata, kept
        coarse = np.full((256, 256), 8.0, dtype=np.float32)
        coarse[0, 1] = nodata
        out = _backfill_from_coarser(
            fine, (0, 0, 0, 0), coarse, (0, 0, 0, 0), 15, 14, nodata=nodata
        )
        assert out[0, 0] == pytest.approx(8.0)   # filled from coarse
        assert out[0, 2] == nodata               # coarse nodata -> stays hole
        assert out[0, 1] == pytest.approx(5.0)   # pre-existing value untouched


class TestAutoMergeSave:
    def _verts(self):
        return [(140.09, 36.21), (140.12, 36.21), (140.12, 36.24), (140.09, 36.24)]

    def test_dem10b_backfills_when_no_5m_coverage(self, tmp_path):
        import rasterio
        out = tmp_path / "dem.tif"

        def responder(dem_type, zoom, x, y):
            if dem_type == "dem10b":
                return _txt_resp(42.0)
            return _resp(404)                     # no dem5a / dem5b

        with patch("voxcity.downloader.gsi.requests.get",
                   side_effect=_dispatch(responder)):
            save_gsi_dem_as_geotiff(self._verts(), str(out), sleep=0)
        with rasterio.open(str(out)) as src:
            assert np.allclose(src.read(1), 42.0)

    def test_dem5a_used_no_fallback_requests(self, tmp_path):
        import rasterio
        out = tmp_path / "dem.tif"
        seen = set()

        def responder(dem_type, zoom, x, y):
            seen.add(dem_type)
            if dem_type == "dem5a":
                return _txt_resp(6.0)
            return _resp(404)

        with patch("voxcity.downloader.gsi.requests.get",
                   side_effect=_dispatch(responder)):
            save_gsi_dem_as_geotiff(self._verts(), str(out), sleep=0)
        assert "dem5b" not in seen and "dem10b" not in seen
        with rasterio.open(str(out)) as src:
            assert np.allclose(src.read(1), 6.0)

    def test_fallback_disabled_does_not_request_dem10b(self, tmp_path):
        out = tmp_path / "dem.tif"
        seen = set()

        def responder(dem_type, zoom, x, y):
            seen.add(dem_type)
            if dem_type == "dem10b":
                return _txt_resp(1.0)
            return _resp(404)                     # no 5 m coverage

        with patch("voxcity.downloader.gsi.requests.get",
                   side_effect=_dispatch(responder)):
            with pytest.raises(ValueError):
                save_gsi_dem_as_geotiff(
                    self._verts(), str(out), sleep=0,
                    include_dem10b_fallback=False,
                )
        assert "dem10b" not in seen

    def test_all_missing_raises(self, tmp_path):
        out = tmp_path / "dem.tif"
        with patch("voxcity.downloader.gsi.requests.get",
                   return_value=_resp(404)):
            with pytest.raises(ValueError):
                save_gsi_dem_as_geotiff(self._verts(), str(out), sleep=0)


class TestPackageExport:
    def test_exported_from_downloader(self):
        import voxcity.downloader as dl
        assert hasattr(dl, "save_gsi_dem_as_geotiff")
        assert "save_gsi_dem_as_geotiff" in dl.__all__
