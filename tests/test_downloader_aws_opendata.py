"""Tests for voxcity.downloader.aws_opendata (offline)."""
import logging
import os
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
import requests as _requests
from pyproj import Transformer
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin

from voxcity.downloader.aws_opendata import (
    CHMV2_BASE_URL,
    _quadkey,
    _quadkeys_for_bbox,
    _remote_exists,
    chmv2_tile_urls,
    fetch_chmv2_canopy_height,
    fetch_cog_crop,
)


class TestQuadkey:
    def test_zoom1_corners(self):
        # Bing quadkey digits: 0=NW, 1=NE, 2=SW, 3=SE
        assert _quadkey(0, 0, 1) == "0"
        assert _quadkey(1, 0, 1) == "1"
        assert _quadkey(0, 1, 1) == "2"
        assert _quadkey(1, 1, 1) == "3"

    def test_length_equals_zoom(self):
        assert len(_quadkey(523, 341, 10)) == 10


class TestQuadkeysForBbox:
    def test_berlin_point_matches_live_tile_key(self):
        # Verified against the bucket's tiles.geojson on 2026-10-23.
        assert _quadkeys_for_bbox(13.3, 52.1, 13.3, 52.1) == ["1202120103"]

    def test_bbox_straddling_tile_edge_yields_two_keys(self):
        assert _quadkeys_for_bbox(13.35, 52.1, 13.37, 52.1) == [
            "1202120103",
            "1202120112",
        ]

    def test_bbox_straddling_both_axes_yields_four_sorted_unique_keys(self):
        keys = _quadkeys_for_bbox(13.35, 52.26, 13.37, 52.27)
        assert keys == ["1202120101", "1202120103", "1202120110", "1202120112"]
        assert keys == sorted(set(keys))
        assert all(len(k) == 10 for k in keys)

    def test_global_bbox_is_clamped_to_web_mercator(self):
        assert _quadkeys_for_bbox(-180.0, -90.0, 180.0, 90.0, zoom=1) == ["0", "1", "2", "3"]


def test_base_url_has_no_trailing_slash():
    assert not CHMV2_BASE_URL.endswith("/")


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


class TestChmv2TileUrls:
    BBOX_VERTS = [(13.35, 52.1), (13.37, 52.1), (13.37, 52.1), (13.35, 52.1)]

    def test_keeps_only_tiles_that_exist(self):
        # Two candidate keys: 1202120103 (exists) and 1202120112 (pretend missing).
        def fake_head(url, timeout):
            return _Resp(200 if url.endswith("/chm/1202120103.tif") else 404)

        with patch("voxcity.downloader.aws_opendata.requests.head", side_effect=fake_head) as head:
            urls = chmv2_tile_urls(self.BBOX_VERTS)

        assert urls == [f"{CHMV2_BASE_URL}/chm/1202120103.tif"]
        assert head.call_count == 2

    def test_request_exception_counts_as_missing(self):
        with patch(
            "voxcity.downloader.aws_opendata.requests.head",
            side_effect=_requests.ConnectionError("offline"),
        ):
            assert chmv2_tile_urls(self.BBOX_VERTS) == []

    def test_custom_base_url_trailing_slash_is_normalised(self):
        with patch("voxcity.downloader.aws_opendata.requests.head", return_value=_Resp(200)):
            urls = chmv2_tile_urls([(13.3, 52.1)], base_url="https://example.org/mirror/")
        assert urls == ["https://example.org/mirror/chm/1202120103.tif"]


@pytest.fixture
def module_caplog(caplog):
    """voxcity loggers don't propagate to root, so attach caplog's handler directly."""
    from voxcity.downloader import aws_opendata

    aws_opendata._logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger=aws_opendata._logger.name)
    yield caplog
    aws_opendata._logger.removeHandler(caplog.handler)


class TestRemoteExists:
    URL = "https://example.org/chm/x.tif"

    def test_200_true_and_404_false_silently(self, module_caplog):
        with patch("voxcity.downloader.aws_opendata.requests.head", return_value=_Resp(200)):
            assert _remote_exists(self.URL, 5) is True
        with patch("voxcity.downloader.aws_opendata.requests.head", return_value=_Resp(404)):
            assert _remote_exists(self.URL, 5) is False
        assert not [r for r in module_caplog.records if r.levelno >= logging.WARNING]

    def test_503_warns(self, module_caplog):
        with patch("voxcity.downloader.aws_opendata.requests.head", return_value=_Resp(503)):
            assert _remote_exists(self.URL, 5) is False
        warnings = [r for r in module_caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert self.URL in warnings[0].getMessage() and "503" in warnings[0].getMessage()

    def test_request_exception_warns(self, module_caplog):
        with patch(
            "voxcity.downloader.aws_opendata.requests.head",
            side_effect=_requests.ConnectionError("offline"),
        ):
            assert _remote_exists(self.URL, 5) is False
        assert any(r.levelno == logging.WARNING and self.URL in r.getMessage() for r in module_caplog.records)


_TO_LONLAT = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def _merc_rect(x0, y0, x1, y1):
    """Rectangle vertices (lon, lat) for an EPSG:3857 box."""
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [tuple(_TO_LONLAT.transform(x, y)) for x, y in corners]


def _write_tile(path, left, top, values, res=1.0):
    """Write a uint8 EPSG:3857 GeoTIFF with its top-left corner at (left, top)."""
    values = np.asarray(values, dtype="uint8")
    transform = from_origin(left, top, res, res)
    with rasterio.open(
        path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
        count=1, dtype="uint8", crs="EPSG:3857", transform=transform,
    ) as ds:
        ds.write(values, 1)
    return str(path)


# Tiles live near Berlin in EPSG:3857 so lat/lon <-> metres stays well-conditioned.
_X0, _Y0 = 1_480_000.0, 6_860_000.0   # top-left of tile A (metres)


@pytest.fixture
def tile_a(tmp_path):
    # 20x20 px, value = row index (0..19), 1 m/px, top-left (_X0, _Y0)
    vals = np.repeat(np.arange(20, dtype="uint8")[:, None], 20, axis=1)
    return _write_tile(tmp_path / "a.tif", _X0, _Y0, vals)


@pytest.fixture
def tile_b(tmp_path):
    # Same rows, immediately east of tile A, constant 200
    vals = np.full((20, 20), 200, dtype="uint8")
    return _write_tile(tmp_path / "b.tif", _X0 + 20, _Y0, vals)


class TestFetchCogCrop:
    def test_single_tile_crop_has_expected_extent_and_values(self, tile_a, tmp_path):
        # Box 5..15 m east, 5..15 m south of the tile origin -> rows 5..14
        rect = _merc_rect(_X0 + 5, _Y0 - 15, _X0 + 15, _Y0 - 5)
        out = fetch_cog_crop([tile_a], rect, str(tmp_path / "out.tif"))
        assert out == str(tmp_path / "out.tif")
        with rasterio.open(out) as ds:
            assert ds.crs.to_epsg() == 3857
            assert ds.dtypes[0] == "uint8"
            assert ds.res == (1.0, 1.0)
            # One-pixel padding on every side: rows 4..15 -> 12 px tall
            assert ds.height == 12 and ds.width == 12
            arr = ds.read(1)
        assert arr[0, 0] == 4 and arr[-1, 0] == 15

    def test_two_tiles_mosaic_seamlessly(self, tile_a, tile_b, tmp_path):
        # Box straddling the A|B boundary at x = _X0 + 20
        rect = _merc_rect(_X0 + 15, _Y0 - 10, _X0 + 25, _Y0 - 5)
        out = fetch_cog_crop([tile_a, tile_b], rect, str(tmp_path / "out.tif"))
        with rasterio.open(out) as ds:
            arr = ds.read(1)
            left = ds.transform.c
        col_of_boundary = int(round((_X0 + 20) - left))
        assert arr[0, col_of_boundary - 1] < 200   # from A (row value)
        assert arr[0, col_of_boundary] == 200       # from B

    def test_bbox_outside_tile_is_zero_filled(self, tile_a, tmp_path):
        # Box extends 10 m north of tile A (nothing there)
        rect = _merc_rect(_X0 + 2, _Y0 - 5, _X0 + 8, _Y0 + 10)
        out = fetch_cog_crop([tile_a], rect, str(tmp_path / "out.tif"))
        with rasterio.open(out) as ds:
            arr = ds.read(1)
        assert arr[0, :].max() == 0          # north rim: padding
        assert arr[-2, :].max() > 0          # inside tile A (row ~4)

    def test_unopenable_source_is_skipped(self, tile_a, tmp_path):
        rect = _merc_rect(_X0 + 5, _Y0 - 15, _X0 + 15, _Y0 - 5)
        out = fetch_cog_crop(
            [str(tmp_path / "missing.tif"), tile_a], rect, str(tmp_path / "out.tif")
        )
        assert out is not None and os.path.exists(out)

    def test_all_sources_unopenable_returns_none(self, tmp_path):
        rect = _merc_rect(_X0 + 5, _Y0 - 15, _X0 + 15, _Y0 - 5)
        out = fetch_cog_crop([str(tmp_path / "missing.tif")], rect, str(tmp_path / "out.tif"))
        assert out is None
        assert not os.path.exists(tmp_path / "out.tif")

    def test_creates_output_directory(self, tile_a, tmp_path):
        rect = _merc_rect(_X0 + 5, _Y0 - 15, _X0 + 15, _Y0 - 5)
        out = fetch_cog_crop([tile_a], rect, str(tmp_path / "nested" / "dir" / "out.tif"))
        assert os.path.exists(out)

    def test_http_url_gets_vsicurl_prefix(self, tile_a, tmp_path, monkeypatch):
        opened = []
        real_open = rasterio.open

        def spy_open(path, *a, **k):
            opened.append(path)
            if isinstance(path, str) and path.startswith("/vsicurl/"):
                raise RasterioIOError("no network in tests")
            return real_open(path, *a, **k)

        monkeypatch.setattr("voxcity.downloader.aws_opendata.rasterio.open", spy_open)
        rect = _merc_rect(_X0 + 5, _Y0 - 15, _X0 + 15, _Y0 - 5)
        fetch_cog_crop(["https://example.org/t.tif", tile_a], rect, str(tmp_path / "out.tif"))
        assert opened[0] == "/vsicurl/https://example.org/t.tif"


class TestFetchCogCropResample:
    def test_res_max_coarsens_and_takes_block_maximum(self, tile_a, tmp_path):
        # Rows 0..19 have value = row. Crop rows 0..9 (+1 px pad -> 0..10 with
        # row -1 zero-filled), then 5 m blocks with 'max'.
        rect = _merc_rect(_X0 + 0, _Y0 - 10, _X0 + 10, _Y0 - 0)
        out = fetch_cog_crop([tile_a], rect, str(tmp_path / "out.tif"), res=5.0, resampling="max")
        with rasterio.open(out) as ds:
            assert ds.res == (5.0, 5.0)
            arr = ds.read(1)
        # First 5 m block covers padded row -1 (0) and rows 0..3 -> max 3
        assert arr[0, 0] == 3
        # Second block covers rows 4..8 -> max 8
        assert arr[1, 0] == 8

    def test_res_average_is_between_min_and_max(self, tile_a, tmp_path):
        rect = _merc_rect(_X0 + 0, _Y0 - 10, _X0 + 10, _Y0 - 0)
        out = fetch_cog_crop([tile_a], rect, str(tmp_path / "out.tif"), res=5.0, resampling="average")
        with rasterio.open(out) as ds:
            arr = ds.read(1)
        assert 4 <= arr[1, 0] <= 8

    def test_res_nearest_keeps_integer_values(self, tile_a, tmp_path):
        rect = _merc_rect(_X0 + 0, _Y0 - 10, _X0 + 10, _Y0 - 0)
        out = fetch_cog_crop([tile_a], rect, str(tmp_path / "out.tif"), res=2.0, resampling="nearest")
        with rasterio.open(out) as ds:
            arr = ds.read(1)
        assert set(np.unique(arr)) <= set(range(0, 20))

    def test_invalid_resampling_name_raises(self, tile_a, tmp_path):
        rect = _merc_rect(_X0 + 0, _Y0 - 10, _X0 + 10, _Y0 - 0)
        with pytest.raises(ValueError, match="resampling"):
            fetch_cog_crop([tile_a], rect, str(tmp_path / "out.tif"), res=5.0, resampling="cubic")


class TestFetchChmv2CanopyHeight:
    VERTS = [(13.3, 52.1), (13.31, 52.1), (13.31, 52.11), (13.3, 52.11)]

    def test_writes_canopy_height_tif_in_output_dir(self, tmp_path):
        calls = {}

        def fake_tile_urls(verts, base_url=CHMV2_BASE_URL, timeout=30):
            return ["https://example.org/chm/1202120103.tif"]

        def fake_fetch(urls, verts, out_path, *, res=None, resampling="nearest", timeout=60):
            calls.update(urls=urls, out_path=out_path, res=res, resampling=resampling)
            open(out_path, "wb").close()
            return out_path

        with patch("voxcity.downloader.aws_opendata.chmv2_tile_urls", side_effect=fake_tile_urls), \
             patch("voxcity.downloader.aws_opendata.fetch_cog_crop", side_effect=fake_fetch):
            out = fetch_chmv2_canopy_height(
                self.VERTS, str(tmp_path / "out"), res=5.0, resampling="max"
            )

        assert out == str(tmp_path / "out" / "canopy_height.tif")
        assert os.path.exists(out)
        assert calls["urls"] == ["https://example.org/chm/1202120103.tif"]
        assert calls["res"] == 5.0 and calls["resampling"] == "max"

    def test_no_tiles_returns_none_without_fetching(self, tmp_path):
        with patch("voxcity.downloader.aws_opendata.chmv2_tile_urls", return_value=[]), \
             patch("voxcity.downloader.aws_opendata.fetch_cog_crop") as fetch:
            assert fetch_chmv2_canopy_height(self.VERTS, str(tmp_path)) is None
        fetch.assert_not_called()

    def test_fetch_failure_propagates_none(self, tmp_path):
        with patch("voxcity.downloader.aws_opendata.chmv2_tile_urls", return_value=["u"]), \
             patch("voxcity.downloader.aws_opendata.fetch_cog_crop", return_value=None):
            assert fetch_chmv2_canopy_height(self.VERTS, str(tmp_path)) is None
