"""Offline tests for voxcity.downloader.cog."""
import logging
import os
import zipfile
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from voxcity.downloader.cog import (
    _GDAL_HTTP_ENV,
    _MAX_TILES,
    _gdal_path,
    _lat_tag,
    _lon_tag,
    _tile_origins,
    fetch_cog_crop,
    probe_existing,
    tile_origins_for,
)


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


def _write(path, values, left, top, res, nodata=None, dtype="float32", crs="EPSG:4326"):
    values = np.asarray(values, dtype=dtype)
    with rasterio.open(
        path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0], count=1,
        dtype=dtype, crs=crs, transform=from_origin(left, top, res, res), nodata=nodata,
    ) as ds:
        ds.write(values, 1)
    return str(path)


def _rect(w, s, e, n):
    return [(w, s), (e, s), (e, n), (w, n)]


class TestTags:
    def test_lat_tags(self):
        assert _lat_tag(52) == "N52"
        assert _lat_tag(0) == "N00"
        assert _lat_tag(-5) == "S05"

    def test_lon_tags(self):
        assert _lon_tag(13) == "E013"
        assert _lon_tag(0) == "E000"
        assert _lon_tag(-74) == "W074"


class TestTileOrigins:
    def test_berlin_one_degree(self):
        assert _tile_origins(13.366, 52.513, 13.374, 52.518, 1) == [(52, 13)]

    def test_straddles_longitude_line(self):
        assert _tile_origins(-74.5, 40.2, -73.5, 40.8, 1) == [(40, -75), (40, -74)]

    def test_three_degree_tiles(self):
        assert _tile_origins(13.366, 52.513, 13.374, 52.518, 3) == [(51, 12)]

    def test_around_origin(self):
        assert _tile_origins(-0.5, -0.5, 0.5, 0.5, 1) == [(-1, -1), (-1, 0), (0, -1), (0, 0)]

    def test_guard_rejects_antimeridian_span(self, propagate_voxcity_logs, caplog):
        with caplog.at_level(logging.WARNING, logger="voxcity"):
            assert tile_origins_for(_rect(-179.9, 10.0, 179.9, 10.5), 1, "Test DEM") == []
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_guard_rejects_too_many_tiles(self):
        # 5 x 5 one-degree tiles = 25 > _MAX_TILES
        assert _MAX_TILES == 16
        assert tile_origins_for(_rect(10.5, 40.5, 14.5, 44.5), 1, "Test DEM") == []

    def test_guard_passes_normal_area(self):
        assert tile_origins_for(_rect(13.366, 52.513, 13.374, 52.518), 1, "Test DEM") == [(52, 13)]


class TestProbeExisting:
    def test_keeps_200_only_and_warns_on_unexpected_status(self, propagate_voxcity_logs, caplog):
        status = {"https://x/a.tif": 200, "https://x/b.tif": 404, "https://x/c.tif": 503}
        with caplog.at_level(logging.WARNING, logger="voxcity"), patch(
            "voxcity.downloader.cog.requests.head",
            side_effect=lambda url, timeout: _Resp(status[url]),
        ):
            assert probe_existing(list(status)) == ["https://x/a.tif"]
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and "c.tif" in warnings[0]


class TestZipSupport:
    def test_zip_extension_allowed_for_vsicurl(self):
        assert ".zip" in _GDAL_HTTP_ENV["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"].split(",")
        assert ".tif" in _GDAL_HTTP_ENV["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"].split(",")

    def test_vsi_paths_pass_through(self):
        p = "/vsizip//vsicurl/https://example.org/a.zip/b.tif"
        assert _gdal_path(p) == p

    def test_reads_tif_inside_local_zip(self, tmp_path):
        tif = _write(tmp_path / "t.tif", np.full((100, 100), 7.0), 13.0, 53.0, 0.001)
        zpath = tmp_path / "pack.zip"
        with zipfile.ZipFile(zpath, "w") as z:
            z.write(tif, arcname="N52E013_test.tif")
        out = fetch_cog_crop(
            [f"/vsizip/{zpath.as_posix()}/N52E013_test.tif"],
            _rect(13.02, 52.92, 13.05, 52.95), str(tmp_path / "out.tif"),
        )
        with rasterio.open(out) as ds:
            assert np.all(ds.read(1) == 7.0)


class TestPadPixels:
    def test_pad_widens_crop_by_expected_pixels(self, tmp_path):
        src = _write(tmp_path / "s.tif", np.ones((200, 200)), 13.0, 53.0, 0.001)
        rect = _rect(13.05, 52.85, 13.10, 52.90)
        with rasterio.open(fetch_cog_crop([src], rect, str(tmp_path / "p1.tif"))) as ds:
            w1, h1 = ds.width, ds.height
        with rasterio.open(fetch_cog_crop([src], rect, str(tmp_path / "p5.tif"), pad_pixels=5)) as ds:
            w5, h5 = ds.width, ds.height
        assert (w5 - w1, h5 - h1) == (8, 8)


class TestFillNodata:
    def test_fills_hole_and_uncovered_strip_but_keeps_real_zero(self, tmp_path):
        vals = np.full((100, 100), 30.0)
        vals[40:60, 40:60] = -999999.0   # nodata hole
        vals[0, 0] = 0.0                 # a real sea-level value
        src = _write(tmp_path / "s.tif", vals, 13.0, 53.0, 0.001, nodata=-999999.0)
        # Box runs 0.02 deg west of the raster (uncovered strip).
        rect = _rect(12.98, 52.92, 13.08, 52.999)
        out = fetch_cog_crop([src], rect, str(tmp_path / "o.tif"), fill_nodata=True)
        with rasterio.open(out) as ds:
            a = ds.read(1)
            assert ds.dtypes[0] == "float32"
            r0, c0 = ds.index(13.0005, 52.9995)      # the real-zero pixel
        assert np.isfinite(a).all()
        assert a.min() >= 0.0 and a.max() <= 30.0
        assert a[r0, c0] == 0.0
        assert not np.any(a == -999999.0)

    def test_all_nodata_returns_none_and_writes_nothing(self, tmp_path):
        src = _write(tmp_path / "s.tif", np.full((50, 50), -999999.0), 13.0, 53.0, 0.001, nodata=-999999.0)
        out_path = tmp_path / "o.tif"
        assert fetch_cog_crop([src], _rect(13.01, 52.96, 13.02, 52.97), str(out_path), fill_nodata=True) is None
        assert not out_path.exists()

    def test_default_keeps_old_behaviour(self, tmp_path):
        src = _write(tmp_path / "s.tif", np.full((50, 50), 5, dtype="uint8"), 13.0, 53.0, 0.001, dtype="uint8")
        out = fetch_cog_crop([src], _rect(13.01, 52.96, 13.02, 52.97), str(tmp_path / "o.tif"))
        with rasterio.open(out) as ds:
            assert ds.dtypes[0] == "uint8"
