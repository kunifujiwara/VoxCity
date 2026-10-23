"""Offline tests for the Copernicus, WorldCover and USGS 3DEP adapters."""
import os
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from voxcity.downloader import aws_opendata as aod

BERLIN = [(13.366, 52.513), (13.374, 52.513), (13.374, 52.518), (13.366, 52.518)]
MANHATTAN = [(-73.99, 40.75), (-73.97, 40.75), (-73.97, 40.765), (-73.99, 40.765)]
HOUSTON = [(-95.56, 29.735), (-95.54, 29.735), (-95.54, 29.75), (-95.56, 29.75)]


def _echo(urls, timeout=10):
    return list(urls)


class TestCopernicusNames:
    def test_berlin(self):
        assert aod._copernicus_tile_name(52, 13) == "Copernicus_DSM_COG_10_N52_00_E013_00_DEM"

    def test_manhattan(self):
        assert aod._copernicus_tile_name(40, -74) == "Copernicus_DSM_COG_10_N40_00_W074_00_DEM"

    def test_southern_hemisphere(self):
        assert aod._copernicus_tile_name(-34, 151) == "Copernicus_DSM_COG_10_S34_00_E151_00_DEM"

    def test_urls_are_probed(self):
        with patch("voxcity.downloader.aws_opendata.probe_existing", side_effect=_echo) as probe:
            urls = aod.copernicus_tile_urls(BERLIN)
        assert urls == [
            "https://copernicus-dem-30m.s3.amazonaws.com/"
            "Copernicus_DSM_COG_10_N52_00_E013_00_DEM/Copernicus_DSM_COG_10_N52_00_E013_00_DEM.tif"
        ]
        probe.assert_called_once()


class TestUsgsNames:
    def test_manhattan_uses_north_west_corner(self):
        assert aod._usgs_3dep_tile_tag(40, -74) == "n41w074"

    def test_houston(self):
        assert aod._usgs_3dep_tile_tag(29, -96) == "n30w096"

    def test_eastern_hemisphere_format(self):
        assert aod._usgs_3dep_tile_tag(13, 144) == "n14e144"

    def test_urls(self):
        with patch("voxcity.downloader.aws_opendata.probe_existing", side_effect=_echo):
            urls = aod.usgs_3dep_10m_tile_urls(MANHATTAN)
        assert urls == [
            "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/"
            "n41w074/USGS_13_n41w074.tif"
        ]


class TestDemFetchWiring:
    @pytest.mark.parametrize(
        "fetch_name, urls_name, expected",
        [
            ("fetch_copernicus_dem", "copernicus_tile_urls", {"pad_pixels": 4}),
            ("fetch_usgs_3dep_10m", "usgs_3dep_10m_tile_urls", {"pad_pixels": 10, "fill_nodata": True}),
        ],
    )
    def test_writes_dem_tif_with_dataset_options(self, tmp_path, fetch_name, urls_name, expected):
        captured = {}

        def fake_crop(urls, verts, out_path, **kwargs):
            captured.update(kwargs, urls=urls, out_path=out_path)
            open(out_path, "wb").close()
            return out_path

        with patch(f"voxcity.downloader.aws_opendata.{urls_name}", return_value=["u1"]), \
             patch("voxcity.downloader.aws_opendata.fetch_cog_crop", side_effect=fake_crop):
            out = getattr(aod, fetch_name)(BERLIN, str(tmp_path / "o"))

        assert out == str(tmp_path / "o" / "dem.tif")
        assert captured["urls"] == ["u1"]
        for k, v in expected.items():
            assert captured[k] == v

    @pytest.mark.parametrize(
        "fetch_name, urls_name",
        [("fetch_copernicus_dem", "copernicus_tile_urls"), ("fetch_usgs_3dep_10m", "usgs_3dep_10m_tile_urls")],
    )
    def test_no_tiles_returns_none(self, tmp_path, fetch_name, urls_name):
        with patch(f"voxcity.downloader.aws_opendata.{urls_name}", return_value=[]), \
             patch("voxcity.downloader.aws_opendata.fetch_cog_crop") as crop:
            assert getattr(aod, fetch_name)(BERLIN, str(tmp_path)) is None
        crop.assert_not_called()
