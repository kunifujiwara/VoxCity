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


from voxcity.geoprocessor.raster import create_land_cover_grid_from_geotiff_polygon
from voxcity.utils.lc import get_land_cover_classes

_ESA_CLASSES = get_land_cover_classes("ESA WorldCover")
_ESA_NAMES = list(_ESA_CLASSES.values())
_CODES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
_CODE_TO_NAME = dict(zip(_CODES, _ESA_NAMES))
_CODE_TO_NAME[0] = "Open water"


class TestWorldCoverNames:
    def test_berlin(self):
        assert aod._worldcover_tile_name(51, 12) == "ESA_WorldCover_10m_2021_v200_N51E012_Map.tif"

    def test_southern_western(self):
        assert aod._worldcover_tile_name(-36, -60) == "ESA_WorldCover_10m_2021_v200_S36W060_Map.tif"

    def test_urls_use_three_degree_tiles(self):
        with patch("voxcity.downloader.aws_opendata.probe_existing", side_effect=_echo):
            urls = aod.worldcover_tile_urls(BERLIN)
        assert urls == [
            "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
            "ESA_WorldCover_10m_2021_v200_N51E012_Map.tif"
        ]


class TestWorldCoverRoundTrip:
    """Codes -> RGB land_cover.tif -> VoxCity's own colour lookup -> class names."""

    @pytest.mark.parametrize("code", _CODES + [0])
    def test_every_code_maps_to_its_class(self, tmp_path, code):
        codes_tif = tmp_path / "codes.tif"
        with rasterio.open(
            codes_tif, "w", driver="GTiff", width=200, height=200, count=1, dtype="uint8",
            crs="EPSG:4326", transform=from_origin(13.36, 52.52, 0.0001, 0.0001), nodata=0,
        ) as ds:
            ds.write(np.full((200, 200), code, dtype="uint8"), 1)

        with patch("voxcity.downloader.aws_opendata.worldcover_tile_urls", return_value=[str(codes_tif)]):
            out = aod.fetch_esa_worldcover(BERLIN, str(tmp_path / "o"))

        assert out == str(tmp_path / "o" / "land_cover.tif")
        assert not os.path.exists(tmp_path / "o" / "_esa_worldcover_codes.tif")
        with rasterio.open(out) as ds:
            assert ds.count == 3 and ds.dtypes[0] == "uint8"
        grid = create_land_cover_grid_from_geotiff_polygon(out, 5.0, _ESA_CLASSES, BERLIN)
        assert np.all(grid == _CODE_TO_NAME[code])

    def test_no_tiles_returns_none(self, tmp_path):
        with patch("voxcity.downloader.aws_opendata.worldcover_tile_urls", return_value=[]):
            assert aod.fetch_esa_worldcover(BERLIN, str(tmp_path)) is None
