"""Offline tests for voxcity.downloader.fabdem."""
from unittest.mock import patch

from voxcity.downloader import fabdem

BERLIN = [(13.366, 52.513), (13.374, 52.513), (13.374, 52.518), (13.366, 52.518)]
MANHATTAN = [(-73.99, 40.75), (-73.97, 40.75), (-73.97, 40.765), (-73.99, 40.765)]


class TestNames:
    def test_zip_berlin(self):
        assert fabdem._fabdem_zip_name(52, 13) == "N50E010-N60E020_FABDEM_V1-2.zip"

    def test_zip_manhattan(self):
        assert fabdem._fabdem_zip_name(40, -74) == "N40W080-N50W070_FABDEM_V1-2.zip"

    def test_zip_southern(self):
        assert fabdem._fabdem_zip_name(-5, 20) == "S10E020-N00E030_FABDEM_V1-2.zip"

    def test_members(self):
        assert fabdem._fabdem_member_name(52, 13) == "N52E013_FABDEM_V1-2.tif"
        assert fabdem._fabdem_member_name(40, -74) == "N40W074_FABDEM_V1-2.tif"


class TestPaths:
    def test_berlin_path(self):
        assert fabdem.fabdem_paths(BERLIN) == [
            "/vsizip//vsicurl/https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/"
            "N50E010-N60E020_FABDEM_V1-2.zip/N52E013_FABDEM_V1-2.tif"
        ]

    def test_manhattan_path(self):
        assert fabdem.fabdem_paths(MANHATTAN) == [
            "/vsizip//vsicurl/https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/"
            "N40W080-N50W070_FABDEM_V1-2.zip/N40W074_FABDEM_V1-2.tif"
        ]


class TestFetch:
    def test_writes_dem_tif_with_fill_and_padding(self, tmp_path):
        captured = {}

        def fake_crop(paths, verts, out_path, **kwargs):
            captured.update(kwargs, paths=paths, out_path=out_path)
            open(out_path, "wb").close()
            return out_path

        with patch("voxcity.downloader.fabdem.fetch_cog_crop", side_effect=fake_crop):
            out = fabdem.fetch_fabdem(BERLIN, str(tmp_path / "o"))

        assert out == str(tmp_path / "o" / "dem.tif")
        assert captured["paths"] == fabdem.fabdem_paths(BERLIN)
        assert captured["pad_pixels"] == 4 and captured["fill_nodata"] is True

    def test_guard_rejection_returns_none(self, tmp_path):
        huge = [(-179.9, 10.0), (179.9, 10.0), (179.9, 10.5), (-179.9, 10.5)]
        with patch("voxcity.downloader.fabdem.fetch_cog_crop") as crop:
            assert fabdem.fetch_fabdem(huge, str(tmp_path)) is None
        crop.assert_not_called()
