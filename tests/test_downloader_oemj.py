"""Tests for voxcity.downloader.oemj."""
import pytest

import voxcity.downloader.oemj as oemj
from voxcity.errors import ProcessingError

POLYGON = [(139.76, 35.68), (139.76, 35.69), (139.77, 35.69), (139.77, 35.68)]


class TestSaveOemjAsGeotiffPropagates:
    """A failed write must reach the caller, not be printed and swallowed.

    Regression: save_oemj_as_geotiff wrapped its whole body in `except
    Exception` and only printed. With a GDAL lacking numpy bindings the
    write failed *after* the file was created, leaving a header-only TIFF
    that get_land_cover_grid happily read as valid land cover.
    """

    def test_writer_failure_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(oemj, "download_tiles", lambda *a, **k: ({(0, 0): object()}, (0, 0, 1, 1)))
        monkeypatch.setattr(oemj, "compose_image", lambda *a, **k: object())
        monkeypatch.setattr(oemj, "crop_image", lambda *a, **k: (object(), (0, 0, 4, 4)))

        def boom(*a, **k):
            raise ModuleNotFoundError("No module named '_gdal_array'")

        monkeypatch.setattr(oemj, "save_as_geotiff", boom)

        with pytest.raises(ProcessingError, match="OEMJ"):
            oemj.save_oemj_as_geotiff(POLYGON, str(tmp_path / "lc.tif"))

    def test_no_tiles_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(oemj, "download_tiles", lambda *a, **k: ({}, (0, 0, 1, 1)))

        with pytest.raises(ProcessingError, match="no tiles"):
            oemj.save_oemj_as_geotiff(POLYGON, str(tmp_path / "lc.tif"))
