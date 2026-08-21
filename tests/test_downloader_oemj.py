"""Tests for voxcity.downloader.oemj."""
import pytest
import numpy as np
import rasterio
from PIL import Image

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


class TestSaveAsGeotiffUsesRasterio:
    """The writer must not need GDAL, and must georeference north-up in 3857."""

    def test_writes_readable_georeferenced_raster_without_gdal(self, monkeypatch, tmp_path):
        monkeypatch.setattr(oemj, "gdal", None, raising=False)
        monkeypatch.setattr(oemj, "osr", None, raising=False)

        arr = np.zeros((8, 6, 3), dtype=np.uint8)
        arr[0, 0] = (255, 0, 0)          # asymmetric marker: north-west pixel
        image = Image.fromarray(arr)
        out = tmp_path / "lc.tif"

        oemj.save_as_geotiff(image, POLYGON, 16, (0, 0, 6, 8), (58000, 25800, 58001, 25801), str(out))

        with rasterio.open(out) as src:
            assert src.count == 3
            assert (src.width, src.height) == (6, 8)
            assert src.crs.to_epsg() == 3857
            assert src.transform.e < 0          # north-up
            assert tuple(src.read()[:, 0, 0]) == (255, 0, 0)


class TestGetLandCoverGridRejectsUnreadableRaster:
    """Existence is not readability: a header-only TIFF used to sail through."""

    def test_unreadable_geotiff_raises(self, monkeypatch, tmp_path):
        from voxcity.generator import grids

        geotiff = tmp_path / "land_cover.tif"

        def fake_save(polygon, filepath, **kwargs):
            open(filepath, "wb").write(b"II*\x00")   # TIFF magic, nothing else

        monkeypatch.setattr(grids, "save_oemj_as_geotiff", fake_save)

        with pytest.raises(ProcessingError, match="could not be read"):
            grids.get_land_cover_grid(
                POLYGON, 5, "OpenEarthMapJapan", str(tmp_path),
                print_class_info=False, gridvis=False,
            )

