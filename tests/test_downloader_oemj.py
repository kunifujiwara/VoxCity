"""Tests for voxcity.downloader.oemj."""
import sys

import pytest
import numpy as np
import pyproj
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

    def test_partial_write_is_deleted(self, monkeypatch, tmp_path):
        """A writer that creates the file before failing must not leave it behind.

        A mid-write rasterio failure leaves a valid, readable, all-zero
        GeoTIFF on disk that passes get_land_cover_grid's readability check.
        The unlink-on-failure cleanup is the only thing preventing that
        silently-garbage grid, so it must be tested directly.
        """
        monkeypatch.setattr(oemj, "download_tiles", lambda *a, **k: ({(0, 0): object()}, (0, 0, 1, 1)))
        monkeypatch.setattr(oemj, "compose_image", lambda *a, **k: object())
        monkeypatch.setattr(oemj, "crop_image", lambda *a, **k: (object(), (0, 0, 4, 4)))

        out = tmp_path / "lc.tif"

        def create_then_boom(*a, **k):
            open(out, "wb").write(b"partial-tiff-bytes")
            raise ModuleNotFoundError("No module named '_gdal_array'")

        monkeypatch.setattr(oemj, "save_as_geotiff", create_then_boom)

        with pytest.raises(ProcessingError, match="OEMJ"):
            oemj.save_oemj_as_geotiff(POLYGON, str(out))

        assert not out.exists()


class TestSaveAsGeotiffUsesRasterio:
    """The writer must not need GDAL, and must georeference north-up in 3857."""

    def test_writes_readable_georeferenced_raster_without_gdal(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "osgeo", None)

        arr = np.zeros((8, 6, 3), dtype=np.uint8)
        arr[0, 0] = (255, 0, 0)          # asymmetric marker: north-west pixel
        image = Image.fromarray(arr)
        out = tmp_path / "lc.tif"
        bounds = (58000, 25800, 58001, 25801)
        bbox = (0, 0, 6, 8)
        zoom = 16

        oemj.save_as_geotiff(image, POLYGON, zoom, bbox, bounds, str(out))

        min_x, min_y, _, _ = bounds
        lon_ul, lat_ul = oemj.num2deg(min_x + bbox[0] / 256, min_y + bbox[1] / 256, zoom)
        lon_lr, lat_lr = oemj.num2deg(min_x + bbox[2] / 256, min_y + bbox[3] / 256, zoom)
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        expected_ul_x, expected_ul_y = transformer.transform(lon_ul, lat_ul)
        expected_lr_x, expected_lr_y = transformer.transform(lon_lr, lat_lr)
        expected_pixel_width = (expected_lr_x - expected_ul_x) / image.width

        with rasterio.open(out) as src:
            assert src.count == 3
            assert (src.width, src.height) == (6, 8)
            assert src.crs.to_epsg() == 3857
            assert src.transform.e < 0          # north-up
            assert src.transform.c == pytest.approx(expected_ul_x)
            assert src.transform.f == pytest.approx(expected_ul_y)
            assert src.transform.a == pytest.approx(expected_pixel_width)
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

