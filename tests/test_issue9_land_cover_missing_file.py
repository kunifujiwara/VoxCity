"""Regression test for issue #9: land_cover.tif never created.

When a GEE land-cover exporter silently fails to write the file, get_land_cover_grid
must raise a clear ProcessingError naming the source, not a cryptic RasterioIOError.
An unknown source name must raise a clear ValueError.

Note: initialize_earth_engine/get_roi/save_geotiff_esa_land_cover etc. are imported
LOCALLY inside get_land_cover_grid (from ..downloader.gee), not bound at module level
in voxcity.generator.grids. So monkeypatching grids_mod.initialize_earth_engine has no
effect on the function body -- the local import re-resolves the name from the
voxcity.downloader.gee module at call time. We therefore patch the source module
attributes directly.
"""

import pytest

from voxcity.generator import grids as grids_mod
from voxcity.errors import ProcessingError


def test_esa_export_writes_no_file_raises_processing_error(monkeypatch, tmp_path):
    rect = [(0.0, 0.0), (0.001, 0.0), (0.001, 0.001), (0.0, 0.001)]
    monkeypatch.setattr("voxcity.downloader.gee.initialize_earth_engine", lambda *a, **k: None)
    monkeypatch.setattr("voxcity.downloader.gee.get_roi", lambda *a, **k: object())
    monkeypatch.setattr("voxcity.downloader.gee.save_geotiff_esa_land_cover", lambda *a, **k: None)  # writes nothing
    with pytest.raises(ProcessingError):
        grids_mod.get_land_cover_grid(rect, 5.0, "ESA WorldCover", str(tmp_path), gridvis=False, quiet=True)


def test_unknown_land_cover_source_raises_value_error(monkeypatch, tmp_path):
    rect = [(0.0, 0.0), (0.001, 0.0), (0.001, 0.001), (0.0, 0.001)]
    monkeypatch.setattr("voxcity.downloader.gee.initialize_earth_engine", lambda *a, **k: None)
    with pytest.raises(ValueError):
        grids_mod.get_land_cover_grid(rect, 5.0, "esa_worldcover", str(tmp_path), gridvis=False, quiet=True)
