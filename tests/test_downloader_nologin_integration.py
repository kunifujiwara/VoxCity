"""Live tests for the login-free DEM and land cover sources. Skipped by default.

Enable with:  VOXCITY_LIVE_AWS=1 pytest tests/test_downloader_nologin_integration.py
No account or credentials are needed.
"""
import os

import numpy as np
import pytest

LIVE = os.environ.get("VOXCITY_LIVE_AWS") == "1"
pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(not LIVE, reason="set VOXCITY_LIVE_AWS=1 to run"),
]

BERLIN = [(13.366, 52.513), (13.374, 52.513), (13.374, 52.518), (13.366, 52.518)]
MANHATTAN = [(-73.99, 40.75), (-73.97, 40.75), (-73.97, 40.765), (-73.99, 40.765)]


def _read(path):
    import rasterio
    with rasterio.open(path) as ds:
        return ds.read(), ds.crs.to_epsg(), ds.dtypes[0]


def test_copernicus_berlin(tmp_path):
    from voxcity.downloader.aws_opendata import fetch_copernicus_dem
    a, epsg, dtype = _read(fetch_copernicus_dem(BERLIN, str(tmp_path)))
    assert epsg == 4326 and dtype == "float32"
    assert 20 < np.median(a) < 80


def test_usgs_manhattan(tmp_path):
    from voxcity.downloader.aws_opendata import fetch_usgs_3dep_10m
    a, epsg, _ = _read(fetch_usgs_3dep_10m(MANHATTAN, str(tmp_path)))
    assert epsg == 4269
    assert np.isfinite(a).all() and a.min() > -50 and a.max() < 60


def test_fabdem_berlin(tmp_path):
    from voxcity.downloader.fabdem import fetch_fabdem
    a, epsg, _ = _read(fetch_fabdem(BERLIN, str(tmp_path)))
    assert epsg == 4326
    assert 25 < np.median(a) < 50


def test_worldcover_berlin(tmp_path):
    from voxcity.downloader.aws_opendata import fetch_esa_worldcover
    a, epsg, dtype = _read(fetch_esa_worldcover(BERLIN, str(tmp_path)))
    assert epsg == 4326 and dtype == "uint8" and a.shape[0] == 3
    trees = (a[0] == 0) & (a[1] == 112) & (a[2] == 0)
    assert trees.mean() > 0.3   # Tiergarten is mostly woodland
