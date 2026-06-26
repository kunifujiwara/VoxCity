"""Live network integration test for GSI DEM download. Skipped by default.

Enable with:  VOXCITY_LIVE_GSI=1 pytest tests/test_downloader_gsi_integration.py
"""
import os

import numpy as np
import pytest

LIVE = os.environ.get("VOXCITY_LIVE_GSI") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="set VOXCITY_LIVE_GSI=1 to run")


def test_tsukuba_download(tmp_path):
    import rasterio
    from voxcity.downloader.gsi import save_gsi_dem_as_geotiff

    verts = [(140.09, 36.21), (140.12, 36.21), (140.12, 36.24), (140.09, 36.24)]
    out = tmp_path / "tsukuba_dem.tif"
    save_gsi_dem_as_geotiff(verts, str(out))
    assert out.exists()
    with rasterio.open(str(out)) as src:
        assert src.crs.to_epsg() == 3857
        data = src.read(1)
    # At least some real (non-nodata) elevation present.
    assert np.any(data > -1000)
