"""Live network integration test for GSI DEM download. Skipped by default.

Enable with:  VOXCITY_LIVE_GSI=1 pytest tests/test_downloader_gsi_integration.py
"""
import os

import numpy as np
import pytest

LIVE = os.environ.get("VOXCITY_LIVE_GSI") == "1"
pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(not LIVE, reason="set VOXCITY_LIVE_GSI=1 to run"),
]


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


def test_partial_5m_area_comes_back_uniformly_10m(tmp_path):
    """Boso peninsula ROI: 5 m products cover only ~23%, so the downloader must
    return one uniform 10 m raster rather than a patched 5 m one."""
    import rasterio
    from voxcity.downloader.gsi import save_gsi_dem_as_geotiff, GSI_NODATA

    verts = [(140.256410, 35.326512), (140.256410, 35.342736),
             (140.269610, 35.342736), (140.269610, 35.326512)]
    out = tmp_path / "boso_dem.tif"
    save_gsi_dem_as_geotiff(verts, str(out))

    with rasterio.open(str(out)) as src:
        # z14 pixel size => the 10 m product was chosen for the whole ROI.
        expected = (2 * 20037508.342789244) / (2.0 ** 14) / 256
        assert src.transform.a == pytest.approx(expected)
        data = src.read(1)

    # Measured 2026-10-05: 100% valid, elevation 31.33-166.65 m.
    assert (data != GSI_NODATA).mean() > 0.99
    valid = data[data != GSI_NODATA]
    assert valid.max() - valid.min() > 50   # real relief, not a flattened plain
