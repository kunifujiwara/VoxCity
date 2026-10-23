"""Live network integration test for the CHMv2 AWS Open Data download. Skipped by default.

Enable with:  VOXCITY_LIVE_AWS=1 pytest tests/test_downloader_aws_opendata_integration.py
No AWS account or credentials are needed; the bucket is public.
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


def test_berlin_window_download(tmp_path):
    import rasterio
    from voxcity.downloader.aws_opendata import fetch_chmv2_canopy_height

    # ~300 m x 300 m in Tiergarten, Berlin (tile 1202120103)
    verts = [(13.350, 52.512), (13.3545, 52.512), (13.3545, 52.5147), (13.350, 52.5147)]
    out = fetch_chmv2_canopy_height(verts, str(tmp_path))
    assert out == str(tmp_path / "canopy_height.tif")
    with rasterio.open(out) as ds:
        assert ds.crs.to_epsg() == 3857
        assert ds.dtypes[0] == "uint8"
        arr = ds.read(1)
    assert arr.shape[0] > 200 and arr.shape[1] > 200   # ~1.19 m/px over ~300 m
    assert arr.max() <= 255
    assert np.any((arr >= 1) & (arr <= 100))            # some trees in the park
