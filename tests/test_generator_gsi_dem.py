"""Tests for the GSI DEM source wired into get_dem_grid."""
import os
from unittest.mock import patch

import numpy as np

from voxcity.generator.grids import get_dem_grid


VERTS = [(140.09, 36.21), (140.12, 36.21), (140.12, 36.24), (140.09, 36.24)]


def test_gsi_source_calls_downloader_and_builds_grid(tmp_path):
    """source='GSI DEM Japan' must route to the GSI downloader (no Earth
    Engine) and return a grid from the written GeoTIFF."""
    called = {}

    def fake_save(rectangle_vertices, filepath, dem_type=None, **kwargs):
        called["dem_type"] = dem_type
        called["filepath"] = filepath
        # Write a tiny valid EPSG:3857 GeoTIFF so the grid builder can read it.
        from voxcity.downloader.gsi import save_dem_as_geotiff
        arr = np.full((256, 256), 12.0, dtype=np.float32)
        save_dem_as_geotiff(arr, (29000, 12900, 29000, 12900), 15, filepath)
        return filepath

    with patch("voxcity.downloader.gsi.save_gsi_dem_as_geotiff", side_effect=fake_save), \
         patch("voxcity.generator.grids.initialize_earth_engine") as init_ee:
        grid = get_dem_grid(
            VERTS, meshsize=10, source="GSI DEM Japan",
            output_dir=str(tmp_path), gsi_dem_type="dem10b", gridvis=False,
        )

    init_ee.assert_not_called()
    assert called["dem_type"] == "dem10b"
    assert os.path.basename(called["filepath"]) == "dem.tif"
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
