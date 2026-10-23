"""Offline tests: login-free DEM / land cover sources in grids.py and api.py."""
import os
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from voxcity.generator.grids import get_dem_grid, get_land_cover_grid
from voxcity.geoprocessor.raster.core import compute_grid_shape

VERTS = [(13.3660, 52.5130), (13.3660, 52.5150), (13.3690, 52.5150), (13.3690, 52.5130)]  # canonical SW, NW, NE, SE
MESH = 10.0
EE = "voxcity.downloader.gee.initialize_earth_engine"

DEM_FETCHERS = {
    "COPERNICUS": "voxcity.downloader.aws_opendata.fetch_copernicus_dem",
    "USGS 3DEP 10m": "voxcity.downloader.aws_opendata.fetch_usgs_3dep_10m",
    "FABDEM": "voxcity.downloader.fabdem.fetch_fabdem",
}


def _fake_dem(value):
    def _fetch(rectangle_vertices, output_dir, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "dem.tif")
        with rasterio.open(
            path, "w", driver="GTiff", width=60, height=40, count=1, dtype="float32",
            crs="EPSG:4326", transform=from_origin(13.365, 52.516, 0.0001, 0.0001),
        ) as ds:
            ds.write(np.full((40, 60), value, dtype="float32"), 1)
        return path
    return _fetch


@pytest.mark.parametrize("source", list(DEM_FETCHERS))
def test_dem_source_reads_without_earth_engine(tmp_path, source):
    with patch(DEM_FETCHERS[source], side_effect=_fake_dem(42.0)) as fetch, \
         patch(EE) as init_ee:
        grid = get_dem_grid(VERTS, MESH, source, str(tmp_path), gridvis=False)
    fetch.assert_called_once()
    init_ee.assert_not_called()
    assert grid.shape == compute_grid_shape(VERTS, MESH)
    np.testing.assert_allclose(grid, 42.0)


@pytest.mark.parametrize("source", list(DEM_FETCHERS))
def test_dem_source_failure_falls_back_to_flat(tmp_path, source):
    with patch(DEM_FETCHERS[source], return_value=None), patch(EE) as init_ee:
        grid = get_dem_grid(VERTS, MESH, source, str(tmp_path), gridvis=False)
    init_ee.assert_not_called()
    assert grid.shape == compute_grid_shape(VERTS, MESH)
    assert np.all(grid == 0)


from voxcity.utils.lc import get_land_cover_classes

WC_FETCH = "voxcity.downloader.aws_opendata.fetch_esa_worldcover"


def _fake_worldcover(rgb):
    def _fetch(rectangle_vertices, output_dir, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "land_cover.tif")
        with rasterio.open(
            path, "w", driver="GTiff", width=60, height=40, count=3, dtype="uint8",
            crs="EPSG:4326", transform=from_origin(13.365, 52.516, 0.0001, 0.0001),
        ) as ds:
            for band, value in enumerate(rgb, start=1):
                ds.write(np.full((40, 60), value, dtype="uint8"), band)
        return path
    return _fetch


def test_worldcover_reads_without_earth_engine(tmp_path):
    trees_rgb = list(get_land_cover_classes("ESA WorldCover"))[0]   # (0, 112, 0) = 'Trees'
    with patch(WC_FETCH, side_effect=_fake_worldcover(trees_rgb)) as fetch, patch(EE) as init_ee:
        grid = get_land_cover_grid(VERTS, MESH, "ESA WorldCover", str(tmp_path), gridvis=False, quiet=True)
    fetch.assert_called_once()
    init_ee.assert_not_called()
    assert grid.shape == compute_grid_shape(VERTS, MESH)
    assert len(np.unique(grid)) == 1


class _ReachedOSM(Exception):
    pass


def test_worldcover_failure_falls_back_to_openstreetmap(tmp_path):
    with patch(WC_FETCH, return_value=None), patch(EE) as init_ee, \
         patch("voxcity.generator.grids.load_land_cover_gdf_from_osm", side_effect=_ReachedOSM):
        with pytest.raises(_ReachedOSM):
            get_land_cover_grid(VERTS, MESH, "ESA WorldCover", str(tmp_path), gridvis=False, quiet=True)
    init_ee.assert_not_called()
