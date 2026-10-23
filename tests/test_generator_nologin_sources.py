"""Offline tests: login-free DEM / land cover sources in grids.py and api.py."""
import os
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from voxcity.generator.grids import get_dem_grid, get_land_cover_grid
from voxcity.geoprocessor.raster.core import compute_grid_shape
from voxcity.utils.lc import get_land_cover_classes

VERTS = [(13.3660, 52.5130), (13.3660, 52.5150), (13.3690, 52.5150), (13.3690, 52.5130)]  # canonical SW, NW, NE, SE
MESH = 10.0
EE = "voxcity.downloader.gee.initialize_earth_engine"
WC_FETCH = "voxcity.downloader.aws_opendata.fetch_esa_worldcover"

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
    with patch(DEM_FETCHERS["COPERNICUS"], return_value=None) as cop,          patch(DEM_FETCHERS["USGS 3DEP 10m"], return_value=None) as usgs,          patch(DEM_FETCHERS["FABDEM"], return_value=None) as fab,          patch(EE) as init_ee:
        grid = get_dem_grid(VERTS, MESH, source, str(tmp_path), gridvis=False)
    init_ee.assert_not_called()
    assert grid.shape == compute_grid_shape(VERTS, MESH)
    assert np.all(grid == 0)
    # the requested source is tried once; the chain never calls it again
    assert {"COPERNICUS": cop, "USGS 3DEP 10m": usgs, "FABDEM": fab}[source].call_count == 1


def test_usgs_failure_tries_fabdem_before_copernicus(tmp_path):
    with patch(DEM_FETCHERS["USGS 3DEP 10m"], return_value=None),          patch(DEM_FETCHERS["FABDEM"], side_effect=_fake_dem(42.0)) as fab,          patch(DEM_FETCHERS["COPERNICUS"]) as cop, patch(EE) as init_ee:
        grid = get_dem_grid(VERTS, MESH, "USGS 3DEP 10m", str(tmp_path), gridvis=False)
    fab.assert_called_once()
    cop.assert_not_called()
    init_ee.assert_not_called()
    np.testing.assert_allclose(grid, 42.0)


def test_fabdem_failure_tries_copernicus(tmp_path):
    with patch(DEM_FETCHERS["FABDEM"], return_value=None),          patch(DEM_FETCHERS["COPERNICUS"], side_effect=_fake_dem(7.0)), patch(EE):
        grid = get_dem_grid(VERTS, MESH, "FABDEM", str(tmp_path), gridvis=False)
    np.testing.assert_allclose(grid, 7.0)


def test_all_dems_failing_gives_flat_each_called_once(tmp_path):
    with patch(DEM_FETCHERS["USGS 3DEP 10m"], return_value=None) as usgs,          patch(DEM_FETCHERS["FABDEM"], return_value=None) as fab,          patch(DEM_FETCHERS["COPERNICUS"], return_value=None) as cop, patch(EE):
        grid = get_dem_grid(VERTS, MESH, "USGS 3DEP 10m", str(tmp_path), gridvis=False)
    assert np.all(grid == 0)
    usgs.assert_called_once()
    fab.assert_called_once()
    cop.assert_called_once()



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


class _StopBeforeBuild(Exception):
    """Sentinel to abort get_voxcity right after source resolution."""


def _resolved_sources_without_ee(rectangle_vertices):
    """get_voxcity's auto-selected sources with Earth Engine unavailable (same
    pattern as tests/test_generator_gsi_dem.py)."""
    from voxcity.generator import api

    captured = {}

    def _capture(_verts, _b, _bc, land_cover, _canopy, dem):
        captured.update(land_cover=land_cover, dem=dem)
        raise _StopBeforeBuild

    with patch(EE, side_effect=Exception("EE unavailable")),          patch.object(api, "_warn_source_coverage", side_effect=_capture):
        try:
            api.get_voxcity(rectangle_vertices, meshsize=10)
        except _StopBeforeBuild:
            pass
    return captured


USA_VERTS = [(-122.42, 37.77), (-122.41, 37.77), (-122.41, 37.78), (-122.42, 37.78)]
GERMANY_VERTS = VERTS
JAPAN_VERTS = [(139.75, 35.68), (139.76, 35.68), (139.76, 35.69), (139.75, 35.69)]


def test_no_ee_dem_usa_is_usgs_10m():
    assert _resolved_sources_without_ee(USA_VERTS)["dem"] == "USGS 3DEP 10m"


def test_no_ee_dem_germany_is_fabdem():
    assert _resolved_sources_without_ee(GERMANY_VERTS)["dem"] == "FABDEM"


def test_no_ee_dem_japan_stays_gsi():
    assert _resolved_sources_without_ee(JAPAN_VERTS)["dem"] == "GSI DEM Japan"


def test_usgs_10m_registered():
    from voxcity.generator.api import _DEM_COVERAGE, _SOURCE_URLS
    assert _SOURCE_URLS["USGS 3DEP 10m"].startswith("https://www.usgs.gov/")
    assert "USGS 3DEP 10m" in _DEM_COVERAGE
