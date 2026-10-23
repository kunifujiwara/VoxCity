"""Tests for the CHMv2 (AWS Open Data) canopy source wired into get_canopy_height_grid."""
import os
from unittest.mock import patch

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from voxcity.generator.grids import get_canopy_height_grid
from voxcity.geoprocessor.raster.core import compute_grid_shape

CHMV2 = "High Resolution 1m Global Canopy Height Maps v2"

# ~100 m x 100 m box in Berlin
VERTS = [(13.3000, 52.1000), (13.3015, 52.1000), (13.3015, 52.1009), (13.3000, 52.1009)]
MESH = 10.0

_TO_MERC = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def _write_constant_tif(path, value):
    """Write a 1 m EPSG:3857 raster of ``value`` generously covering VERTS."""
    xs, ys = zip(*[_TO_MERC.transform(lon, lat) for lon, lat in VERTS])
    left, top = min(xs) - 20, max(ys) + 20
    width = int(max(xs) - min(xs)) + 40
    height = int(max(ys) - min(ys)) + 40
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1, dtype="uint8",
        crs="EPSG:3857", transform=from_origin(left, top, 1.0, 1.0),
    ) as ds:
        ds.write(np.full((height, width), value, dtype="uint8"), 1)
    return str(path)


def _fake_fetch_returning(value):
    def _fetch(rectangle_vertices, output_dir, *, res=None, resampling="nearest", base_url=None):
        os.makedirs(output_dir, exist_ok=True)
        return _write_constant_tif(os.path.join(output_dir, "canopy_height.tif"), value)
    return _fetch


def test_chmv2_builds_grid_without_earth_engine(tmp_path):
    with patch("voxcity.downloader.aws_opendata.fetch_chmv2_canopy_height",
               side_effect=_fake_fetch_returning(12)) as fetch, \
         patch("voxcity.downloader.gee.initialize_earth_engine",
               side_effect=AssertionError("Earth Engine must not be touched")):
        top, bottom = get_canopy_height_grid(VERTS, MESH, CHMV2, str(tmp_path), gridvis=False)

    assert top.shape == compute_grid_shape(VERTS, MESH)
    assert np.all(top == 12)
    np.testing.assert_allclose(bottom, top * (11.76 / 19.98))
    # kwargs defaults forwarded
    _, kwargs = fetch.call_args
    assert kwargs == {"res": None, "resampling": "nearest"}


def test_chmv2_forwards_resampling_kwargs(tmp_path):
    with patch("voxcity.downloader.aws_opendata.fetch_chmv2_canopy_height",
               side_effect=_fake_fetch_returning(12)) as fetch:
        get_canopy_height_grid(
            VERTS, MESH, CHMV2, str(tmp_path), gridvis=False,
            canopy_resample_res=MESH, canopy_resampling="max",
        )
    _, kwargs = fetch.call_args
    assert kwargs == {"res": MESH, "resampling": "max"}


def test_chmv2_255_sentinel_becomes_zero(tmp_path):
    with patch("voxcity.downloader.aws_opendata.fetch_chmv2_canopy_height",
               side_effect=_fake_fetch_returning(255)):
        top, _ = get_canopy_height_grid(VERTS, MESH, CHMV2, str(tmp_path), gridvis=False)
    assert np.all(top == 0)


def test_chmv2_no_tiles_falls_back_to_static(tmp_path):
    shape = compute_grid_shape(VERTS, MESH)
    land_cover = np.zeros(shape, dtype=int)   # no tree class anywhere
    with patch("voxcity.downloader.aws_opendata.fetch_chmv2_canopy_height", return_value=None), \
         patch("voxcity.downloader.gee.initialize_earth_engine",
               side_effect=AssertionError("Earth Engine must not be touched")):
        top, bottom = get_canopy_height_grid(
            VERTS, MESH, CHMV2, str(tmp_path), gridvis=False,
            land_cover_like=land_cover, land_cover_source="OpenStreetMap",
        )
    assert top.shape == shape
    assert np.all(top == 0) and np.all(bottom == 0)
