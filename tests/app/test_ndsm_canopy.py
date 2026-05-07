from __future__ import annotations

import math

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from app.backend import main as main_mod
from voxcity.geoprocessor.raster.core import compute_cell_center_coords


def _rotated_rectangle(center_lon: float, center_lat: float) -> list[list[float]]:
    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    to_geo = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    cx, cy = to_merc.transform(center_lon, center_lat)
    base = [(-50.0, -30.0), (-50.0, 30.0), (50.0, 30.0), (50.0, -30.0)]
    angle = math.radians(30.0)
    vertices = []
    for x, y in base:
        rx = x * math.cos(angle) - y * math.sin(angle)
        ry = x * math.sin(angle) + y * math.cos(angle)
        lon, lat = to_geo.transform(cx + rx, cy + ry)
        vertices.append([lon, lat])
    return vertices


def test_load_ndsm_grid_samples_rotated_grid_cell_centers(tmp_path, monkeypatch):
    rectangle = _rotated_rectangle(139.75, 35.68)
    meshsize = 10.0
    cc = compute_cell_center_coords(rectangle, meshsize)
    expected_shape = cc["grid_size"]

    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = to_merc.transform(cc["lons"].ravel(), cc["lats"].ravel())
    minx = math.floor(float(np.min(xs)) / meshsize) * meshsize - meshsize
    maxx = math.ceil(float(np.max(xs)) / meshsize) * meshsize + meshsize
    miny = math.floor(float(np.min(ys)) / meshsize) * meshsize - meshsize
    maxy = math.ceil(float(np.max(ys)) / meshsize) * meshsize + meshsize
    width = int(round((maxx - minx) / meshsize))
    height = int(round((maxy - miny) / meshsize))

    transform = from_origin(minx, maxy, meshsize, meshsize)
    rows, cols = np.indices((height, width))
    data = (rows * 1000 + cols).astype(np.float32)
    tif_path = tmp_path / "ndsm.tif"
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    expected_rows, expected_cols = rasterio.transform.rowcol(transform, xs, ys)
    expected = data[np.asarray(expected_rows), np.asarray(expected_cols)].reshape(expected_shape)

    monkeypatch.setattr(main_mod, "NDSM_COG_PATH", str(tif_path))

    actual = main_mod._load_ndsm_grid(rectangle, meshsize)

    assert actual.shape == expected_shape
    np.testing.assert_array_equal(actual, expected)


def test_sanitize_ndsm_canopy_clamps_plausible_tree_range():
    canopy = np.array(
        [
            [np.nan, 1.0, 12.0],
            [35.0, 60.0, 0.0],
        ],
        dtype=float,
    )
    tree_mask = np.array(
        [
            [False, True, True],
            [True, True, False],
        ],
        dtype=bool,
    )
    building_heights = np.zeros_like(canopy)

    actual = main_mod._sanitize_ndsm_canopy(
        canopy,
        building_heights=building_heights,
        tree_mask=tree_mask,
        replacement_m=10.0,
        min_tree_height_m=2.0,
        max_tree_height_m=35.0,
    )

    expected = np.array(
        [
            [np.nan, 10.0, 12.0],
            [35.0, 35.0, 0.0],
        ],
        dtype=float,
    )
    np.testing.assert_array_equal(actual, expected)


def test_sanitize_ndsm_canopy_replaces_local_tree_height_spikes():
    canopy = np.zeros((5, 5), dtype=float)
    canopy[1:4, 1:4] = np.array(
        [
            [10.0, 11.0, 10.0],
            [10.0, 45.0, 11.0],
            [10.0, 9.0, 10.0],
        ]
    )
    tree_mask = canopy > 0
    building_heights = np.zeros_like(canopy)
    building_heights[2, 0] = 30.0

    actual = main_mod._sanitize_ndsm_canopy(
        canopy,
        building_heights=building_heights,
        tree_mask=tree_mask,
        replacement_m=10.0,
        max_tree_height_m=60.0,
        local_outlier_margin_m=12.0,
    )

    assert actual[2, 2] == 10.0
    unchanged = tree_mask.copy()
    unchanged[2, 2] = False
    np.testing.assert_array_equal(actual[unchanged], canopy[unchanged])