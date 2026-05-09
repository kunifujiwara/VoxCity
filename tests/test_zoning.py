"""Tests for voxcity.utils.zoning — lon/lat polygon helpers."""

import numpy as np
import pytest

from voxcity.geoprocessor.raster.core import compute_grid_geometry
from voxcity.utils.zoning import (
    mask_from_lonlat_ring,
    polygon_lonlat_to_cells,
    points_in_polygon_lonlat,
)


# Axis-aligned ~4 km × 4 km rectangle in Tokyo, vertex order SW/NW/NE/SE.
# Same shape as tests/test_geo_projector.py for consistency.
_RECT = [
    [139.680, 35.680],  # 0 SW
    [139.680, 35.716],  # 1 NW
    [139.716, 35.716],  # 2 NE
    [139.716, 35.680],  # 3 SE
]
_MESHSIZE = 50.0


@pytest.fixture(scope="module")
def geom():
    g = compute_grid_geometry(_RECT, _MESHSIZE)
    assert g is not None
    return g


# ── mask_from_lonlat_ring ────────────────────────────────────────────────────

def test_mask_from_lonlat_ring_inner_square(geom):
    """A square ring centred on the rectangle produces a contiguous block of True cells."""
    nx, ny = geom["grid_size"]
    # Ring covers the central ~half of the rectangle
    ring = [
        [139.689, 35.689],
        [139.689, 35.707],
        [139.707, 35.707],
        [139.707, 35.689],
    ]
    mask = mask_from_lonlat_ring(ring, geom)
    assert mask.shape == (nx, ny)
    assert mask.dtype == np.bool_
    assert mask.any(), "expected at least one True cell"
    assert not mask.all(), "expected at least one False cell"
    # The True region should be contiguous: True count equals the bounding-box
    # count for the True cells (within ±1 row/col, allowing for rasterization).
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    bbox_area = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
    assert mask.sum() >= 0.95 * bbox_area  # nearly fills its bounding box


def test_mask_from_lonlat_ring_all_outside(geom):
    """A ring entirely outside the rectangle produces all-False mask."""
    nx, ny = geom["grid_size"]
    ring = [
        [140.000, 36.000],
        [140.000, 36.001],
        [140.001, 36.001],
        [140.001, 36.000],
    ]
    mask = mask_from_lonlat_ring(ring, geom)
    assert mask.shape == (nx, ny)
    assert not mask.any()


def test_mask_from_lonlat_ring_too_few_points(geom):
    """Rings with fewer than 3 points must return all-False, not raise."""
    nx, ny = geom["grid_size"]
    assert not mask_from_lonlat_ring([], geom).any()
    assert not mask_from_lonlat_ring([[139.7, 35.7]], geom).any()
    assert not mask_from_lonlat_ring([[139.7, 35.7], [139.71, 35.71]], geom).any()


# ── polygon_lonlat_to_cells ──────────────────────────────────────────────────

def test_polygon_lonlat_to_cells_matches_argwhere(geom):
    """polygon_lonlat_to_cells == np.argwhere(mask).tolist()."""
    ring = [
        [139.689, 35.689],
        [139.689, 35.707],
        [139.707, 35.707],
        [139.707, 35.689],
    ]
    mask = mask_from_lonlat_ring(ring, geom)
    cells = polygon_lonlat_to_cells(ring, geom)
    expected = [(int(i), int(j)) for i, j in np.argwhere(mask)]
    assert cells == expected


def test_polygon_lonlat_to_cells_empty_outside(geom):
    cells = polygon_lonlat_to_cells(
        [[140.0, 36.0], [140.0, 36.001], [140.001, 36.001]], geom,
    )
    assert cells == []


# ── points_in_polygon_lonlat ─────────────────────────────────────────────────

def test_points_in_polygon_lonlat_basic():
    ring = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
    pts = np.array([
        [0.5, 0.5],   # inside
        [-0.5, 0.5],  # outside
        [0.5, 1.5],   # outside
    ])
    out = points_in_polygon_lonlat(pts, ring)
    assert out.tolist() == [True, False, False]


def test_points_in_polygon_lonlat_empty_inputs():
    ring = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
    # Empty point array
    assert points_in_polygon_lonlat(np.empty((0, 2)), ring).shape == (0,)
    # Ring with too few points
    out = points_in_polygon_lonlat(np.array([[0.5, 0.5]]), [[0.0, 0.0]])
    assert out.tolist() == [False]
