"""Unit tests for app.backend.zoning helpers."""
import numpy as np
import pytest

from app.backend.zoning import (
    polygon_lonlat_to_cells,
    points_in_polygon_lonlat,
    stats_from_values,
)


# ---- Fixtures --------------------------------------------------------------

@pytest.fixture
def grid_geom_axis_aligned():
    """A trivial 10x10 axis-aligned grid: cell centres at (i+0.5, j+0.5)."""
    return {
        "origin":    [0.0, 0.0],
        "u_vec":     [1.0, 0.0],
        "v_vec":     [0.0, 1.0],
        "adj_mesh":  [1.0, 1.0],
        "grid_size": [10, 10],
    }


# ---- polygon_lonlat_to_cells ----------------------------------------------

def test_rect_polygon_returns_inside_cells(grid_geom_axis_aligned):
    # Square covering centres (0.5..3.5, 0.5..3.5) -> 4x4 = 16 cells.
    ring = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
    cells = polygon_lonlat_to_cells(ring, grid_geom_axis_aligned)
    assert len(cells) == 16
    assert (0, 0) in cells and (3, 3) in cells


def test_polygon_outside_grid_returns_empty(grid_geom_axis_aligned):
    ring = [[100.0, 100.0], [101.0, 100.0], [101.0, 101.0]]
    assert polygon_lonlat_to_cells(ring, grid_geom_axis_aligned) == []


def test_degenerate_polygon_returns_empty(grid_geom_axis_aligned):
    assert polygon_lonlat_to_cells([[0.0, 0.0], [1.0, 0.0]], grid_geom_axis_aligned) == []


# ---- stats_from_values -----------------------------------------------------

def test_stats_finite_values():
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    s = stats_from_values("z1", cell_count=4, values=vals)
    assert s.cell_count == 4
    assert s.valid_count == 4
    assert s.mean == pytest.approx(2.5)
    assert s.min == 1.0 and s.max == 4.0
    assert s.std == pytest.approx(np.std(vals))


def test_stats_with_nan_values():
    vals = np.array([1.0, np.nan, 3.0, np.inf])
    s = stats_from_values("z1", cell_count=4, values=vals)
    assert s.cell_count == 4
    assert s.valid_count == 2
    assert s.mean == pytest.approx(2.0)


def test_stats_empty_zone():
    s = stats_from_values("z1", cell_count=0, values=np.array([], dtype=float))
    assert s.cell_count == 0 and s.valid_count == 0
    assert s.mean is None and s.std is None


# ---- points_in_polygon_lonlat ---------------------------------------------

def test_points_in_polygon():
    ring = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    pts = np.array([[5.0, 5.0], [-1.0, 5.0], [11.0, 5.0]])
    mask = points_in_polygon_lonlat(pts, ring)
    assert mask.tolist() == [True, False, False]
