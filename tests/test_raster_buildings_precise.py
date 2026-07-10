"""Characterization tests for the precise building rasterization path.

Written against the pre-optimization _process_with_geometry_intersection
and kept green through the optimization. The import is updated to
buildings_precise when the module split lands (later task).

Grid frame used throughout: origin (0,0), u_vec=(0,1), v_vec=(1,0),
meshsize (1,1), grid 3x3 -> cell (i, j) spans x in [j, j+1], y in [i, i+1].
Fixtures deliberately avoid equal-height overlaps: the tie-break order is
the one documented behavior change.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from voxcity.geoprocessor.raster.buildings_precise import (
    _process_with_geometry_intersection as _process,
)

ORIGIN = np.array([0.0, 0.0])
U_VEC = np.array([0.0, 1.0])
V_VEC = np.array([1.0, 0.0])
MESH = (1.0, 1.0)
GRID = (3, 3)


def _rect(x0, y0, x1, y1):
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _run(gdf, complement_height=None):
    return _process(gdf, GRID, MESH, ORIGIN, U_VEC, V_VEC, complement_height)


def test_max_height_wins_and_id_is_last_processed():
    # A (h=10, id=1) covers cells (0..1, 0..1); B (h=30, id=2, min_height=2)
    # covers only cell (0,0). Processing order in a cell is height-desc, so
    # in (0,0): B first (sets h=30, id=2, [[2,30]]), then A (appends [0,10],
    # id becomes 1, height stays 30). id = last-processed = the shorter one.
    gdf = gpd.GeoDataFrame({
        "height": [10.0, 30.0],
        "min_height": [0.0, 2.0],
        "is_inner": [False, False],
        "id": [1, 2],
        "geometry": [_rect(0, 0, 2, 2), _rect(0, 0, 1, 1)],
    })
    h, mh, bid, _ = _run(gdf)
    assert h[0, 0] == 30.0
    assert bid[0, 0] == 1
    assert mh[0, 0] == [[2.0, 30.0], [0.0, 10.0]]
    for (i, j) in [(0, 1), (1, 0), (1, 1)]:
        assert h[i, j] == 10.0
        assert bid[i, j] == 1
        assert mh[i, j] == [[0.0, 10.0]]
    assert h[2, 2] == 0.0 and mh[2, 2] == [] and bid[2, 2] == 0.0


def test_inner_building_zeroes_cell_and_short_circuits():
    # A (h=10, id=1) covers (0,0); C (h=5, id=3, is_inner) also covers (0,0).
    # Height-desc order: A assigns first; then C (inner) resets the cell to
    # h=0, min list [[0,0]], and breaks. id keeps A's value (inner path does
    # not touch the id grid).
    gdf = gpd.GeoDataFrame({
        "height": [10.0, 5.0],
        "min_height": [0.0, 0.0],
        "is_inner": [False, True],
        "id": [1, 3],
        "geometry": [_rect(0, 0, 1, 1), _rect(0, 0, 1, 1)],
    })
    h, mh, bid, _ = _run(gdf)
    assert h[0, 0] == 0.0
    assert mh[0, 0] == [[0, 0]]
    assert bid[0, 0] == 1


def test_sliver_below_threshold_leaves_ground():
    # 20% of cell (0,0) < _CELL_INTERSECTION_THRESHOLD (0.3) -> untouched.
    gdf = gpd.GeoDataFrame({
        "height": [50.0],
        "min_height": [0.0],
        "is_inner": [False],
        "id": [9],
        "geometry": [_rect(0, 0, 0.2, 1)],
    })
    h, mh, bid, _ = _run(gdf)
    assert h[0, 0] == 0.0 and mh[0, 0] == [] and bid[0, 0] == 0.0


def test_nan_height_marks_cell_nan():
    gdf = gpd.GeoDataFrame({
        "height": [np.nan],
        "min_height": [0.0],
        "is_inner": [False],
        "id": [4],
        "geometry": [_rect(0, 0, 1, 1)],
    })
    h, mh, bid, _ = _run(gdf)  # complement_height=None
    assert np.isnan(h[0, 0])
    assert bid[0, 0] == 4
    assert len(mh[0, 0]) == 1
    assert mh[0, 0][0][0] == 0.0 and np.isnan(mh[0, 0][0][1])


def test_complement_height_substitutes_missing():
    gdf = gpd.GeoDataFrame({
        "height": [np.nan],
        "min_height": [0.0],
        "is_inner": [False],
        "id": [4],
        "geometry": [_rect(0, 0, 1, 1)],
    })
    h, mh, bid, _ = _run(gdf, complement_height=7.5)
    assert h[0, 0] == 7.5
    assert mh[0, 0] == [[0.0, 7.5]]


def test_invalid_bowtie_polygon_is_repaired():
    # buffer(0) repairs the bowtie into a single triangle
    # (0.5,0.5)-(1,1)-(1,0) of area 0.25 (NOT two triangles totalling 0.5 --
    # for this vertex ordering buffer(0) keeps only one lobe). That area is
    # entirely within cell (0,0) (cell_area=1), giving ratio 0.25 < 0.3, so
    # the (repaired, non-crashing) polygon falls below the coverage
    # threshold and the cell is left untouched. This still demonstrates the
    # invalid polygon is handled without raising.
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    assert not bowtie.is_valid
    gdf = gpd.GeoDataFrame({
        "height": [12.0],
        "min_height": [0.0],
        "is_inner": [False],
        "id": [5],
        "geometry": [bowtie],
    })
    h, mh, bid, _ = _run(gdf)
    assert h[0, 0] == 0.0
    assert mh[0, 0] == []
    assert bid[0, 0] == 0.0


def test_missing_optional_columns_use_defaults():
    # No min_height/is_inner/id columns: min_height -> 0, is_inner -> False,
    # feature_id -> the DataFrame index label.
    gdf = gpd.GeoDataFrame(
        {"height": [20.0], "geometry": [_rect(1, 1, 2, 2)]}, index=[7]
    )
    h, mh, bid, _ = _run(gdf)
    assert h[1, 1] == 20.0
    assert bid[1, 1] == 7
    assert mh[1, 1] == [[0, 20.0]]


def test_empty_gdf_returns_untouched_grids():
    gdf = gpd.GeoDataFrame({"height": [], "geometry": []})
    h, mh, bid, _ = _run(gdf)
    assert h.shape == GRID and not h.any()
    assert all(mh.flat[k] == [] for k in range(mh.size))


def test_rotated_grid_frame():
    # 30-degree-rotated unit frame. A building exactly equal to cell (1,1)'s
    # parallelogram covers it 100% and touches no other cell above threshold.
    from voxcity.geoprocessor.raster.core import create_cell_polygon

    ang = np.deg2rad(30.0)
    u = np.array([np.sin(ang), np.cos(ang)])   # rotated "north"
    v = np.array([np.cos(ang), -np.sin(ang)])  # rotated "east"
    cell_11 = create_cell_polygon(ORIGIN, 1, 1, MESH, u, v)
    gdf = gpd.GeoDataFrame({
        "height": [15.0],
        "min_height": [0.0],
        "is_inner": [False],
        "id": [6],
        "geometry": [cell_11],
    })
    h, mh, bid, _ = _process(gdf, GRID, MESH, ORIGIN, u, v, None)
    assert h[1, 1] == 15.0
    assert bid[1, 1] == 6
    others = [(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]
    assert all(h[i, j] == 0.0 for (i, j) in others)


def test_backcompat_import_from_buildings():
    """The legacy import path must keep working (buildings re-exports)."""
    buildings_mod = pytest.importorskip("voxcity.geoprocessor.raster.buildings")
    assert buildings_mod._process_with_geometry_intersection is _process
    assert buildings_mod._CELL_INTERSECTION_THRESHOLD == pytest.approx(0.3)


def test_nan_height_candidate_does_not_perturb_ordering():
    """NaN height sorts lowest (like None), giving a deterministic order.

    Regression guard: the candidate list is a superset of strictly
    overlapping buildings, so a NaN-height building can share a cell's
    candidate list. If NaN leaked into the sort key it would make the
    per-cell processing order non-deterministic. Instead NaN is mapped to
    -inf, exactly like a None height in the legacy code: it sorts last in
    the height-descending order. Here real (h=10, id=1) and a NaN-height
    building (id=2) both fully cover cell (0,0). Deterministic outcome:
    real processed first, NaN last -> the NaN building wins the last-wins
    id, and the real height still wins the height grid.
    """
    gdf = gpd.GeoDataFrame({
        "height": [10.0, np.nan],
        "min_height": [0.0, 0.0],
        "is_inner": [False, False],
        "id": [1, 2],
        "geometry": [_rect(0, 0, 1, 1), _rect(0, 0, 1, 1)],
    })
    h, mh, bid, _ = _run(gdf)
    # Real height wins the height grid (NaN never sets it).
    assert h[0, 0] == 10.0
    # NaN sorts last -> processed last -> wins the id (same as None would).
    assert bid[0, 0] == 2
    # min-height list: real (h=10) appended first, NaN last.
    assert mh[0, 0][0] == [0.0, 10.0]
    assert mh[0, 0][1][0] == 0.0 and np.isnan(mh[0, 0][1][1])
