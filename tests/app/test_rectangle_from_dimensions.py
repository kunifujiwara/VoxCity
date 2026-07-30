"""/api/rectangle-from-dimensions must return geodesically exact rectangles.

The previous implementation rotated the rectangle in Web Mercator, which
distorts ground side lengths with latitude (~±2-3 m at Tokyo for 45°), so a
900×900 m request produced a 451×449 grid at meshsize 2 instead of 450×450.
"""
import pytest
from fastapi.testclient import TestClient
from pyproj import Geod

import app.backend.main as main

geod = Geod(ellps="WGS84")
TOKYO = {"center_lon": 139.767125, "center_lat": 35.681236}


def _vertices(rot):
    client = TestClient(main.app)
    r = client.post("/api/rectangle-from-dimensions", json={
        **TOKYO, "width_m": 900.0, "height_m": 900.0, "rotation_deg": rot,
    })
    assert r.status_code == 200, r.text
    v = r.json()["vertices"]
    assert len(v) == 4
    return v


def _side_lengths(v):
    _, _, d01 = geod.inv(v[0][0], v[0][1], v[1][0], v[1][1])
    _, _, d03 = geod.inv(v[0][0], v[0][1], v[3][0], v[3][1])
    return d01, d03


@pytest.mark.parametrize("rot", [0, 15, 30, 45, 60, -45])
def test_side_lengths_exact_at_any_rotation(rot):
    d01, d03 = _side_lengths(_vertices(rot))
    assert d01 == pytest.approx(900.0, abs=1e-3)
    assert d03 == pytest.approx(900.0, abs=1e-3)


@pytest.mark.parametrize("rot", [0, 45, 60])
def test_grid_size_is_450x450_at_meshsize_2(rot):
    # Mirrors calculate_grid_size() rounding in voxcity/geoprocessor/raster/core.py
    d01, d03 = _side_lengths(_vertices(rot))
    assert int(d01 / 2 + 0.5) == 450
    assert int(d03 / 2 + 0.5) == 450


def test_rotation_convention_matches_old_mercator_direction():
    # Positive rotation turns the v0→v1 (SW→NW) edge to azimuth +rot —
    # the same visual direction the Mercator implementation produced.
    v = _vertices(30)
    az01, _, _ = geod.inv(v[0][0], v[0][1], v[1][0], v[1][1])
    assert az01 == pytest.approx(30.0, abs=0.2)


def test_vertex_order_sw_nw_ne_se_when_unrotated():
    sw, nw, ne, se = _vertices(0)
    assert sw[1] < nw[1] and se[1] < ne[1]  # north rows above south rows
    assert sw[0] < se[0] and nw[0] < ne[0]  # east columns right of west
