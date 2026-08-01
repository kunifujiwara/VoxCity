"""Tests for /api/rectangle-from-dimensions.

Extracted from ``test_generate_lod2.py``: that module's ``_dimension_rectangle``
helper drove this endpoint as a side effect of exercising the LOD2
axis-alignment guard. The guard is gone (VoxCityGML now supports rotated
rectangles), and deleting its tests would have left this endpoint with no
backend coverage at all — so the incidental coverage is made explicit here.

The property under test is the one the geodesic-construction fix established:
side lengths match the requested dimensions at *any* rotation and latitude.
Rotating in a projected CRS instead distorts ground lengths with latitude
(a 900 m side measured ~902 m at Tokyo at 45 deg).
"""
import pytest
from fastapi.testclient import TestClient
from pyproj import Geod

from backend.main import app

_GEOD = Geod(ellps="WGS84")


def _vertices(center_lon, center_lat, width_m, height_m, rotation_deg=0.0):
    resp = TestClient(app).post("/api/rectangle-from-dimensions", json={
        "center_lon": center_lon, "center_lat": center_lat,
        "width_m": width_m, "height_m": height_m,
        "rotation_deg": rotation_deg,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()["vertices"]


def _side_m(a, b):
    return _GEOD.inv(a[0], a[1], b[0], b[1])[2]


def test_returns_four_lonlat_vertices():
    verts = _vertices(139.77, 35.65, 1000, 800)
    assert len(verts) == 4
    assert all(len(v) == 2 for v in verts)


@pytest.mark.parametrize("center_lat", [24.0, 35.65, 45.52])
@pytest.mark.parametrize("rotation_deg", [0.0, 30.0, 45.0, -30.0, 90.0])
def test_side_lengths_match_requested_dimensions(center_lat, rotation_deg):
    """Vertices are [SW, NW, NE, SE], so SW->NW is the height side and
    NW->NE the width side. Both must hold at any rotation/latitude."""
    width_m, height_m = 900.0, 500.0
    sw, nw, ne, se = _vertices(139.77, center_lat, width_m, height_m,
                               rotation_deg)

    assert _side_m(sw, nw) == pytest.approx(height_m, abs=0.01)
    assert _side_m(se, ne) == pytest.approx(height_m, abs=0.01)
    assert _side_m(nw, ne) == pytest.approx(width_m, abs=0.01)
    assert _side_m(sw, se) == pytest.approx(width_m, abs=0.01)


def test_rotation_zero_is_essentially_axis_aligned():
    """At rotation 0 the sides run north/south and east/west. They are not
    *exactly* constant-lon/lat -- geodesic construction flares the north edge
    slightly wider in longitude -- so this asserts the bearing, not equality."""
    sw, nw, ne, _se = _vertices(139.77, 35.65, 1000, 1000, 0.0)
    assert _GEOD.inv(sw[0], sw[1], nw[0], nw[1])[0] == pytest.approx(0.0, abs=0.2)
    assert _GEOD.inv(nw[0], nw[1], ne[0], ne[1])[0] == pytest.approx(90.0, abs=0.2)


def test_positive_rotation_turns_the_first_edge_to_that_azimuth():
    """Pins the documented convention: positive rotation_deg turns the v0->v1
    (SW->NW) edge to azimuth +rotation_deg."""
    sw, nw, *_ = _vertices(139.77, 35.65, 1000, 1000, 30.0)
    assert _GEOD.inv(sw[0], sw[1], nw[0], nw[1])[0] == pytest.approx(30.0, abs=0.2)
