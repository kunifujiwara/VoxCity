"""_rotate_vertices must preserve geodesic side lengths.

Rotating in Web Mercator (the previous implementation) stretched/shrank
ground side lengths with latitude — ±2-3 m for a 900 m square at Tokyo, 45°.
"""
import pytest
from geopy import distance
from pyproj import Geod

from voxcity.geoprocessor.draw.rectangle import _rotate_vertices

geod = Geod(ellps="WGS84")
LAT_C, LON_C = 35.681236, 139.767125  # Tokyo


def _base_square(side_m=900.0):
    """Axis-aligned square built geodesically, like the notebook widgets do."""
    north = distance.distance(meters=side_m / 2).destination((LAT_C, LON_C), bearing=0)
    south = distance.distance(meters=side_m / 2).destination((LAT_C, LON_C), bearing=180)
    east = distance.distance(meters=side_m / 2).destination((LAT_C, LON_C), bearing=90)
    west = distance.distance(meters=side_m / 2).destination((LAT_C, LON_C), bearing=270)
    return [
        (west.longitude, south.latitude),
        (west.longitude, north.latitude),
        (east.longitude, north.latitude),
        (east.longitude, south.latitude),
    ]


def _side_lengths(v):
    _, _, d01 = geod.inv(v[0][0], v[0][1], v[1][0], v[1][1])
    _, _, d03 = geod.inv(v[0][0], v[0][1], v[3][0], v[3][1])
    return d01, d03


@pytest.mark.parametrize("angle", [15, 30, 45, 60, -45])
def test_rotation_preserves_side_lengths(angle):
    base = _base_square()
    d01_before, d03_before = _side_lengths(base)
    rotated = _rotate_vertices(base, angle)
    d01_after, d03_after = _side_lengths(rotated)
    assert d01_after == pytest.approx(d01_before, abs=1e-3)
    assert d03_after == pytest.approx(d03_before, abs=1e-3)


def test_zero_rotation_returns_copy():
    base = _base_square()
    assert _rotate_vertices(base, 0) == list(base)


def test_rotation_direction_unchanged():
    # Positive angle turns the v0→v1 edge to azimuth +angle (old convention).
    rotated = _rotate_vertices(_base_square(), 30)
    az01, _, _ = geod.inv(rotated[0][0], rotated[0][1], rotated[1][0], rotated[1][1])
    assert az01 == pytest.approx(30.0, abs=0.2)
