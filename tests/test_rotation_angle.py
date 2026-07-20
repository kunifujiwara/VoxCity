"""Tests for geoprocessor.utils.compute_rotation_angle (sign convention pinned)."""

import math

import pytest

from voxcity.geoprocessor.utils import compute_rotation_angle


def _rotated_rect(angle_deg, size_deg=0.01, origin=(0.0, 0.0)):
    """Rectangle whose v0->v1 (SW->NW) edge bears `angle_deg` clockwise from north.

    Built near the equator where Web Mercator x/y are proportional to
    lon/lat, so the bearing in lon/lat equals the Mercator bearing.
    """
    a = math.radians(angle_deg)
    ox, oy = origin
    # v0->v1: length size_deg at bearing angle_deg (clockwise from north)
    d1 = (size_deg * math.sin(a), size_deg * math.cos(a))
    # v0->v3: perpendicular, 90 deg further clockwise
    d2 = (size_deg * math.cos(a), -size_deg * math.sin(a))
    v0 = (ox, oy)
    v1 = (ox + d1[0], oy + d1[1])
    v3 = (ox + d2[0], oy + d2[1])
    v2 = (ox + d1[0] + d2[0], oy + d1[1] + d2[1])
    return [v0, v1, v2, v3]


class TestComputeRotationAngle:
    def test_axis_aligned_is_zero(self):
        rect = [(0.0, 0.0), (0.0, 0.01), (0.01, 0.01), (0.01, 0.0)]
        assert compute_rotation_angle(rect) == 0

    def test_clockwise_30_is_positive_30(self):
        # Sign convention pin: NW corner displaced toward the east
        # (clockwise rotation) must give +30.
        assert compute_rotation_angle(_rotated_rect(30.0)) == pytest.approx(30.0, abs=1e-3)

    def test_counterclockwise_30_is_negative_30(self):
        assert compute_rotation_angle(_rotated_rect(-30.0)) == pytest.approx(-30.0, abs=1e-3)

    def test_none_and_short_input_return_zero(self):
        assert compute_rotation_angle(None) == 0
        assert compute_rotation_angle([(0.0, 0.0)]) == 0

    def test_generator_api_still_exposes_it(self):
        # generator/api.py must keep working after the move.
        from voxcity.generator.api import _compute_rotation_angle
        assert _compute_rotation_angle(_rotated_rect(10.0)) == pytest.approx(10.0, abs=1e-3)
