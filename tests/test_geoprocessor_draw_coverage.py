"""
Tests for voxcity.geoprocessor.draw - rotate_rectangle function.
"""

import math
import pytest
from unittest.mock import MagicMock

from voxcity.geoprocessor.draw import rotate_rectangle


class TestRotateRectangle:
    def _make_mock_map(self):
        m = MagicMock()
        m.add_layer = MagicMock()
        return m

    def test_no_vertices(self):
        m = self._make_mock_map()
        result = rotate_rectangle(m, [], 45)
        assert result is None

    def test_zero_rotation(self):
        m = self._make_mock_map()
        verts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        result = rotate_rectangle(m, verts, 0)
        assert result is not None
        assert len(result) == 4
        # Zero rotation should return nearly the same vertices
        for orig, rotated in zip(verts, result):
            assert orig[0] == pytest.approx(rotated[0], abs=1e-6)
            assert orig[1] == pytest.approx(rotated[1], abs=1e-6)

    def test_360_rotation(self):
        m = self._make_mock_map()
        verts = [(139.75, 35.67), (139.76, 35.67), (139.76, 35.68), (139.75, 35.68)]
        result = rotate_rectangle(m, verts, 360)
        assert result is not None
        for orig, rotated in zip(verts, result):
            assert orig[0] == pytest.approx(rotated[0], abs=1e-4)
            assert orig[1] == pytest.approx(rotated[1], abs=1e-4)

    def test_90_rotation_preserves_shape(self):
        m = self._make_mock_map()
        verts = [(0.0, 0.0), (0.01, 0.0), (0.01, 0.01), (0.0, 0.01)]
        result = rotate_rectangle(m, verts, 90)
        assert result is not None
        assert len(result) == 4
        # Centroid should remain roughly the same
        orig_cx = sum(v[0] for v in verts) / 4
        orig_cy = sum(v[1] for v in verts) / 4
        rot_cx = sum(v[0] for v in result) / 4
        rot_cy = sum(v[1] for v in result) / 4
        assert orig_cx == pytest.approx(rot_cx, abs=1e-6)
        assert orig_cy == pytest.approx(rot_cy, abs=1e-6)

    def test_adds_layer_to_map(self):
        m = self._make_mock_map()
        verts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        rotate_rectangle(m, verts, 45)
        m.add_layer.assert_called_once()

    def test_negative_rotation(self):
        m = self._make_mock_map()
        verts = [(0.0, 0.0), (0.01, 0.0), (0.01, 0.01), (0.0, 0.01)]
        result = rotate_rectangle(m, verts, -45)
        assert result is not None
        assert len(result) == 4
