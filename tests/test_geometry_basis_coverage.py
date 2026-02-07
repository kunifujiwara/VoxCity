"""
Tests for geometry.py remaining coverage:
  - _build_face_basis (lines 70-96)
  - rotate_vector_axis_angle (lines 44-65)
"""
import numpy as np
import pytest
from voxcity.simulator.common.geometry import (
    _build_face_basis,
    rotate_vector_axis_angle,
)


class TestBuildFaceBasis:

    def test_upward_normal(self):
        """Normal pointing straight up -> u,v in XY plane."""
        u, v, n = _build_face_basis(np.array([0.0, 0.0, 1.0]))
        # n should equal input (normalized)
        np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-12)
        # u, v should be orthogonal to n
        assert abs(np.dot(u, n)) < 1e-10
        assert abs(np.dot(v, n)) < 1e-10
        # u, v orthogonal to each other
        assert abs(np.dot(u, v)) < 1e-10

    def test_sideways_normal(self):
        """Normal pointing along X axis."""
        u, v, n = _build_face_basis(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(n, [1.0, 0.0, 0.0], atol=1e-12)
        assert abs(np.dot(u, n)) < 1e-10
        assert abs(np.dot(v, n)) < 1e-10
        assert abs(np.dot(u, v)) < 1e-10

    def test_diagonal_normal(self):
        """Non-axis-aligned normal."""
        raw = np.array([1.0, 1.0, 1.0])
        u, v, n = _build_face_basis(raw)
        expected_n = raw / np.linalg.norm(raw)
        np.testing.assert_allclose(n, expected_n, atol=1e-12)
        assert abs(np.dot(u, n)) < 1e-10
        assert abs(np.dot(v, n)) < 1e-10

    def test_unit_vectors(self):
        """All returned vectors should be unit length."""
        u, v, n = _build_face_basis(np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-12)

    def test_zero_normal(self):
        """Zero-length normal -> fallback identity basis."""
        u, v, n = _build_face_basis(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(u, [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-12)

    def test_nearly_z_aligned(self):
        """Normal very close to Z triggers helper swap (abs(nz) >= 0.999)."""
        u, v, n = _build_face_basis(np.array([0.0, 0.001, 1.0]))
        assert abs(np.dot(u, n)) < 1e-8
        assert abs(np.dot(v, n)) < 1e-8

    def test_negative_normal(self):
        """Negative normal direction."""
        u, v, n = _build_face_basis(np.array([0.0, 0.0, -2.0]))
        np.testing.assert_allclose(n, [0.0, 0.0, -1.0], atol=1e-12)
        assert abs(np.dot(u, n)) < 1e-10


class TestRotateVectorAxisAngle:

    def test_zero_angle(self):
        """No rotation -> same vector."""
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, 0.0)
        np.testing.assert_allclose(result, vec, atol=1e-12)

    def test_90_degrees_around_z(self):
        """Rotate (1,0,0) 90° around Z -> (0,1,0)."""
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, np.pi / 2)
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-10)

    def test_180_degrees(self):
        """Rotate (1,0,0) 180° around Z -> (-1,0,0)."""
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, np.pi)
        np.testing.assert_allclose(result, [-1.0, 0.0, 0.0], atol=1e-10)

    def test_rotation_around_own_axis(self):
        """Rotating a vector around itself -> same vector."""
        vec = np.array([1.0, 0.0, 0.0])
        result = rotate_vector_axis_angle(vec, vec, np.pi / 3)
        np.testing.assert_allclose(result, vec, atol=1e-10)

    def test_zero_axis(self):
        """Zero-length axis -> returns original vector."""
        vec = np.array([1.0, 2.0, 3.0])
        axis = np.array([0.0, 0.0, 0.0])
        result = rotate_vector_axis_angle(vec, axis, np.pi / 4)
        np.testing.assert_allclose(result, vec, atol=1e-12)

    def test_preserves_length(self):
        """Rotation preserves vector magnitude."""
        vec = np.array([3.0, 4.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, 1.23)
        np.testing.assert_allclose(np.linalg.norm(result), np.linalg.norm(vec), atol=1e-10)

    def test_non_unit_axis(self):
        """Non-unit axis should be normalized internally."""
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 5.0])  # 5× Z
        result = rotate_vector_axis_angle(vec, axis, np.pi / 2)
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-10)
