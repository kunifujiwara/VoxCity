"""
Tests for voxcity.simulator.common.geometry functions.
"""

import numpy as np
import pytest

from voxcity.simulator.common.geometry import (
    _generate_ray_directions_grid,
    _generate_ray_directions_fibonacci,
    rotate_vector_axis_angle,
    _build_face_basis,
)


class TestGenerateRayDirectionsGrid:
    def test_basic_shape(self):
        dirs = _generate_ray_directions_grid(8, 4, 0.0, 90.0)
        assert dirs.shape == (32, 3)  # 8 * 4

    def test_unit_vectors(self):
        dirs = _generate_ray_directions_grid(10, 5, 0.0, 90.0)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_single_azimuth(self):
        dirs = _generate_ray_directions_grid(1, 1, 45.0, 45.0)
        assert dirs.shape == (1, 3)
        assert dirs[0, 2] > 0  # z should be positive for 45 degrees elevation

    def test_zenith_elevation(self):
        dirs = _generate_ray_directions_grid(4, 1, 90.0, 90.0)
        # At 90 degrees, z should be 1
        for d in dirs:
            assert d[2] == pytest.approx(1.0, abs=1e-10)

    def test_horizon_elevation(self):
        dirs = _generate_ray_directions_grid(4, 1, 0.0, 0.0)
        for d in dirs:
            assert d[2] == pytest.approx(0.0, abs=1e-10)

    def test_increasing_azimuth_coverage(self):
        dirs_low = _generate_ray_directions_grid(4, 2, 0.0, 45.0)
        dirs_high = _generate_ray_directions_grid(16, 2, 0.0, 45.0)
        assert dirs_high.shape[0] > dirs_low.shape[0]


class TestGenerateRayDirectionsFibonacci:
    def test_basic_shape(self):
        dirs = _generate_ray_directions_fibonacci(100, 0.0, 90.0)
        assert dirs.shape == (100, 3)

    def test_unit_vectors(self):
        dirs = _generate_ray_directions_fibonacci(50, 0.0, 90.0)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_hemisphere(self):
        dirs = _generate_ray_directions_fibonacci(200, 0.0, 90.0)
        # All z values should be positive (upper hemisphere)
        assert np.all(dirs[:, 2] >= 0)

    def test_narrow_band(self):
        dirs = _generate_ray_directions_fibonacci(20, 30.0, 60.0)
        # z should be between sin(30) and sin(60)
        assert np.all(dirs[:, 2] >= np.sin(np.deg2rad(30)) - 0.1)
        assert np.all(dirs[:, 2] <= np.sin(np.deg2rad(60)) + 0.1)

    def test_single_ray(self):
        dirs = _generate_ray_directions_fibonacci(1, 0.0, 90.0)
        assert dirs.shape == (1, 3)

    def test_zero_rays(self):
        dirs = _generate_ray_directions_fibonacci(0, 0.0, 90.0)
        assert dirs.shape[0] == 1  # max(1, 0) = 1


class TestRotateVectorAxisAngle:
    def test_zero_angle(self):
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, 0.0)
        np.testing.assert_allclose(result, vec, atol=1e-10)

    def test_90_degrees_z(self):
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, np.pi / 2)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_180_degrees(self):
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, np.pi)
        expected = np.array([-1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_zero_axis(self):
        vec = np.array([1.0, 2.0, 3.0])
        axis = np.array([0.0, 0.0, 0.0])
        result = rotate_vector_axis_angle(vec, axis, np.pi / 4)
        np.testing.assert_allclose(result, vec, atol=1e-10)

    def test_rotation_preserves_length(self):
        vec = np.array([3.0, 4.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, 1.234)
        np.testing.assert_allclose(np.linalg.norm(result), np.linalg.norm(vec), atol=1e-10)

    def test_rotate_around_x(self):
        vec = np.array([0.0, 1.0, 0.0])
        axis = np.array([1.0, 0.0, 0.0])
        result = rotate_vector_axis_angle(vec, axis, np.pi / 2)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestBuildFaceBasis:
    def test_z_normal(self):
        normal = np.array([0.0, 0.0, 1.0])
        u, v, n = _build_face_basis(normal)
        # n should be the same as normal
        np.testing.assert_allclose(n, normal, atol=1e-10)
        # u, v, n should be orthogonal
        assert abs(np.dot(u, v)) < 1e-10
        assert abs(np.dot(u, n)) < 1e-10
        assert abs(np.dot(v, n)) < 1e-10

    def test_x_normal(self):
        normal = np.array([1.0, 0.0, 0.0])
        u, v, n = _build_face_basis(normal)
        assert abs(np.dot(u, v)) < 1e-10
        assert abs(np.dot(u, n)) < 1e-10

    def test_arbitrary_normal(self):
        normal = np.array([1.0, 2.0, 3.0])
        u, v, n = _build_face_basis(normal)
        # n should be normalized
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-10)
        assert np.linalg.norm(u) == pytest.approx(1.0, abs=1e-10)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-10)

    def test_zero_normal(self):
        normal = np.array([0.0, 0.0, 0.0])
        u, v, n = _build_face_basis(normal)
        # Should return default basis
        assert np.linalg.norm(u) > 0
        assert np.linalg.norm(v) > 0

    def test_negative_normal(self):
        normal = np.array([0.0, 0.0, -1.0])
        u, v, n = _build_face_basis(normal)
        np.testing.assert_allclose(n, np.array([0.0, 0.0, -1.0]), atol=1e-10)
        assert abs(np.dot(u, v)) < 1e-10
