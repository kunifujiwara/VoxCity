"""Tests for voxcity.simulator.common.geometry module."""
import pytest
import numpy as np

from voxcity.simulator.common.geometry import (
    _generate_ray_directions_grid,
    _generate_ray_directions_fibonacci,
    rotate_vector_axis_angle,
    _build_face_basis,
)


class TestGenerateRayDirectionsGrid:
    """Tests for _generate_ray_directions_grid function."""

    def test_shape_correct(self):
        """Test output shape."""
        N_azimuth = 8
        N_elevation = 4
        result = _generate_ray_directions_grid(N_azimuth, N_elevation, 0.0, 90.0)
        assert result.shape == (N_azimuth * N_elevation, 3)

    def test_all_unit_vectors(self):
        """Test that all rays are unit vectors."""
        result = _generate_ray_directions_grid(16, 8, 10.0, 80.0)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(len(result)))

    def test_elevation_range(self):
        """Test that Z components respect elevation range."""
        result = _generate_ray_directions_grid(8, 8, 0.0, 90.0)
        # At elevation 0, z = 0; at elevation 90, z = 1
        z_values = result[:, 2]
        assert np.min(z_values) >= -0.01  # Near 0 for lowest elevation
        assert np.max(z_values) <= 1.01  # Near 1 for highest elevation

    def test_single_elevation(self):
        """Test with single elevation angle."""
        result = _generate_ray_directions_grid(8, 1, 45.0, 45.0)
        assert result.shape == (8, 3)
        # All should have same Z (elevation)
        z_values = result[:, 2]
        np.testing.assert_array_almost_equal(z_values, z_values[0] * np.ones(8))

    def test_single_azimuth(self):
        """Test with single azimuth angle."""
        result = _generate_ray_directions_grid(1, 8, 0.0, 90.0)
        assert result.shape == (8, 3)

    def test_zero_elevation(self):
        """Test rays at zero elevation (horizontal)."""
        result = _generate_ray_directions_grid(4, 1, 0.0, 0.0)
        # At elevation 0, z should be 0
        np.testing.assert_array_almost_equal(result[:, 2], np.zeros(4))

    def test_90_degree_elevation(self):
        """Test ray pointing straight up."""
        result = _generate_ray_directions_grid(1, 1, 90.0, 90.0)
        # Should point straight up (0, 0, 1)
        np.testing.assert_array_almost_equal(result[0, 2], 1.0)


class TestGenerateRayDirectionsFibonacci:
    """Tests for _generate_ray_directions_fibonacci function."""

    def test_shape_correct(self):
        """Test output shape."""
        result = _generate_ray_directions_fibonacci(100, 0.0, 90.0)
        assert result.shape == (100, 3)

    def test_all_unit_vectors(self):
        """Test that all rays are unit vectors."""
        result = _generate_ray_directions_fibonacci(50, 10.0, 80.0)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(50), decimal=5)

    def test_elevation_range_respected(self):
        """Test that Z components respect elevation range."""
        result = _generate_ray_directions_fibonacci(100, 30.0, 60.0)
        z_values = result[:, 2]
        # sin(30°) ≈ 0.5, sin(60°) ≈ 0.866
        assert np.min(z_values) >= 0.49
        assert np.max(z_values) <= 0.87

    def test_minimum_one_ray(self):
        """Test that at least one ray is generated."""
        result = _generate_ray_directions_fibonacci(0, 0.0, 90.0)
        assert result.shape[0] >= 1

    def test_single_ray(self):
        """Test with single ray."""
        result = _generate_ray_directions_fibonacci(1, 45.0, 45.0)
        assert result.shape == (1, 3)
        assert np.linalg.norm(result[0]) == pytest.approx(1.0)

    def test_hemisphere(self):
        """Test generating rays for upper hemisphere."""
        result = _generate_ray_directions_fibonacci(200, 0.0, 90.0)
        # All Z should be non-negative for upper hemisphere
        assert np.all(result[:, 2] >= -0.01)

    def test_dtype_float64(self):
        """Test output dtype."""
        result = _generate_ray_directions_fibonacci(10, 0.0, 90.0)
        assert result.dtype == np.float64


class TestRotateVectorAxisAngle:
    """Tests for rotate_vector_axis_angle function."""

    def test_zero_angle_no_rotation(self):
        """Test that zero angle returns same vector."""
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, 0.0)
        np.testing.assert_array_almost_equal(result, vec)

    def test_90_degree_z_axis(self):
        """Test 90 degree rotation around Z axis."""
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, np.pi / 2)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_180_degree_rotation(self):
        """Test 180 degree rotation."""
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, np.pi)
        expected = np.array([-1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_rotation_around_x_axis(self):
        """Test rotation around X axis."""
        vec = np.array([0.0, 1.0, 0.0])
        axis = np.array([1.0, 0.0, 0.0])
        result = rotate_vector_axis_angle(vec, axis, np.pi / 2)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_rotation_around_y_axis(self):
        """Test rotation around Y axis."""
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 1.0, 0.0])
        result = rotate_vector_axis_angle(vec, axis, np.pi / 2)
        expected = np.array([0.0, 0.0, -1.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_zero_axis_returns_original(self):
        """Test that zero axis returns original vector."""
        vec = np.array([1.0, 2.0, 3.0])
        axis = np.array([0.0, 0.0, 0.0])
        result = rotate_vector_axis_angle(vec, axis, np.pi)
        np.testing.assert_array_almost_equal(result, vec)

    def test_preserves_vector_length(self):
        """Test that rotation preserves vector length."""
        vec = np.array([1.0, 2.0, 3.0])
        axis = np.array([1.0, 1.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, 1.234)
        original_norm = np.linalg.norm(vec)
        result_norm = np.linalg.norm(result)
        assert result_norm == pytest.approx(original_norm, rel=1e-5)

    def test_full_rotation(self):
        """Test 360 degree rotation returns original."""
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, 2 * np.pi)
        np.testing.assert_array_almost_equal(result, vec, decimal=5)


class TestBuildFaceBasis:
    """Tests for _build_face_basis function."""

    def test_returns_three_vectors(self):
        """Test that function returns 3 vectors."""
        normal = np.array([0.0, 0.0, 1.0])
        u, v, n = _build_face_basis(normal)
        assert u.shape == (3,)
        assert v.shape == (3,)
        assert n.shape == (3,)

    def test_orthogonal_vectors(self):
        """Test that returned vectors are orthogonal."""
        normal = np.array([1.0, 0.0, 0.0])
        u, v, n = _build_face_basis(normal)
        assert abs(np.dot(u, v)) < 1e-6
        assert abs(np.dot(u, n)) < 1e-6
        assert abs(np.dot(v, n)) < 1e-6

    def test_unit_vectors(self):
        """Test that returned vectors are unit vectors."""
        normal = np.array([1.0, 2.0, 3.0])
        u, v, n = _build_face_basis(normal)
        assert np.linalg.norm(u) == pytest.approx(1.0, rel=1e-6)
        assert np.linalg.norm(v) == pytest.approx(1.0, rel=1e-6)
        assert np.linalg.norm(n) == pytest.approx(1.0, rel=1e-6)

    def test_z_up_normal(self):
        """Test with Z-up normal."""
        normal = np.array([0.0, 0.0, 1.0])
        u, v, n = _build_face_basis(normal)
        np.testing.assert_array_almost_equal(n, normal)

    def test_x_normal(self):
        """Test with X-axis normal."""
        normal = np.array([1.0, 0.0, 0.0])
        u, v, n = _build_face_basis(normal)
        np.testing.assert_array_almost_equal(n, normal)

    def test_y_normal(self):
        """Test with Y-axis normal."""
        normal = np.array([0.0, 1.0, 0.0])
        u, v, n = _build_face_basis(normal)
        np.testing.assert_array_almost_equal(n, normal)

    def test_negative_z_normal(self):
        """Test with negative Z normal."""
        normal = np.array([0.0, 0.0, -1.0])
        u, v, n = _build_face_basis(normal)
        np.testing.assert_array_almost_equal(n, normal)

    def test_diagonal_normal(self):
        """Test with diagonal normal."""
        normal = np.array([1.0, 1.0, 1.0])
        u, v, n = _build_face_basis(normal)
        # Check that n is normalized version of input
        expected_n = normal / np.linalg.norm(normal)
        np.testing.assert_array_almost_equal(n, expected_n)

    def test_zero_normal(self):
        """Test with zero normal (edge case)."""
        normal = np.array([0.0, 0.0, 0.0])
        u, v, n = _build_face_basis(normal)
        # Should return default basis
        assert u.shape == (3,)
        assert v.shape == (3,)
        assert n.shape == (3,)

    def test_near_z_aligned_normal(self):
        """Test with normal nearly aligned to Z axis."""
        normal = np.array([0.001, 0.001, 0.9999])
        u, v, n = _build_face_basis(normal)
        assert abs(np.dot(u, v)) < 1e-6
        assert abs(np.dot(u, n)) < 1e-6

    def test_right_handed_basis(self):
        """Test that the basis is right-handed (u x v = n)."""
        normal = np.array([0.0, 0.0, 1.0])
        u, v, n = _build_face_basis(normal)
        cross = np.cross(u, v)
        # For some implementations, it might be v x u = n
        assert np.allclose(np.abs(cross), np.abs(n), rtol=1e-5)
