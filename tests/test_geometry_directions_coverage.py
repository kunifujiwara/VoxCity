"""Tests for simulator.common.geometry: ray direction generators and axis-angle rotation."""
import numpy as np
import pytest

from voxcity.simulator.common.geometry import (
    _generate_ray_directions_grid,
    _generate_ray_directions_fibonacci,
    rotate_vector_axis_angle,
    _build_face_basis,
)


# ---------------------------------------------------------------------------
# _generate_ray_directions_grid
# ---------------------------------------------------------------------------
class TestGenerateRayDirectionsGrid:
    def test_output_shape(self):
        dirs = _generate_ray_directions_grid(N_azimuth=8, N_elevation=4, elevation_min_degrees=0, elevation_max_degrees=90)
        assert dirs.shape == (8 * 4, 3)

    def test_unit_vectors(self):
        dirs = _generate_ray_directions_grid(N_azimuth=12, N_elevation=6, elevation_min_degrees=0, elevation_max_degrees=90)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_hemisphere_only(self):
        dirs = _generate_ray_directions_grid(N_azimuth=8, N_elevation=5, elevation_min_degrees=0, elevation_max_degrees=90)
        assert np.all(dirs[:, 2] >= -1e-12)

    def test_full_sphere(self):
        dirs = _generate_ray_directions_grid(N_azimuth=8, N_elevation=5, elevation_min_degrees=-90, elevation_max_degrees=90)
        assert np.any(dirs[:, 2] < 0)
        assert np.any(dirs[:, 2] > 0)

    def test_single_direction(self):
        dirs = _generate_ray_directions_grid(N_azimuth=1, N_elevation=1, elevation_min_degrees=45, elevation_max_degrees=45)
        assert dirs.shape == (1, 3)
        np.testing.assert_allclose(dirs[0, 2], np.sin(np.deg2rad(45)), atol=1e-10)

    def test_azimuth_symmetry(self):
        dirs = _generate_ray_directions_grid(N_azimuth=4, N_elevation=1, elevation_min_degrees=30, elevation_max_degrees=30)
        # 4 azimuth at 0, 90, 180, 270 → z components should all be equal
        np.testing.assert_allclose(dirs[:, 2], dirs[0, 2], atol=1e-10)


# ---------------------------------------------------------------------------
# _generate_ray_directions_fibonacci
# ---------------------------------------------------------------------------
class TestGenerateRayDirectionsFibonacci:
    def test_output_shape(self):
        dirs = _generate_ray_directions_fibonacci(N_rays=100, elevation_min_degrees=0, elevation_max_degrees=90)
        assert dirs.shape == (100, 3)

    def test_unit_vectors(self):
        dirs = _generate_ray_directions_fibonacci(N_rays=200, elevation_min_degrees=0, elevation_max_degrees=90)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_hemisphere(self):
        dirs = _generate_ray_directions_fibonacci(N_rays=200, elevation_min_degrees=0, elevation_max_degrees=90)
        assert np.all(dirs[:, 2] >= -0.01)

    def test_min_one_ray(self):
        dirs = _generate_ray_directions_fibonacci(N_rays=0, elevation_min_degrees=0, elevation_max_degrees=90)
        assert dirs.shape[0] == 1

    def test_elevation_range_reversed(self):
        dirs = _generate_ray_directions_fibonacci(N_rays=50, elevation_min_degrees=60, elevation_max_degrees=30)
        # function handles min/max swap internally
        z_vals = dirs[:, 2]
        assert np.all(z_vals >= np.sin(np.deg2rad(30)) - 0.05)

    def test_dtype(self):
        dirs = _generate_ray_directions_fibonacci(N_rays=10, elevation_min_degrees=0, elevation_max_degrees=90)
        assert dirs.dtype == np.float64


# ---------------------------------------------------------------------------
# rotate_vector_axis_angle
# ---------------------------------------------------------------------------
class TestRotateVectorAxisAngle:
    def test_identity_rotation(self):
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, 0.0)
        np.testing.assert_allclose(result, vec, atol=1e-10)

    def test_90_degree_z_axis(self):
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, np.pi / 2)
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-10)

    def test_180_degree(self):
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, np.pi)
        np.testing.assert_allclose(result, [-1.0, 0.0, 0.0], atol=1e-10)

    def test_zero_axis_returns_original(self):
        vec = np.array([1.0, 2.0, 3.0])
        axis = np.array([0.0, 0.0, 0.0])
        result = rotate_vector_axis_angle(vec, axis, np.pi / 4)
        np.testing.assert_allclose(result, vec, atol=1e-10)

    def test_rotation_around_self(self):
        vec = np.array([0.0, 0.0, 1.0])
        axis = np.array([0.0, 0.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, np.pi / 3)
        np.testing.assert_allclose(result, vec, atol=1e-10)

    def test_preserves_length(self):
        vec = np.array([3.0, 4.0, 0.0])
        axis = np.array([1.0, 1.0, 1.0])
        result = rotate_vector_axis_angle(vec, axis, 1.23)
        np.testing.assert_allclose(np.linalg.norm(result), np.linalg.norm(vec), atol=1e-10)


# ---------------------------------------------------------------------------
# _build_face_basis (already tested in another file, add edge cases)
# ---------------------------------------------------------------------------
class TestBuildFaceBasisEdgeCases:
    def test_z_aligned_normal(self):
        """When normal is along z, helper should switch to (1,0,0)."""
        u, v, n = _build_face_basis(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(np.dot(u, n), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.dot(v, n), 0.0, atol=1e-10)

    def test_x_aligned_normal(self):
        u, v, n = _build_face_basis(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(np.dot(u, n), 0.0, atol=1e-10)

    def test_zero_normal(self):
        u, v, n = _build_face_basis(np.array([0.0, 0.0, 0.0]))
        # Should return default basis
        np.testing.assert_allclose(u, [1, 0, 0], atol=1e-10)
        np.testing.assert_allclose(v, [0, 1, 0], atol=1e-10)
        np.testing.assert_allclose(n, [0, 0, 1], atol=1e-10)

    def test_unnormalized_input(self):
        u, v, n = _build_face_basis(np.array([0.0, 0.0, 5.0]))
        np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-10)
