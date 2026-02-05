"""Tests for voxcity.simulator.common modules."""
import pytest
import numpy as np

from voxcity.simulator.common.raytracing import (
    calculate_transmittance,
    trace_ray_generic,
)
from voxcity.simulator.common.geometry import (
    _generate_ray_directions_grid,
    _generate_ray_directions_fibonacci,
    _build_face_basis,
    rotate_vector_axis_angle,
)


class TestCalculateTransmittance:
    def test_zero_length_full_transmittance(self):
        """Zero path length should give full transmittance (1.0)."""
        result = calculate_transmittance(0.0, tree_k=0.6, tree_lad=1.0)
        assert result == pytest.approx(1.0)

    def test_positive_length_reduces_transmittance(self):
        """Positive path length should reduce transmittance."""
        t0 = calculate_transmittance(0.0, tree_k=0.6, tree_lad=1.0)
        t1 = calculate_transmittance(1.0, tree_k=0.6, tree_lad=1.0)
        t2 = calculate_transmittance(2.0, tree_k=0.6, tree_lad=1.0)
        
        assert t1 < t0
        assert t2 < t1

    def test_monotonic_decrease(self):
        """Transmittance should monotonically decrease with length."""
        lengths = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        transmittances = [calculate_transmittance(l, tree_k=0.6, tree_lad=1.0) for l in lengths]
        
        for i in range(len(transmittances) - 1):
            assert transmittances[i] >= transmittances[i + 1]

    def test_bounded_zero_to_one(self):
        """Transmittance should always be between 0 and 1."""
        for length in [0.0, 0.1, 1.0, 10.0, 100.0]:
            t = calculate_transmittance(length, tree_k=0.6, tree_lad=1.0)
            assert 0.0 <= t <= 1.0

    def test_higher_k_lower_transmittance(self):
        """Higher extinction coefficient should give lower transmittance."""
        t_low_k = calculate_transmittance(1.0, tree_k=0.3, tree_lad=1.0)
        t_high_k = calculate_transmittance(1.0, tree_k=0.9, tree_lad=1.0)
        
        assert t_high_k < t_low_k

    def test_higher_lad_lower_transmittance(self):
        """Higher leaf area density should give lower transmittance."""
        t_low_lad = calculate_transmittance(1.0, tree_k=0.6, tree_lad=0.5)
        t_high_lad = calculate_transmittance(1.0, tree_k=0.6, tree_lad=2.0)
        
        assert t_high_lad < t_low_lad

    def test_beer_lambert_formula(self):
        """Should follow Beer-Lambert law: T = exp(-k * LAD * L)."""
        length = 2.0
        tree_k = 0.6
        tree_lad = 1.0
        
        expected = np.exp(-tree_k * tree_lad * length)
        result = calculate_transmittance(length, tree_k, tree_lad)
        
        assert result == pytest.approx(expected)


class TestGenerateRayDirectionsGrid:
    def test_output_shape(self):
        """Should generate correct number of rays."""
        N_azimuth = 8
        N_elevation = 4
        
        rays = _generate_ray_directions_grid(N_azimuth, N_elevation, 0.0, 90.0)
        
        assert rays.shape == (N_azimuth * N_elevation, 3)

    def test_unit_vectors(self):
        """All ray directions should be unit vectors."""
        rays = _generate_ray_directions_grid(8, 4, 0.0, 90.0)
        
        magnitudes = np.linalg.norm(rays, axis=1)
        np.testing.assert_allclose(magnitudes, 1.0, rtol=1e-6)

    def test_elevation_bounds(self):
        """Ray z-components should respect elevation bounds."""
        rays = _generate_ray_directions_grid(8, 4, 0.0, 45.0)
        
        # Elevation 0 gives z=0, elevation 45 gives z ≈ 0.707
        assert rays[:, 2].min() >= -0.01
        assert rays[:, 2].max() <= np.sin(np.deg2rad(45.0)) + 0.01

    def test_full_azimuth_coverage(self):
        """Should cover full 360 degrees azimuth."""
        rays = _generate_ray_directions_grid(16, 1, 45.0, 45.0)
        
        # At constant elevation, should have rays pointing in all directions
        # Check that we have positive and negative x and y components
        assert rays[:, 0].max() > 0
        assert rays[:, 0].min() < 0
        assert rays[:, 1].max() > 0
        assert rays[:, 1].min() < 0


class TestGenerateRayDirectionsFibonacci:
    def test_output_shape(self):
        """Should generate requested number of rays."""
        N_rays = 100
        rays = _generate_ray_directions_fibonacci(N_rays, 0.0, 90.0)
        
        assert rays.shape == (N_rays, 3)

    def test_unit_vectors(self):
        """All ray directions should be unit vectors."""
        rays = _generate_ray_directions_fibonacci(100, 0.0, 90.0)
        
        magnitudes = np.linalg.norm(rays, axis=1)
        np.testing.assert_allclose(magnitudes, 1.0, rtol=1e-6)

    def test_elevation_bounds(self):
        """Ray z-components should respect elevation bounds."""
        rays = _generate_ray_directions_fibonacci(100, 10.0, 80.0)
        
        z_min_expected = np.sin(np.deg2rad(10.0))
        z_max_expected = np.sin(np.deg2rad(80.0))
        
        assert rays[:, 2].min() >= z_min_expected - 0.05
        assert rays[:, 2].max() <= z_max_expected + 0.05

    def test_minimum_rays(self):
        """Should handle minimum number of rays."""
        rays = _generate_ray_directions_fibonacci(1, 0.0, 90.0)
        assert rays.shape == (1, 3)

    def test_handles_zero_rays(self):
        """Should handle zero rays (returns 1 minimum)."""
        rays = _generate_ray_directions_fibonacci(0, 0.0, 90.0)
        assert rays.shape[0] >= 1

    def test_hemisphere_distribution(self):
        """Fibonacci sampling should give good hemisphere coverage."""
        rays = _generate_ray_directions_fibonacci(500, 0.0, 90.0)
        
        # Check rough uniformity - all octants should have rays
        assert np.sum((rays[:, 0] > 0) & (rays[:, 1] > 0)) > 50
        assert np.sum((rays[:, 0] > 0) & (rays[:, 1] < 0)) > 50
        assert np.sum((rays[:, 0] < 0) & (rays[:, 1] > 0)) > 50
        assert np.sum((rays[:, 0] < 0) & (rays[:, 1] < 0)) > 50


class TestTraceRayGeneric:
    """Tests for trace_ray_generic voxel raytracing function."""
    
    def test_empty_voxel_grid(self):
        """Ray through empty grid should not hit anything."""
        voxel_data = np.zeros((10, 10, 10), dtype=np.int32)
        origin = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        hit_values = np.array([-1], dtype=np.int32)  # Building code
        
        hit, transmittance = trace_ray_generic(
            voxel_data, origin, direction, hit_values,
            meshsize=1.0, tree_k=0.6, tree_lad=1.0
        )
        
        assert not hit
        assert transmittance == pytest.approx(1.0)  # No trees, full transmittance

    def test_hits_building(self):
        """Ray should hit a building voxel."""
        voxel_data = np.zeros((10, 10, 10), dtype=np.int32)
        voxel_data[5, 0, 0] = -1  # Building at (5,0,0)
        origin = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        hit_values = np.array([-1], dtype=np.int32)
        
        hit, transmittance = trace_ray_generic(
            voxel_data, origin, direction, hit_values,
            meshsize=1.0, tree_k=0.6, tree_lad=1.0
        )
        
        assert hit
        assert transmittance == pytest.approx(1.0)  # No trees before building

    def test_tree_reduces_transmittance(self):
        """Ray through tree voxels should have reduced transmittance."""
        voxel_data = np.zeros((10, 10, 10), dtype=np.int32)
        voxel_data[3, 0, 0] = -2  # Tree at (3,0,0)
        voxel_data[8, 0, 0] = -1  # Building at (8,0,0)
        origin = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        hit_values = np.array([-1], dtype=np.int32)
        
        hit, transmittance = trace_ray_generic(
            voxel_data, origin, direction, hit_values,
            meshsize=1.0, tree_k=0.6, tree_lad=1.0
        )
        
        assert hit
        assert transmittance < 1.0  # Tree reduced transmittance

    def test_zero_length_direction(self):
        """Zero direction vector should return early."""
        voxel_data = np.zeros((10, 10, 10), dtype=np.int32)
        origin = np.array([5.5, 5.5, 5.5], dtype=np.float64)
        direction = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        hit_values = np.array([-1], dtype=np.int32)
        
        hit, transmittance = trace_ray_generic(
            voxel_data, origin, direction, hit_values,
            meshsize=1.0, tree_k=0.6, tree_lad=1.0
        )
        
        assert not hit
        assert transmittance == pytest.approx(1.0)

    def test_diagonal_ray(self):
        """Ray at 45 degrees should traverse correctly."""
        voxel_data = np.zeros((10, 10, 10), dtype=np.int32)
        voxel_data[5, 5, 5] = -1  # Building at diagonal
        origin = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        direction = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        hit_values = np.array([-1], dtype=np.int32)
        
        hit, transmittance = trace_ray_generic(
            voxel_data, origin, direction, hit_values,
            meshsize=1.0, tree_k=0.6, tree_lad=1.0
        )
        
        assert hit

    def test_ray_exits_grid(self):
        """Ray that exits grid bounds should not hit."""
        voxel_data = np.zeros((5, 5, 5), dtype=np.int32)
        origin = np.array([2.5, 2.5, 2.5], dtype=np.float64)
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # Up
        hit_values = np.array([-1], dtype=np.int32)
        
        hit, transmittance = trace_ray_generic(
            voxel_data, origin, direction, hit_values,
            meshsize=1.0, tree_k=0.6, tree_lad=1.0
        )
        
        assert not hit


class TestBuildFaceBasis:
    """Tests for _build_face_basis function."""
    
    def test_z_up_normal(self):
        """Test basis for upward-facing normal (0, 0, 1)."""
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        u, v, n = _build_face_basis(normal)
        
        # n should be normalized normal
        np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-10)
        
        # u, v, n should be orthonormal
        assert abs(np.dot(u, v)) < 1e-10
        assert abs(np.dot(u, n)) < 1e-10
        assert abs(np.dot(v, n)) < 1e-10
        
        # All should be unit vectors
        np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-10)

    def test_x_normal(self):
        """Test basis for x-facing normal (1, 0, 0)."""
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u, v, n = _build_face_basis(normal)
        
        np.testing.assert_allclose(n, [1.0, 0.0, 0.0], atol=1e-10)
        
        # Orthonormality
        assert abs(np.dot(u, v)) < 1e-10
        assert abs(np.dot(u, n)) < 1e-10
        assert abs(np.dot(v, n)) < 1e-10

    def test_y_normal(self):
        """Test basis for y-facing normal (0, 1, 0)."""
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u, v, n = _build_face_basis(normal)
        
        np.testing.assert_allclose(n, [0.0, 1.0, 0.0], atol=1e-10)
        
        # Orthonormality
        assert abs(np.dot(u, v)) < 1e-10
        assert abs(np.dot(u, n)) < 1e-10
        assert abs(np.dot(v, n)) < 1e-10

    def test_diagonal_normal(self):
        """Test basis for diagonal normal."""
        normal = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        u, v, n = _build_face_basis(normal)
        
        # n should be normalized
        expected_n = normal / np.linalg.norm(normal)
        np.testing.assert_allclose(n, expected_n, atol=1e-10)
        
        # Orthonormality
        assert abs(np.dot(u, v)) < 1e-10
        assert abs(np.dot(u, n)) < 1e-10
        assert abs(np.dot(v, n)) < 1e-10

    def test_unnormalized_input(self):
        """Test that unnormalized input is handled correctly."""
        normal = np.array([2.0, 0.0, 0.0], dtype=np.float64)  # Not normalized
        u, v, n = _build_face_basis(normal)
        
        # n should be normalized
        np.testing.assert_allclose(n, [1.0, 0.0, 0.0], atol=1e-10)

    def test_near_zero_normal(self):
        """Test handling of near-zero normal."""
        normal = np.array([1e-15, 1e-15, 1e-15], dtype=np.float64)
        u, v, n = _build_face_basis(normal)
        
        # Should return default orthogonal basis
        assert np.linalg.norm(u) > 0
        assert np.linalg.norm(v) > 0

    def test_negative_z_normal(self):
        """Test basis for downward-facing normal (0, 0, -1)."""
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        u, v, n = _build_face_basis(normal)
        
        np.testing.assert_allclose(n, [0.0, 0.0, -1.0], atol=1e-10)
        
        # Orthonormality
        assert abs(np.dot(u, v)) < 1e-10
        assert abs(np.dot(u, n)) < 1e-10
        assert abs(np.dot(v, n)) < 1e-10


class TestRotateVectorAxisAngle:
    """Tests for rotate_vector_axis_angle function."""
    
    def test_zero_rotation(self):
        """Zero angle rotation should return original vector."""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        angle = 0.0
        
        result = rotate_vector_axis_angle(vec, axis, angle)
        np.testing.assert_allclose(result, vec, atol=1e-10)

    def test_90_degree_rotation_z_axis(self):
        """90 degree rotation around z-axis."""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        angle = np.pi / 2  # 90 degrees
        
        result = rotate_vector_axis_angle(vec, axis, angle)
        # x-axis rotated 90 degrees around z should give y-axis
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-10)

    def test_180_degree_rotation(self):
        """180 degree rotation should flip the vector."""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        angle = np.pi  # 180 degrees
        
        result = rotate_vector_axis_angle(vec, axis, angle)
        np.testing.assert_allclose(result, [-1.0, 0.0, 0.0], atol=1e-10)

    def test_360_degree_rotation(self):
        """360 degree rotation should return to original."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        angle = 2 * np.pi
        
        result = rotate_vector_axis_angle(vec, axis, angle)
        np.testing.assert_allclose(result, vec, atol=1e-10)

    def test_rotation_around_parallel_axis(self):
        """Rotation around parallel axis should not change vector."""
        vec = np.array([0.0, 0.0, 5.0], dtype=np.float64)
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        angle = np.pi / 4
        
        result = rotate_vector_axis_angle(vec, axis, angle)
        np.testing.assert_allclose(result, vec, atol=1e-10)

    def test_rotation_preserves_magnitude(self):
        """Rotation should preserve vector magnitude."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        axis = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        angle = np.pi / 3
        
        result = rotate_vector_axis_angle(vec, axis, angle)
        np.testing.assert_allclose(np.linalg.norm(result), np.linalg.norm(vec), atol=1e-10)

    def test_rotation_around_x_axis(self):
        """Rotation around x-axis."""
        vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        angle = np.pi / 2
        
        result = rotate_vector_axis_angle(vec, axis, angle)
        # y-axis rotated 90 degrees around x should give z-axis
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-10)

    def test_rotation_around_y_axis(self):
        """Rotation around y-axis."""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        angle = np.pi / 2
        
        result = rotate_vector_axis_angle(vec, axis, angle)
        # x-axis rotated 90 degrees around y should give -z-axis
        np.testing.assert_allclose(result, [0.0, 0.0, -1.0], atol=1e-10)

    def test_zero_axis_returns_original(self):
        """Zero-length axis should return original vector."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        axis = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        angle = np.pi / 4
        
        result = rotate_vector_axis_angle(vec, axis, angle)
        np.testing.assert_allclose(result, vec, atol=1e-10)

    def test_unnormalized_axis(self):
        """Unnormalized axis should work correctly."""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis = np.array([0.0, 0.0, 2.0], dtype=np.float64)  # Length 2
        angle = np.pi / 2
        
        result = rotate_vector_axis_angle(vec, axis, angle)
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-10)
