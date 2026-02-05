"""Tests for voxcity.simulator.common modules."""
import pytest
import numpy as np

from voxcity.simulator.common.raytracing import calculate_transmittance
from voxcity.simulator.common.geometry import (
    _generate_ray_directions_grid,
    _generate_ray_directions_fibonacci,
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
