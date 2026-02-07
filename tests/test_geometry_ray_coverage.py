"""Round 6 – cover geometry.py uncovered lines 44-96: _generate_ray_directions_grid, _generate_ray_directions_fibonacci."""
from __future__ import annotations

import numpy as np
import pytest
from voxcity.simulator.common.geometry import (
    _generate_ray_directions_grid,
    _generate_ray_directions_fibonacci,
)


# ===========================================================================
# Tests for _generate_ray_directions_grid
# ===========================================================================

class TestGenerateRayDirectionsGrid:
    """Cover geometry.py lines 44-65."""

    def test_basic_output_shape(self):
        N_az, N_el = 8, 4
        dirs = _generate_ray_directions_grid(N_az, N_el, 0.0, 90.0)
        assert dirs.shape == (N_az * N_el, 3)

    def test_all_unit_vectors(self):
        dirs = _generate_ray_directions_grid(12, 6, 10.0, 80.0)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_elevation_range(self):
        """All z-components should be between sin(min_elev) and sin(max_elev)."""
        dirs = _generate_ray_directions_grid(16, 8, 10.0, 70.0)
        z_min = np.sin(np.deg2rad(10.0))
        z_max = np.sin(np.deg2rad(70.0))
        assert dirs[:, 2].min() >= z_min - 1e-10
        assert dirs[:, 2].max() <= z_max + 1e-10

    def test_single_direction(self):
        dirs = _generate_ray_directions_grid(1, 1, 45.0, 45.0)
        assert dirs.shape == (1, 3)
        assert np.linalg.norm(dirs[0]) == pytest.approx(1.0)

    def test_hemisphere_coverage(self):
        """With full azimuth, directions should span all quadrants in XY."""
        dirs = _generate_ray_directions_grid(36, 1, 45.0, 45.0)
        assert dirs[:, 0].max() > 0
        assert dirs[:, 0].min() < 0
        assert dirs[:, 1].max() > 0
        assert dirs[:, 1].min() < 0


# ===========================================================================
# Tests for _generate_ray_directions_fibonacci
# ===========================================================================

class TestGenerateRayDirectionsFibonacci:
    """Cover geometry.py lines 70-96."""

    def test_basic_output_shape(self):
        dirs = _generate_ray_directions_fibonacci(100, 0.0, 90.0)
        assert dirs.shape == (100, 3)

    def test_all_unit_vectors(self):
        dirs = _generate_ray_directions_fibonacci(200, 5.0, 85.0)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_elevation_bounds(self):
        dirs = _generate_ray_directions_fibonacci(500, 10.0, 60.0)
        z_min = np.sin(np.deg2rad(10.0))
        z_max = np.sin(np.deg2rad(60.0))
        # z should be within [z_min, z_max] (with small tolerance for bin edges)
        assert dirs[:, 2].min() >= z_min - 0.05
        assert dirs[:, 2].max() <= z_max + 0.05

    def test_single_ray(self):
        dirs = _generate_ray_directions_fibonacci(1, 45.0, 45.0)
        assert dirs.shape == (1, 3)
        assert np.linalg.norm(dirs[0]) == pytest.approx(1.0)

    def test_full_hemisphere(self):
        dirs = _generate_ray_directions_fibonacci(1000, 0.0, 90.0)
        # All z should be non-negative (upper hemisphere)
        assert dirs[:, 2].min() >= -0.01

    def test_xy_distribution(self):
        """Golden-angle spiral should give good angular coverage."""
        dirs = _generate_ray_directions_fibonacci(500, 0.0, 90.0)
        # Should have directions in all four xy quadrants
        assert (dirs[:, 0] > 0).any() and (dirs[:, 0] < 0).any()
        assert (dirs[:, 1] > 0).any() and (dirs[:, 1] < 0).any()
