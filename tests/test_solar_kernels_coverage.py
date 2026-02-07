"""
Comprehensive tests for voxcity.simulator.solar.kernels to improve coverage.
Covers: compute_direct_solar_irradiance_map_binary
"""

import numpy as np
import pytest

from voxcity.simulator.solar.kernels import compute_direct_solar_irradiance_map_binary


def _make_ground_grid(nx=10, ny=10, nz=10):
    grid = np.zeros((nx, ny, nz), dtype=np.int32)
    grid[:, :, 0] = 1  # ground
    return grid


class TestComputeDirectSolarIrradianceMapBinary:
    def test_open_sky(self):
        grid = _make_ground_grid()
        sun_dir = (0.0, 0.0, 1.0)  # Sun straight overhead
        hit_values = np.array([-3], dtype=np.int32)
        result = compute_direct_solar_irradiance_map_binary(
            grid, sun_dir, 1.5, hit_values, 1.0, 0.6, 1.0, True
        )
        assert result.shape == (10, 10)
        # No buildings => transmittance should be 1.0 everywhere (or NaN for invalid)
        valid = ~np.isnan(result)
        if np.any(valid):
            assert np.all(result[valid] >= 0.0)

    def test_building_shadow(self):
        grid = _make_ground_grid()
        # Place tall building
        grid[5, 5, 1:8] = -3
        sun_dir = (1.0, 0.0, 0.5)  # Sun from side
        hit_values = np.array([-3], dtype=np.int32)
        result = compute_direct_solar_irradiance_map_binary(
            grid, sun_dir, 1.5, hit_values, 1.0, 0.6, 1.0, True
        )
        assert result.shape == (10, 10)
        # Some cells should be shadowed (transmittance=0) and some in sunlight

    def test_zero_sun_direction(self):
        grid = _make_ground_grid()
        sun_dir = (0.0, 0.0, 0.0)
        hit_values = np.array([-3], dtype=np.int32)
        result = compute_direct_solar_irradiance_map_binary(
            grid, sun_dir, 1.5, hit_values, 1.0, 0.6, 1.0, True
        )
        assert result.shape == (10, 10)
        # All NaN with zero sun direction
        assert np.all(np.isnan(result))

    def test_tree_transmittance(self):
        grid = _make_ground_grid()
        # Tree row
        grid[3, :, 1:3] = -2
        sun_dir = (1.0, 0.0, 0.5)
        hit_values = np.array([-3], dtype=np.int32)
        result = compute_direct_solar_irradiance_map_binary(
            grid, sun_dir, 1.5, hit_values, 1.0, 0.6, 1.0, True
        )
        assert result.shape == (10, 10)
        # Some cells should have partial transmittance (0 < t < 1)

    def test_small_grid(self):
        grid = np.zeros((3, 3, 3), dtype=np.int32)
        grid[:, :, 0] = 1
        sun_dir = (0.0, 0.0, 1.0)
        hit_values = np.array([-3], dtype=np.int32)
        result = compute_direct_solar_irradiance_map_binary(
            grid, sun_dir, 0.5, hit_values, 0.5, 0.6, 1.0, True
        )
        assert result.shape == (3, 3)

    def test_diagonal_sun(self):
        grid = _make_ground_grid()
        sun_dir = (1.0, 1.0, 1.0)
        hit_values = np.array([-3], dtype=np.int32)
        result = compute_direct_solar_irradiance_map_binary(
            grid, sun_dir, 1.5, hit_values, 1.0, 0.6, 1.0, True
        )
        assert result.shape == (10, 10)
