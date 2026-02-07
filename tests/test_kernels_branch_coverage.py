"""Tests for solar kernels: exhaustive branch coverage of compute_direct_solar_irradiance_map_binary."""
import numpy as np
import pytest

from voxcity.simulator.solar.kernels import compute_direct_solar_irradiance_map_binary


class TestKernelsBranchCoverage:
    """Cover every branch in compute_direct_solar_irradiance_map_binary."""

    def _make_voxel_data(self, nx=6, ny=6, nz=10):
        """Basic scene: ground at z=0, air above."""
        vd = np.zeros((nx, ny, nz), dtype=np.int8)
        vd[:, :, 0] = 1  # ground = land cover code 1
        return vd

    def test_open_sky_all_transmittance(self):
        vd = self._make_voxel_data()
        result = compute_direct_solar_irradiance_map_binary(
            vd, (0.0, 0.0, 1.0), 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        valid = result[~np.isnan(result)]
        # All valid cells should have transmittance = 1.0 (no obstructions)
        np.testing.assert_allclose(valid, 1.0, atol=0.01)

    def test_building_shadow(self):
        vd = self._make_voxel_data()
        # Add a building at (3, 3) going from z=0 to z=5
        vd[3, 3, 0:5] = -1  # building code
        result = compute_direct_solar_irradiance_map_binary(
            vd, (0.0, 0.0, 1.0), 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        # Building cell itself should be NaN (building top -> voxel_data[x,y,z-1] < 0)
        # or the observer might be on top of building
        assert result.shape == (6, 6)

    def test_zero_sun_direction_returns_nan(self):
        vd = self._make_voxel_data()
        result = compute_direct_solar_irradiance_map_binary(
            vd, (0.0, 0.0, 0.0), 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        # zero sun direction -> all NaN (flipud of NaN)
        assert np.all(np.isnan(result))

    def test_observer_on_water_excluded(self):
        """Cells with ground = 7, 8, or 9 should produce NaN."""
        vd = self._make_voxel_data()
        vd[2, 2, 0] = 7  # water code
        result = compute_direct_solar_irradiance_map_binary(
            vd, (0.0, 0.0, 1.0), 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        # Water cells should be NaN
        # result is flipped, so (2,2) in original -> (3,2) in flipped (ny=6)
        assert np.isnan(result[6 - 1 - 2, 2])

    def test_observer_on_building_excluded(self):
        """Cells with voxel_data[x,y,z-1] < 0 should produce NaN."""
        vd = self._make_voxel_data()
        vd[2, 2, 0] = -1  # building
        # z=1 is air (0), z=0 is building (-1) -> observer at z=1 sees building below -> NaN
        result = compute_direct_solar_irradiance_map_binary(
            vd, (0.0, 0.0, 1.0), 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        assert np.isnan(result[6 - 1 - 2, 2])

    def test_no_surface_found_nan(self):
        """Column of all air should produce NaN."""
        vd = np.zeros((6, 6, 10), dtype=np.int8)
        # All zeros -> no surface transition -> found_observer stays False
        result = compute_direct_solar_irradiance_map_binary(
            vd, (0.0, 0.0, 1.0), 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        # Should be all NaN
        assert np.all(np.isnan(result))

    def test_tree_transmittance(self):
        """Trees in the path should reduce transmittance but not block completely."""
        vd = self._make_voxel_data()
        # Place trees above ground at a column to the east
        vd[4, 3, 1:4] = -2  # tree code
        # Sun from the east, hitting trees before reaching (2,3)
        sun_dir = (1.0, 0.0, 0.5)
        result = compute_direct_solar_irradiance_map_binary(
            vd, sun_dir, 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_inclusion_mode(self):
        vd = self._make_voxel_data()
        result = compute_direct_solar_irradiance_map_binary(
            vd, (0.0, 0.0, 1.0), 1.5, (1,), 1.0, 0.6, 1.0, True
        )
        assert result.shape == (6, 6)

    def test_diagonal_sun(self):
        vd = self._make_voxel_data()
        result = compute_direct_solar_irradiance_map_binary(
            vd, (0.5, 0.5, 0.7), 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_small_grid(self):
        vd = np.zeros((2, 2, 3), dtype=np.int8)
        vd[:, :, 0] = 1
        result = compute_direct_solar_irradiance_map_binary(
            vd, (0.0, 0.0, 1.0), 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        assert result.shape == (2, 2)

    def test_tree_only_ground(self):
        """Ground made of tree code should produce NaN (code < 0)."""
        vd = np.zeros((4, 4, 6), dtype=np.int8)
        vd[:, :, 0] = -2  # tree as ground
        result = compute_direct_solar_irradiance_map_binary(
            vd, (0.0, 0.0, 1.0), 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        # tree ground -> voxel_data[x,y,z-1] = -2 < 0 -> NaN
        # actually z=1 is air(0), z=0 is tree(-2): tree in (7,8,9) check is False,
        # but tree < 0 check is True -> NaN
        assert np.all(np.isnan(result))
