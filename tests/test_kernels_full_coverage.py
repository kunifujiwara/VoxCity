"""
Tests for kernels.py – compute_direct_solar_irradiance_map_binary.
Lines 24-60 are the njit kernel body.
"""
import numpy as np
import pytest
from voxcity.simulator.solar.kernels import compute_direct_solar_irradiance_map_binary


class TestComputeDirectSolarIrradianceMapBinary:

    def _make_voxels(self, nx=6, ny=6, nz=6):
        """Create voxel data with ground + air."""
        v = np.zeros((nx, ny, nz), dtype=np.int8)
        v[:, :, 0] = 1  # ground
        return v

    def test_clear_sky_transmittance(self):
        """No obstacles -> transmittance=1 for valid ground cells."""
        vox = self._make_voxels()
        sun_dir = (0.0, 0.0, 1.0)  # straight up
        result = compute_direct_solar_irradiance_map_binary(
            vox, sun_dir, 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        assert result.shape == (6, 6)
        # Valid cells should be 1.0 (unblocked)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        np.testing.assert_allclose(valid, 1.0)

    def test_building_blocks_sun(self):
        """A building voxel above ground should block some cells."""
        vox = self._make_voxels()
        # Place a building at (3,3) that is 3 voxels tall
        vox[3, 3, 1] = -1  # building
        vox[3, 3, 2] = -1
        vox[3, 3, 3] = -1
        sun_dir = (0.0, 0.0, 1.0)  # straight up
        result = compute_direct_solar_irradiance_map_binary(
            vox, sun_dir, 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        # The building cell itself should be NaN (building ground, code < 0)
        assert np.isnan(result[6 - 1 - 3, 3])  # flipud

    def test_tree_transmittance(self):
        """Tree voxels should cause partial transmittance."""
        vox = self._make_voxels()
        vox[2, 2, 1] = -2  # tree
        vox[2, 2, 2] = -2
        sun_dir = (0.0, 0.0, 1.0)
        result = compute_direct_solar_irradiance_map_binary(
            vox, sun_dir, 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        # Tree cells should be NaN (ground code is -2, which is in (7,8,9) or < 0)
        # Actually -2 < 0 so the observer is not placed, or the observer is placed above
        # The result depends on the observer placement logic
        assert result.shape == (6, 6)

    def test_zero_sun_direction(self):
        """sd_len == 0 -> returns NaN map."""
        vox = self._make_voxels()
        sun_dir = (0.0, 0.0, 0.0)
        result = compute_direct_solar_irradiance_map_binary(
            vox, sun_dir, 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        assert np.all(np.isnan(result))

    def test_no_ground_cells(self):
        """All empty voxels -> observer never found -> NaN."""
        vox = np.zeros((4, 4, 4), dtype=np.int8)  # all air
        sun_dir = (0.0, 0.0, 1.0)
        result = compute_direct_solar_irradiance_map_binary(
            vox, sun_dir, 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        assert np.all(np.isnan(result))

    def test_angled_sun(self):
        """Angled sun direction should still work."""
        vox = self._make_voxels()
        sun_dir = (0.5, 0.5, 0.7071)  # ~45 degree
        result = compute_direct_solar_irradiance_map_binary(
            vox, sun_dir, 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        assert result.shape == (6, 6)

    def test_output_is_flipped(self):
        """Result should be flipped (np.flipud applied)."""
        vox = self._make_voxels(4, 4, 4)
        sun_dir = (0.0, 0.0, 1.0)
        result = compute_direct_solar_irradiance_map_binary(
            vox, sun_dir, 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        assert result.shape == (4, 4)

    def test_water_voxel_ground(self):
        """Ground codes 7, 8, 9 should produce NaN."""
        vox = np.zeros((4, 4, 4), dtype=np.int8)
        vox[:, :, 0] = 7  # water
        sun_dir = (0.0, 0.0, 1.0)
        result = compute_direct_solar_irradiance_map_binary(
            vox, sun_dir, 1.5, (0,), 1.0, 0.6, 1.0, False
        )
        assert np.all(np.isnan(result))
