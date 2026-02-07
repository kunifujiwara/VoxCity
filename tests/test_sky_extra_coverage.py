"""
Tests for sky.py additional coverage:
  - bin_sun_positions_to_patches (non-tregenza methods)
  - visualize_sky_patches with show=True
  - generate_tregenza_patches call
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


class TestBinSunPositionsToPatches:
    """Cover bin_sun_positions_to_patches for all sky discretization methods."""

    def test_tregenza_method(self):
        from voxcity.simulator.solar.sky import bin_sun_positions_to_patches
        az = np.array([180.0, 200.0, 220.0, 180.0])
        el = np.array([45.0, 30.0, 60.0, -5.0])  # last is below horizon
        dni = np.array([800.0, 600.0, 400.0, 100.0])
        dirs, cum_dni, solid, hours = bin_sun_positions_to_patches(az, el, dni, method="tregenza")
        assert dirs.shape[0] == 145
        assert cum_dni.shape == (145,)
        assert np.sum(hours) == 3  # 3 valid, 1 below horizon
        assert np.sum(cum_dni) == pytest.approx(1800.0)

    def test_reinhart_method(self):
        from voxcity.simulator.solar.sky import bin_sun_positions_to_patches
        az = np.array([90.0, 270.0])
        el = np.array([45.0, 30.0])
        dni = np.array([500.0, 300.0])
        dirs, cum_dni, solid, hours = bin_sun_positions_to_patches(az, el, dni, method="reinhart", mf=2)
        assert dirs.shape[0] > 145  # reinhart has more patches
        assert np.sum(cum_dni) == pytest.approx(800.0)

    def test_uniform_method(self):
        from voxcity.simulator.solar.sky import bin_sun_positions_to_patches
        az = np.array([0.0, 90.0, 180.0])
        el = np.array([30.0, 60.0, 10.0])
        dni = np.array([200.0, 400.0, 100.0])
        dirs, cum_dni, solid, hours = bin_sun_positions_to_patches(
            az, el, dni, method="uniform", n_azimuth=12, n_elevation=6
        )
        assert np.sum(cum_dni) == pytest.approx(700.0)

    def test_fibonacci_method(self):
        from voxcity.simulator.solar.sky import bin_sun_positions_to_patches
        az = np.array([45.0])
        el = np.array([50.0])
        dni = np.array([1000.0])
        dirs, cum_dni, solid, hours = bin_sun_positions_to_patches(
            az, el, dni, method="fibonacci", n_patches=100
        )
        assert dirs.shape[0] == 100
        assert np.sum(cum_dni) == pytest.approx(1000.0)

    def test_unknown_method_raises(self):
        from voxcity.simulator.solar.sky import bin_sun_positions_to_patches
        with pytest.raises(ValueError, match="Unknown"):
            bin_sun_positions_to_patches(
                np.array([0.0]), np.array([45.0]), np.array([500.0]),
                method="nonexistent"
            )


class TestVisualizeSkyPatches:
    """Cover visualize_sky_patches show=True branch (line 666)."""

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots")
    def test_show_true(self, mock_subplots, mock_show):
        from voxcity.simulator.solar.sky import visualize_sky_patches
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        ax = visualize_sky_patches(method="tregenza", show=True)
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots")
    def test_show_false(self, mock_subplots, mock_show):
        from voxcity.simulator.solar.sky import visualize_sky_patches
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        ax = visualize_sky_patches(method="tregenza", show=False)
        mock_show.assert_not_called()


class TestGetTregenzaPatchIndex:
    """Cover pure-python get_tregenza_patch_index (non-njit version)."""

    def test_below_horizon(self):
        from voxcity.simulator.solar.sky import get_tregenza_patch_index
        assert get_tregenza_patch_index(0.0, -1.0) == -1

    def test_zenith(self):
        from voxcity.simulator.solar.sky import get_tregenza_patch_index
        idx = get_tregenza_patch_index(0.0, 89.0)
        assert idx == 144  # zenith patch

    def test_low_elevation(self):
        from voxcity.simulator.solar.sky import get_tregenza_patch_index
        idx = get_tregenza_patch_index(0.0, 5.0)
        assert 0 <= idx < 30  # first band (0-12 deg)

    def test_mid_elevation(self):
        from voxcity.simulator.solar.sky import get_tregenza_patch_index
        idx = get_tregenza_patch_index(180.0, 40.0)
        assert 0 <= idx < 144  # valid non-zenith patch


class TestGeneratePatches:
    """Cover patch generation functions."""

    def test_tregenza_patches(self):
        from voxcity.simulator.solar.sky import generate_tregenza_patches
        patches, directions, solid_angles = generate_tregenza_patches()
        assert patches.shape == (145, 2)
        assert directions.shape == (145, 3)
        assert solid_angles.shape == (145,)

    def test_reinhart_patches(self):
        from voxcity.simulator.solar.sky import generate_reinhart_patches
        patches, directions, solid_angles = generate_reinhart_patches(mf=2)
        assert patches.shape[0] > 145

    def test_uniform_grid_patches(self):
        from voxcity.simulator.solar.sky import generate_uniform_grid_patches
        patches, directions, solid_angles = generate_uniform_grid_patches(n_azimuth=12, n_elevation=6)
        assert patches.shape[0] == 72  # 12 * 6

    def test_fibonacci_patches(self):
        from voxcity.simulator.solar.sky import generate_fibonacci_patches
        patches, directions, solid_angles = generate_fibonacci_patches(n_patches=50)
        assert patches.shape[0] == 50
