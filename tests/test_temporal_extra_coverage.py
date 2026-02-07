"""
Tests for temporal.py remaining branches:
  - get_cumulative_global_solar_irradiance: obj_export, show_each_timestep, sky-patch direct loop
  - get_cumulative_building_solar_irradiance: slow path (fast_path=False), below horizon
"""
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


# ── helpers ──────────────────────────────────────────────────────────────

def _make_voxcity(nx=5, ny=5, nz=4, meshsize=1.0):
    classes = np.zeros((nx, ny, nz), dtype=np.int8)
    classes[:, :, 0] = 1  # ground
    meta = SimpleNamespace(meshsize=meshsize, bounds=(0, 0, nx * meshsize, ny * meshsize), crs="EPSG:4326")
    voxels = SimpleNamespace(classes=classes, meta=meta)
    dem = SimpleNamespace(elevation=np.zeros((nx, ny)))
    return SimpleNamespace(voxels=voxels, dem=dem)


def _make_epw_data(n_hours=3):
    """Create EPW-like weather DataFrame with tz-naive index.
    Uses Jan 15 to avoid leap-year dayofyear mismatch."""
    base = datetime(2023, 1, 15, 10, 0, 0)
    times = pd.date_range(base, periods=n_hours, freq="h")
    df = pd.DataFrame({
        "DNI": [800.0] * n_hours,
        "DHI": [100.0] * n_hours,
    }, index=times)
    return df


def _solar_side_effect(elevation_values):
    """Return a side_effect for get_solar_positions_astral mock that
    creates a DataFrame with the correct (tz-aware) index from the call args."""
    def _side_effect(times, lon, lat):
        n = len(times)
        elevs = elevation_values[:n]
        return pd.DataFrame({
            "azimuth": [180.0] * n,
            "elevation": elevs,
        }, index=times)
    return _side_effect


def _make_building_svf_mesh(n_faces=4):
    mesh = MagicMock()
    mesh.faces = np.zeros((n_faces, 3), dtype=int)
    mesh.triangles_center = np.random.rand(n_faces, 3).astype(np.float64) * 2
    mesh.face_normals = np.tile([0.0, 0.0, 1.0], (n_faces, 1)).astype(np.float64)
    mesh.metadata = {"svf": np.full(n_faces, 0.5, dtype=np.float64)}
    copied = MagicMock()
    copied.metadata = {}
    copied.name = ""
    mesh.copy.return_value = copied
    return mesh


# ══════════════════════════════════════════════════════════════════════════
# get_cumulative_global_solar_irradiance – obj_export branch
# Signature: (voxcity, df, lon, lat, tz, ..., **kwargs)
# kwargs include: start_time, end_time, show_plot, obj_export, etc.
# ══════════════════════════════════════════════════════════════════════════

class TestCumulativeGlobalObjExport:

    @patch("voxcity.simulator.solar.temporal.grid_to_obj")
    @patch("voxcity.simulator.solar.temporal.get_direct_solar_irradiance_map")
    @patch("voxcity.simulator.solar.temporal.get_diffuse_solar_irradiance_map")
    @patch("voxcity.simulator.solar.temporal.get_solar_positions_astral")
    def test_obj_export_triggered(self, mock_solar, mock_diffuse, mock_direct, mock_obj):
        vc = _make_voxcity()
        df = _make_epw_data(2)
        mock_diffuse.return_value = np.full((5, 5), 0.5)
        mock_direct.return_value = np.full((5, 5), 300.0)
        mock_solar.side_effect = _solar_side_effect([45.0, 45.0])

        from voxcity.simulator.solar.temporal import get_cumulative_global_solar_irradiance
        get_cumulative_global_solar_irradiance(
            vc, df,
            lon=139.75, lat=35.68, tz=0,
            start_time="01-15 10:00:00", end_time="01-15 11:00:00",
            show_plot=False, obj_export=True,
        )
        mock_obj.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════
# get_cumulative_global_solar_irradiance – show_each_timestep
# ══════════════════════════════════════════════════════════════════════════

class TestCumulativeGlobalShowEachTimestep:

    @patch("voxcity.simulator.solar.temporal.plt")
    @patch("voxcity.simulator.solar.temporal.get_direct_solar_irradiance_map")
    @patch("voxcity.simulator.solar.temporal.get_diffuse_solar_irradiance_map")
    @patch("voxcity.simulator.solar.temporal.get_solar_positions_astral")
    def test_show_each_timestep(self, mock_solar, mock_diffuse, mock_direct, mock_plt):
        vc = _make_voxcity()
        df = _make_epw_data(2)
        mock_diffuse.return_value = np.full((5, 5), 0.5)
        mock_direct.return_value = np.full((5, 5), 300.0)
        mock_cmap = MagicMock()
        mock_plt.cm.get_cmap.return_value = mock_cmap
        mock_cmap.copy.return_value = mock_cmap
        mock_solar.side_effect = _solar_side_effect([45.0, 45.0])

        from voxcity.simulator.solar.temporal import get_cumulative_global_solar_irradiance
        get_cumulative_global_solar_irradiance(
            vc, df,
            lon=139.75, lat=35.68, tz=0,
            start_time="01-15 10:00:00", end_time="01-15 11:00:00",
            show_plot=False,
            show_each_timestep=True,
        )
        # plt.show called for each timestep
        assert mock_plt.show.call_count >= 2


# ══════════════════════════════════════════════════════════════════════════
# get_cumulative_global_solar_irradiance – sky patch path
# ══════════════════════════════════════════════════════════════════════════

class TestCumulativeGlobalSkyPatchPath:

    @patch("voxcity.simulator.solar.temporal.get_direct_solar_irradiance_map")
    @patch("voxcity.simulator.solar.temporal.get_diffuse_solar_irradiance_map")
    @patch("voxcity.simulator.solar.temporal.get_solar_positions_astral")
    @patch("voxcity.simulator.solar.temporal._aggregate_weather_to_sky_patches")
    def test_sky_patch_direct_loop(self, mock_agg, mock_solar, mock_diffuse, mock_direct):
        vc = _make_voxcity()
        df = _make_epw_data(3)

        mock_diffuse.return_value = np.full((5, 5), 0.5)
        mock_direct.return_value = np.full((5, 5), 1.0)  # transmittance
        mock_solar.side_effect = _solar_side_effect([45.0, 50.0, 55.0])

        # Mock sky patch aggregation
        n_patches = 145
        patches = np.zeros((n_patches, 2))
        patches[:, 0] = np.linspace(0, 360, n_patches)
        patches[:, 1] = np.linspace(10, 80, n_patches)
        active_mask = np.zeros(n_patches, dtype=bool)
        active_mask[0] = True
        active_mask[10] = True
        mock_agg.return_value = {
            "patches": patches,
            "patch_cumulative_dni": np.ones(n_patches) * 100.0,
            "active_mask": active_mask,
            "n_original_timesteps": 3,
            "n_active_patches": 2,
            "method": "tregenza",
            "total_cumulative_dhi": 300.0,
        }

        from voxcity.simulator.solar.temporal import get_cumulative_global_solar_irradiance
        result = get_cumulative_global_solar_irradiance(
            vc, df,
            lon=139.75, lat=35.68, tz=0,
            start_time="01-15 10:00:00", end_time="01-15 12:00:00",
            show_plot=False,
            use_sky_patches=True,
            sky_discretization="tregenza",
            progress_report=True,
        )
        assert result.shape == (5, 5)
        mock_agg.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════
# get_cumulative_building_solar_irradiance – slow path
# Signature: (voxcity, building_svf_mesh, weather_df, lon, lat, tz, **kwargs)
# kwargs include: period_start, period_end, fast_path, etc.
# ══════════════════════════════════════════════════════════════════════════

class TestCumulativeBuildingSlowPath:

    @patch("voxcity.simulator.solar.temporal.get_building_solar_irradiance")
    @patch("voxcity.simulator.solar.temporal.get_solar_positions_astral")
    def test_slow_path_per_timestep(self, mock_solar, mock_bsi):
        vc = _make_voxcity()
        df = _make_epw_data(2)
        mesh = _make_building_svf_mesh(4)

        mock_solar.side_effect = _solar_side_effect([45.0, 50.0])

        irr = MagicMock()
        irr.metadata = {
            "direct": np.array([100.0, 0.0, 50.0, np.nan]),
            "diffuse": np.array([30.0, 30.0, 30.0, np.nan]),
            "global": np.array([130.0, 30.0, 80.0, np.nan]),
        }
        mock_bsi.return_value = irr

        from voxcity.simulator.solar.temporal import get_cumulative_building_solar_irradiance
        result = get_cumulative_building_solar_irradiance(
            vc, mesh, df,
            lon=139.75, lat=35.68, tz=0,
            period_start="01-15 10:00:00", period_end="01-15 11:00:00",
            fast_path=False,
            progress_report=True,
        )
        assert result.metadata["direct"] is not None
        assert mock_bsi.call_count == 2

    @patch("voxcity.simulator.solar.temporal.get_building_solar_irradiance")
    @patch("voxcity.simulator.solar.temporal.get_solar_positions_astral")
    def test_slow_path_below_horizon(self, mock_solar, mock_bsi):
        """Timestep below horizon -> diffuse only, no call to get_building_solar_irradiance."""
        vc = _make_voxcity()
        df = _make_epw_data(1)
        mesh = _make_building_svf_mesh(4)

        mock_solar.side_effect = _solar_side_effect([-5.0])

        from voxcity.simulator.solar.temporal import get_cumulative_building_solar_irradiance
        result = get_cumulative_building_solar_irradiance(
            vc, mesh, df,
            lon=139.75, lat=35.68, tz=0,
            period_start="01-15 10:00:00", period_end="01-15 10:00:00",
            fast_path=False,
        )
        # Below horizon -> get_building_solar_irradiance never called
        mock_bsi.assert_not_called()
        assert result.metadata["diffuse"] is not None
