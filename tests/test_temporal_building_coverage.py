"""Tests for temporal.py: sky-patch optimization path, fast-path batching, and remaining branches."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
import trimesh

from voxcity.models import VoxCity, VoxelGrid, GridMetadata, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid


def _make_voxcity(nx=5, ny=5, nz=10, meshsize=1.0):
    voxel_data = np.zeros((nx, ny, nz), dtype=np.int8)
    voxel_data[:, :, 0] = 1
    meta = GridMetadata(meshsize=meshsize, bounds=(0.0, 0.0, 1.0, 1.0), crs="EPSG:4326")
    voxels = VoxelGrid(classes=voxel_data, meta=meta)
    bh = np.zeros((nx, ny))
    buildings = BuildingGrid(heights=bh, min_heights=np.empty((nx, ny), dtype=object), ids=np.zeros((nx, ny), dtype=int), meta=meta)
    land = LandCoverGrid(classes=np.ones((nx, ny), dtype=int), meta=meta)
    dem = DemGrid(elevation=np.zeros((nx, ny)), meta=meta)
    canopy = CanopyGrid(top=np.zeros((nx, ny)), bottom=None, meta=meta)
    return VoxCity(voxels=voxels, buildings=buildings, land_cover=land, dem=dem, tree_canopy=canopy, extras={})


def _make_epw_df():
    idx = pd.date_range("2023-01-01 00:00:00", periods=8760, freq="h")
    rng = np.random.RandomState(42)
    return pd.DataFrame({"DNI": rng.uniform(0, 800, len(idx)), "DHI": rng.uniform(0, 300, len(idx))}, index=idx)


def _make_building_mesh():
    vertices = np.array([
        [0.5, 0.5, 1.0], [1.5, 0.5, 1.0], [1.0, 1.5, 1.0],
        [2.0, 2.0, 2.0], [3.0, 2.0, 2.0], [2.5, 3.0, 2.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.metadata['svf'] = np.array([0.8, 0.6], dtype=np.float64)
    return mesh


def _make_isolated_building_voxcity():
    voxel_data = np.zeros((6, 6, 4), dtype=np.int32)
    voxel_data[3, 3, 1:3] = -3
    building_ids = np.zeros((6, 6), dtype=np.int32)
    building_ids[3, 3] = 1
    return SimpleNamespace(
        voxels=SimpleNamespace(
            classes=voxel_data,
            meta=SimpleNamespace(meshsize=1.0),
        ),
        buildings=SimpleNamespace(ids=building_ids),
        extras={"rotation_angle": 0},
        dem=None,
    )


class TestCumulativeBuildingFastPathBatching:
    """Test the fast path with time-batching in get_cumulative_building_solar_irradiance."""

    @patch("voxcity.simulator.solar.temporal.get_solar_positions_astral")
    @patch("voxcity.simulator.solar.temporal.get_diffuse_solar_irradiance_map")
    @patch("voxcity.simulator.solar.radiation.compute_cumulative_solar_irradiance_faces_masked_timeseries")
    def test_fast_path_batching(self, mock_cum_kernel, mock_diffuse, mock_solar_pos):
        from voxcity.simulator.solar.temporal import get_cumulative_building_solar_irradiance
        vc = _make_voxcity(nx=4, ny=4, nz=8)
        df = _make_epw_df()
        mesh = _make_building_mesh()
        n_faces = 2

        # Solar positions
        pos_df = pd.DataFrame({
            'azimuth': np.full(8760, 180.0),
            'elevation': np.full(8760, 45.0),
        })
        mock_solar_pos.return_value = pos_df

        # The kernel returns (dir, diff, glob) arrays per face
        mock_cum_kernel.return_value = (
            np.full(n_faces, 100.0),
            np.full(n_faces, 50.0),
            np.full(n_faces, 150.0),
        )

        result = get_cumulative_building_solar_irradiance(
            vc, mesh, df, lon=139.75, lat=35.65, tz=9.0,
            period_start="07-01 06:00:00", period_end="07-01 18:00:00",
            fast_path=True, show_plot=False, obj_export=False,
        )
        assert result is not None
        assert 'direct' in result.metadata or 'global' in result.metadata

    @patch("voxcity.simulator.solar.temporal.get_solar_positions_astral")
    @patch("voxcity.simulator.solar.temporal.get_diffuse_solar_irradiance_map")
    @patch("voxcity.simulator.solar.radiation.compute_solar_irradiance_for_all_faces")
    def test_slow_path_per_timestep(self, mock_slow_kernel, mock_diffuse, mock_solar_pos):
        from voxcity.simulator.solar.temporal import get_cumulative_building_solar_irradiance
        vc = _make_voxcity(nx=4, ny=4, nz=8)
        df = _make_epw_df()
        mesh = _make_building_mesh()
        n_faces = 2

        pos_df = pd.DataFrame({
            'azimuth': np.full(8760, 180.0),
            'elevation': np.full(8760, 45.0),
        })
        mock_solar_pos.return_value = pos_df
        mock_slow_kernel.return_value = (
            np.full(n_faces, 100.0),
            np.full(n_faces, 50.0),
            np.full(n_faces, 150.0),
        )
        result = get_cumulative_building_solar_irradiance(
            vc, mesh, df, lon=139.75, lat=35.65, tz=9.0,
            period_start="07-01 06:00:00", period_end="07-01 18:00:00",
            fast_path=False, obj_export=False,
        )
        assert result is not None


class TestCumulativeBuildingWithSkyPatches:
    """Test the sky-patch optimization path for building irradiance."""

    def test_diffuse_only_uses_sky_hemisphere_factor(self):
        from voxcity.simulator.solar.temporal import get_cumulative_building_solar_irradiance
        from voxcity.simulator.visibility import get_surface_view_factor

        voxcity = _make_isolated_building_voxcity()
        svf_mesh = get_surface_view_factor(
            voxcity,
            value_name="svf",
            sky_diffuse=True,
            N_azimuth=72,
            N_elevation=18,
            fast_path=True,
        )
        weather = pd.DataFrame(
            {"DNI": [0.0, 0.0, 0.0], "DHI": [0.0, 50.0, 100.0]},
            index=pd.date_range("2024-01-01 00:00:00", periods=3, freq="h"),
        )
        solar_positions = pd.DataFrame(
            {"azimuth": [0.0, 0.0, 0.0], "elevation": [-10.0, -10.0, -10.0]},
            index=weather.index,
        )

        result = get_cumulative_building_solar_irradiance(
            voxcity,
            svf_mesh,
            weather,
            lon=0.0,
            lat=0.0,
            tz=0.0,
            period_start="01-01 00:00:00",
            period_end="01-01 02:00:00",
            time_step_hours=1.0,
            use_sky_patches=True,
            precomputed_solar_positions=solar_positions,
        )

        normals = svf_mesh.face_normals
        roof_faces = normals[:, 2] > 0.9
        vertical_faces = np.abs(normals[:, 2]) < 0.1
        bottom_faces = normals[:, 2] < -0.9

        assert np.nanmax(result.metadata["diffuse"][roof_faces]) == pytest.approx(150.0, abs=15.0)
        assert np.nanmax(result.metadata["diffuse"][vertical_faces]) == pytest.approx(75.0, abs=20.0)
        assert np.nanmax(np.nan_to_num(result.metadata["diffuse"][bottom_faces], nan=0.0)) == pytest.approx(0.0)

    @patch("voxcity.simulator.solar.temporal.get_solar_positions_astral")
    @patch("voxcity.simulator.solar.temporal.get_diffuse_solar_irradiance_map")
    @patch("voxcity.simulator.solar.temporal._aggregate_weather_to_sky_patches")
    @patch("voxcity.simulator.solar.radiation.compute_solar_irradiance_for_all_faces_masked")
    def test_sky_patches_path(self, mock_face_masked, mock_agg, mock_diffuse, mock_solar_pos):
        from voxcity.simulator.solar.temporal import get_cumulative_building_solar_irradiance
        vc = _make_voxcity(nx=4, ny=4, nz=8)
        df = _make_epw_df()
        mesh = _make_building_mesh()
        n_faces = 2

        pos_df = pd.DataFrame({
            'azimuth': np.full(8760, 180.0),
            'elevation': np.full(8760, 45.0),
        })
        mock_solar_pos.return_value = pos_df

        # Fake patch data
        patches = np.array([[180.0, 45.0], [90.0, 30.0]], dtype=np.float64)
        mock_agg.return_value = {
            'patches': patches,
            'active_mask': np.array([True, False]),
            'patch_cumulative_dni': np.array([500.0, 0.0]),
            'total_cumulative_dhi': 200.0,
            'n_original_timesteps': 13,
            'n_active_patches': 1,
            'method': 'tregenza',
        }
        mock_face_masked.return_value = (
            np.full(n_faces, 0.5),
            np.zeros(n_faces),
            np.full(n_faces, 0.5),
        )
        result = get_cumulative_building_solar_irradiance(
            vc, mesh, df, lon=139.75, lat=35.65, tz=9.0,
            period_start="07-01 06:00:00", period_end="07-01 18:00:00",
            use_sky_patches=True, show_plot=False, obj_export=False,
        )
        assert result is not None
        mock_agg.assert_called_once()
        mock_face_masked.assert_called()


class TestCumulativeGlobalWithSkyPatches:
    """Test the sky-patches path in get_cumulative_global_solar_irradiance."""

    @patch("voxcity.simulator.solar.temporal.get_direct_solar_irradiance_map")
    @patch("voxcity.simulator.solar.temporal.get_diffuse_solar_irradiance_map")
    def test_sky_patches_global(self, mock_diffuse, mock_direct):
        from voxcity.simulator.solar.temporal import get_cumulative_global_solar_irradiance
        vc = _make_voxcity(nx=4, ny=4, nz=8)
        df = _make_epw_df()
        mock_direct.return_value = np.full((4, 4), 100.0)
        mock_diffuse.return_value = np.full((4, 4), 0.5)

        result = get_cumulative_global_solar_irradiance(
            vc, df, lon=139.75, lat=35.65, tz=9.0,
            start_time="07-15 06:00:00", end_time="07-15 18:00:00",
            use_sky_patches=True, show_plot=False, obj_export=False,
        )
        assert result.shape == (4, 4)
