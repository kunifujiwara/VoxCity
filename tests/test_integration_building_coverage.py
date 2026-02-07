"""Tests for integration.py: building-level EPW paths and additional branches."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import pytz
import trimesh

from voxcity.models import VoxCity, VoxelGrid, GridMetadata, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid


def _make_voxcity(nx=5, ny=5, nz=10, meshsize=1.0):
    voxel_data = np.zeros((nx, ny, nz), dtype=np.int8)
    voxel_data[:, :, 0] = 1  # ground
    meta = GridMetadata(meshsize=meshsize, bounds=(139.7, 35.6, 139.8, 35.7), crs="EPSG:4326")
    voxels = VoxelGrid(classes=voxel_data, meta=meta)
    bh = np.zeros((nx, ny))
    buildings = BuildingGrid(heights=bh, min_heights=np.empty((nx, ny), dtype=object), ids=np.zeros((nx, ny), dtype=int), meta=meta)
    land = LandCoverGrid(classes=np.ones((nx, ny), dtype=int), meta=meta)
    dem = DemGrid(elevation=np.zeros((nx, ny)), meta=meta)
    canopy = CanopyGrid(top=np.zeros((nx, ny)), bottom=None, meta=meta)
    extras = {"rectangle_vertices": [(139.7, 35.6), (139.7, 35.7), (139.8, 35.7), (139.8, 35.6)]}
    return VoxCity(voxels=voxels, buildings=buildings, land_cover=land, dem=dem, tree_canopy=canopy, extras=extras)


def _make_epw_df():
    idx = pd.date_range("2023-01-01 00:00:00", periods=8760, freq="h")
    rng = np.random.RandomState(42)
    return pd.DataFrame({"DNI": rng.uniform(0, 800, len(idx)), "DHI": rng.uniform(0, 300, len(idx))}, index=idx)


def _make_building_mesh():
    vertices = np.array([[0.5, 0.5, 1.0], [1.5, 0.5, 1.0], [1.0, 1.5, 1.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.metadata['svf'] = np.array([0.8], dtype=np.float64)
    return mesh


class TestBuildingEpwInstantaneous:
    """get_building_global_solar_irradiance_using_epw: instantaneous path (full)."""

    @patch("voxcity.simulator.solar.integration.read_epw_for_solar_simulation")
    @patch("voxcity.simulator.solar.integration.get_building_solar_irradiance")
    @patch("voxcity.simulator.solar.integration.get_surface_view_factor")
    def test_instantaneous_with_building_svf(self, mock_svf, mock_bld_irr, mock_read):
        from voxcity.simulator.solar.integration import get_building_global_solar_irradiance_using_epw
        vc = _make_voxcity()
        df = _make_epw_df()
        mock_read.return_value = (df, 139.75, 35.65, 9.0, 40.0)
        mesh = _make_building_mesh()
        mock_svf.return_value = mesh
        result_mesh = mesh.copy()
        result_mesh.metadata['direct'] = np.array([100.0])
        result_mesh.metadata['diffuse'] = np.array([50.0])
        result_mesh.metadata['global'] = np.array([150.0])
        mock_bld_irr.return_value = result_mesh
        result = get_building_global_solar_irradiance_using_epw(
            vc, calc_type="instantaneous", epw_file_path="test.epw",
            calc_time="07-15 12:00:00",
        )
        assert result is not None
        mock_svf.assert_called_once()
        mock_bld_irr.assert_called_once()

    @patch("voxcity.simulator.solar.integration.read_epw_for_solar_simulation")
    @patch("voxcity.simulator.solar.integration.get_building_solar_irradiance")
    def test_with_precomputed_building_svf_mesh(self, mock_bld_irr, mock_read):
        from voxcity.simulator.solar.integration import get_building_global_solar_irradiance_using_epw
        vc = _make_voxcity()
        df = _make_epw_df()
        mock_read.return_value = (df, 139.75, 35.65, 9.0, 40.0)
        mesh = _make_building_mesh()
        result_mesh = mesh.copy()
        result_mesh.metadata['direct'] = np.array([100.0])
        result_mesh.metadata['diffuse'] = np.array([50.0])
        result_mesh.metadata['global'] = np.array([150.0])
        mock_bld_irr.return_value = result_mesh
        result = get_building_global_solar_irradiance_using_epw(
            vc, calc_type="instantaneous", epw_file_path="test.epw",
            calc_time="07-15 12:00:00", building_svf_mesh=mesh,
        )
        assert result is not None

    @patch("voxcity.simulator.solar.integration.read_epw_for_solar_simulation")
    @patch("voxcity.simulator.solar.integration.get_cumulative_building_solar_irradiance")
    def test_cumulative_path(self, mock_cum, mock_read):
        from voxcity.simulator.solar.integration import get_building_global_solar_irradiance_using_epw
        vc = _make_voxcity()
        df = _make_epw_df()
        mock_read.return_value = (df, 139.75, 35.65, 9.0, 40.0)
        mesh = _make_building_mesh()
        mock_cum.return_value = mesh
        result = get_building_global_solar_irradiance_using_epw(
            vc, calc_type="cumulative", epw_file_path="test.epw",
            building_svf_mesh=mesh,
        )
        assert result is not None
        mock_cum.assert_called_once()


class TestDownloadNearestEpwBranch:
    """Test download_nearest_epw kwarg path in integration."""

    @patch("voxcity.simulator.solar.integration.read_epw_for_solar_simulation")
    @patch("voxcity.simulator.solar.integration.get_global_solar_irradiance_map")
    @patch("voxcity.simulator.solar.integration.get_nearest_epw_from_climate_onebuilding")
    def test_download_nearest_epw(self, mock_get_epw, mock_global, mock_read):
        from voxcity.simulator.solar.integration import get_global_solar_irradiance_using_epw
        vc = _make_voxcity()
        df = _make_epw_df()
        mock_get_epw.return_value = ("auto.epw", None, None)
        mock_read.return_value = (df, 139.75, 35.65, 9.0, 40.0)
        mock_global.return_value = np.full((5, 5), 250.0)
        result = get_global_solar_irradiance_using_epw(
            vc, calc_type="instantaneous",
            download_nearest_epw=True,
            calc_time="07-15 12:00:00",
        )
        assert result.shape == (5, 5)
        mock_get_epw.assert_called_once()

    @patch("voxcity.simulator.solar.integration.read_epw_for_solar_simulation")
    @patch("voxcity.simulator.solar.integration.get_global_solar_irradiance_map")
    @patch("voxcity.simulator.solar.integration.get_nearest_epw_from_climate_onebuilding")
    def test_download_nearest_no_rect_returns_none(self, mock_get_epw, mock_global, mock_read):
        from voxcity.simulator.solar.integration import get_global_solar_irradiance_using_epw
        vc = _make_voxcity()
        vc.extras = {}  # no rectangle_vertices
        result = get_global_solar_irradiance_using_epw(
            vc, calc_type="instantaneous",
            download_nearest_epw=True,
        )
        assert result is None


class TestSaveMeshBranch:
    """Test save_mesh kwarg in building EPW integration."""

    @patch("voxcity.simulator.solar.integration.read_epw_for_solar_simulation")
    @patch("voxcity.simulator.solar.integration.get_building_solar_irradiance")
    @patch("voxcity.simulator.solar.integration.save_irradiance_mesh")
    def test_save_mesh(self, mock_save, mock_bld, mock_read, tmp_path):
        from voxcity.simulator.solar.integration import get_building_global_solar_irradiance_using_epw
        vc = _make_voxcity()
        df = _make_epw_df()
        mock_read.return_value = (df, 139.75, 35.65, 9.0, 40.0)
        mesh = _make_building_mesh()
        result_mesh = mesh.copy()
        result_mesh.metadata['direct'] = np.array([100.0])
        result_mesh.metadata['diffuse'] = np.array([50.0])
        result_mesh.metadata['global'] = np.array([150.0])
        mock_bld.return_value = result_mesh
        out_path = str(tmp_path / "irr.pkl")
        result = get_building_global_solar_irradiance_using_epw(
            vc, calc_type="instantaneous", epw_file_path="test.epw",
            calc_time="07-15 12:00:00", building_svf_mesh=mesh,
            save_mesh=True, mesh_output_path=out_path,
        )
        assert result is not None
        mock_save.assert_called_once()
