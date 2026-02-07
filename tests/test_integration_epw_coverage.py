"""Tests for solar integration: get_global_solar_irradiance_using_epw and related."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import pytz

from voxcity.models import VoxCity, VoxelGrid, GridMetadata, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid


def _make_voxcity(nx=5, ny=5, nz=10, meshsize=1.0):
    voxel_data = np.zeros((nx, ny, nz), dtype=np.int8)
    voxel_data[:, :, 0] = 1
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
    """Create a small EPW-like DataFrame for testing."""
    # Create hourly index for a full year
    idx = pd.date_range("2023-01-01 00:00:00", periods=8760, freq="h")
    df = pd.DataFrame(
        {"DNI": np.random.uniform(0, 800, len(idx)),
         "DHI": np.random.uniform(0, 300, len(idx))},
        index=idx,
    )
    return df


# ---------------------------------------------------------------------------
# get_global_solar_irradiance_using_epw - instantaneous
# ---------------------------------------------------------------------------
class TestGetGlobalSolarIrradianceUsingEpwInstantaneous:
    @patch("voxcity.simulator.solar.integration.read_epw_for_solar_simulation")
    @patch("voxcity.simulator.solar.integration.get_global_solar_irradiance_map")
    def test_instantaneous_returns_map(self, mock_global_map, mock_read_epw):
        from voxcity.simulator.solar.integration import get_global_solar_irradiance_using_epw
        vc = _make_voxcity()
        df = _make_epw_df()
        mock_read_epw.return_value = (df, 139.75, 35.65, 9.0, 40.0)
        mock_global_map.return_value = np.full((5, 5), 250.0)
        result = get_global_solar_irradiance_using_epw(
            vc, calc_type="instantaneous", epw_file_path="test.epw",
            calc_time="07-15 12:00:00",
        )
        assert result.shape == (5, 5)
        mock_global_map.assert_called_once()

    @patch("voxcity.simulator.solar.integration.read_epw_for_solar_simulation")
    def test_no_epw_path_raises(self, mock_read_epw):
        from voxcity.simulator.solar.integration import get_global_solar_irradiance_using_epw
        vc = _make_voxcity()
        with pytest.raises(ValueError, match="epw_file_path must be provided"):
            get_global_solar_irradiance_using_epw(vc, calc_type="instantaneous")

    @patch("voxcity.simulator.solar.integration.read_epw_for_solar_simulation")
    def test_bad_calc_time_format(self, mock_read_epw):
        from voxcity.simulator.solar.integration import get_global_solar_irradiance_using_epw
        vc = _make_voxcity()
        df = _make_epw_df()
        mock_read_epw.return_value = (df, 139.75, 35.65, 9.0, 40.0)
        with pytest.raises(ValueError, match="calc_time must be in format"):
            get_global_solar_irradiance_using_epw(
                vc, calc_type="instantaneous", epw_file_path="test.epw",
                calc_time="2023-07-15",
            )

    @patch("voxcity.simulator.solar.integration.read_epw_for_solar_simulation")
    def test_empty_epw_raises(self, mock_read_epw):
        from voxcity.simulator.solar.integration import get_global_solar_irradiance_using_epw
        vc = _make_voxcity()
        mock_read_epw.return_value = (pd.DataFrame(), 139.75, 35.65, 9.0, 40.0)
        with pytest.raises(ValueError, match="No data in EPW"):
            get_global_solar_irradiance_using_epw(
                vc, calc_type="instantaneous", epw_file_path="test.epw",
            )

    @patch("voxcity.simulator.solar.integration.read_epw_for_solar_simulation")
    def test_invalid_calc_type_raises(self, mock_read_epw):
        from voxcity.simulator.solar.integration import get_global_solar_irradiance_using_epw
        vc = _make_voxcity()
        df = _make_epw_df()
        mock_read_epw.return_value = (df, 139.75, 35.65, 9.0, 40.0)
        with pytest.raises(ValueError, match="calc_type must be"):
            get_global_solar_irradiance_using_epw(
                vc, calc_type="invalid", epw_file_path="test.epw",
            )


# ---------------------------------------------------------------------------
# get_global_solar_irradiance_using_epw - cumulative
# ---------------------------------------------------------------------------
class TestGetGlobalSolarIrradianceUsingEpwCumulative:
    @patch("voxcity.simulator.solar.integration.read_epw_for_solar_simulation")
    @patch("voxcity.simulator.solar.integration.get_cumulative_global_solar_irradiance")
    def test_cumulative_returns_map(self, mock_cumulative, mock_read_epw):
        from voxcity.simulator.solar.integration import get_global_solar_irradiance_using_epw
        vc = _make_voxcity()
        df = _make_epw_df()
        mock_read_epw.return_value = (df, 139.75, 35.65, 9.0, 40.0)
        mock_cumulative.return_value = np.full((5, 5), 1000.0)
        result = get_global_solar_irradiance_using_epw(
            vc, calc_type="cumulative", epw_file_path="test.epw",
        )
        assert result.shape == (5, 5)
        mock_cumulative.assert_called_once()


# ---------------------------------------------------------------------------
# get_building_global_solar_irradiance_using_epw
# ---------------------------------------------------------------------------
class TestGetBuildingGlobalSolarIrradianceUsingEpw:
    @patch("voxcity.simulator.solar.integration.read_epw_for_solar_simulation")
    @patch("voxcity.simulator.solar.integration.get_building_solar_irradiance")
    @patch("voxcity.simulator.solar.integration.get_surface_view_factor")
    def test_instantaneous_building(self, mock_svf, mock_building_irr, mock_read_epw):
        import trimesh
        from voxcity.simulator.solar.integration import get_building_global_solar_irradiance_using_epw
        vc = _make_voxcity()
        df = _make_epw_df()
        mock_read_epw.return_value = (df, 139.75, 35.65, 9.0, 40.0)
        
        # Create a mock building SVF mesh
        vertices = np.array([[0.5, 0.5, 1.0], [1.5, 0.5, 1.0], [1.0, 1.5, 1.0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]])
        svf_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        svf_mesh.metadata['svf'] = np.array([0.8], dtype=np.float64)
        mock_svf.return_value = svf_mesh
        
        result_mesh = svf_mesh.copy()
        result_mesh.metadata['direct'] = np.array([100.0])
        result_mesh.metadata['diffuse'] = np.array([50.0])
        result_mesh.metadata['global'] = np.array([150.0])
        mock_building_irr.return_value = result_mesh
        
        result = get_building_global_solar_irradiance_using_epw(
            vc, calc_type="instantaneous", epw_file_path="test.epw",
            calc_time="07-15 12:00:00",
        )
        assert 'direct' in result.metadata

    def test_no_voxcity_raises(self):
        from voxcity.simulator.solar.integration import get_building_global_solar_irradiance_using_epw
        with pytest.raises(ValueError, match="voxcity"):
            get_building_global_solar_irradiance_using_epw(calc_type="instantaneous")

    @patch("voxcity.simulator.solar.integration.read_epw_for_solar_simulation")
    def test_no_epw_raises(self, mock_read_epw):
        from voxcity.simulator.solar.integration import get_building_global_solar_irradiance_using_epw
        vc = _make_voxcity()
        with pytest.raises(ValueError, match="epw_file_path"):
            get_building_global_solar_irradiance_using_epw(vc, calc_type="instantaneous")

    @patch("voxcity.simulator.solar.integration.read_epw_for_solar_simulation")
    def test_invalid_calc_type_raises(self, mock_read_epw):
        from voxcity.simulator.solar.integration import get_building_global_solar_irradiance_using_epw
        vc = _make_voxcity()
        df = _make_epw_df()
        mock_read_epw.return_value = (df, 139.75, 35.65, 9.0, 40.0)
        
        # Mock SVF so it doesn't try to compute
        with patch("voxcity.simulator.solar.integration.get_surface_view_factor") as mock_svf:
            import trimesh
            vertices = np.array([[0.5, 0.5, 1.0], [1.5, 0.5, 1.0], [1.0, 1.5, 1.0]], dtype=np.float64)
            faces = np.array([[0, 1, 2]])
            svf_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            svf_mesh.metadata['svf'] = np.array([0.8], dtype=np.float64)
            mock_svf.return_value = svf_mesh
            
            with pytest.raises(ValueError, match="calc_type must be"):
                get_building_global_solar_irradiance_using_epw(
                    vc, calc_type="wrong", epw_file_path="test.epw",
                )
