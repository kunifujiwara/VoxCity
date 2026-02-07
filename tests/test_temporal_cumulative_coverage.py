"""Tests for temporal.py: get_cumulative_global_solar_irradiance and get_cumulative_building_solar_irradiance."""
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
    voxel_data[:, :, 0] = 1
    meta = GridMetadata(meshsize=meshsize, bounds=(0.0, 0.0, 1.0, 1.0), crs="EPSG:4326")
    voxels = VoxelGrid(classes=voxel_data, meta=meta)
    bh = np.zeros((nx, ny))
    buildings = BuildingGrid(heights=bh, min_heights=np.empty((nx, ny), dtype=object), ids=np.zeros((nx, ny), dtype=int), meta=meta)
    land = LandCoverGrid(classes=np.ones((nx, ny), dtype=int), meta=meta)
    dem = DemGrid(elevation=np.zeros((nx, ny)), meta=meta)
    canopy = CanopyGrid(top=np.zeros((nx, ny)), bottom=None, meta=meta)
    return VoxCity(voxels=voxels, buildings=buildings, land_cover=land, dem=dem, tree_canopy=canopy, extras={})


def _make_epw_df_small():
    """Create a full-year EPW-like DataFrame with hourly data."""
    idx = pd.date_range("2023-01-01 00:00:00", periods=8760, freq="h")
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "DNI": rng.uniform(0, 800, len(idx)),
        "DHI": rng.uniform(0, 300, len(idx)),
    }, index=idx)
    return df


def _make_building_svf_mesh():
    vertices = np.array([
        [0.5, 0.5, 1.0],
        [1.5, 0.5, 1.0],
        [1.0, 1.5, 1.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.metadata['svf'] = np.array([0.8], dtype=np.float64)
    return mesh


# ---------------------------------------------------------------------------
# get_cumulative_global_solar_irradiance
# ---------------------------------------------------------------------------
class TestGetCumulativeGlobalSolarIrradiance:
    @patch("voxcity.simulator.solar.temporal.get_direct_solar_irradiance_map")
    @patch("voxcity.simulator.solar.temporal.get_diffuse_solar_irradiance_map")
    def test_basic_per_timestep(self, mock_diffuse, mock_direct):
        from voxcity.simulator.solar.temporal import get_cumulative_global_solar_irradiance
        vc = _make_voxcity(nx=4, ny=4, nz=8)
        df = _make_epw_df_small()
        mock_direct.return_value = np.full((4, 4), 100.0)
        mock_diffuse.return_value = np.full((4, 4), 0.5)  # SVF-normalized
        
        result = get_cumulative_global_solar_irradiance(
            vc, df, lon=139.75, lat=35.65, tz=9.0,
            start_time="07-15 06:00:00", end_time="07-15 18:00:00",
            show_plot=False, obj_export=False,
        )
        assert result.shape == (4, 4)
        # Should be cumulative (positive)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    @patch("voxcity.simulator.solar.temporal.get_direct_solar_irradiance_map")
    @patch("voxcity.simulator.solar.temporal.get_diffuse_solar_irradiance_map")
    def test_empty_df_raises(self, mock_diffuse, mock_direct):
        from voxcity.simulator.solar.temporal import get_cumulative_global_solar_irradiance
        vc = _make_voxcity()
        with pytest.raises(ValueError, match="No data"):
            get_cumulative_global_solar_irradiance(
                vc, pd.DataFrame(), lon=139.75, lat=35.65, tz=9.0,
            )

    @patch("voxcity.simulator.solar.temporal.get_direct_solar_irradiance_map")
    @patch("voxcity.simulator.solar.temporal.get_diffuse_solar_irradiance_map")
    def test_bad_time_format(self, mock_diffuse, mock_direct):
        from voxcity.simulator.solar.temporal import get_cumulative_global_solar_irradiance
        vc = _make_voxcity()
        df = _make_epw_df_small()
        with pytest.raises(ValueError, match="format"):
            get_cumulative_global_solar_irradiance(
                vc, df, lon=139.75, lat=35.65, tz=9.0,
                start_time="bad-format",
            )


# ---------------------------------------------------------------------------
# get_cumulative_building_solar_irradiance
# ---------------------------------------------------------------------------
class TestGetCumulativeBuildingSolarIrradiance:
    def test_fast_path_basic(self):
        from voxcity.simulator.solar.temporal import get_cumulative_building_solar_irradiance
        vc = _make_voxcity(nx=5, ny=5, nz=10)
        mesh = _make_building_svf_mesh()
        df = _make_epw_df_small()
        
        result = get_cumulative_building_solar_irradiance(
            vc, mesh, df, lon=139.75, lat=35.65, tz=9.0,
            period_start="07-15 06:00:00", period_end="07-15 18:00:00",
            fast_path=True,
            progress_report=False,
        )
        assert 'direct' in result.metadata
        assert 'diffuse' in result.metadata
        assert 'global' in result.metadata
        assert len(result.metadata['direct']) == 1

    def test_bad_period_format_raises(self):
        from voxcity.simulator.solar.temporal import get_cumulative_building_solar_irradiance
        vc = _make_voxcity()
        mesh = _make_building_svf_mesh()
        df = _make_epw_df_small()
        with pytest.raises(ValueError, match="format"):
            get_cumulative_building_solar_irradiance(
                vc, mesh, df, lon=139.75, lat=35.65, tz=9.0,
                period_start="bad",
            )

    def test_empty_period_raises(self):
        from voxcity.simulator.solar.temporal import get_cumulative_building_solar_irradiance
        vc = _make_voxcity()
        mesh = _make_building_svf_mesh()
        # Create data for January only
        idx = pd.date_range("2023-01-01 00:00:00", periods=24, freq="h")
        df = pd.DataFrame({"DNI": [100]*24, "DHI": [50]*24}, index=idx)
        # Request July period -> empty
        with pytest.raises(ValueError, match="No weather data"):
            get_cumulative_building_solar_irradiance(
                vc, mesh, df, lon=139.75, lat=35.65, tz=9.0,
                period_start="07-15 00:00:00", period_end="07-15 23:59:59",
            )

    @patch("voxcity.simulator.solar.temporal.get_building_solar_irradiance")
    def test_slow_path(self, mock_building_irr):
        from voxcity.simulator.solar.temporal import get_cumulative_building_solar_irradiance
        vc = _make_voxcity(nx=5, ny=5, nz=10)
        mesh = _make_building_svf_mesh()
        df = _make_epw_df_small()
        
        # Mock building irradiance to return mesh with metadata
        def mock_irr(*args, **kwargs):
            m = mesh.copy()
            m.metadata['direct'] = np.array([10.0])
            m.metadata['diffuse'] = np.array([5.0])
            m.metadata['global'] = np.array([15.0])
            return m
        mock_building_irr.side_effect = mock_irr
        
        result = get_cumulative_building_solar_irradiance(
            vc, mesh, df, lon=139.75, lat=35.65, tz=9.0,
            period_start="07-15 06:00:00", period_end="07-15 18:00:00",
            fast_path=False,
            progress_report=False,
        )
        assert 'direct' in result.metadata
        assert 'global' in result.metadata

    def test_precomputed_geometry_and_masks(self):
        from voxcity.simulator.solar.temporal import get_cumulative_building_solar_irradiance
        vc = _make_voxcity(nx=5, ny=5, nz=10)
        mesh = _make_building_svf_mesh()
        df = _make_epw_df_small()
        
        precomp_geo = {
            'face_centers': mesh.triangles_center.astype(np.float64),
            'face_normals': mesh.face_normals.astype(np.float64),
            'face_svf': np.array([0.8], dtype=np.float64),
            'grid_bounds_real': np.array([[0, 0, 0], [5, 5, 10]], dtype=np.float64),
            'boundary_epsilon': 0.05,
        }
        precomp_masks = {
            'vox_is_tree': (vc.voxels.classes == -2),
            'vox_is_opaque': (vc.voxels.classes != 0) & (vc.voxels.classes != -2),
            'att': float(np.exp(-0.6 * 1.0 * 1.0)),
        }
        
        result = get_cumulative_building_solar_irradiance(
            vc, mesh, df, lon=139.75, lat=35.65, tz=9.0,
            period_start="07-15 10:00:00", period_end="07-15 14:00:00",
            fast_path=True,
            precomputed_geometry=precomp_geo,
            precomputed_masks=precomp_masks,
        )
        assert 'direct' in result.metadata
