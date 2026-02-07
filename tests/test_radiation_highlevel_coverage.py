"""Tests for radiation.py high-level functions (get_direct/diffuse/global/building solar irradiance)."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import trimesh

from voxcity.models import VoxCity, VoxelGrid, GridMetadata, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid


def _make_voxcity(nx=5, ny=5, nz=10, meshsize=1.0):
    """Create a minimal VoxCity with simple voxel data."""
    voxel_data = np.zeros((nx, ny, nz), dtype=np.int8)
    # Ground layer
    voxel_data[:, :, 0] = 1  # land cover
    meta = GridMetadata(meshsize=meshsize, bounds=(0.0, 0.0, 1.0, 1.0), crs="EPSG:4326")
    voxels = VoxelGrid(classes=voxel_data, meta=meta)
    bh = np.zeros((nx, ny))
    buildings = BuildingGrid(heights=bh, min_heights=np.empty((nx, ny), dtype=object), ids=np.zeros((nx, ny), dtype=int), meta=meta)
    land = LandCoverGrid(classes=np.ones((nx, ny), dtype=int), meta=meta)
    dem = DemGrid(elevation=np.zeros((nx, ny)), meta=meta)
    canopy = CanopyGrid(top=np.zeros((nx, ny)), bottom=None, meta=meta)
    return VoxCity(voxels=voxels, buildings=buildings, land_cover=land, dem=dem, tree_canopy=canopy, extras={})


def _make_building_svf_mesh():
    """Create a tiny trimesh with SVF metadata."""
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
# get_direct_solar_irradiance_map
# ---------------------------------------------------------------------------
class TestGetDirectSolarIrradianceMap:
    def test_returns_2d_array(self):
        from voxcity.simulator.solar.radiation import get_direct_solar_irradiance_map
        vc = _make_voxcity(nx=5, ny=5, nz=10)
        result = get_direct_solar_irradiance_map(
            vc,
            azimuth_degrees_ori=180,
            elevation_degrees=45,
            direct_normal_irradiance=500.0,
            show_plot=False,
        )
        assert result.shape == (5, 5)

    def test_irradiance_values_bounded(self):
        from voxcity.simulator.solar.radiation import get_direct_solar_irradiance_map
        vc = _make_voxcity(nx=5, ny=5, nz=10)
        DNI = 800.0
        result = get_direct_solar_irradiance_map(
            vc,
            azimuth_degrees_ori=180,
            elevation_degrees=60,
            direct_normal_irradiance=DNI,
            show_plot=False,
        )
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.all(valid >= 0)
            assert np.all(valid <= DNI * 1.01)

    def test_zero_dni(self):
        from voxcity.simulator.solar.radiation import get_direct_solar_irradiance_map
        vc = _make_voxcity(nx=5, ny=5, nz=10)
        result = get_direct_solar_irradiance_map(
            vc, 180, 45, direct_normal_irradiance=0.0, show_plot=False
        )
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            np.testing.assert_allclose(valid, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# get_diffuse_solar_irradiance_map  
# ---------------------------------------------------------------------------
class TestGetDiffuseSolarIrradianceMap:
    @patch("voxcity.simulator.solar.radiation.get_sky_view_factor_map")
    def test_returns_svf_times_dhi(self, mock_svf):
        from voxcity.simulator.solar.radiation import get_diffuse_solar_irradiance_map
        vc = _make_voxcity(nx=4, ny=4, nz=8)
        mock_svf.return_value = np.full((4, 4), 0.5)
        result = get_diffuse_solar_irradiance_map(vc, diffuse_irradiance=200.0, show_plot=False)
        np.testing.assert_allclose(result, 100.0)

    @patch("voxcity.simulator.solar.radiation.get_sky_view_factor_map")
    def test_svf_one_equals_full_dhi(self, mock_svf):
        from voxcity.simulator.solar.radiation import get_diffuse_solar_irradiance_map
        vc = _make_voxcity(nx=3, ny=3, nz=6)
        mock_svf.return_value = np.ones((3, 3))
        result = get_diffuse_solar_irradiance_map(vc, diffuse_irradiance=300.0, show_plot=False)
        np.testing.assert_allclose(result, 300.0)


# ---------------------------------------------------------------------------
# get_global_solar_irradiance_map
# ---------------------------------------------------------------------------
class TestGetGlobalSolarIrradianceMap:
    @patch("voxcity.simulator.solar.radiation.get_diffuse_solar_irradiance_map")
    @patch("voxcity.simulator.solar.radiation.get_direct_solar_irradiance_map")
    def test_combines_direct_and_diffuse(self, mock_direct, mock_diffuse):
        from voxcity.simulator.solar.radiation import get_global_solar_irradiance_map
        vc = _make_voxcity(nx=4, ny=4, nz=8)
        mock_direct.return_value = np.full((4, 4), 200.0)
        mock_diffuse.return_value = np.full((4, 4), 100.0)
        result = get_global_solar_irradiance_map(
            vc, 180, 45, 500, 200, show_plot=False
        )
        np.testing.assert_allclose(result, 300.0)

    @patch("voxcity.simulator.solar.radiation.get_diffuse_solar_irradiance_map")
    @patch("voxcity.simulator.solar.radiation.get_direct_solar_irradiance_map")
    def test_nan_direct_uses_diffuse_only(self, mock_direct, mock_diffuse):
        from voxcity.simulator.solar.radiation import get_global_solar_irradiance_map
        vc = _make_voxcity(nx=4, ny=4, nz=8)
        direct = np.full((4, 4), np.nan)
        mock_direct.return_value = direct
        mock_diffuse.return_value = np.full((4, 4), 50.0)
        result = get_global_solar_irradiance_map(
            vc, 180, 45, 500, 200, show_plot=False
        )
        np.testing.assert_allclose(result, 50.0)


# ---------------------------------------------------------------------------
# get_building_solar_irradiance
# ---------------------------------------------------------------------------
class TestGetBuildingSolarIrradiance:
    def test_returns_mesh_with_irradiance_metadata(self):
        from voxcity.simulator.solar.radiation import get_building_solar_irradiance
        vc = _make_voxcity(nx=5, ny=5, nz=10)
        mesh = _make_building_svf_mesh()
        result = get_building_solar_irradiance(
            vc, mesh,
            azimuth_degrees=180,
            elevation_degrees=45,
            direct_normal_irradiance=500.0,
            diffuse_irradiance=100.0,
            fast_path=True,
        )
        assert 'direct' in result.metadata
        assert 'diffuse' in result.metadata
        assert 'global' in result.metadata
        assert len(result.metadata['direct']) == len(mesh.faces)

    def test_slow_path(self):
        from voxcity.simulator.solar.radiation import get_building_solar_irradiance
        vc = _make_voxcity(nx=5, ny=5, nz=10)
        mesh = _make_building_svf_mesh()
        result = get_building_solar_irradiance(
            vc, mesh,
            azimuth_degrees=180,
            elevation_degrees=45,
            direct_normal_irradiance=500.0,
            diffuse_irradiance=100.0,
            fast_path=False,
        )
        assert 'direct' in result.metadata
        assert len(result.metadata['global']) == 1

    def test_precomputed_geometry(self):
        from voxcity.simulator.solar.radiation import get_building_solar_irradiance
        vc = _make_voxcity(nx=5, ny=5, nz=10)
        mesh = _make_building_svf_mesh()
        precomp_geo = {
            'face_centers': mesh.triangles_center.astype(np.float64),
            'face_normals': mesh.face_normals.astype(np.float64),
            'grid_bounds_real': np.array([[0, 0, 0], [5, 5, 10]], dtype=np.float64),
            'boundary_epsilon': 0.05,
        }
        precomp_masks = {
            'vox_is_tree': (vc.voxels.classes == -2),
            'vox_is_opaque': (vc.voxels.classes != 0) & (vc.voxels.classes != -2),
            'att': float(np.exp(-0.6 * 1.0 * 1.0)),
        }
        result = get_building_solar_irradiance(
            vc, mesh, 180, 45, 500.0, 100.0,
            fast_path=True,
            precomputed_geometry=precomp_geo,
            precomputed_masks=precomp_masks,
        )
        assert 'direct' in result.metadata

    def test_no_svf_in_metadata(self):
        from voxcity.simulator.solar.radiation import get_building_solar_irradiance
        vc = _make_voxcity(nx=5, ny=5, nz=10)
        vertices = np.array([[0.5, 0.5, 1.0], [1.5, 0.5, 1.0], [1.0, 1.5, 1.0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        # no svf in metadata
        result = get_building_solar_irradiance(
            vc, mesh, 180, 45, 500.0, 100.0, fast_path=True
        )
        assert 'svf' in result.metadata
        np.testing.assert_allclose(result.metadata['svf'], 0.0)  # defaults to zeros
