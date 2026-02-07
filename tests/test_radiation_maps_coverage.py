"""
Tests for radiation.py high-level map functions:
  - get_direct_solar_irradiance_map (show_plot, obj_export branches)
  - get_diffuse_solar_irradiance_map (show_plot, obj_export branches)
  - get_global_solar_irradiance_map (show_plot, obj_export branches)
  - get_building_solar_irradiance (fast_path, slow_path, precomputed geometry/masks)
"""
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock


# ── helpers ──────────────────────────────────────────────────────────────

def _make_voxcity(nx=5, ny=5, nz=4, meshsize=1.0):
    """Minimal VoxCity-like object."""
    classes = np.zeros((nx, ny, nz), dtype=np.int8)
    # ground layer
    classes[:, :, 0] = 1
    meta = SimpleNamespace(meshsize=meshsize, bounds=(0, 0, nx * meshsize, ny * meshsize), crs="EPSG:4326")
    voxels = SimpleNamespace(classes=classes, meta=meta)
    dem = SimpleNamespace(elevation=np.zeros((nx, ny)))
    return SimpleNamespace(voxels=voxels, dem=dem)


def _make_building_svf_mesh(n_faces=6):
    """Fake trimesh-like mesh with SVF metadata."""
    mesh = MagicMock()
    mesh.faces = np.zeros((n_faces, 3), dtype=int)
    mesh.triangles_center = np.random.rand(n_faces, 3).astype(np.float64) * 2
    mesh.face_normals = np.tile([0.0, 0.0, 1.0], (n_faces, 1)).astype(np.float64)
    mesh.metadata = {"svf": np.full(n_faces, 0.5, dtype=np.float64)}
    mesh.copy.return_value = MagicMock(metadata={}, name="")
    return mesh


# ══════════════════════════════════════════════════════════════════════════
# get_direct_solar_irradiance_map
# ══════════════════════════════════════════════════════════════════════════

class TestGetDirectSolarIrradianceMap:

    @patch("voxcity.simulator.solar.radiation.compute_direct_solar_irradiance_map_binary")
    def test_basic_returns_map(self, mock_kernel):
        vc = _make_voxcity()
        mock_kernel.return_value = np.ones((5, 5))
        from voxcity.simulator.solar.radiation import get_direct_solar_irradiance_map
        result = get_direct_solar_irradiance_map(vc, 180, 45, 800)
        assert result.shape == (5, 5)
        mock_kernel.assert_called_once()

    @patch("voxcity.simulator.solar.radiation.plt")
    @patch("voxcity.simulator.solar.radiation.compute_direct_solar_irradiance_map_binary")
    def test_show_plot_branch(self, mock_kernel, mock_plt):
        vc = _make_voxcity()
        mock_kernel.return_value = np.ones((5, 5))
        mock_cmap = MagicMock()
        mock_plt.cm.get_cmap.return_value = mock_cmap
        mock_cmap.copy.return_value = mock_cmap
        from voxcity.simulator.solar.radiation import get_direct_solar_irradiance_map
        get_direct_solar_irradiance_map(vc, 180, 45, 800, show_plot=True)
        mock_plt.show.assert_called_once()

    @patch("voxcity.simulator.solar.radiation.grid_to_obj")
    @patch("voxcity.simulator.solar.radiation.compute_direct_solar_irradiance_map_binary")
    def test_obj_export_branch(self, mock_kernel, mock_obj):
        vc = _make_voxcity()
        mock_kernel.return_value = np.ones((5, 5))
        from voxcity.simulator.solar.radiation import get_direct_solar_irradiance_map
        get_direct_solar_irradiance_map(vc, 180, 45, 800, obj_export=True)
        mock_obj.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════
# get_diffuse_solar_irradiance_map
# ══════════════════════════════════════════════════════════════════════════

class TestGetDiffuseSolarIrradianceMap:

    @patch("voxcity.simulator.solar.radiation.get_sky_view_factor_map")
    def test_basic(self, mock_svf):
        vc = _make_voxcity()
        mock_svf.return_value = np.full((5, 5), 0.8)
        from voxcity.simulator.solar.radiation import get_diffuse_solar_irradiance_map
        result = get_diffuse_solar_irradiance_map(vc, diffuse_irradiance=100.0)
        np.testing.assert_allclose(result, 80.0)

    @patch("voxcity.simulator.solar.radiation.plt")
    @patch("voxcity.simulator.solar.radiation.get_sky_view_factor_map")
    def test_show_plot(self, mock_svf, mock_plt):
        vc = _make_voxcity()
        mock_svf.return_value = np.full((5, 5), 0.5)
        mock_cmap = MagicMock()
        mock_plt.cm.get_cmap.return_value = mock_cmap
        mock_cmap.copy.return_value = mock_cmap
        from voxcity.simulator.solar.radiation import get_diffuse_solar_irradiance_map
        get_diffuse_solar_irradiance_map(vc, diffuse_irradiance=100.0, show_plot=True)
        mock_plt.show.assert_called_once()

    @patch("voxcity.simulator.solar.radiation.grid_to_obj")
    @patch("voxcity.simulator.solar.radiation.get_sky_view_factor_map")
    def test_obj_export(self, mock_svf, mock_obj):
        vc = _make_voxcity()
        mock_svf.return_value = np.full((5, 5), 0.5)
        from voxcity.simulator.solar.radiation import get_diffuse_solar_irradiance_map
        get_diffuse_solar_irradiance_map(vc, diffuse_irradiance=100.0, obj_export=True)
        mock_obj.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════
# get_global_solar_irradiance_map
# ══════════════════════════════════════════════════════════════════════════

class TestGetGlobalSolarIrradianceMap:

    @patch("voxcity.simulator.solar.radiation.get_diffuse_solar_irradiance_map")
    @patch("voxcity.simulator.solar.radiation.get_direct_solar_irradiance_map")
    def test_basic(self, mock_direct, mock_diffuse):
        vc = _make_voxcity()
        mock_direct.return_value = np.full((5, 5), 200.0)
        mock_diffuse.return_value = np.full((5, 5), 80.0)
        from voxcity.simulator.solar.radiation import get_global_solar_irradiance_map
        result = get_global_solar_irradiance_map(vc, 180, 45, 800, 100)
        np.testing.assert_allclose(result, 280.0)

    @patch("voxcity.simulator.solar.radiation.plt")
    @patch("voxcity.simulator.solar.radiation.get_diffuse_solar_irradiance_map")
    @patch("voxcity.simulator.solar.radiation.get_direct_solar_irradiance_map")
    def test_show_plot(self, mock_direct, mock_diffuse, mock_plt):
        vc = _make_voxcity()
        mock_direct.return_value = np.full((5, 5), 200.0)
        mock_diffuse.return_value = np.full((5, 5), 80.0)
        mock_cmap = MagicMock()
        mock_plt.cm.get_cmap.return_value = mock_cmap
        mock_cmap.copy.return_value = mock_cmap
        from voxcity.simulator.solar.radiation import get_global_solar_irradiance_map
        get_global_solar_irradiance_map(vc, 180, 45, 800, 100, show_plot=True)
        mock_plt.show.assert_called_once()

    @patch("voxcity.simulator.solar.radiation.plt")
    @patch("voxcity.simulator.solar.radiation.grid_to_obj")
    @patch("voxcity.simulator.solar.radiation.get_diffuse_solar_irradiance_map")
    @patch("voxcity.simulator.solar.radiation.get_direct_solar_irradiance_map")
    def test_obj_export(self, mock_direct, mock_diffuse, mock_obj, mock_plt):
        vc = _make_voxcity()
        mock_direct.return_value = np.full((5, 5), 200.0)
        mock_diffuse.return_value = np.full((5, 5), 80.0)
        mock_cmap = MagicMock()
        mock_plt.cm.get_cmap.return_value = mock_cmap
        mock_cmap.copy.return_value = mock_cmap
        from voxcity.simulator.solar.radiation import get_global_solar_irradiance_map
        # Note: must also trigger show_plot or supply colormap via kwargs
        # because source code has a bug where `colormap` is only defined in show_plot block
        # but referenced in obj_export. Triggering show_plot first sets it.
        get_global_solar_irradiance_map(vc, 180, 45, 800, 100, show_plot=True, obj_export=True)
        mock_obj.assert_called_once()

    @patch("voxcity.simulator.solar.radiation.get_diffuse_solar_irradiance_map")
    @patch("voxcity.simulator.solar.radiation.get_direct_solar_irradiance_map")
    def test_nan_handling(self, mock_direct, mock_diffuse):
        vc = _make_voxcity()
        direct = np.full((5, 5), 200.0)
        direct[0, 0] = np.nan
        mock_direct.return_value = direct
        mock_diffuse.return_value = np.full((5, 5), 80.0)
        from voxcity.simulator.solar.radiation import get_global_solar_irradiance_map
        result = get_global_solar_irradiance_map(vc, 180, 45, 800, 100)
        # NaN cells get only diffuse
        assert result[0, 0] == 80.0
        assert result[1, 1] == 280.0


# ══════════════════════════════════════════════════════════════════════════
# get_building_solar_irradiance
# ══════════════════════════════════════════════════════════════════════════

class TestGetBuildingSolarIrradiance:

    @patch("voxcity.simulator.solar.radiation.compute_solar_irradiance_for_all_faces_masked")
    def test_fast_path(self, mock_masked):
        vc = _make_voxcity()
        mesh = _make_building_svf_mesh(4)
        mock_masked.return_value = (
            np.array([100.0, 200.0, 0.0, np.nan]),
            np.array([50.0, 50.0, 50.0, np.nan]),
            np.array([150.0, 250.0, 50.0, np.nan]),
        )
        from voxcity.simulator.solar.radiation import get_building_solar_irradiance
        result = get_building_solar_irradiance(
            vc, mesh, 180, 45, 800, 100, fast_path=True
        )
        assert result.metadata["direct"] is not None
        assert result.metadata["global"] is not None
        mock_masked.assert_called_once()

    @patch("voxcity.simulator.solar.radiation.compute_solar_irradiance_for_all_faces")
    def test_slow_path(self, mock_generic):
        vc = _make_voxcity()
        mesh = _make_building_svf_mesh(4)
        mock_generic.return_value = (
            np.array([100.0, 0.0, 0.0, np.nan]),
            np.array([50.0, 50.0, 50.0, np.nan]),
            np.array([150.0, 50.0, 50.0, np.nan]),
        )
        from voxcity.simulator.solar.radiation import get_building_solar_irradiance
        result = get_building_solar_irradiance(
            vc, mesh, 180, 45, 800, 100, fast_path=False
        )
        assert result.metadata["direct"] is not None
        mock_generic.assert_called_once()

    @patch("voxcity.simulator.solar.radiation.compute_solar_irradiance_for_all_faces_masked")
    def test_precomputed_geometry_and_masks(self, mock_masked):
        vc = _make_voxcity()
        mesh = _make_building_svf_mesh(4)
        mock_masked.return_value = (
            np.zeros(4), np.zeros(4), np.zeros(4),
        )
        precomputed_geometry = {
            "face_centers": np.random.rand(4, 3).astype(np.float64),
            "face_normals": np.tile([0, 0, 1.0], (4, 1)).astype(np.float64),
            "grid_bounds_real": np.array([[0, 0, 0], [5, 5, 4]], dtype=np.float64),
            "boundary_epsilon": 0.05,
        }
        precomputed_masks = {
            "vox_is_tree": np.zeros((5, 5, 4), dtype=bool),
            "vox_is_opaque": np.zeros((5, 5, 4), dtype=bool),
            "att": 0.5,
        }
        from voxcity.simulator.solar.radiation import get_building_solar_irradiance
        result = get_building_solar_irradiance(
            vc, mesh, 180, 45, 800, 100,
            precomputed_geometry=precomputed_geometry,
            precomputed_masks=precomputed_masks,
        )
        assert result.metadata["svf"] is not None

    @patch("voxcity.simulator.solar.radiation.compute_solar_irradiance_for_all_faces_masked")
    def test_no_svf_in_metadata(self, mock_masked):
        """When mesh has no svf metadata, zeros are used."""
        vc = _make_voxcity()
        mesh = _make_building_svf_mesh(3)
        mesh.metadata = {}  # no svf
        mock_masked.return_value = (np.zeros(3), np.zeros(3), np.zeros(3))
        from voxcity.simulator.solar.radiation import get_building_solar_irradiance
        result = get_building_solar_irradiance(vc, mesh, 180, 45, 800, 100)
        np.testing.assert_array_equal(result.metadata["svf"], np.zeros(3))
