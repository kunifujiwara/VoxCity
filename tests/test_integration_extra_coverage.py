"""
Tests for integration.py additional coverage:
  - download_nearest_epw branch
  - instantaneous calc_type path
  - cumulative calc_type path
  - save_mesh branch
  - invalid calc_type
  - no svf_mesh computes one
"""
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from datetime import datetime

# The function under test uses *args/**kwargs. VoxCity isinstance check
# requires passing voxcity as a keyword arg when using SimpleNamespace.
# _configure_num_threads is a LOCAL import from .temporal → mock at source.
# get_solar_positions_astral is also a LOCAL import from .temporal → mock at source.

_INTEG = "voxcity.simulator.solar.integration"
_TEMPORAL = "voxcity.simulator.solar.temporal"


def _make_voxcity(nx=5, ny=5, nz=4, meshsize=1.0):
    classes = np.zeros((nx, ny, nz), dtype=np.int8)
    classes[:, :, 0] = 1  # ground
    meta = SimpleNamespace(meshsize=meshsize, bounds=(0, 0, nx * meshsize, ny * meshsize), crs="EPSG:4326")
    voxels = SimpleNamespace(classes=classes, meta=meta)
    dem = SimpleNamespace(elevation=np.zeros((nx, ny)))
    extras = {"rectangle_vertices": [(0, 0), (0, 5), (5, 5), (5, 0)]}
    return SimpleNamespace(voxels=voxels, dem=dem, extras=extras)


def _make_epw_df(n_hours=24):
    """Jan 15 data with tz-naive index."""
    base = datetime(2023, 1, 15, 0, 0, 0)
    times = pd.date_range(base, periods=n_hours, freq="h")
    return pd.DataFrame({
        "DNI": [500.0] * n_hours,
        "DHI": [80.0] * n_hours,
    }, index=times)


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


def _solar_side_effect(times, lon, lat):
    """Return solar positions DataFrame matching the index passed in."""
    n = len(times)
    return pd.DataFrame({"azimuth": [180.0] * n, "elevation": [45.0] * n}, index=times)


class TestBuildingGlobalSolarIrradianceUsingEpw:
    """Tests for get_building_global_solar_irradiance_using_epw."""

    @patch(f"{_INTEG}.save_irradiance_mesh")
    @patch(f"{_INTEG}.get_building_solar_irradiance")
    @patch(f"{_TEMPORAL}.get_solar_positions_astral", side_effect=_solar_side_effect)
    @patch(f"{_TEMPORAL}._configure_num_threads")
    @patch(f"{_INTEG}.read_epw_for_solar_simulation")
    def test_instantaneous_with_save_mesh(self, mock_read, mock_cfg, mock_solar, mock_bsi, mock_save):
        vc = _make_voxcity()
        df = _make_epw_df(24)
        mock_read.return_value = (df, 139.75, 35.68, 9, 10.0)

        irr = MagicMock()
        irr.metadata = {"direct": np.array([100.0]), "diffuse": np.array([30.0]), "global": np.array([130.0])}
        mock_bsi.return_value = irr

        mesh = _make_building_svf_mesh(4)
        from voxcity.simulator.solar.integration import get_building_global_solar_irradiance_using_epw
        result = get_building_global_solar_irradiance_using_epw(
            voxcity=vc,
            building_svf_mesh=mesh,
            calc_type="instantaneous",
            epw_file_path="test.epw",
            calc_time="01-15 12:00:00",
            save_mesh=True,
            mesh_output_path="test_output.pkl",
            progress_report=True,
        )
        assert result is not None
        mock_save.assert_called_once()
        mock_cfg.assert_called_once()

    @patch(f"{_INTEG}.get_cumulative_building_solar_irradiance")
    @patch(f"{_TEMPORAL}._configure_num_threads")
    @patch(f"{_INTEG}.read_epw_for_solar_simulation")
    def test_cumulative_calc_type(self, mock_read, mock_cfg, mock_cum):
        vc = _make_voxcity()
        df = _make_epw_df(24)
        mock_read.return_value = (df, 139.75, 35.68, 9, 10.0)

        cum_mesh = MagicMock()
        cum_mesh.metadata = {"direct": np.array([500.0]), "diffuse": np.array([100.0]), "global": np.array([600.0])}
        mock_cum.return_value = cum_mesh

        mesh = _make_building_svf_mesh(4)
        from voxcity.simulator.solar.integration import get_building_global_solar_irradiance_using_epw
        result = get_building_global_solar_irradiance_using_epw(
            voxcity=vc,
            building_svf_mesh=mesh,
            calc_type="cumulative",
            epw_file_path="test.epw",
        )
        assert result is not None
        mock_cum.assert_called_once()

    @patch(f"{_TEMPORAL}._configure_num_threads")
    @patch(f"{_INTEG}.read_epw_for_solar_simulation")
    def test_invalid_calc_type(self, mock_read, mock_cfg):
        vc = _make_voxcity()
        df = _make_epw_df(24)
        mock_read.return_value = (df, 139.75, 35.68, 9, 10.0)

        mesh = _make_building_svf_mesh(4)
        from voxcity.simulator.solar.integration import get_building_global_solar_irradiance_using_epw
        with pytest.raises(ValueError, match="calc_type"):
            get_building_global_solar_irradiance_using_epw(
                voxcity=vc,
                building_svf_mesh=mesh,
                calc_type="invalid",
                epw_file_path="test.epw",
            )

    @patch(f"{_INTEG}.get_nearest_epw_from_climate_onebuilding")
    @patch(f"{_INTEG}.get_building_solar_irradiance")
    @patch(f"{_TEMPORAL}.get_solar_positions_astral", side_effect=_solar_side_effect)
    @patch(f"{_TEMPORAL}._configure_num_threads")
    @patch(f"{_INTEG}.read_epw_for_solar_simulation")
    def test_download_nearest_epw(self, mock_read, mock_cfg, mock_solar, mock_bsi, mock_epw_dl):
        vc = _make_voxcity()
        df = _make_epw_df(24)
        mock_epw_dl.return_value = ("downloaded.epw", None, None)
        mock_read.return_value = (df, 139.75, 35.68, 9, 10.0)

        irr = MagicMock()
        irr.metadata = {"direct": np.array([100.0]), "diffuse": np.array([30.0]), "global": np.array([130.0])}
        mock_bsi.return_value = irr

        mesh = _make_building_svf_mesh(4)
        from voxcity.simulator.solar.integration import get_building_global_solar_irradiance_using_epw
        result = get_building_global_solar_irradiance_using_epw(
            voxcity=vc,
            building_svf_mesh=mesh,
            calc_type="instantaneous",
            download_nearest_epw=True,
            rectangle_vertices=[(139.7, 35.6), (139.7, 35.7), (139.8, 35.7), (139.8, 35.6)],
            calc_time="01-15 12:00:00",
        )
        assert result is not None
        mock_epw_dl.assert_called_once()

    @patch(f"{_TEMPORAL}._configure_num_threads")
    @patch(f"{_INTEG}.read_epw_for_solar_simulation")
    def test_download_nearest_epw_no_rectangle(self, mock_read, mock_cfg):
        """download_nearest_epw=True without rectangle_vertices returns None."""
        vc = SimpleNamespace(
            voxels=SimpleNamespace(classes=np.zeros((5, 5, 4), dtype=np.int8), meta=SimpleNamespace(meshsize=1.0)),
            dem=SimpleNamespace(elevation=np.zeros((5, 5))),
            extras=None,
        )
        df = _make_epw_df(24)
        mock_read.return_value = (df, 139.75, 35.68, 9, 10.0)

        mesh = _make_building_svf_mesh(4)
        from voxcity.simulator.solar.integration import get_building_global_solar_irradiance_using_epw
        result = get_building_global_solar_irradiance_using_epw(
            voxcity=vc,
            building_svf_mesh=mesh,
            calc_type="instantaneous",
            download_nearest_epw=True,
        )
        assert result is None

    @patch(f"{_TEMPORAL}._configure_num_threads")
    @patch(f"{_INTEG}.read_epw_for_solar_simulation")
    def test_no_epw_raises(self, mock_read, mock_cfg):
        """No epw_file_path and download_nearest_epw=False raises ValueError."""
        vc = _make_voxcity()
        df = _make_epw_df(24)
        mock_read.return_value = (df, 139.75, 35.68, 9, 10.0)

        mesh = _make_building_svf_mesh(4)
        from voxcity.simulator.solar.integration import get_building_global_solar_irradiance_using_epw
        with pytest.raises(ValueError, match="epw_file_path"):
            get_building_global_solar_irradiance_using_epw(
                voxcity=vc,
                building_svf_mesh=mesh,
                calc_type="instantaneous",
            )

    @patch(f"{_INTEG}.get_surface_view_factor")
    @patch(f"{_INTEG}.get_building_solar_irradiance")
    @patch(f"{_TEMPORAL}.get_solar_positions_astral", side_effect=_solar_side_effect)
    @patch(f"{_TEMPORAL}._configure_num_threads")
    @patch(f"{_INTEG}.read_epw_for_solar_simulation")
    def test_no_svf_mesh_computes_one(self, mock_read, mock_cfg, mock_solar, mock_bsi, mock_svf):
        """building_svf_mesh=None triggers get_surface_view_factor."""
        vc = _make_voxcity()
        df = _make_epw_df(24)
        mock_read.return_value = (df, 139.75, 35.68, 9, 10.0)

        svf_mesh = _make_building_svf_mesh(4)
        mock_svf.return_value = svf_mesh

        irr = MagicMock()
        irr.metadata = {"direct": np.array([100.0]), "diffuse": np.array([30.0]), "global": np.array([130.0])}
        mock_bsi.return_value = irr

        from voxcity.simulator.solar.integration import get_building_global_solar_irradiance_using_epw
        result = get_building_global_solar_irradiance_using_epw(
            voxcity=vc,
            building_svf_mesh=None,
            calc_type="instantaneous",
            epw_file_path="test.epw",
            calc_time="01-15 12:00:00",
            progress_report=True,
        )
        assert result is not None
        mock_svf.assert_called_once()

    def test_no_voxcity_raises(self):
        """Calling without voxcity raises ValueError."""
        from voxcity.simulator.solar.integration import get_building_global_solar_irradiance_using_epw
        with pytest.raises(ValueError, match="voxcity"):
            get_building_global_solar_irradiance_using_epw(calc_type="instantaneous")
