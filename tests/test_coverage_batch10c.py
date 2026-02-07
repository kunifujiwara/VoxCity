"""
Coverage batch 10c – tests targeting remaining gaps in:
  temporal.py, integration.py, pipeline.py
"""

import os
import sys
import types
import tempfile
from unittest.mock import patch, MagicMock, PropertyMock, call

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Polygon, Point, box


# ---------------------------------------------------------------------------
# temporal.py – _configure_num_threads edge cases
# ---------------------------------------------------------------------------

class TestTemporalConfigureThreads:
    """Targets: lines 51-52, 56, 64-65."""

    def test_cpu_count_fails_fallback(self):
        """Lines 51-52 – os.cpu_count raises → cores = 4."""
        from voxcity.simulator.solar.temporal import _configure_num_threads

        with patch("os.cpu_count", side_effect=RuntimeError("no cpu")):
            result = _configure_num_threads(progress=False)
        assert result == 4

    def test_set_num_threads_fails(self):
        """Line 56 – numba.set_num_threads fails → silently passes."""
        from voxcity.simulator.solar.temporal import _configure_num_threads

        with patch("voxcity.simulator.solar.temporal.numba") as mock_numba:
            mock_numba.set_num_threads.side_effect = RuntimeError("fail")
            result = _configure_num_threads(desired_threads=2, progress=False)
        assert result == 2

    def test_progress_print_get_num_threads_fails(self):
        """Lines 64-65 – progress=True, get_num_threads fails → fallback print."""
        from voxcity.simulator.solar.temporal import _configure_num_threads

        with patch("voxcity.simulator.solar.temporal.numba") as mock_numba:
            mock_numba.set_num_threads = MagicMock()
            mock_numba.get_num_threads.side_effect = RuntimeError("fail")
            result = _configure_num_threads(desired_threads=2, progress=True)
        assert result == 2


# ---------------------------------------------------------------------------
# pipeline.py – DemSourceFactory
# ---------------------------------------------------------------------------

class TestPipelineEdgeCases:
    """Targets: lines 641-645."""

    def test_dem_source_factory_exception_fallback(self):
        """DemSourceFactory.create handles exception in normalization."""
        from voxcity.generator.pipeline import DemSourceFactory

        class BadStr:
            def strip(self):
                raise RuntimeError("boom")

        result = DemSourceFactory.create(BadStr())
        assert result is not None

    def test_dem_source_factory_none_source(self):
        """source=None → FlatDemStrategy."""
        from voxcity.generator.pipeline import DemSourceFactory

        result = DemSourceFactory.create(None)
        assert result is not None

    def test_dem_source_factory_flat_string(self):
        """'flat' string → FlatDemStrategy."""
        from voxcity.generator.pipeline import DemSourceFactory

        result = DemSourceFactory.create("flat")
        assert result is not None

    def test_dem_source_factory_none_string(self):
        """'none' string → FlatDemStrategy."""
        from voxcity.generator.pipeline import DemSourceFactory

        result = DemSourceFactory.create("none")
        assert result is not None

    def test_dem_source_factory_normal_source(self):
        """normal DEM source string → SourceDemStrategy."""
        from voxcity.generator.pipeline import DemSourceFactory

        result = DemSourceFactory.create("OpenTopography")
        assert result is not None


# ---------------------------------------------------------------------------
# pipeline.py – LandCoverSourceFactory, CanopySourceFactory
# ---------------------------------------------------------------------------

class TestLandCoverSourceFactory:

    def test_lc_source_factory(self):
        from voxcity.generator.pipeline import LandCoverSourceFactory
        result = LandCoverSourceFactory.create("OpenStreetMap")
        assert result is not None

    def test_lc_source_factory_none(self):
        from voxcity.generator.pipeline import LandCoverSourceFactory
        result = LandCoverSourceFactory.create(None)
        assert result is not None


class TestCanopySourceFactory:

    def test_canopy_source_factory(self):
        from voxcity.generator.pipeline import CanopySourceFactory
        mock_cfg = MagicMock()
        result = CanopySourceFactory.create("OpenStreetMap", mock_cfg)
        assert result is not None

    def test_canopy_source_factory_static(self):
        from voxcity.generator.pipeline import CanopySourceFactory
        mock_cfg = MagicMock()
        result = CanopySourceFactory.create("Static", mock_cfg)
        assert result is not None


# ---------------------------------------------------------------------------
# integration.py – get_global_solar_irradiance_using_epw edge cases
# ---------------------------------------------------------------------------

class TestIntegrationEdgeCases:
    """Targets: lines 62-65 (no data at time), 79 (empty df), 120 (invalid calc_type)."""

    def test_invalid_calc_type_raises(self):
        """invalid calc_type → ValueError."""
        from voxcity.simulator.solar.integration import (
            get_global_solar_irradiance_using_epw,
        )

        mock_voxcity = MagicMock()
        mock_voxcity.extras = {}

        dates = pd.date_range("2023-01-01", periods=8760, freq="h")
        df = pd.DataFrame({"DNI": [300.0] * 8760, "DHI": [100.0] * 8760}, index=dates)

        with patch(
            "voxcity.simulator.solar.integration.read_epw_for_solar_simulation",
            return_value=(df, 139.0, 35.0, 9.0, 10.0),
        ):
            with pytest.raises(ValueError, match="calc_type"):
                get_global_solar_irradiance_using_epw(
                    mock_voxcity,
                    calc_type="invalid_type",
                    epw_file_path="fake.epw",
                )

    def test_no_epw_path_raises(self):
        """No epw_file_path and download_nearest_epw=False → ValueError."""
        from voxcity.simulator.solar.integration import (
            get_global_solar_irradiance_using_epw,
        )

        mock_voxcity = MagicMock()
        mock_voxcity.extras = {}

        with pytest.raises(ValueError, match="epw_file_path"):
            get_global_solar_irradiance_using_epw(
                mock_voxcity,
                calc_type="instantaneous",
            )

    def test_empty_epw_raises(self):
        """empty EPW DataFrame → ValueError."""
        from voxcity.simulator.solar.integration import (
            get_global_solar_irradiance_using_epw,
        )

        mock_voxcity = MagicMock()
        mock_voxcity.extras = {}

        empty_df = pd.DataFrame(columns=["DNI", "DHI"])

        with patch(
            "voxcity.simulator.solar.integration.read_epw_for_solar_simulation",
            return_value=(empty_df, 139.0, 35.0, 9.0, 10.0),
        ):
            with pytest.raises(ValueError, match="No data"):
                get_global_solar_irradiance_using_epw(
                    mock_voxcity,
                    calc_type="instantaneous",
                    epw_file_path="fake.epw",
                )

    def test_download_epw_no_rectangle_returns_none(self):
        """download_nearest_epw=True but no rectangle_vertices → None."""
        from voxcity.simulator.solar.integration import (
            get_global_solar_irradiance_using_epw,
        )

        mock_voxcity = MagicMock()
        mock_voxcity.extras = {}

        result = get_global_solar_irradiance_using_epw(
            mock_voxcity,
            calc_type="instantaneous",
            download_nearest_epw=True,
        )
        assert result is None


# ---------------------------------------------------------------------------
# integration.py – building irradiance function
# ---------------------------------------------------------------------------

class TestBuildingIntegrationEdgeCases:

    def test_no_voxcity_raises(self):
        """voxcity not provided → ValueError."""
        from voxcity.simulator.solar.integration import (
            get_building_global_solar_irradiance_using_epw,
        )

        with pytest.raises(ValueError, match="voxcity"):
            get_building_global_solar_irradiance_using_epw(
                calc_type="instantaneous",
            )

    def test_building_download_no_rectangle(self):
        """download_nearest_epw=True but no rectangle_vertices → returns None."""
        from voxcity.simulator.solar.integration import (
            get_building_global_solar_irradiance_using_epw,
        )

        mock_voxcity = MagicMock()
        mock_voxcity.extras = {}

        result = get_building_global_solar_irradiance_using_epw(
            voxcity=mock_voxcity,
            calc_type="instantaneous",
            download_nearest_epw=True,
        )
        assert result is None


# ---------------------------------------------------------------------------
# integration.py – save/load irradiance mesh
# ---------------------------------------------------------------------------

class TestSaveLoadIrradianceMesh:

    def test_save_and_load_irradiance_mesh(self):
        """save_irradiance_mesh / load_irradiance_mesh round trip."""
        from voxcity.simulator.solar.integration import (
            save_irradiance_mesh,
            load_irradiance_mesh,
        )
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_mesh.pkl")
            save_irradiance_mesh(mesh, path)
            assert os.path.exists(path)
            loaded = load_irradiance_mesh(path)
            assert loaded is not None


# ---------------------------------------------------------------------------
# temporal.py – _auto_time_batch_size
# ---------------------------------------------------------------------------

class TestAutoTimeBatchSize:

    def test_auto_batch_size_default(self):
        from voxcity.simulator.solar.temporal import _auto_time_batch_size
        result = _auto_time_batch_size(n_faces=1000, total_steps=100)
        assert isinstance(result, int)
        assert result > 0

    def test_auto_batch_size_user_value(self):
        from voxcity.simulator.solar.temporal import _auto_time_batch_size
        result = _auto_time_batch_size(n_faces=1000, total_steps=100, user_value=50)
        assert result == 50

    def test_auto_batch_size_large_mesh(self):
        from voxcity.simulator.solar.temporal import _auto_time_batch_size
        result = _auto_time_batch_size(n_faces=1_000_000, total_steps=1000)
        assert isinstance(result, int)
        assert result > 0


# ---------------------------------------------------------------------------
# temporal.py – get_solar_positions_astral
# ---------------------------------------------------------------------------

class TestSolarPositionsAstral:

    def test_solar_positions_basic(self):
        from voxcity.simulator.solar.temporal import get_solar_positions_astral
        import pytz

        times = pd.date_range("2023-06-21 06:00", periods=5, freq="h", tz=pytz.UTC)
        result = get_solar_positions_astral(times, lon=139.0, lat=35.0)
        assert "elevation" in result.columns
        assert "azimuth" in result.columns
        assert len(result) == 5
