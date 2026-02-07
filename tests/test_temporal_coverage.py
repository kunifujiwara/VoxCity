"""
Comprehensive tests for voxcity.simulator.solar.temporal to improve coverage.
Covers: get_solar_positions_astral, _configure_num_threads, _auto_time_batch_size,
_aggregate_weather_to_sky_patches
"""

import numpy as np
import pytest
from datetime import datetime
import pytz


from voxcity.simulator.solar.temporal import (
    get_solar_positions_astral,
    _configure_num_threads,
    _auto_time_batch_size,
    _aggregate_weather_to_sky_patches,
)


class TestGetSolarPositionsAstral:
    def test_basic_output(self):
        import pandas as pd
        tz = pytz.UTC
        times = pd.date_range("2024-06-21 06:00", periods=5, freq="1h", tz=tz)
        result = get_solar_positions_astral(times, lon=139.7, lat=35.68)
        assert len(result) == 5
        assert 'azimuth' in result.columns
        assert 'elevation' in result.columns

    def test_night_low_elevation(self):
        import pandas as pd
        tz = pytz.UTC
        # 18:00 UTC = 03:00 JST (next day), well after sunset in Tokyo
        times = pd.date_range("2024-01-01 18:00", periods=3, freq="1h", tz=tz)
        result = get_solar_positions_astral(times, lon=139.7, lat=35.68)
        # At 3-5 AM JST, elevation should be negative (below horizon)
        assert all(result['elevation'] < 0)

    def test_noon_positive_elevation(self):
        import pandas as pd
        tz = pytz.UTC
        # Noon UTC is ~21:00 JST, but let's pick ~03:00 UTC = noon JST
        times = pd.date_range("2024-06-21 03:00", periods=1, freq="1h", tz=tz)
        result = get_solar_positions_astral(times, lon=139.7, lat=35.68)
        assert result['elevation'].iloc[0] > 0


class TestConfigureNumThreads:
    def test_default_uses_cpu_count(self):
        import os
        result = _configure_num_threads()
        assert result > 0

    def test_specific_count(self):
        result = _configure_num_threads(desired_threads=2)
        assert result == 2

    def test_progress_output(self, capsys):
        _configure_num_threads(desired_threads=2, progress=True)
        captured = capsys.readouterr()
        assert 'Numba threads' in captured.out or 'threads' in captured.out


class TestAutoTimeBatchSize:
    def test_user_value_override(self):
        assert _auto_time_batch_size(1000, 100, user_value=10) == 10

    def test_user_value_minimum_1(self):
        assert _auto_time_batch_size(1000, 100, user_value=0) == 1

    def test_zero_steps(self):
        assert _auto_time_batch_size(1000, 0) == 1

    def test_small_faces(self):
        result = _auto_time_batch_size(100, 100)
        assert result >= 1

    def test_medium_faces(self):
        result = _auto_time_batch_size(10000, 100)
        assert result >= 1

    def test_large_faces(self):
        result = _auto_time_batch_size(100000, 1000)
        assert result >= 1

    def test_very_large_faces(self):
        result = _auto_time_batch_size(500000, 1000)
        assert result >= 1


class TestAggregateWeatherToSkyPatches:
    def _make_weather_data(self, n=100):
        """Create synthetic weather data with sun above horizon."""
        np.random.seed(42)
        azimuths = np.random.uniform(90, 270, n)
        elevations = np.random.uniform(5, 70, n)
        dnis = np.random.uniform(100, 900, n)
        dhis = np.random.uniform(50, 300, n)
        return azimuths, elevations, dnis, dhis

    def test_tregenza(self):
        az, el, dni, dhi = self._make_weather_data()
        result = _aggregate_weather_to_sky_patches(az, el, dni, dhi, sky_discretization="tregenza")
        assert 'patch_directions' in result
        assert 'patch_cumulative_dni' in result
        assert 'total_cumulative_dhi' in result
        assert result['total_cumulative_dhi'] > 0

    def test_reinhart(self):
        az, el, dni, dhi = self._make_weather_data()
        result = _aggregate_weather_to_sky_patches(az, el, dni, dhi, sky_discretization="reinhart", mf=2)
        assert result['n_patches'] > 0
        assert result['patch_directions'].shape[0] > 0

    def test_uniform(self):
        az, el, dni, dhi = self._make_weather_data()
        result = _aggregate_weather_to_sky_patches(az, el, dni, dhi, sky_discretization="uniform", n_azimuth=12, n_elevation=6)
        assert result['n_patches'] > 0

    def test_fibonacci(self):
        az, el, dni, dhi = self._make_weather_data()
        result = _aggregate_weather_to_sky_patches(az, el, dni, dhi, sky_discretization="fibonacci", n_patches=100)
        assert result['n_patches'] > 0

    def test_unknown_method_raises(self):
        az, el, dni, dhi = self._make_weather_data(10)
        with pytest.raises(ValueError, match="Unknown sky discretization"):
            _aggregate_weather_to_sky_patches(az, el, dni, dhi, sky_discretization="invalid")

    def test_below_horizon_ignored(self):
        # All sun positions below horizon
        az = np.array([180.0, 180.0])
        el = np.array([-10.0, -5.0])
        dni = np.array([500.0, 500.0])
        dhi = np.array([100.0, 100.0])
        result = _aggregate_weather_to_sky_patches(az, el, dni, dhi, sky_discretization="tregenza")
        # No DNI should be accumulated (sun below horizon)
        assert np.sum(result['patch_cumulative_dni']) == 0.0

    def test_zero_dni_ignored(self):
        az = np.array([180.0])
        el = np.array([45.0])
        dni = np.array([0.0])
        dhi = np.array([200.0])
        result = _aggregate_weather_to_sky_patches(az, el, dni, dhi, sky_discretization="tregenza")
        assert np.sum(result['patch_cumulative_dni']) == 0.0
        assert result['total_cumulative_dhi'] > 0

    def test_time_step_scaling(self):
        az = np.array([180.0])
        el = np.array([45.0])
        dni = np.array([500.0])
        dhi = np.array([200.0])
        r1 = _aggregate_weather_to_sky_patches(az, el, dni, dhi, time_step_hours=1.0)
        r2 = _aggregate_weather_to_sky_patches(az, el, dni, dhi, time_step_hours=2.0)
        assert r2['total_cumulative_dhi'] == pytest.approx(2 * r1['total_cumulative_dhi'])
