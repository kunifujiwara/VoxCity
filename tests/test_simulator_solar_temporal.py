"""Tests for voxcity.simulator.solar.temporal module."""
import pytest
import numpy as np
from datetime import datetime
import pytz


class TestConfigureNumThreads:
    """Tests for _configure_num_threads function."""

    def test_default_uses_cpu_count(self):
        """Without desired_threads, should use os.cpu_count()."""
        from voxcity.simulator.solar.temporal import _configure_num_threads
        import os
        
        result = _configure_num_threads()
        # Should be at least 1
        assert result >= 1

    def test_explicit_thread_count(self):
        """Should respect explicit thread count."""
        from voxcity.simulator.solar.temporal import _configure_num_threads
        
        result = _configure_num_threads(desired_threads=2)
        assert result == 2

    def test_progress_output(self, capsys):
        """With progress=True, should print thread info."""
        from voxcity.simulator.solar.temporal import _configure_num_threads
        
        _configure_num_threads(desired_threads=2, progress=True)
        captured = capsys.readouterr()
        assert "threads" in captured.out.lower() or "numba" in captured.out.lower()


class TestAutoTimeBatchSize:
    """Tests for _auto_time_batch_size function."""

    def test_user_value_respected(self):
        """Should use user_value when provided."""
        from voxcity.simulator.solar.temporal import _auto_time_batch_size
        
        result = _auto_time_batch_size(1000, 100, user_value=10)
        assert result == 10

    def test_user_value_minimum_one(self):
        """User value should be at least 1."""
        from voxcity.simulator.solar.temporal import _auto_time_batch_size
        
        result = _auto_time_batch_size(1000, 100, user_value=0)
        assert result == 1

    def test_small_faces_fewer_batches(self):
        """Small face count should result in fewer batches."""
        from voxcity.simulator.solar.temporal import _auto_time_batch_size
        
        result_small = _auto_time_batch_size(1000, 100)  # n_faces <= 5000
        result_large = _auto_time_batch_size(100000, 100)  # n_faces > 50000
        
        # Larger faces should have smaller batch size (more batches)
        assert result_small >= result_large

    def test_zero_steps_returns_one(self):
        """Zero total_steps should return 1."""
        from voxcity.simulator.solar.temporal import _auto_time_batch_size
        
        result = _auto_time_batch_size(1000, 0)
        assert result == 1

    def test_returns_at_least_one(self):
        """Should always return at least 1."""
        from voxcity.simulator.solar.temporal import _auto_time_batch_size
        
        for n_faces in [10, 1000, 10000, 100000, 1000000]:
            for total_steps in [1, 10, 100]:
                result = _auto_time_batch_size(n_faces, total_steps)
                assert result >= 1


class TestGetSolarPositionsAstral:
    """Tests for get_solar_positions_astral function."""

    def test_returns_dataframe_with_correct_columns(self):
        """Should return DataFrame with azimuth and elevation columns."""
        from voxcity.simulator.solar.temporal import get_solar_positions_astral
        
        tz = pytz.timezone('Asia/Tokyo')
        times = [tz.localize(datetime(2024, 6, 21, 12, 0))]
        lon, lat = 139.7, 35.7  # Tokyo
        
        result = get_solar_positions_astral(times, lon, lat)
        
        assert 'azimuth' in result.columns
        assert 'elevation' in result.columns

    def test_noon_summer_high_elevation(self):
        """Noon in summer should have high sun elevation."""
        from voxcity.simulator.solar.temporal import get_solar_positions_astral
        
        tz = pytz.timezone('Asia/Tokyo')
        times = [tz.localize(datetime(2024, 6, 21, 12, 0))]  # Summer solstice
        lon, lat = 139.7, 35.7  # Tokyo
        
        result = get_solar_positions_astral(times, lon, lat)
        
        # At noon on summer solstice, elevation should be high
        assert result.iloc[0]['elevation'] > 60

    def test_midnight_negative_elevation(self):
        """Midnight should have negative (below horizon) elevation."""
        from voxcity.simulator.solar.temporal import get_solar_positions_astral
        
        tz = pytz.timezone('Asia/Tokyo')
        times = [tz.localize(datetime(2024, 6, 21, 0, 0))]  # Midnight
        lon, lat = 139.7, 35.7  # Tokyo
        
        result = get_solar_positions_astral(times, lon, lat)
        
        # At midnight, sun should be below horizon
        assert result.iloc[0]['elevation'] < 0

    def test_multiple_times(self):
        """Should handle multiple timestamps."""
        from voxcity.simulator.solar.temporal import get_solar_positions_astral
        
        tz = pytz.timezone('Asia/Tokyo')
        times = [
            tz.localize(datetime(2024, 6, 21, 6, 0)),
            tz.localize(datetime(2024, 6, 21, 12, 0)),
            tz.localize(datetime(2024, 6, 21, 18, 0)),
        ]
        lon, lat = 139.7, 35.7
        
        result = get_solar_positions_astral(times, lon, lat)
        
        assert len(result) == 3
        # Noon should have highest elevation
        assert result.iloc[1]['elevation'] > result.iloc[0]['elevation']
        assert result.iloc[1]['elevation'] > result.iloc[2]['elevation']


class TestAggregateWeatherToSkyPatches:
    """Tests for _aggregate_weather_to_sky_patches function."""

    def test_tregenza_returns_expected_keys(self):
        """Should return dict with all expected keys."""
        from voxcity.simulator.solar.temporal import _aggregate_weather_to_sky_patches
        
        # Simple test with one sunny timestep
        azimuth = np.array([180.0])
        elevation = np.array([45.0])
        dni = np.array([500.0])
        dhi = np.array([100.0])
        
        result = _aggregate_weather_to_sky_patches(
            azimuth, elevation, dni, dhi,
            time_step_hours=1.0,
            sky_discretization="tregenza"
        )
        
        assert 'patches' in result
        assert 'patch_directions' in result
        assert 'patch_cumulative_dni' in result
        assert 'patch_solid_angles' in result
        assert 'patch_hours' in result
        assert 'active_mask' in result
        assert 'n_active_patches' in result
        assert 'total_cumulative_dhi' in result
        assert 'n_patches' in result
        assert 'n_original_timesteps' in result

    def test_accumulates_dhi(self):
        """Should accumulate DHI correctly."""
        from voxcity.simulator.solar.temporal import _aggregate_weather_to_sky_patches
        
        # Multiple timesteps with DHI
        azimuth = np.array([0.0, 90.0, 180.0])
        elevation = np.array([30.0, 45.0, 60.0])
        dni = np.array([0.0, 0.0, 0.0])  # No direct
        dhi = np.array([100.0, 150.0, 200.0])  # Diffuse only
        
        result = _aggregate_weather_to_sky_patches(
            azimuth, elevation, dni, dhi,
            time_step_hours=1.0,
            sky_discretization="tregenza"
        )
        
        # Total DHI should be sum * time_step_hours = 450 Wh/m²
        assert result['total_cumulative_dhi'] == pytest.approx(450.0)

    def test_timestep_scaling(self):
        """Time step hours should scale accumulation."""
        from voxcity.simulator.solar.temporal import _aggregate_weather_to_sky_patches
        
        azimuth = np.array([180.0])
        elevation = np.array([45.0])
        dni = np.array([0.0])
        dhi = np.array([100.0])
        
        result1 = _aggregate_weather_to_sky_patches(
            azimuth, elevation, dni, dhi,
            time_step_hours=1.0,
            sky_discretization="tregenza"
        )
        
        result2 = _aggregate_weather_to_sky_patches(
            azimuth, elevation, dni, dhi,
            time_step_hours=0.5,
            sky_discretization="tregenza"
        )
        
        # DHI with 0.5 hour should be half
        assert result2['total_cumulative_dhi'] == pytest.approx(result1['total_cumulative_dhi'] / 2)

    def test_below_horizon_no_dni_accumulation(self):
        """Sun below horizon should not accumulate DNI."""
        from voxcity.simulator.solar.temporal import _aggregate_weather_to_sky_patches
        
        azimuth = np.array([180.0])
        elevation = np.array([-10.0])  # Below horizon
        dni = np.array([500.0])
        dhi = np.array([50.0])
        
        result = _aggregate_weather_to_sky_patches(
            azimuth, elevation, dni, dhi,
            time_step_hours=1.0,
            sky_discretization="tregenza"
        )
        
        # No DNI should accumulate for below-horizon sun
        assert np.sum(result['patch_cumulative_dni']) == pytest.approx(0.0)
        # But DHI still accumulates
        assert result['total_cumulative_dhi'] == pytest.approx(50.0)

    def test_unknown_discretization_raises_error(self):
        """Unknown sky discretization should raise ValueError."""
        from voxcity.simulator.solar.temporal import _aggregate_weather_to_sky_patches
        
        azimuth = np.array([180.0])
        elevation = np.array([45.0])
        dni = np.array([500.0])
        dhi = np.array([100.0])
        
        with pytest.raises(ValueError, match="Unknown sky discretization"):
            _aggregate_weather_to_sky_patches(
                azimuth, elevation, dni, dhi,
                sky_discretization="invalid_method"
            )

    def test_fibonacci_method(self):
        """Should work with fibonacci discretization."""
        from voxcity.simulator.solar.temporal import _aggregate_weather_to_sky_patches
        
        azimuth = np.array([180.0])
        elevation = np.array([45.0])
        dni = np.array([500.0])
        dhi = np.array([100.0])
        
        result = _aggregate_weather_to_sky_patches(
            azimuth, elevation, dni, dhi,
            sky_discretization="fibonacci",
            n_patches=100
        )
        
        assert result['n_patches'] == 100
        assert result['method'] == "fibonacci"

    def test_uniform_method(self):
        """Should work with uniform grid discretization."""
        from voxcity.simulator.solar.temporal import _aggregate_weather_to_sky_patches
        
        azimuth = np.array([180.0])
        elevation = np.array([45.0])
        dni = np.array([500.0])
        dhi = np.array([100.0])
        
        result = _aggregate_weather_to_sky_patches(
            azimuth, elevation, dni, dhi,
            sky_discretization="uniform",
            n_azimuth=18,
            n_elevation=5
        )
        
        # Should have 18 * 5 = 90 patches
        assert result['n_patches'] == 90
        assert result['method'] == "uniform"

    def test_reinhart_method(self):
        """Should work with reinhart discretization."""
        from voxcity.simulator.solar.temporal import _aggregate_weather_to_sky_patches
        
        azimuth = np.array([180.0])
        elevation = np.array([45.0])
        dni = np.array([500.0])
        dhi = np.array([100.0])
        
        result = _aggregate_weather_to_sky_patches(
            azimuth, elevation, dni, dhi,
            sky_discretization="reinhart",
            mf=2
        )
        
        assert result['method'] == "reinhart"
        assert result['n_patches'] > 145  # More patches than Tregenza

    def test_active_patches_count(self):
        """Should correctly count active patches with DNI."""
        from voxcity.simulator.solar.temporal import _aggregate_weather_to_sky_patches
        
        # Sun at specific position
        azimuth = np.array([180.0])
        elevation = np.array([45.0])
        dni = np.array([500.0])
        dhi = np.array([0.0])
        
        result = _aggregate_weather_to_sky_patches(
            azimuth, elevation, dni, dhi,
            sky_discretization="tregenza"
        )
        
        # Only one patch should be active
        assert result['n_active_patches'] == 1


import pandas as pd
from voxcity.simulator_gpu.solar.integration.utils import filter_df_to_period


def _year_df():
    idx = pd.date_range("2000-01-01 00:00:00", "2000-12-31 23:00:00", freq="h")
    return pd.DataFrame({"DNI": 1.0, "DHI": 1.0}, index=idx)


def test_filter_df_daily_hour_window_crosses_date_range():
    df = _year_df()
    out = filter_df_to_period(
        df, "07-01 00:00:00", "08-31 23:00:00", tz=0.0,
        daily_start_hour=6, daily_end_hour=11,
    )
    hours = out.index.hour.unique().tolist()
    assert min(hours) >= 6 and max(hours) <= 11
    months = out.index.month.unique().tolist()
    assert set(months) == {7, 8}


def test_filter_df_daily_hour_wraparound():
    df = _year_df()
    out = filter_df_to_period(
        df, "07-01 00:00:00", "07-02 23:00:00", tz=0.0,
        daily_start_hour=22, daily_end_hour=3,
    )
    assert set(out.index.hour.unique().tolist()) <= {22, 23, 0, 1, 2, 3}


def test_filter_df_none_daily_hours_unchanged():
    df = _year_df()
    out = filter_df_to_period(df, "07-01 06:00:00", "07-01 11:00:00", tz=0.0)
    # Continuous span: all 6 hours present, no daily narrowing.
    assert len(out) == 6


def test_filter_df_non_leap_year_post_february_window():
    # Regression: the [start,end] window must key off the EPW's own (non-leap)
    # year, not a hardcoded leap year. A post-February single day used to compute
    # its day-of-year against datetime(2000) (leap), landing one day off and
    # selecting nothing -> "No weather data in the specified period".
    idx = pd.date_range("2023-06-21 00:00:00", "2023-06-21 23:00:00", freq="h")
    df = pd.DataFrame({"DNI": 1.0, "DHI": 1.0}, index=idx)
    out = filter_df_to_period(df, "06-21 00:00:00", "06-21 23:00:00", tz=0.0)
    assert len(out) == 24


def test_get_hour_range_from_period_uses_given_year():
    from datetime import datetime as _dt

    from voxcity.simulator_gpu.solar.integration.utils import get_hour_range_from_period
    # non-leap year: the window's hour-of-year must match that year's calendar
    start_hour, end_hour = get_hour_range_from_period(
        "06-21 00:00:00", "06-21 23:00:00", year=2023)
    doy = _dt(2023, 6, 21).timetuple().tm_yday
    assert start_hour == (doy - 1) * 24 + 0 + 1
    assert end_hour == (doy - 1) * 24 + 23 + 1


from unittest.mock import patch


def test_cumulative_global_forwards_daily_hours():
    """Ground cumulative entry point must forward daily hour kwargs to filter_df_to_period."""
    import voxcity.simulator_gpu.solar.integration as integ
    # Patch filter_df_to_period as resolved inside the caller's own module.
    target_module = integ.get_cumulative_global_solar_irradiance.__module__
    with patch(f"{target_module}.filter_df_to_period") as mock_filter:
        # Stop execution right after the filter call so no heavy work runs.
        mock_filter.side_effect = RuntimeError("stop after filter")
        try:
            integ.get_cumulative_global_solar_irradiance(
                voxcity=None, df=_year_df(), lon=0.0, lat=0.0, tz=0.0,
                start_time="07-01 00:00:00", end_time="08-31 23:00:00",
                daily_start_hour=6, daily_end_hour=11,
            )
        except Exception:
            pass
        assert mock_filter.called, "filter_df_to_period was not reached"
        _, called_kwargs = mock_filter.call_args
        assert called_kwargs.get("daily_start_hour") == 6
        assert called_kwargs.get("daily_end_hour") == 11


def test_cumulative_volumetric_applies_daily_hours():
    """Volumetric cumulative entry point filters inline; verify the daily window is applied.

    volumetric.get_cumulative_volumetric_solar_irradiance does NOT call
    filter_df_to_period (it filters inline), so we prove the daily-hour bounds
    reach the filtering by inspecting the timestamps passed to the first
    downstream consumer (get_solar_positions_astral).
    """
    import voxcity.simulator_gpu.solar.integration as integ
    mod = integ.get_cumulative_volumetric_solar_irradiance.__module__
    captured = {}

    def fake_solar_positions(index, lon, lat):
        captured["hours"] = sorted(set(index.hour.tolist()))
        raise RuntimeError("stop after filter")

    with patch(f"{mod}.get_solar_positions_astral", side_effect=fake_solar_positions):
        try:
            integ.get_cumulative_volumetric_solar_irradiance(
                voxcity=None, df=_year_df(), lon=0.0, lat=0.0, tz=0.0,
                start_time="07-01 00:00:00", end_time="08-31 23:00:00",
                daily_start_hour=6, daily_end_hour=11,
            )
        except Exception:
            pass
    assert captured.get("hours"), "get_solar_positions_astral was not reached"
    assert set(captured["hours"]) == {6, 7, 8, 9, 10, 11}


def test_building_cumulative_forwards_daily_hours():
    """Building-surface cumulative entry point must forward daily hour kwargs
    to filter_df_to_period."""
    import voxcity.simulator_gpu.solar.integration.building as bld
    with patch.object(bld, "filter_df_to_period") as mock_filter:
        # Stop execution right after the filter call so no heavy work runs.
        mock_filter.side_effect = RuntimeError("stop after filter")
        try:
            bld.get_cumulative_building_solar_irradiance(
                voxcity=None,
                building_svf_mesh=None,
                weather_df=_year_df(),
                lon=0.0, lat=0.0, tz=0.0,
                period_start="07-01 00:00:00", period_end="08-31 23:00:00",
                daily_start_hour=6, daily_end_hour=11,
            )
        except Exception:
            pass
        assert mock_filter.called, "filter_df_to_period was not reached"
        _, called_kwargs = mock_filter.call_args
        assert called_kwargs.get("daily_start_hour") == 6
        assert called_kwargs.get("daily_end_hour") == 11
