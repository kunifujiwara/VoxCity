"""Top-level entry points pass computation_mask all the way through."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

pytestmark = pytest.mark.gpu


def _tiny_voxel_data():
    arr = np.zeros((8, 8, 4), dtype=np.int32)
    arr[0:2, 0:2, 0:3] = -3  # building
    arr[:, :, 0] = 1  # ground
    return arr


class _FakeVoxCity:
    def __init__(self, classes, meshsize=1.0):
        from types import SimpleNamespace
        self.voxels = SimpleNamespace(
            classes=classes,
            meta=SimpleNamespace(meshsize=meshsize),
        )
        self.extras = {}


def test_get_global_solar_irradiance_using_epw_respects_mask():
    """computation_mask is respected: non-mask cells get NaN in the result."""
    from voxcity.simulator_gpu.solar.integration.volumetric import (
        get_global_solar_irradiance_using_epw,
    )

    vc = _FakeVoxCity(_tiny_voxel_data())

    # We use monkeypatching at the ground.py level to avoid actual EPW loading.
    # We patch get_cumulative_global_solar_irradiance (the function that
    # get_global_solar_irradiance_using_epw calls for temporal_mode='cumulative')
    # to return a full map, then check that the mask was applied to the final result.

    # Alternatively: patch load_epw_data to return synthetic data.
    import voxcity.simulator_gpu.solar.integration.volumetric as vm

    fake_result = np.full((8, 8), 100.0, dtype=np.float32)

    # We'll patch get_cumulative_global_solar_irradiance to capture kwargs and return fake data
    call_kwargs_captured = {}

    import voxcity.simulator_gpu.solar.integration.ground as gmod
    original_cumulative = gmod.get_cumulative_global_solar_irradiance

    def fake_cumulative(voxcity, df, lon, lat, tz, **kwargs):
        call_kwargs_captured.update(kwargs)
        # Simulate mask application (as the real function does)
        result = fake_result.copy()
        if 'computation_mask' in kwargs and kwargs['computation_mask'] is not None:
            mask = kwargs['computation_mask']
            result = np.where(mask, result, np.nan)
        return result

    mask = np.zeros((8, 8), dtype=bool)
    mask[3:6, 3:6] = True

    import unittest.mock as mock
    with mock.patch.object(gmod, 'get_cumulative_global_solar_irradiance', fake_cumulative):
        # Also patch load_epw_data to return synthetic data
        def fake_load_epw(*args, **kwargs):
            idx = pd.date_range('2024-06-21 09:00', periods=3, freq='1h', tz='UTC')
            df = pd.DataFrame({'DNI': [800, 850, 700], 'DHI': [100, 110, 90]}, index=idx)
            return df, 139.7, 35.6, 9.0

        with mock.patch.object(vm, 'load_epw_data', fake_load_epw):
            result = get_global_solar_irradiance_using_epw(
                vc,
                temporal_mode='cumulative',
                spatial_mode='horizontal',
                start_time='06-21 09:00:00',
                end_time='06-21 11:00:00',
                epw_file_path='unused.epw',
                computation_mask=mask,
            )

    assert result.shape == (8, 8)
    assert np.all(np.isnan(result[~mask]))
    assert 'computation_mask' in call_kwargs_captured, "computation_mask was not forwarded"
