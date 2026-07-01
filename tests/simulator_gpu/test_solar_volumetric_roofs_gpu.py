import numpy as np
import pandas as pd
import pytest

pytest.importorskip("taichi")

from tests.simulator._roof_helpers import make_voxcity_with_building
from voxcity.simulator_gpu.solar.integration.volumetric import (
    get_volumetric_solar_irradiance_map,
    get_cumulative_volumetric_solar_irradiance,
)


def _instant(vc, include_roofs):
    return get_volumetric_solar_irradiance_map(
        vc, azimuth_degrees_ori=180.0, elevation_degrees=60.0,
        direct_normal_irradiance=900.0, diffuse_irradiance=100.0,
        volumetric_height=1.5, with_reflections=False, show_plot=False,
        include_building_roofs=include_roofs,
    )


def _small_epw_df():
    idx = pd.date_range("2000-06-21 09:00:00", periods=3, freq="h")
    return pd.DataFrame({"DNI": [800.0, 850.0, 800.0], "DHI": [100.0, 110.0, 100.0]}, index=idx)


def _cumulative(vc, include_roofs):
    return get_cumulative_volumetric_solar_irradiance(
        vc, _small_epw_df(), lon=0.0, lat=0.0, tz=0.0,
        volumetric_height=1.5, with_reflections=False, show_plot=False,
        start_time="06-21 08:00:00", end_time="06-21 20:00:00",
        include_building_roofs=include_roofs,
    )


def test_instant_volumetric_excludes_building_by_default():
    vc = make_voxcity_with_building()
    m = _instant(vc, include_roofs=False)
    assert np.isnan(m[2, 2])          # rooftop cell NaN'd (terrain-following)
    assert np.isfinite(m[0, 0])       # street cell evaluated


def test_instant_volumetric_includes_building_roof_when_enabled():
    vc = make_voxcity_with_building()
    m = _instant(vc, include_roofs=True)
    assert np.isfinite(m[2, 2])       # rooftop cell now evaluated


def test_cumulative_volumetric_excludes_building_by_default():
    vc = make_voxcity_with_building()
    m = _cumulative(vc, include_roofs=False)
    assert np.isnan(m[2, 2])
    assert np.isfinite(m[0, 0])


def test_cumulative_volumetric_includes_building_roof_when_enabled():
    vc = make_voxcity_with_building()
    m = _cumulative(vc, include_roofs=True)
    assert np.isfinite(m[2, 2])
