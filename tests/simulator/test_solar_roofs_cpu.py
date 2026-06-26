import numpy as np

from tests.simulator._roof_helpers import make_voxcity_with_building
from voxcity.simulator.solar.radiation import (
    get_direct_solar_irradiance_map, get_global_solar_irradiance_map,
)


def _direct(vc, include_roofs):
    return get_direct_solar_irradiance_map(
        vc, azimuth_degrees_ori=180.0, elevation_degrees=60.0,
        direct_normal_irradiance=900.0, show_plot=False,
        include_building_roofs=include_roofs,
    )


def test_direct_excludes_building_by_default():
    vc = make_voxcity_with_building()
    m = _direct(vc, include_roofs=False)
    assert np.isnan(m[2, 2])
    assert np.isfinite(m[0, 0])


def test_direct_includes_building_roof_when_enabled():
    vc = make_voxcity_with_building()
    m = _direct(vc, include_roofs=True)
    assert np.isfinite(m[2, 2])


def test_global_includes_building_roof_when_enabled():
    vc = make_voxcity_with_building()
    g = get_global_solar_irradiance_map(
        vc, azimuth_degrees_ori=180.0, elevation_degrees=60.0,
        direct_normal_irradiance=900.0, diffuse_irradiance=100.0,
        show_plot=False, N_azimuth=8, N_elevation=3,
        include_building_roofs=True,
    )
    assert np.isfinite(g[2, 2])
