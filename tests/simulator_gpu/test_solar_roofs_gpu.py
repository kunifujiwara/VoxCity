import numpy as np
import pytest

pytest.importorskip("taichi")

from tests.simulator._roof_helpers import make_voxcity_with_building
from voxcity.simulator_gpu.solar.integration.ground import get_global_solar_irradiance_map


def _global(vc, include_roofs):
    return get_global_solar_irradiance_map(
        vc, azimuth_degrees_ori=180.0, elevation_degrees=60.0,
        direct_normal_irradiance=900.0, diffuse_irradiance=100.0,
        show_plot=False, with_reflections=False,
        N_azimuth=8, N_elevation=3,
        include_building_roofs=include_roofs,
    )


def test_gpu_global_excludes_building_by_default():
    vc = make_voxcity_with_building()
    g = _global(vc, include_roofs=False)
    assert np.isnan(g[2, 2])
    assert np.isfinite(g[0, 0])


def test_gpu_global_includes_building_roof_when_enabled():
    vc = make_voxcity_with_building()
    g = _global(vc, include_roofs=True)
    assert np.isfinite(g[2, 2])
