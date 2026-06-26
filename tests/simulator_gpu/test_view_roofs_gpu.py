import numpy as np
import pytest

pytest.importorskip("taichi")

from tests.simulator._roof_helpers import make_voxcity_with_building
from voxcity.simulator_gpu.visibility.integration import get_sky_view_factor_map


def _svf(vc, include_roofs):
    return get_sky_view_factor_map(
        vc, show_plot=False, N_azimuth=8, N_elevation=3,
        include_building_roofs=include_roofs,
    )


def test_gpu_svf_excludes_building_by_default():
    vc = make_voxcity_with_building()
    m = _svf(vc, include_roofs=False)
    assert np.isnan(m[2, 2])
    assert np.isfinite(m[0, 0])


def test_gpu_svf_includes_building_roof_when_enabled():
    vc = make_voxcity_with_building()
    m = _svf(vc, include_roofs=True)
    assert np.isfinite(m[2, 2])
