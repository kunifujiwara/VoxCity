import numpy as np

from tests.simulator._roof_helpers import make_voxcity_with_building, make_voxcity_with_pilotis
from voxcity.simulator.visibility.view import get_sky_view_factor_map


def _svf(vc, include_roofs):
    return get_sky_view_factor_map(vc, N_azimuth=8, N_elevation=3, include_building_roofs=include_roofs)


def test_pilotis_off_uses_open_floor():
    vc = make_voxcity_with_pilotis()
    m = _svf(vc, include_roofs=False)
    assert np.isfinite(m[2, 2])


def test_pilotis_on_uses_roof_not_floor():
    vc = make_voxcity_with_pilotis()
    on = _svf(vc, include_roofs=True)
    off = _svf(vc, include_roofs=False)
    assert np.isfinite(on[2, 2])
    assert on[2, 2] > off[2, 2] + 1e-3


def test_building_on_ground_unchanged_by_topmost():
    vc = make_voxcity_with_building()
    m = _svf(vc, include_roofs=True)
    assert np.isfinite(m[2, 2])


def test_street_cells_unaffected():
    vc = make_voxcity_with_pilotis()
    off = _svf(vc, include_roofs=False)
    on = _svf(vc, include_roofs=True)
    mask = np.ones_like(off, dtype=bool)
    mask[2, 2] = False
    np.testing.assert_allclose(off[mask], on[mask], equal_nan=True)
