import numpy as np
import pytest

from tests.simulator._roof_helpers import make_voxcity_with_building
from voxcity.simulator.visibility.view import get_sky_view_factor_map
from voxcity.simulator.common.raytracing import (
    compute_vi_map_generic, _compute_vi_map_generic_fast, _prepare_masks_for_vi,
)


def _svf(voxcity, include_roofs):
    return get_sky_view_factor_map(
        voxcity, N_azimuth=8, N_elevation=3, include_building_roofs=include_roofs
    )


def test_svf_excludes_building_by_default():
    vc = make_voxcity_with_building()
    m = _svf(vc, include_roofs=False)
    assert np.isnan(m[2, 2])          # building cell blank by default
    assert np.isfinite(m[0, 0])       # street cell valid


def test_svf_includes_building_roof_when_enabled():
    vc = make_voxcity_with_building()
    m = _svf(vc, include_roofs=True)
    assert np.isfinite(m[2, 2])       # rooftop observer now present
    assert m[2, 2] > 0.0


def test_svf_street_cells_unchanged_by_flag():
    vc = make_voxcity_with_building()
    off = _svf(vc, include_roofs=False)
    on = _svf(vc, include_roofs=True)
    mask = np.ones_like(off, dtype=bool)
    mask[2, 2] = False
    np.testing.assert_allclose(off[mask], on[mask], equal_nan=True)


def test_fast_path_matches_flag_behaviour():
    vc = make_voxcity_with_building()
    voxel = vc.voxels.classes
    ms = vc.voxels.meta.meshsize
    hit_values = (0,)
    is_tree, is_target, is_allowed, is_blocker = _prepare_masks_for_vi(voxel, hit_values, False)
    import numpy as _np
    rays = _np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.2]], dtype=_np.float64)
    off = _compute_vi_map_generic_fast(
        voxel, rays, 1, ms, 0.6, 1.0, is_tree,
        _np.zeros(1, dtype=_np.bool_), is_allowed, _np.zeros(1, dtype=_np.bool_),
        False, False, include_building_roofs=False,
    )
    on = _compute_vi_map_generic_fast(
        voxel, rays, 1, ms, 0.6, 1.0, is_tree,
        _np.zeros(1, dtype=_np.bool_), is_allowed, _np.zeros(1, dtype=_np.bool_),
        False, False, include_building_roofs=True,
    )
    assert np.isnan(off[2, 2])
    assert np.isfinite(on[2, 2])


def test_non_building_negative_surface_stays_nan_when_flag_on():
    """Surfaces with negative codes other than -3 must never become observers."""
    vc = make_voxcity_with_building()
    # Overwrite the building column with code -1 (ground/underground - should stay invalid)
    vc.voxels.classes[2, 2, 1:5] = -1
    m = _svf(vc, include_roofs=True)
    assert np.isnan(m[2, 2])   # -1 surface must remain invalid even with flag on
