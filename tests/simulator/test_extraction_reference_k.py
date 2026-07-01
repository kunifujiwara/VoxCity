import numpy as np
from voxcity.simulator_gpu.solar.integration.volumetric import (
    _compute_ground_k_from_voxels,
    _compute_surface_k_from_voxels,
    _compute_extraction_reference_k,
)

AIR, TREE, BUILDING, GROUND, WATER = 0, -2, -3, 1, 8


def _column_world():
    # shape (2, 2, 6)
    v = np.zeros((2, 2, 6), dtype=np.int32)
    v[:, :, 0] = GROUND                 # flat terrain everywhere at k0
    v[0, 1, 1:5] = BUILDING             # building col at (0,1): roof top k=4
    v[1, 0, 1:3] = TREE                 # tree over terrain at (1,0): not solid
    v[1, 1, 0] = WATER                  # water col at (1,1): invalid ground
    return v


def test_surface_k_equals_ground_k_on_plain_terrain():
    v = _column_world()
    g = _compute_ground_k_from_voxels(v)
    s = _compute_surface_k_from_voxels(v)
    assert g[0, 0] == 1 and s[0, 0] == 1            # first air above terrain


def test_surface_k_lands_above_roof_where_ground_k_is_invalid():
    v = _column_world()
    g = _compute_ground_k_from_voxels(v)
    s = _compute_surface_k_from_voxels(v)
    assert g[0, 1] == -1                            # building footprint excluded
    assert s[0, 1] == 5                             # first air above roof (k=4)


def test_surface_k_ignores_tree_canopy():
    v = _column_world()
    s = _compute_surface_k_from_voxels(v)
    assert s[1, 0] == 1                             # tree is not opaque solid


def test_water_column_has_no_surface():
    v = _column_world()
    s = _compute_surface_k_from_voxels(v)
    assert s[1, 1] == -1


def test_dispatcher_selects_reference():
    v = _column_world()
    ground = _compute_extraction_reference_k(v, include_building_roofs=False)
    roof = _compute_extraction_reference_k(v, include_building_roofs=True)
    np.testing.assert_array_equal(ground, _compute_ground_k_from_voxels(v))
    np.testing.assert_array_equal(roof, _compute_surface_k_from_voxels(v))


def test_water_codes_7_and_9_also_invalid():
    v = np.zeros((2, 1, 6), dtype=np.int32)
    v[:, :, 0] = 1           # terrain
    v[0, 0, 0] = 7           # water code 7
    v[1, 0, 0] = 9           # water code 9
    s = _compute_surface_k_from_voxels(v)
    assert s[0, 0] == -1     # water 7 → invalid
    assert s[1, 0] == -1     # water 9 → invalid
