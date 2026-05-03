from types import SimpleNamespace

import numpy as np


def _fake_voxcity(voxel_data, meshsize=1.0):
    return SimpleNamespace(
        voxels=SimpleNamespace(
            classes=voxel_data,
            meta=SimpleNamespace(meshsize=meshsize),
        ),
        extras={"rotation_angle": 0},
        dem=None,
    )


def _ground_grid(shape=(4, 1, 4)):
    voxel_data = np.zeros(shape, dtype=np.int32)
    voxel_data[:, :, 0] = 1
    return voxel_data


def test_cpu_direct_solar_north_azimuth_uses_positive_u_and_preserves_uv_layout():
    from voxcity.simulator.solar.radiation import get_direct_solar_irradiance_map

    voxel_data = _ground_grid()
    voxel_data[3, 0, 1:3] = -3
    voxcity = _fake_voxcity(voxel_data)

    result = get_direct_solar_irradiance_map(
        voxcity,
        azimuth_degrees_ori=0,
        elevation_degrees=45,
        direct_normal_irradiance=1000,
        view_point_height=0,
    )

    assert result.shape == voxel_data.shape[:2]
    assert np.isnan(result[3, 0])
    assert result[2, 0] == 0
    assert result[1, 0] > 0


def test_cpu_visibility_maps_preserve_uv_layout():
    from voxcity.simulator.common.raytracing import (
        _compute_vi_map_generic_fast,
        _prepare_masks_for_vi,
        compute_vi_map_generic,
    )

    voxel_data = _ground_grid()
    voxel_data[3, 0, 1:3] = -3
    ray_directions = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    hit_values = (0,)
    meshsize = 1.0
    tree_k = 0.6
    tree_lad = 1.0
    inclusion_mode = False

    slow_result = compute_vi_map_generic(
        voxel_data,
        ray_directions,
        0,
        hit_values,
        meshsize,
        tree_k,
        tree_lad,
        inclusion_mode,
    )

    is_tree, is_target, is_allowed, is_blocker_inc = _prepare_masks_for_vi(
        voxel_data, hit_values, inclusion_mode
    )
    fast_result = _compute_vi_map_generic_fast(
        voxel_data,
        ray_directions,
        0,
        meshsize,
        tree_k,
        tree_lad,
        is_tree,
        np.zeros(1, dtype=np.bool_) if is_target is None else is_target,
        np.zeros(1, dtype=np.bool_) if is_allowed is None else is_allowed,
        np.zeros(1, dtype=np.bool_) if is_blocker_inc is None else is_blocker_inc,
        inclusion_mode,
        False,
    )

    for result in (slow_result, fast_result):
        assert result.shape == voxel_data.shape[:2]
        assert np.isnan(result[3, 0])
        assert result[2, 0] == 1.0
        assert result[0, 0] == 1.0
