"""Verify that a computation_mask is applied as a post-filter pass so that
non-mask cells receive NaN in the returned result."""
import numpy as np
import pytest

from voxcity.simulator_gpu.solar.integration.caching import (
    compute_direct_transmittance_map_gpu,
)

pytestmark = pytest.mark.gpu  # marker used elsewhere in the suite for CUDA tests


def _tiny_voxel_data():
    # 8 x 8 x 4 mostly-empty grid with a single tall building in the SW corner.
    arr = np.zeros((8, 8, 4), dtype=np.int32)
    arr[0:2, 0:2, 0:3] = -3  # building
    return arr


def test_computation_mask_marks_outside_cells_nan():
    voxels = _tiny_voxel_data()
    sun_dir = (0.5, 0.5, -1.0)
    meshsize = 1.0

    # Mask: only compute the NE quadrant.
    mask = np.zeros(voxels.shape[:2], dtype=bool)
    mask[4:, 4:] = True

    result = compute_direct_transmittance_map_gpu(
        voxel_data=voxels,
        sun_direction=sun_dir,
        view_point_height=1.5,
        meshsize=meshsize,
        computation_mask=mask,
    )
    assert result.shape == voxels.shape[:2]
    assert np.all(np.isnan(result[~mask]))
    assert np.all(np.isfinite(result[mask]))


def test_no_mask_preserves_legacy_behavior():
    voxels = _tiny_voxel_data()
    legacy = compute_direct_transmittance_map_gpu(
        voxel_data=voxels,
        sun_direction=(0.5, 0.5, -1.0),
        view_point_height=1.5,
        meshsize=1.0,
    )
    with_full_mask = compute_direct_transmittance_map_gpu(
        voxel_data=voxels,
        sun_direction=(0.5, 0.5, -1.0),
        view_point_height=1.5,
        meshsize=1.0,
        computation_mask=np.ones(voxels.shape[:2], dtype=bool),
    )
    np.testing.assert_allclose(legacy, with_full_mask, equal_nan=True)


def test_all_false_mask_returns_all_nan():
    voxels = _tiny_voxel_data()
    mask = np.zeros(voxels.shape[:2], dtype=bool)  # all False
    result = compute_direct_transmittance_map_gpu(
        voxel_data=voxels,
        sun_direction=(0.5, 0.5, -1.0),
        view_point_height=1.5,
        meshsize=1.0,
        computation_mask=mask,
    )
    assert result.shape == voxels.shape[:2]
    assert np.all(np.isnan(result))


def test_trace_rays_kernel_does_not_touch_out_of_mask_cells():
    """When mask is partial, out-of-mask cells in the transmittance field
    keep NaN — proving the inner kernel skipped them rather than being
    NaN'd by a post-pass."""
    import taichi as ti
    from voxcity.simulator_gpu.solar.integration.caching import (
        compute_direct_transmittance_map_gpu, get_or_create_gpu_ray_tracer,
        reset_solar_taichi_cache,
    )
    reset_solar_taichi_cache()

    voxels = _tiny_voxel_data()
    meshsize = 1.0
    sun_dir = (0.5, 0.5, -1.0)

    # Prime the cache so we can poke trans_field with a sentinel.
    _ = compute_direct_transmittance_map_gpu(
        voxel_data=voxels, sun_direction=sun_dir,
        view_point_height=1.5, meshsize=meshsize,
    )
    cache = get_or_create_gpu_ray_tracer(voxels, meshsize, 1.0)

    @ti.kernel
    def fill_sentinel(f: ti.template(), value: ti.f32):
        for i, j in f:
            f[i, j] = value
    SENTINEL = -42.0
    fill_sentinel(cache.transmittance_field, SENTINEL)

    mask = np.zeros(voxels.shape[:2], dtype=bool)
    mask[4:, 4:] = True
    result = compute_direct_transmittance_map_gpu(
        voxel_data=voxels, sun_direction=sun_dir,
        view_point_height=1.5, meshsize=meshsize,
        computation_mask=mask,
    )
    # Out-of-mask cells should be NaN (kernel wrote NaN at head-of-loop)
    assert np.all(np.isnan(result[~mask])), (
        "Expected non-mask cells to be NaN'd by the kernel itself"
    )
    # In-mask cells should be finite transmittance values.
    assert np.all(np.isfinite(result[mask]))


def test_apply_mask_nan_kernel_is_removed():
    """Confirms the obsolete post-pass kernel is gone from caching.py."""
    import voxcity.simulator_gpu.solar.integration.caching as caching_mod
    assert not hasattr(caching_mod, "_apply_mask_nan_kernel"), (
        "_apply_mask_nan_kernel should have been removed in favor of "
        "in-kernel mask gating in trace_rays_kernel."
    )
