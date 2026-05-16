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
