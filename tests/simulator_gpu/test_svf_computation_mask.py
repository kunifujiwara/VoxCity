"""SVF map and view index respect computation_mask."""
import numpy as np
import pytest
from voxcity.simulator_gpu.visibility.integration import get_sky_view_factor_map

pytestmark = pytest.mark.gpu


def _tiny_svf_voxel_data():
    """8x8x4 grid with a ground layer (val=1) and a small building in the SW corner.

    val=1 (paved) cells at z=0 are solid AND walkable, which lets the SVF
    kernel place observers above them.  The building (val=-3) cells are solid
    but NOT walkable, so they correctly yield NaN without a mask.
    """
    arr = np.zeros((8, 8, 4), dtype=np.int32)
    arr[:, :, 0] = 1        # paved ground (solid, walkable)
    arr[0:2, 0:2, 0] = -3  # building footprint (solid, not walkable)
    arr[0:2, 0:2, 1] = -3  # building body (solid)
    arr[0:2, 0:2, 2] = -3  # building top (solid)
    return arr


class _FakeVoxCity:
    """Minimal voxcity stand-in."""
    def __init__(self, classes, meshsize=1.0):
        from types import SimpleNamespace
        self.voxels = SimpleNamespace(
            classes=classes,
            meta=SimpleNamespace(meshsize=meshsize),
        )
        self.extras = {}


def test_svf_map_respects_computation_mask():
    vc = _FakeVoxCity(_tiny_svf_voxel_data())
    mask = np.zeros(vc.voxels.classes.shape[:2], dtype=bool)
    mask[2:5, 2:5] = True
    svf_map = get_sky_view_factor_map(vc, computation_mask=mask, show_plot=False)
    assert svf_map.shape == vc.voxels.classes.shape[:2]
    assert np.all(np.isnan(svf_map[~mask]))
    assert np.all(np.isfinite(svf_map[mask]))


def test_svf_map_all_false_mask_returns_all_nan():
    vc = _FakeVoxCity(_tiny_svf_voxel_data())
    mask = np.zeros(vc.voxels.classes.shape[:2], dtype=bool)
    svf_map = get_sky_view_factor_map(vc, computation_mask=mask, show_plot=False)
    assert np.all(np.isnan(svf_map))


def test_svf_map_no_mask_preserves_behavior():
    vc = _FakeVoxCity(_tiny_svf_voxel_data())
    svf_no_mask = get_sky_view_factor_map(vc, show_plot=False)
    full_mask = np.ones(vc.voxels.classes.shape[:2], dtype=bool)
    svf_full_mask = get_sky_view_factor_map(vc, computation_mask=full_mask, show_plot=False)
    np.testing.assert_allclose(svf_no_mask, svf_full_mask, equal_nan=True)


def test_view_kernel_does_not_touch_out_of_mask_cells():
    """The GPU kernel itself must write NaN for non-mask cells.

    This test proves the gating is done inside ``_compute_vi_map_kernel``
    (via ``mask_f``), NOT by the ``np.where`` post-filter in
    ``compute_view_index``.  It reads ``ws.vi_map`` directly after the call
    so that the raw kernel output — not the returned numpy array — is checked.
    """
    from voxcity.simulator_gpu.visibility.integration import (
        _get_or_create_view_workspace,
        get_view_index,
    )

    voxels = _tiny_svf_voxel_data()
    nx, ny, nz = voxels.shape
    vc = _FakeVoxCity(voxels)

    # Warm the workspace cache so _get_or_create_view_workspace hits cache.
    _ = get_view_index(vc, mode='green', show_plot=False)

    ws = _get_or_create_view_workspace(
        nx=nx, ny=ny, nz=nz, meshsize=1.0,
        n_azimuth=120, n_elevation=20,
        ray_sampling='grid', n_rays=None,
        elevation_min_degrees=-30.0, elevation_max_degrees=30.0,
    )

    # Pre-seed vi_map with a distinctive sentinel so we can tell whether
    # the kernel touched (overwrote) a cell.
    SENTINEL = -77.0
    ws.vi_map.fill(SENTINEL)

    mask = np.zeros((nx, ny), dtype=bool)
    mask[2:5, 2:5] = True  # walkable paved cells only (building is at 0:2, 0:2)

    result = get_view_index(vc, mode='green', show_plot=False, computation_mask=mask)

    # KEY assertion: the raw Taichi field must have NaN for non-mask cells.
    # Before Task 2 the kernel writes computed values everywhere (sentinel is
    # gone and no NaN is written by the kernel) → this FAILS.
    # After Task 2 the kernel writes NaN for mask_f==0 cells → this PASSES.
    raw_map = ws.vi_map.to_numpy()
    assert np.all(np.isnan(raw_map[~mask])), (
        "Kernel should write NaN for non-mask cells (mask_f gate missing)"
    )

    # Sanity: returned result must also reflect masking.
    assert np.all(np.isnan(result[~mask])), "Non-mask cells should be NaN in returned result"
    assert np.all(np.isfinite(result[mask])), "Mask cells should be finite in returned result"
