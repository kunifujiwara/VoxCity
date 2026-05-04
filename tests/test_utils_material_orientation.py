"""Coordinate-frame regression test for `set_building_material_by_id`.

Phase 3 contract: voxel arrays are uv layout (axis 0 = u = north). Both
``voxelcity_grid`` and ``building_id_grid`` arrive in the same uv layout, so the
function must index them with the same (i, j) — no flip needed.
"""
import numpy as np


def test_set_building_material_marks_uv_aligned_cells():
    from voxcity.utils.material import set_building_material_by_id

    # Building footprint at uv(u=2, v=1). The grid is asymmetric along axis 0
    # so a stale flipud lands marks at u=N-1-2=u_orig-flipped row instead.
    voxelcity_grid = np.zeros((5, 4, 3), dtype=np.int32)
    voxelcity_grid[2, 1, :] = -3                  # building voxels (only here)

    building_id_grid = np.zeros((5, 4), dtype=np.int32)
    building_id_grid[2, 1] = 99

    set_building_material_by_id(
        voxelcity_grid, building_id_grid, [99], mark=-3,
        window_ratio=0.125,
    )

    # All voxels at uv(2, 1) should remain marked (or window-painted on top of
    # the same column). Crucially, no other (u, v) column should have been
    # touched — a flipud bug would mark uv(2, 1) -> uv(N-1-2, 1) = uv(2, 1)
    # when N=5 (5-1-2=2, coincidence), so use N=5 with the building at u=2 to
    # avoid that mid-grid coincidence... let's be explicit and pick u=1.
    # Re-do with u=1, where 5-1-1=3, an obviously different row.
    voxelcity_grid = np.zeros((5, 4, 3), dtype=np.int32)
    voxelcity_grid[1, 1, :] = -3
    building_id_grid = np.zeros((5, 4), dtype=np.int32)
    building_id_grid[1, 1] = 99

    set_building_material_by_id(
        voxelcity_grid, building_id_grid, [99], mark=7,
        window_ratio=1.0,  # max density window placement -> exercises the
                           # window-pattern branch where x % x_mod == 0.
    )

    # The building column at uv(1, 1) must have its base material applied.
    # voxels are now either `mark` (7) or glass_id (-16, the default), but
    # in particular they must NOT remain -3 (= unmarked building).
    column_uv_1_1 = voxelcity_grid[1, 1, :]
    assert not np.any(column_uv_1_1 == -3), (
        f"uv(1,1) column should be fully painted; got {column_uv_1_1!r}"
    )

    # The mirror column the buggy flipud would target — uv(N-1-1, 1) = uv(3, 1)
    # — was never a building voxel and must be untouched (still 0).
    column_uv_3_1 = voxelcity_grid[3, 1, :]
    assert np.all(column_uv_3_1 == 0), (
        f"uv(3,1) column should be untouched; got {column_uv_3_1!r}"
    )
