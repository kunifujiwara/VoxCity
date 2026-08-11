import numpy as np
import pytest

ti = pytest.importorskip("taichi")


@pytest.fixture(scope="module")
def _ti():
    from voxcity.simulator_gpu.init_taichi import ensure_initialized
    ensure_initialized()


def _table():
    from voxcity.simulator_gpu.solar.surface_override import SurfaceOverride
    r2 = float(np.sqrt(0.5))
    return SurfaceOverride(
        cell=np.array([[2, 2, 1], [2, 2, 1]], dtype=np.int32),
        face=np.array([4, 2], dtype=np.int8),          # IEAST, INORTH
        origin=np.array([[3.0, 2.5, 1.5], [2.5, 3.0, 1.5]], dtype=np.float32),
        normal=np.array([[r2, r2, 0.0], [r2, r2, 0.0]], dtype=np.float32),
        patch=np.array([5, 5], dtype=np.int32),
        area=np.array([1.41, 1.41], dtype=np.float32),
    )


def test_one_cell_can_emit_two_surfaces(_ti):
    """A corner voxel cut by two walls is the point of the (cell, patch) key."""
    from voxcity.simulator_gpu.solar.domain import surfaces_from_override

    surfaces = surfaces_from_override(_table(), default_albedo=0.3)
    assert surfaces.count == 2

    r2 = float(np.sqrt(0.5))
    assert np.allclose(surfaces.normal.to_numpy()[:2], [[r2, r2, 0.0]] * 2, atol=1e-5)
    assert np.array_equal(surfaces.patch_id.to_numpy()[:2], [5, 5])
    assert np.array_equal(surfaces.position.to_numpy()[:2], [[2, 2, 1], [2, 2, 1]])
    assert np.array_equal(surfaces.direction.to_numpy()[:2], [4, 2])
    assert np.allclose(surfaces.albedo.to_numpy()[:2], 0.3)


def test_center_is_the_supplied_origin_and_area_the_supplied_area(_ti):
    from voxcity.simulator_gpu.solar.domain import surfaces_from_override

    surfaces = surfaces_from_override(_table(), default_albedo=0.2)
    assert np.allclose(surfaces.center.to_numpy()[:2],
                       [[3.0, 2.5, 1.5], [2.5, 3.0, 1.5]], atol=1e-5)
    # the true area crossing the cell, not the axis-face area: the design
    # advertises removing the staircase's sqrt(2) inflation
    assert np.allclose(surfaces.area.to_numpy()[:2], 1.41, atol=1e-5)


def test_a_corner_cell_keeps_two_distinct_patches_and_normals(_ti):
    """The 96%-of-surface-cells case on a non-axis-aligned building: a corner
    voxel is cut by two different wall polygons, each with its own patch id
    and its own true normal. Unlike `_table()` above (same patch, same
    normal, differing only in `face`), this pins that two rows sharing a
    `cell` are never merged or deduplicated -- a refactor that collapsed
    `patch_id` to one value per cell, or dropped one row as a "duplicate",
    would fail this test while still passing the others.
    """
    from voxcity.simulator_gpu.solar.domain import surfaces_from_override
    from voxcity.simulator_gpu.solar.surface_override import SurfaceOverride

    theta = np.deg2rad(20.0)
    normal_a = (float(np.cos(theta)), float(np.sin(theta)), 0.0)
    normal_b = (float(-np.sin(theta)), float(np.cos(theta)), 0.0)

    table = SurfaceOverride(
        cell=np.array([[2, 2, 1], [2, 2, 1]], dtype=np.int32),
        face=np.array([4, 2], dtype=np.int8),          # IEAST, INORTH
        origin=np.array([[3.0, 2.5, 1.5], [2.5, 3.0, 1.5]], dtype=np.float32),
        normal=np.array([normal_a, normal_b], dtype=np.float32),
        patch=np.array([5, 7], dtype=np.int32),
        area=np.array([1.1, 0.9], dtype=np.float32),
    )

    surfaces = surfaces_from_override(table, default_albedo=0.2)
    assert surfaces.count == 2

    patch_ids = surfaces.patch_id.to_numpy()[:2]
    normals = surfaces.normal.to_numpy()[:2]
    assert set(patch_ids.tolist()) == {5, 7}
    assert not np.allclose(normals[0], normals[1], atol=1e-5)

    # Match each row by its patch id (order is not guaranteed) and confirm
    # its normal survived unchanged.
    by_patch = {int(p): n for p, n in zip(patch_ids, normals)}
    assert np.allclose(by_patch[5], normal_a, atol=1e-5)
    assert np.allclose(by_patch[7], normal_b, atol=1e-5)


def test_empty_table_yields_no_surfaces(_ti):
    from voxcity.simulator_gpu.solar.domain import surfaces_from_override
    from voxcity.simulator_gpu.solar.surface_override import SurfaceOverride

    empty = SurfaceOverride(
        cell=np.zeros((0, 3), np.int32), face=np.zeros((0,), np.int8),
        origin=np.zeros((0, 3), np.float32), normal=np.zeros((0, 3), np.float32),
        patch=np.zeros((0,), np.int32), area=np.zeros((0,), np.float32))
    assert surfaces_from_override(empty, default_albedo=0.2).count == 0
