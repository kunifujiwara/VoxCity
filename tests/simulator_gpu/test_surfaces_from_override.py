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


def test_empty_table_yields_no_surfaces(_ti):
    from voxcity.simulator_gpu.solar.domain import surfaces_from_override
    from voxcity.simulator_gpu.solar.surface_override import SurfaceOverride

    empty = SurfaceOverride(
        cell=np.zeros((0, 3), np.int32), face=np.zeros((0,), np.int8),
        origin=np.zeros((0, 3), np.float32), normal=np.zeros((0, 3), np.float32),
        patch=np.zeros((0,), np.int32), area=np.zeros((0,), np.float32))
    assert surfaces_from_override(empty, default_albedo=0.2).count == 0
