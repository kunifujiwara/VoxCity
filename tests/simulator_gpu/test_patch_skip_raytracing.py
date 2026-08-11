import numpy as np
import pytest

ti = pytest.importorskip("taichi")


@pytest.fixture(scope="module")
def _ti():
    from voxcity.simulator_gpu.init_taichi import ensure_initialized
    ensure_initialized()


def _trace(occ_np, patch_np, origin, direction, my_patch):
    """Trace one ray; returns 1 if it hit solid, 0 if it escaped."""
    from voxcity.simulator_gpu.solar.raytracing import ray_voxel_first_hit_skip_patch

    nx, ny, nz = occ_np.shape
    is_solid = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
    cell_patch = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
    out = ti.field(dtype=ti.i32, shape=())
    is_solid.from_numpy(occ_np.astype(np.int32))
    cell_patch.from_numpy(patch_np.astype(np.int32))

    @ti.kernel
    def run(ox: ti.f32, oy: ti.f32, oz: ti.f32,
            dx_: ti.f32, dy_: ti.f32, dz_: ti.f32, mine: ti.i32):
        hit, _, _, _, _ = ray_voxel_first_hit_skip_patch(
            ti.Vector([ox, oy, oz]), ti.Vector([dx_, dy_, dz_]),
            is_solid, cell_patch, mine,
            nx, ny, nz, 1.0, 1.0, 1.0, 100.0)
        out[None] = hit

    run(*[float(v) for v in origin], *[float(v) for v in direction], int(my_patch))
    return out[None]


def test_blocks_a_cell_of_another_patch(_ti):
    occ = np.zeros((8, 3, 3), dtype=np.int32)
    patch = np.full((8, 3, 3), -1, dtype=np.int32)
    occ[4, 1, 1] = 1
    patch[4, 1, 1] = 99                      # someone else's wall
    assert _trace(occ, patch, (0.5, 1.5, 1.5), (1.0, 0.0, 0.0), my_patch=7) == 1


def test_skips_a_cell_of_my_own_patch(_ti):
    occ = np.zeros((8, 3, 3), dtype=np.int32)
    patch = np.full((8, 3, 3), -1, dtype=np.int32)
    occ[4, 1, 1] = 1
    patch[4, 1, 1] = 7                       # my own staircase
    assert _trace(occ, patch, (0.5, 1.5, 1.5), (1.0, 0.0, 0.0), my_patch=7) == 0


def test_my_patch_skipped_but_a_later_neighbour_still_blocks(_ti):
    occ = np.zeros((8, 3, 3), dtype=np.int32)
    patch = np.full((8, 3, 3), -1, dtype=np.int32)
    occ[3, 1, 1] = 1; patch[3, 1, 1] = 7     # mine -- transparent
    occ[6, 1, 1] = 1; patch[6, 1, 1] = 99    # not mine -- opaque
    assert _trace(occ, patch, (0.5, 1.5, 1.5), (1.0, 0.0, 0.0), my_patch=7) == 1


def test_no_patch_ray_behaves_exactly_like_today(_ti):
    """A ray with patch = -1 must skip nothing, including cells marked -1."""
    occ = np.zeros((8, 3, 3), dtype=np.int32)
    patch = np.full((8, 3, 3), -1, dtype=np.int32)
    occ[4, 1, 1] = 1
    assert _trace(occ, patch, (0.5, 1.5, 1.5), (1.0, 0.0, 0.0), my_patch=-1) == 1


def test_canopy_variant_sees_through_own_patch_but_keeps_absorbing(_ti):
    from voxcity.simulator_gpu.solar.raytracing import ray_canopy_absorption_skip_patch

    nx, ny, nz = 8, 3, 3
    occ = np.zeros((nx, ny, nz), dtype=np.int32)
    patch = np.full((nx, ny, nz), -1, dtype=np.int32)
    lad_np = np.zeros((nx, ny, nz), dtype=np.float32)
    occ[3, 1, 1] = 1; patch[3, 1, 1] = 7     # my own wall cell
    lad_np[5, 1, 1] = 2.0                     # a tree beyond it

    is_solid = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
    cell_patch = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
    lad = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
    out = ti.field(dtype=ti.f32, shape=())
    is_solid.from_numpy(occ)
    cell_patch.from_numpy(patch)
    lad.from_numpy(lad_np)

    @ti.kernel
    def run(mine: ti.i32):
        trans, _ = ray_canopy_absorption_skip_patch(
            ti.Vector([0.5, 1.5, 1.5]), ti.Vector([1.0, 0.0, 0.0]),
            lad, is_solid, cell_patch, mine,
            nx, ny, nz, 1.0, 1.0, 1.0, 100.0, 0.6)
        out[None] = trans

    run(-1)
    assert out[None] == pytest.approx(0.0)   # blocked by the wall, as today
    run(7)
    expected = float(np.exp(-0.6 * 2.0 * 1.0))
    assert out[None] == pytest.approx(expected, abs=1e-3)  # skipped wall, absorbed by tree
