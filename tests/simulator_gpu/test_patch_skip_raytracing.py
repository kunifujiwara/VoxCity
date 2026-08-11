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


def test_untagged_solid_cell_still_blocks_a_tagged_ray(_ti):
    """An untagged solid cell (cell_patch == -1, e.g. terrain or anything the
    patch grid never seeded) must still block a ray carrying a real patch id.
    This is the dominant real case -- most solid cells aren't anyone's patch
    -- and it's the one the `and not (...)` clause is easiest to get
    backwards on (a flipped condition would only show up on foreign-tagged
    cells, not untagged ones)."""
    occ = np.zeros((8, 3, 3), dtype=np.int32)
    patch = np.full((8, 3, 3), -1, dtype=np.int32)  # untagged, like terrain
    occ[4, 1, 1] = 1
    assert _trace(occ, patch, (0.5, 1.5, 1.5), (1.0, 0.0, 0.0), my_patch=7) == 1


def test_skips_own_patch_along_diagonal_ray(_ti):
    """All five tests above fire along (1, 0, 0), where t_max_y/t_max_z never
    leave their 1e30 initialisation and the y/z DDA stepping arms never run.
    A diagonal ray (all three direction components non-zero) exercises those
    arms while checking the same skip behaviour."""
    occ = np.zeros((8, 8, 8), dtype=np.int32)
    patch = np.full((8, 8, 8), -1, dtype=np.int32)
    occ[4, 4, 4] = 1
    patch[4, 4, 4] = 7
    d = np.array([1.0, 1.0, 1.0])
    direction = tuple(d / np.linalg.norm(d))
    assert _trace(occ, patch, (0.5, 0.5, 0.5), direction, my_patch=7) == 0

    # Sanity: the same diagonal ray still gets blocked by a foreign patch.
    patch_foreign = np.full((8, 8, 8), -1, dtype=np.int32)
    patch_foreign[4, 4, 4] = 99
    assert _trace(occ, patch_foreign, (0.5, 0.5, 0.5), direction, my_patch=7) == 1


def test_skips_own_patch_along_ray_with_negative_component(_ti):
    """A ray direction with a negative component exercises the negative
    `step_*` DDA arm, which (1, 0, 0) alone never touches."""
    d = np.array([1.0, -1.0, 1.0])
    direction = tuple(d / np.linalg.norm(d))
    occ = np.zeros((8, 8, 8), dtype=np.int32)
    patch = np.full((8, 8, 8), -1, dtype=np.int32)
    occ[4, 3, 4] = 1
    patch[4, 3, 4] = 7
    assert _trace(occ, patch, (0.5, 7.5, 0.5), direction, my_patch=7) == 0


def test_skip_patch_variants_are_bit_identical_when_disabled(_ti):
    """ray_voxel_first_hit_skip_patch and ray_canopy_absorption_skip_patch
    are ~120-line hand copies of ray_voxel_first_hit and ray_canopy_absorption
    (see simulator_gpu/raytracing.py), each with exactly one changed line.
    This is the only test that exercises the copied DDA scaffolding --
    entry/exit, index clamping, step directions, t_max/t_delta setup, the
    min-selection tie-break, and every step arm -- outside that one line.
    The five directional tests above all fire axis-aligned or two-diagonal
    rays; this one throws ~200 random oblique rays (normalised Gaussian
    directions), non-cubic cell sizes, origins both inside and outside the
    domain, and a random sparse occupancy + LAD field at the pair, and
    requires the two implementations to agree exactly (not approximately)
    when my_patch = -1, where they are supposed to be indistinguishable
    from the originals.
    """
    from voxcity.simulator_gpu.raytracing import ray_voxel_first_hit, ray_canopy_absorption
    from voxcity.simulator_gpu.solar.raytracing import (
        ray_voxel_first_hit_skip_patch,
        ray_canopy_absorption_skip_patch,
    )

    nx, ny, nz = 12, 10, 14
    dxv, dyv, dzv = 1.3, 0.9, 1.1
    max_dist = 100.0
    n_rays = 200

    rng = np.random.default_rng(12345)
    occ_np = (rng.random((nx, ny, nz)) < 0.15).astype(np.int32)
    patch_np = np.full((nx, ny, nz), -1, dtype=np.int32)  # irrelevant at my_patch=-1
    lad_np = (rng.random((nx, ny, nz)).astype(np.float32)
              * (rng.random((nx, ny, nz)) < 0.2))

    is_solid = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
    cell_patch = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
    lad = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
    is_solid.from_numpy(occ_np)
    cell_patch.from_numpy(patch_np)
    lad.from_numpy(lad_np.astype(np.float32))

    origins = rng.uniform(
        low=[-2, -2, -2], high=[nx * dxv + 2, ny * dyv + 2, nz * dzv + 2],
        size=(n_rays, 3)
    ).astype(np.float32)
    dirs = rng.normal(size=(n_rays, 3)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    hit_orig = ti.field(dtype=ti.i32, shape=n_rays)
    t_orig = ti.field(dtype=ti.f32, shape=n_rays)
    hit_new = ti.field(dtype=ti.i32, shape=n_rays)
    t_new = ti.field(dtype=ti.f32, shape=n_rays)
    trans_orig = ti.field(dtype=ti.f32, shape=n_rays)
    lad_orig = ti.field(dtype=ti.f32, shape=n_rays)
    trans_new = ti.field(dtype=ti.f32, shape=n_rays)
    lad_new = ti.field(dtype=ti.f32, shape=n_rays)

    ox_f = ti.field(dtype=ti.f32, shape=n_rays)
    oy_f = ti.field(dtype=ti.f32, shape=n_rays)
    oz_f = ti.field(dtype=ti.f32, shape=n_rays)
    dx_f = ti.field(dtype=ti.f32, shape=n_rays)
    dy_f = ti.field(dtype=ti.f32, shape=n_rays)
    dz_f = ti.field(dtype=ti.f32, shape=n_rays)
    ox_f.from_numpy(origins[:, 0])
    oy_f.from_numpy(origins[:, 1])
    oz_f.from_numpy(origins[:, 2])
    dx_f.from_numpy(dirs[:, 0])
    dy_f.from_numpy(dirs[:, 1])
    dz_f.from_numpy(dirs[:, 2])

    @ti.kernel
    def run_all():
        for i in range(n_rays):
            o = ti.Vector([ox_f[i], oy_f[i], oz_f[i]])
            d = ti.Vector([dx_f[i], dy_f[i], dz_f[i]])

            h1, t1, ix1, iy1, iz1 = ray_voxel_first_hit(
                o, d, is_solid, nx, ny, nz, dxv, dyv, dzv, max_dist)
            h2, t2, ix2, iy2, iz2 = ray_voxel_first_hit_skip_patch(
                o, d, is_solid, cell_patch, -1, nx, ny, nz, dxv, dyv, dzv, max_dist)
            hit_orig[i] = h1
            t_orig[i] = t1
            hit_new[i] = h2
            t_new[i] = t2

            tr1, lp1 = ray_canopy_absorption(
                o, d, lad, is_solid, nx, ny, nz, dxv, dyv, dzv, max_dist, 0.5)
            tr2, lp2 = ray_canopy_absorption_skip_patch(
                o, d, lad, is_solid, cell_patch, -1, nx, ny, nz, dxv, dyv, dzv, max_dist, 0.5)
            trans_orig[i] = tr1
            lad_orig[i] = lp1
            trans_new[i] = tr2
            lad_new[i] = lp2

    run_all()

    assert np.array_equal(hit_orig.to_numpy(), hit_new.to_numpy())
    assert np.array_equal(t_orig.to_numpy(), t_new.to_numpy())
    assert np.array_equal(trans_orig.to_numpy(), trans_new.to_numpy())
    assert np.array_equal(lad_orig.to_numpy(), lad_new.to_numpy())


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
