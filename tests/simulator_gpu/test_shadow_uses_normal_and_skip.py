import numpy as np
import pytest

ti = pytest.importorskip("taichi")


@pytest.fixture(scope="module")
def _ti():
    from voxcity.simulator_gpu.init_taichi import ensure_initialized
    ensure_initialized()


def _run_shadow(occ_np, patch_np, centers, dirs_enum, normals, patches, sun):
    """Run compute_direct_shadows over the given surfaces; return shadow_factor."""
    from voxcity.simulator_gpu.solar.domain import Domain, Surfaces
    from voxcity.simulator_gpu.solar.raytracing import RayTracer

    nx, ny, nz = occ_np.shape
    d = Domain(nx=nx, ny=ny, nz=nz, dx=1.0, dy=1.0, dz=1.0,
               origin_lat=35.0, origin_lon=139.0)
    d.is_solid.from_numpy(occ_np.astype(np.int32))
    cell_patch = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
    cell_patch.from_numpy(patch_np.astype(np.int32))

    n = len(centers)
    s = Surfaces(n)

    @ti.kernel
    def fill(c: ti.types.ndarray(), de: ti.types.ndarray(),
             nn: ti.types.ndarray(), pp: ti.types.ndarray()):
        for i in range(n):
            # unused by these kernels; do not reuse this fixture for
            # position-dependent checks
            s.position[i] = ti.math.ivec3(0, 0, 0)
            s.direction[i] = de[i]
            s.center[i] = ti.Vector([c[i, 0], c[i, 1], c[i, 2]])
            s.normal[i] = ti.Vector([nn[i, 0], nn[i, 1], nn[i, 2]])
            s.patch_id[i] = pp[i]

    fill(np.asarray(centers, np.float32), np.asarray(dirs_enum, np.int32),
         np.asarray(normals, np.float32), np.asarray(patches, np.int32))
    s.n_surfaces[None] = n

    tracer = RayTracer(d)
    tracer.compute_direct_shadows(
        s.center, s.normal, s.patch_id, cell_patch,
        ti.Vector([float(sun[0]), float(sun[1]), float(sun[2])]),
        d.is_solid, n, s.shadow_factor)
    return s.shadow_factor.to_numpy()[:n]


def _run_canopy(occ_np, patch_np, lad_np, centers, dirs_enum, normals, patches, sun):
    """Run compute_direct_with_canopy over the given surfaces; return
    (shadow_factor, canopy_transmissivity)."""
    from voxcity.simulator_gpu.solar.domain import Domain, Surfaces
    from voxcity.simulator_gpu.solar.raytracing import RayTracer

    nx, ny, nz = occ_np.shape
    d = Domain(nx=nx, ny=ny, nz=nz, dx=1.0, dy=1.0, dz=1.0,
               origin_lat=35.0, origin_lon=139.0)
    d.is_solid.from_numpy(occ_np.astype(np.int32))
    d.lad.from_numpy(lad_np.astype(np.float32))
    cell_patch = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
    cell_patch.from_numpy(patch_np.astype(np.int32))

    n = len(centers)
    s = Surfaces(n)

    @ti.kernel
    def fill(c: ti.types.ndarray(), de: ti.types.ndarray(),
             nn: ti.types.ndarray(), pp: ti.types.ndarray()):
        for i in range(n):
            # unused by these kernels; do not reuse this fixture for
            # position-dependent checks
            s.position[i] = ti.math.ivec3(0, 0, 0)
            s.direction[i] = de[i]
            s.center[i] = ti.Vector([c[i, 0], c[i, 1], c[i, 2]])
            s.normal[i] = ti.Vector([nn[i, 0], nn[i, 1], nn[i, 2]])
            s.patch_id[i] = pp[i]

    fill(np.asarray(centers, np.float32), np.asarray(dirs_enum, np.int32),
         np.asarray(normals, np.float32), np.asarray(patches, np.int32))
    s.n_surfaces[None] = n

    tracer = RayTracer(d)
    tracer.compute_direct_with_canopy(
        s.center, s.normal, s.patch_id, cell_patch,
        ti.Vector([float(sun[0]), float(sun[1]), float(sun[2])]),
        d.is_solid, d.lad, n, s.shadow_factor, s.canopy_transmissivity)
    return s.shadow_factor.to_numpy()[:n], s.canopy_transmissivity.to_numpy()[:n]


def test_own_staircase_no_longer_shadows(_ti):
    """A surface blocked only by its own patch must come out sunlit.
    This is the blackout: today the solver reports zero beam here."""
    occ = np.zeros((8, 3, 4), dtype=np.int32)
    patch = np.full((8, 3, 4), -1, dtype=np.int32)
    occ[4, 1, 1] = 1; patch[4, 1, 1] = 7          # my own next step
    r2 = float(np.sqrt(0.5))
    sf = _run_shadow(occ, patch,
                     centers=[[0.5, 1.5, 1.5]], dirs_enum=[4],
                     normals=[[r2, r2, 0.0]], patches=[7],
                     sun=(1.0, 0.0, 0.0))
    assert sf[0] == pytest.approx(0.0)            # sunlit


def test_a_neighbour_still_shadows(_ti):
    occ = np.zeros((8, 3, 4), dtype=np.int32)
    patch = np.full((8, 3, 4), -1, dtype=np.int32)
    occ[4, 1, 1] = 1; patch[4, 1, 1] = 99         # someone else's wall
    r2 = float(np.sqrt(0.5))
    sf = _run_shadow(occ, patch,
                     centers=[[0.5, 1.5, 1.5]], dirs_enum=[4],
                     normals=[[r2, r2, 0.0]], patches=[7],
                     sun=(1.0, 0.0, 0.0))
    assert sf[0] == pytest.approx(1.0)            # shaded


def test_facing_test_uses_the_true_normal(_ti):
    """Sun along +y: the axis test for an IEAST (+x) face says 'away from sun',
    but the true 45-degree normal faces it. Must be sunlit, not auto-shaded.
    This is the half of the blackout caused by the facing test, not occlusion."""
    occ = np.zeros((8, 8, 4), dtype=np.int32)
    patch = np.full((8, 8, 4), -1, dtype=np.int32)
    r2 = float(np.sqrt(0.5))
    sf = _run_shadow(occ, patch,
                     centers=[[4.0, 4.5, 1.5]], dirs_enum=[4],
                     normals=[[r2, r2, 0.0]], patches=[7],
                     sun=(0.0, 1.0, 0.0))
    assert sf[0] == pytest.approx(0.0)


def test_unoverridden_surface_behaves_exactly_as_today(_ti):
    """patch = -1 and an axis normal: same facing test, same DDA, same answer."""
    occ = np.zeros((8, 3, 4), dtype=np.int32)
    patch = np.full((8, 3, 4), -1, dtype=np.int32)
    occ[4, 1, 1] = 1
    sf = _run_shadow(occ, patch,
                     centers=[[0.5, 1.5, 1.5]], dirs_enum=[4],
                     normals=[[1.0, 0.0, 0.0]], patches=[-1],
                     sun=(1.0, 0.0, 0.0))
    assert sf[0] == pytest.approx(1.0)            # blocked, as today


def test_canopy_own_staircase_no_longer_shadows(_ti):
    """Same staircase-notch scenario as test_own_staircase_no_longer_shadows,
    but through compute_direct_with_canopy (the live path, since Domain.lad
    is always allocated). With zero LAD everywhere, an unobstructed surface
    must show canopy_transmissivity == 1.0 and shadow_factor == 0.0; the
    own-patch cell must not block it either."""
    occ = np.zeros((8, 3, 4), dtype=np.int32)
    patch = np.full((8, 3, 4), -1, dtype=np.int32)
    lad = np.zeros((8, 3, 4), dtype=np.float32)
    occ[4, 1, 1] = 1; patch[4, 1, 1] = 7          # my own next step
    r2 = float(np.sqrt(0.5))
    sf, trans = _run_canopy(occ, patch, lad,
                            centers=[[0.5, 1.5, 1.5]], dirs_enum=[4],
                            normals=[[r2, r2, 0.0]], patches=[7],
                            sun=(1.0, 0.0, 0.0))
    assert sf[0] == pytest.approx(0.0)             # sunlit
    assert trans[0] == pytest.approx(1.0)          # no canopy present


def test_grazing_ray_escapes_a_multi_step_staircase(_ti):
    """The other tests here fire the sun at (1,0,0) against a 45-degree
    normal -- cos_inc ~= 0.707, a comfortable incidence angle where the ray
    clears its own patch's voxelisation after a single DDA step. The bug
    this task exists to fix (a -30.7% bias, and outright zero-beam
    blackouts) is specifically at grazing azimuths, where a ray leaving the
    facade nearly parallel to its own plane has to cross several of its own
    staircase steps -- not just one -- before it escapes the notch.

    A skip that only clears the *first* same-patch voxel hit along the ray
    (instead of every same-patch voxel encountered during the march) would
    pass every other test in this file: at 45 degrees the ray leaves the
    notch after one cell, so a first-only skip looks indistinguishable from
    a full skip. Only a grazing ray, crossing multiple own-patch cells in a
    row, tells them apart.

    Geometry: sun_dir is built as cos(5deg)*in_plane + sin(5deg)*normal,
    where in_plane = (r2, -r2, 0) is perpendicular to the wall's normal
    (r2, r2, 0) -- i.e. sun_dir sits 5 degrees off the facade plane
    (cos_inc ~= 0.087), not the ~45-degree incidence the other tests use.
    Tracing the kernel's own 3D-DDA from the surface (done with a
    standalone Python replica while designing this test, not shipped) shows
    the ray visits, in order: (10,10,1), (11,10,1), (11,9,1), (12,9,1),
    (12,8,1), ... . The first four of those are marked solid and tagged
    with the surface's own patch id (7); (12,8,1) and beyond are left
    empty. So the ray must cross 4 of its own staircase steps before
    reaching open space.
    """
    nx = ny = 24
    nz = 4
    occ = np.zeros((nx, ny, nz), dtype=np.int32)
    patch = np.full((nx, ny, nz), -1, dtype=np.int32)
    own_steps = [(10, 10, 1), (11, 10, 1), (11, 9, 1), (12, 9, 1)]
    for (i, j, k) in own_steps:
        occ[i, j, k] = 1
        patch[i, j, k] = 7

    r2 = float(np.sqrt(0.5))
    theta = np.radians(5.0)
    in_plane = np.array([r2, -r2, 0.0])
    normal = np.array([r2, r2, 0.0])
    sun = np.cos(theta) * in_plane + np.sin(theta) * normal

    sf = _run_shadow(occ, patch,
                     centers=[[10.5, 10.5, 1.5]], dirs_enum=[4],
                     normals=[[r2, r2, 0.0]], patches=[7],
                     sun=tuple(sun))
    assert sf[0] == pytest.approx(0.0)             # sunlit

    # Control: the same four staircase cells, but belonging to someone
    # else's patch -- must still shadow, since a foreign patch legitimately
    # occludes.
    foreign_patch = np.full((nx, ny, nz), -1, dtype=np.int32)
    for (i, j, k) in own_steps:
        foreign_patch[i, j, k] = 99
    sf_foreign = _run_shadow(occ, foreign_patch,
                             centers=[[10.5, 10.5, 1.5]], dirs_enum=[4],
                             normals=[[r2, r2, 0.0]], patches=[7],
                             sun=tuple(sun))
    assert sf_foreign[0] == pytest.approx(1.0)     # shaded
