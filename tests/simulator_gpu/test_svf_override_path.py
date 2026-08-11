"""Sky view factor must follow the true polygon normal, not the six-way
direction enum, wherever a caller supplied an override (patch_id >= 0).

Direct beam (Task 8's predecessor) already reads surfaces.normal. This task
does the same for the diffuse term: SVFCalculator.compute_svf /
compute_svf_with_canopy gain an override branch that cosine-weights against
the true normal and rescales by the analytic (1 + n_z) / 2 sky fraction --
see svf.py's module docstring on unobstructed_svf_for_normal and the kernel
comments for the physics. The analytic six-way branch (patch_id == -1) must
be completely unperturbed; that is what test_backward_compat_matches_frozen_
old_kernel below pins.
"""
import math

import numpy as np
import pytest

ti = pytest.importorskip("taichi")


@pytest.fixture(scope="module")
def _ti():
    from voxcity.simulator_gpu.init_taichi import ensure_initialized
    ensure_initialized()


# ---------------------------------------------------------------------------
# 1. Module-level reference function (pure NumPy, no Taichi)
# ---------------------------------------------------------------------------

def test_unobstructed_svf_for_normal_matches_closed_form():
    """(1 + n_z) / 2 for beta = 90 (vertical), 30, 0 (flat roof) degrees from
    horizontal. This is the full-sphere reference the kernel's override
    branch is checked against below -- both computed via similar but
    independent discretisations, not against an ideal each on its own."""
    from voxcity.simulator_gpu.solar.svf import unobstructed_svf_for_normal

    cases = [
        (90.0, 0.0),
        (30.0, math.cos(math.radians(30.0))),
        (0.0, 1.0),
    ]
    for beta_deg, n_z in cases:
        normal = (math.sin(math.radians(beta_deg)), 0.0, n_z)
        closed = (1.0 + n_z) / 2.0
        val = unobstructed_svf_for_normal(normal)
        assert val == pytest.approx(closed, abs=0.02), f"beta={beta_deg}"


# ---------------------------------------------------------------------------
# Helper: run SVFCalculator.compute_svf directly on a hand-built Surfaces set
# ---------------------------------------------------------------------------

def _run_svf(occ_np, patch_np, centers, dirs_enum, normals, patches,
             n_azimuth=80, n_elevation=40):
    """Build a Domain + Surfaces set and run the real kernel
    SVFCalculator.compute_svf (not a reimplementation)."""
    from voxcity.simulator_gpu.solar.domain import Domain, Surfaces
    from voxcity.simulator_gpu.solar.svf import SVFCalculator

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
            s.position[i] = ti.math.ivec3(0, 0, 0)
            s.direction[i] = de[i]
            s.center[i] = ti.Vector([c[i, 0], c[i, 1], c[i, 2]])
            s.normal[i] = ti.Vector([nn[i, 0], nn[i, 1], nn[i, 2]])
            s.patch_id[i] = pp[i]

    fill(np.asarray(centers, np.float32), np.asarray(dirs_enum, np.int32),
         np.asarray(normals, np.float32), np.asarray(patches, np.int32))
    s.n_surfaces[None] = n

    calc = SVFCalculator(d, n_azimuth=n_azimuth, n_elevation=n_elevation)
    calc.compute_svf(s.center, s.direction, s.normal, s.patch_id, cell_patch,
                      d.is_solid, n, s.svf)
    return s.svf.to_numpy()[:n]


# ---------------------------------------------------------------------------
# 2. Kernel-level: tilted normals in open air must match the closed form
# ---------------------------------------------------------------------------

def test_kernel_matches_closed_form_for_tilted_normals(_ti):
    """A single overridden surface (real patch id, so the override branch is
    taken) high above the ground in an entirely open-air domain. beta = 90
    degrees (vertical) must give 0.5; beta = 30 degrees must give 0.9330 --
    exercised through the actual SVFCalculator.compute_svf kernel, not the
    Python reference."""
    nx = ny = nz = 20
    occ = np.zeros((nx, ny, nz), dtype=np.int32)
    patch = np.full((nx, ny, nz), -1, dtype=np.int32)

    cases = [
        (90.0, 0.0, 0.5),
        (30.0, math.cos(math.radians(30.0)), (1.0 + math.cos(math.radians(30.0))) / 2.0),
    ]
    for beta_deg, n_z, closed in cases:
        n_xy = math.sin(math.radians(beta_deg))
        normal = [n_xy, 0.0, n_z]
        svf = _run_svf(occ, patch,
                        centers=[[10.5, 10.5, 15.5]], dirs_enum=[4],
                        normals=[normal], patches=[5])
        assert svf[0] == pytest.approx(closed, abs=0.03), f"beta={beta_deg}"


def test_kernel_45_degree_wall_gives_half_not_one(_ti):
    """The kernel's direction grid covers the upper hemisphere only (PALM's
    zenith-measured elevation grid), so before the (1 + n_z) / 2 rescale, an
    unobstructed tilted surface's visible/total ratio is 1.0 regardless of
    tilt -- because every grid direction in front of the surface is, by
    construction, unobstructed sky. A wall whose normal is (r2, r2, 0) is
    still a vertical wall (n_z = 0), just rotated in azimuth, so it must
    come out ~0.5. Reporting ~1.0 here is exactly the bug the rescale
    fixes."""
    nx = ny = nz = 20
    occ = np.zeros((nx, ny, nz), dtype=np.int32)
    patch = np.full((nx, ny, nz), -1, dtype=np.int32)
    r2 = float(np.sqrt(0.5))
    svf = _run_svf(occ, patch,
                    centers=[[10.5, 10.5, 15.5]], dirs_enum=[4],
                    normals=[[r2, r2, 0.0]], patches=[5])
    assert svf[0] == pytest.approx(0.5, abs=0.03)
    assert svf[0] != pytest.approx(1.0, abs=0.1)


def test_kernel_own_patch_obstruction_does_not_reduce_svf(_ti):
    """A solid slab tagged with the surface's own patch, filling the sky
    directly above it, must not reduce SVF at all versus the same surface
    with no obstruction -- the own-patch skip DDA treats it as transparent.
    A control with the identical slab tagged as a *foreign* patch confirms
    the slab really is capable of blocking rays (i.e. the equality above
    isn't vacuous because the geometry never intersected anything)."""
    nx = ny = nz = 20

    occ_free = np.zeros((nx, ny, nz), dtype=np.int32)
    patch_free = np.full((nx, ny, nz), -1, dtype=np.int32)
    svf_free = _run_svf(occ_free, patch_free,
                         centers=[[10.5, 10.5, 5.5]], dirs_enum=[0],
                         normals=[[0.0, 0.0, 1.0]], patches=[5])

    occ_slab = np.zeros((nx, ny, nz), dtype=np.int32)
    occ_slab[:, :, 6:14] = 1  # a slab spanning the whole footprint, right above
    patch_own = np.full((nx, ny, nz), -1, dtype=np.int32)
    patch_own[:, :, 6:14] = 5  # same id as the surface's own patch
    svf_own = _run_svf(occ_slab, patch_own,
                        centers=[[10.5, 10.5, 5.5]], dirs_enum=[0],
                        normals=[[0.0, 0.0, 1.0]], patches=[5])

    assert svf_own[0] == pytest.approx(svf_free[0], abs=1e-5)

    patch_foreign = np.full((nx, ny, nz), -1, dtype=np.int32)
    patch_foreign[:, :, 6:14] = 99  # someone else's patch -- must still occlude
    svf_foreign = _run_svf(occ_slab, patch_foreign,
                            centers=[[10.5, 10.5, 5.5]], dirs_enum=[0],
                            normals=[[0.0, 0.0, 1.0]], patches=[5])
    assert svf_foreign[0] < svf_free[0] - 0.1


def test_kernel_downward_override_normal_gives_near_zero_svf(_ti):
    """No grid direction can satisfy d.n > 0 for a straight-down normal
    (the grid only covers the upper hemisphere), so total_vf stays 0 and the
    kernel must fall back to the clamped analytic value (1 + n_z) / 2 = 0,
    not the generic "no rays traced" fallback of 1.0."""
    nx = ny = nz = 20
    occ = np.zeros((nx, ny, nz), dtype=np.int32)
    patch = np.full((nx, ny, nz), -1, dtype=np.int32)
    svf = _run_svf(occ, patch,
                    centers=[[10.5, 10.5, 15.5]], dirs_enum=[1],
                    normals=[[0.0, 0.0, -1.0]], patches=[5])
    assert svf[0] == pytest.approx(0.0, abs=1e-4)


def test_kernel_with_canopy_matches_closed_form_too(_ti):
    """compute_svf_with_canopy is the live production path (Domain.lad is
    always allocated, see RadiationModel.compute_svf), so the override
    branch must not be left half-done there. With LAD zero everywhere,
    transmissivity is 1.0 for every ray, so both svf and svf_urban should
    match the same (1 + n_z)/2 closed form as compute_svf."""
    from voxcity.simulator_gpu.solar.domain import Domain, Surfaces
    from voxcity.simulator_gpu.solar.svf import SVFCalculator

    nx = ny = nz = 20
    d = Domain(nx=nx, ny=ny, nz=nz, dx=1.0, dy=1.0, dz=1.0,
               origin_lat=35.0, origin_lon=139.0)
    cell_patch = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
    cell_patch.fill(-1)

    r2 = float(np.sqrt(0.5))
    s = Surfaces(1)

    @ti.kernel
    def fill():
        s.direction[0] = 4
        s.center[0] = ti.Vector([10.5, 10.5, 15.5])
        s.normal[0] = ti.Vector([r2, r2, 0.0])  # vertical wall -> 0.5
        s.patch_id[0] = 5

    fill()
    s.n_surfaces[None] = 1

    calc = SVFCalculator(d)
    calc.compute_svf_with_canopy(
        s.center, s.direction, s.normal, s.patch_id, cell_patch,
        d.is_solid, d.lad, 1, 0.6, s.svf, s.svf_urban)

    svf = s.svf.to_numpy()[0]
    svf_urban = s.svf_urban.to_numpy()[0]
    assert svf == pytest.approx(0.5, abs=0.03)
    assert svf_urban == pytest.approx(0.5, abs=0.03)


# ---------------------------------------------------------------------------
# 5. Backward compatibility, measured against a frozen copy of the old kernel
# ---------------------------------------------------------------------------

from voxcity.simulator_gpu.solar.core import Vector3, PI, TWO_PI  # noqa: E402
from voxcity.simulator_gpu.solar.raytracing import ray_voxel_first_hit  # noqa: E402


@ti.data_oriented
class _FrozenOldSVFCalculator:
    """Verbatim copy of SVFCalculator._init_directions + the pre-task-8
    compute_svf kernel (analytic six-way branch only, no override
    parameters), frozen here so the backward-compatibility test below
    compares against genuinely old code rather than trusting that the
    modified module's untouched branch is still untouched."""

    def __init__(self, domain, n_azimuth: int = 80, n_elevation: int = 40):
        self.domain = domain
        self.nx = domain.nx
        self.ny = domain.ny
        self.nz = domain.nz
        self.dx = domain.dx
        self.dy = domain.dy
        self.dz = domain.dz
        self.n_azimuth = n_azimuth
        self.n_elevation = n_elevation
        self.max_dist = math.sqrt((self.nx * self.dx) ** 2 +
                                   (self.ny * self.dy) ** 2 +
                                   (self.nz * self.dz) ** 2)
        self.directions = ti.Vector.field(3, dtype=ti.f32, shape=(n_azimuth, n_elevation))
        self.solid_angles = ti.field(dtype=ti.f32, shape=(n_azimuth, n_elevation))
        self.total_solid_angle = ti.field(dtype=ti.f32, shape=())
        self._init_directions()

    @ti.kernel
    def _init_directions(self):
        total_omega = 0.0
        n_azim_f = ti.cast(self.n_azimuth, ti.f32)
        n_elev_f = ti.cast(self.n_elevation, ti.f32)
        d_azim = TWO_PI / n_azim_f
        for i_azim, i_elev in ti.ndrange(self.n_azimuth, self.n_elevation):
            elev_low = ti.cast(i_elev, ti.f32) * (PI / 2.0) / n_elev_f
            elev_high = ti.cast(i_elev + 1, ti.f32) * (PI / 2.0) / n_elev_f
            elev_center = (elev_low + elev_high) / 2.0
            azim_center = (ti.cast(i_azim, ti.f32) + 0.5) * d_azim
            sin_elev = ti.sin(elev_center)
            cos_elev = ti.cos(elev_center)
            x = sin_elev * ti.sin(azim_center)
            y = sin_elev * ti.cos(azim_center)
            z = cos_elev
            self.directions[i_azim, i_elev] = Vector3(x, y, z)
            vf_up = (ti.cos(2.0 * elev_low) - ti.cos(2.0 * elev_high)) / (2.0 * n_azim_f)
            self.solid_angles[i_azim, i_elev] = vf_up
            total_omega += vf_up
        self.total_solid_angle[None] = total_omega

    @ti.kernel
    def compute_svf(
        self,
        surf_pos: ti.template(),
        surf_dir: ti.template(),
        is_solid: ti.template(),
        n_surf: ti.i32,
        svf: ti.template()
    ):
        n_azim_f = ti.cast(self.n_azimuth, ti.f32)
        n_elev_f = ti.cast(self.n_elevation, ti.f32)
        d_azim = TWO_PI / n_azim_f
        for i in range(n_surf):
            pos = Vector3(surf_pos[i][0], surf_pos[i][1], surf_pos[i][2])
            direction = surf_dir[i]
            normal = Vector3(0.0, 0.0, 0.0)
            normal_azim = 0.0
            if direction == 0:
                normal = Vector3(0.0, 0.0, 1.0)
            elif direction == 1:
                normal = Vector3(0.0, 0.0, -1.0)
            elif direction == 2:
                normal = Vector3(0.0, 1.0, 0.0)
                normal_azim = 0.0
            elif direction == 3:
                normal = Vector3(0.0, -1.0, 0.0)
                normal_azim = PI
            elif direction == 4:
                normal = Vector3(1.0, 0.0, 0.0)
                normal_azim = PI / 2.0
            elif direction == 5:
                normal = Vector3(-1.0, 0.0, 0.0)
                normal_azim = 3.0 * PI / 2.0

            eps = ti.min(self.dx, ti.min(self.dy, self.dz)) * 1e-4
            ray_origin = Vector3(
                pos[0] + normal[0] * eps,
                pos[1] + normal[1] * eps,
                pos[2] + normal[2] * eps,
            )

            visible_vf = 0.0
            total_vf = 0.0

            for i_azim, i_elev in ti.ndrange(self.n_azimuth, self.n_elevation):
                ray_dir = self.directions[i_azim, i_elev]
                cos_angle = ray_dir[0] * normal[0] + ray_dir[1] * normal[1] + ray_dir[2] * normal[2]
                if cos_angle > 0.001:
                    elev_low = ti.cast(i_elev, ti.f32) * (PI / 2.0) / n_elev_f
                    elev_high = ti.cast(i_elev + 1, ti.f32) * (PI / 2.0) / n_elev_f
                    vf_frac = 0.0
                    if direction == 0:
                        vf_frac = (ti.cos(2.0 * elev_low) - ti.cos(2.0 * elev_high)) / (2.0 * n_azim_f)
                    elif direction == 1:
                        vf_frac = (ti.cos(2.0 * elev_low) - ti.cos(2.0 * elev_high)) / (2.0 * n_azim_f)
                    else:
                        azim_low = ti.cast(i_azim, ti.f32) * d_azim
                        azim_high = ti.cast(i_azim + 1, ti.f32) * d_azim
                        az1_rel = azim_low - normal_azim
                        az2_rel = azim_high - normal_azim
                        elev_terms = (elev_high - elev_low
                                      + ti.sin(elev_low) * ti.cos(elev_low)
                                      - ti.sin(elev_high) * ti.cos(elev_high))
                        vf_frac = (ti.sin(az2_rel) - ti.sin(az1_rel)) * elev_terms / TWO_PI
                        if vf_frac < 0.0:
                            vf_frac = 0.0

                    total_vf += vf_frac

                    hit, _, _, _, _ = ray_voxel_first_hit(
                        ray_origin, ray_dir,
                        is_solid,
                        self.nx, self.ny, self.nz,
                        self.dx, self.dy, self.dz,
                        self.max_dist
                    )

                    if hit == 0:
                        visible_vf += vf_frac

            if direction >= 2:
                total_vf = total_vf * 2.0

            if total_vf > 0.001:
                svf[i] = visible_vf / total_vf
            else:
                svf[i] = 1.0


def test_backward_compat_matches_frozen_old_kernel(_ti):
    """Surfaces built by extract_surfaces_from_domain all have patch_id ==
    -1 and axis normals, so the new kernel must take the untouched analytic
    branch and reproduce the frozen pre-task-8 kernel exactly -- this task
    only adds a new branch gated on patch_id >= 0."""
    from voxcity.simulator_gpu.solar.domain import Domain, extract_surfaces_from_domain
    from voxcity.simulator_gpu.solar.svf import SVFCalculator

    nx, ny, nz = 10, 10, 6
    d = Domain(nx=nx, ny=ny, nz=nz, dx=1.0, dy=1.0, dz=1.0,
               origin_lat=35.0, origin_lon=139.0)
    occ = np.zeros((nx, ny, nz), dtype=np.int32)
    occ[3:6, 3:6, 0:3] = 1
    d.is_solid.from_numpy(occ)

    surfaces = extract_surfaces_from_domain(d)
    n = surfaces.count
    assert n > 0
    assert np.all(surfaces.patch_id.to_numpy()[:n] == -1)

    cell_patch = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
    cell_patch.fill(-1)

    new_calc = SVFCalculator(d)
    new_calc.compute_svf(surfaces.center, surfaces.direction, surfaces.normal,
                          surfaces.patch_id, cell_patch, d.is_solid, n, surfaces.svf)
    new_svf = surfaces.svf.to_numpy()[:n].copy()

    old_calc = _FrozenOldSVFCalculator(d)
    old_svf_field = ti.field(dtype=ti.f32, shape=(surfaces.max_surfaces,))
    old_calc.compute_svf(surfaces.center, surfaces.direction, d.is_solid, n, old_svf_field)
    old_svf = old_svf_field.to_numpy()[:n]

    max_diff = float(np.max(np.abs(new_svf - old_svf)))
    print(f"backward-compat: {n} surfaces, max abs diff = {max_diff}")
    assert max_diff == 0.0


# ---------------------------------------------------------------------------
# (f) The diffuse branch in radiation.py must follow the true normal's SVF,
#     not hard-zero on face == IDOWN, for overridden surfaces.
# ---------------------------------------------------------------------------

def test_diffuse_follows_svf_not_face_enum_for_override(_ti):
    """A table can legitimately supply face=IDOWN (direction 1) on a surface
    whose true normal is only partly downward. The old three-way switch
    hard-zeroed diffuse whenever direction == 1; for an overridden surface
    (patch_id >= 0) this must instead follow the SVF already computed from
    the true normal, which is nonzero here (n_z = -0.5 -> svf = 0.25)."""
    from voxcity.simulator_gpu.solar.domain import Domain, Surfaces
    from voxcity.simulator_gpu.solar.radiation import RadiationModel, RadiationConfig

    n = 4
    d = Domain(nx=n, ny=n, nz=n, dx=1.0, dy=1.0, dz=1.0,
               origin_lat=35.0, origin_lon=139.0)
    surfaces = Surfaces(1)

    @ti.kernel
    def fill():
        surfaces.direction[0] = 1  # IDOWN -- the old switch zeroes diffuse here
        surfaces.normal[0] = ti.Vector([0.0, 0.8660254, -0.5])  # n_z = -0.5
        surfaces.patch_id[0] = 0  # overridden: a real patch id
        surfaces.svf[0] = 0.25  # pretend compute_svf already ran: (1 - 0.5) / 2
        surfaces.shadow_factor[0] = 1.0  # fully shadowed -> direct term excluded
        surfaces.canopy_transmissivity[0] = 1.0
        surfaces.center[0] = ti.Vector([2.5, 2.5, 2.5])

    fill()
    surfaces.n_surfaces[None] = 1

    model = RadiationModel(d, RadiationConfig(skip_svf=True), surfaces=surfaces)
    model.solar_calc.sun_direction[None] = (0.0, 0.0, 1.0)

    model._compute_initial_sw_pass(0.0, 100.0, 0.5)

    sw_in_dif = float(model._surfinswdif.to_numpy()[0])
    assert sw_in_dif == pytest.approx(25.0, abs=1e-3)
