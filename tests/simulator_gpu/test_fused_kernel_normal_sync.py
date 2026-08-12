"""The fused kernel `compute_initial_and_reflections_fused` must stay in sync
with the live kernel `_compute_initial_sw_pass` (radiation.py).

That kernel was found still doing the old direction-based axis normal and
the old hard-zero-on-IDOWN diffuse rule after the live kernel was fixed to
(a) read `surfaces.normal` instead of rebuilding an axis normal from the
`direction` enum, and (b) follow SVF unconditionally for the diffuse
contribution of overridden (patch_id >= 0) surfaces. This test drives both
kernels with the same inputs -- the live kernel via a real RadiationModel
run, the fused kernel directly -- and pins them to identical outputs for:

(a) an axis-normal surface with patch_id == -1 (must reproduce the old,
    still-correct staircase behaviour), and
(b) an overridden surface whose true normal is 45 degrees off its tagged
    `direction`, and whose `direction` is IDOWN even though the true normal
    is not straight down (must follow the true normal for cosine, and
    follow SVF unconditionally for diffuse).
"""
import numpy as np
import pytest

pytest.importorskip("taichi")
import taichi as ti


def test_fused_kernel_matches_live_kernel_for_axis_and_overridden_surfaces():
    from voxcity.simulator_gpu.init_taichi import ensure_initialized
    ensure_initialized()

    from voxcity.simulator_gpu.solar.domain import Domain, surfaces_from_override
    from voxcity.simulator_gpu.solar.radiation import RadiationModel, RadiationConfig
    from voxcity.simulator_gpu.solar.surface_override import SurfaceOverride
    from voxcity.simulator_gpu.solar.reflection import compute_initial_and_reflections_fused

    r2 = float(np.sqrt(0.5))

    n = 6
    d = Domain(nx=n, ny=n, nz=4, dx=1.0, dy=1.0, dz=1.0,
               origin_lat=35.0, origin_lon=139.0)
    occ = np.zeros((n, n, 4), dtype=np.int32)
    occ[2, 2, 1] = 1  # one solid cell -- an all-air domain crashes on
                       # zero-sized field allocation (see test_radiation_model_injection.py)
    d.is_solid.from_numpy(occ)

    ov = SurfaceOverride(
        cell=np.array([[2, 2, 1], [2, 2, 1]], dtype=np.int32),
        # Surface 0: IEAST tag with a true axis normal -- ordinary, unpatched.
        # Surface 1: IDOWN tag, but a true normal that is NOT straight down
        # (45 degrees off-axis in the horizontal plane) -- the exact case the
        # SVF-override comment in _compute_initial_sw_pass calls out: a table
        # can legitimately report direction==IDOWN for a surface whose true
        # normal is only partly downward.
        # Both rows use the same open-air origin (east face, proven
        # unobstructed by test_direct_beam_uses_normal.py) so shadow
        # ray-tracing against the one solid cell can't confound the
        # cosine/diffuse comparison this test is actually about.
        face=np.array([4, 1], dtype=np.int8),
        origin=np.array([[3.0, 2.5, 1.5], [3.0, 2.5, 1.5]], dtype=np.float32),
        normal=np.array([[1.0, 0.0, 0.0], [r2, r2, 0.0]], dtype=np.float32),
        patch=np.array([-1, 0], dtype=np.int32),
        area=np.array([1.0, 1.0], dtype=np.float32),
    )
    surfaces = surfaces_from_override(ov)

    model = RadiationModel(
        d,
        RadiationConfig(skip_svf=True, n_reflection_steps=0, surface_reflections=False),
        surfaces=surfaces,
    )

    sun_dir = (r2, r2, 0.0)
    cos_zenith = 0.7  # > min_stable_coszen (0.0262)
    sw_direct = 1000.0
    sw_diffuse = 200.0

    model.solar_calc.sun_direction[None] = sun_dir
    model.solar_calc.cos_zenith[None] = cos_zenith
    model.solar_calc.sun_up[None] = 1

    model.compute_shortwave_radiation(sw_direct=sw_direct, sw_diffuse=sw_diffuse)

    live_dir = model._surfinswdir.to_numpy()
    live_dif = model._surfinswdif.to_numpy()

    # Sanity-check the live kernel actually exercises both regressions this
    # test is meant to catch, so a fused-kernel bug can't hide behind a live
    # kernel that also happens to produce zero/axis-normal answers.
    assert live_dir[0] == pytest.approx(sw_direct * r2, abs=1e-2)   # axis normal, unchanged
    assert live_dir[1] == pytest.approx(sw_direct * 1.0, abs=1e-2)  # true normal, not axis-IDOWN's 0
    assert live_dif[1] == pytest.approx(sw_diffuse * 1.0, abs=1e-2)  # SVF-unconditional, not zeroed by IDOWN

    # ---- Drive the fused kernel directly with the same raw inputs ----
    n_surf = model.n_surfaces
    surf_direction = ti.field(dtype=ti.i32, shape=(n_surf,))
    surf_svf = ti.field(dtype=ti.f32, shape=(n_surf,))
    surf_shadow = ti.field(dtype=ti.f32, shape=(n_surf,))
    surf_canopy_trans = ti.field(dtype=ti.f32, shape=(n_surf,))
    surf_albedo = ti.field(dtype=ti.f32, shape=(n_surf,))
    surf_normal = ti.Vector.field(3, dtype=ti.f32, shape=(n_surf,))
    surf_patch = ti.field(dtype=ti.i32, shape=(n_surf,))

    surf_direction.from_numpy(model.surfaces.direction.to_numpy()[:n_surf])
    surf_svf.from_numpy(model.surfaces.svf.to_numpy()[:n_surf])
    surf_shadow.from_numpy(model.surfaces.shadow_factor.to_numpy()[:n_surf])
    surf_canopy_trans.from_numpy(model.surfaces.canopy_transmissivity.to_numpy()[:n_surf])
    surf_albedo.from_numpy(model.surfaces.albedo.to_numpy()[:n_surf])
    surf_normal.from_numpy(model.surfaces.normal.to_numpy()[:n_surf])
    surf_patch.from_numpy(model.surfaces.patch_id.to_numpy()[:n_surf])

    # Empty SVF matrix -- n_ref_steps=0 means the reflection loop body never
    # runs, but the fields must still exist for the kernel's template params.
    svf_source = ti.field(dtype=ti.i32, shape=(1,))
    svf_target = ti.field(dtype=ti.i32, shape=(1,))
    svf_vf = ti.field(dtype=ti.f32, shape=(1,))
    svf_trans = ti.field(dtype=ti.f32, shape=(1,))

    sw_in_direct = ti.field(dtype=ti.f32, shape=(n_surf,))
    sw_in_diffuse = ti.field(dtype=ti.f32, shape=(n_surf,))
    sw_in_reflected = ti.field(dtype=ti.f32, shape=(n_surf,))
    sw_out_total = ti.field(dtype=ti.f32, shape=(n_surf,))
    surfins_a = ti.field(dtype=ti.f32, shape=(n_surf,))
    surfins_b = ti.field(dtype=ti.f32, shape=(n_surf,))
    surfout = ti.field(dtype=ti.f32, shape=(n_surf,))

    compute_initial_and_reflections_fused(
        surf_direction, surf_svf, surf_shadow, surf_canopy_trans,
        surf_albedo, surf_normal, surf_patch,
        sun_dir[0], sun_dir[1], sun_dir[2], cos_zenith,
        sw_direct, sw_diffuse,
        svf_source, svf_target, svf_vf, svf_trans, 0,
        n_surf, 0,
        sw_in_direct, sw_in_diffuse, sw_in_reflected, sw_out_total,
        surfins_a, surfins_b, surfout,
    )

    fused_dir = sw_in_direct.to_numpy()
    fused_dif = sw_in_diffuse.to_numpy()

    np.testing.assert_allclose(fused_dir, live_dir, atol=1e-2)
    np.testing.assert_allclose(fused_dif, live_dif, atol=1e-2)
