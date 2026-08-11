"""The direct-beam cosine must come from surfaces.normal, not `direction`."""
import numpy as np
import pytest

from voxcity.simulator_gpu.solar.radiation import surface_incidence_cosine

ti = pytest.importorskip("taichi")


def test_cosine_follows_the_stored_normal():
    r2 = float(np.sqrt(0.5))
    assert surface_incidence_cosine((r2, r2, 0.0), (r2, r2, 0.0)) == pytest.approx(1.0, abs=1e-6)


def test_axis_normal_is_unchanged():
    r2 = float(np.sqrt(0.5))
    assert surface_incidence_cosine((1.0, 0.0, 0.0), (r2, r2, 0.0)) == pytest.approx(r2, abs=1e-6)


def test_back_face_clamps_to_zero():
    assert surface_incidence_cosine((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)) == 0.0


@pytest.fixture(scope="module")
def _ti():
    from voxcity.simulator_gpu.init_taichi import ensure_initialized
    ensure_initialized()


def test_kernel_uses_the_true_normal_not_the_axis_normal(_ti):
    """Pin _compute_initial_sw_pass itself, not just the extracted helper.

    A one-row surface table gives face=4 (IEAST), whose axis normal is
    (1, 0, 0), but a true normal of (r2, r2, 0) -- a wall tilted 45 degrees
    off the East axis. The sun sits exactly on that true normal, so:

    - If the kernel reads surfaces.normal (current, correct behaviour),
      cos_incidence = 1.0 and surfinswdir = sw_direct = 1000.0 W/m^2.
    - If the kernel instead rebuilt an axis normal from `direction` (the old
      behaviour Task 5 removed, or a regression that reinstates it),
      cos_incidence = dot((1,0,0), (r2,r2,0)) = r2 and surfinswdir would be
      1000 * r2 ~= 707.1 W/m^2.

    These two outcomes are far enough apart that the assertion below can't
    pass by accident under either implementation -- this is what actually
    exercises the kernel change, since surface_incidence_cosine above is a
    standalone helper the kernel does not call (see its docstring).
    """
    from voxcity.simulator_gpu.solar.domain import Domain, surfaces_from_override
    from voxcity.simulator_gpu.solar.radiation import RadiationModel, RadiationConfig
    from voxcity.simulator_gpu.solar.surface_override import SurfaceOverride

    n = 6
    d = Domain(nx=n, ny=n, nz=4, dx=1.0, dy=1.0, dz=1.0,
               origin_lat=35.0, origin_lon=139.0)
    occ = np.zeros((n, n, 4), dtype=np.int32)
    occ[2, 2, 1] = 1  # one solid cell -- an all-air domain crashes on
                       # zero-sized field allocation (see test_radiation_model_injection.py)
    d.is_solid.from_numpy(occ)

    r2 = float(np.sqrt(0.5))
    ov = SurfaceOverride(
        cell=np.array([[2, 2, 1]], dtype=np.int32),
        face=np.array([4], dtype=np.int8),        # IEAST, axis normal (1, 0, 0)
        origin=np.array([[3.0, 2.5, 1.5]], dtype=np.float32),  # east face, open air beyond it
        normal=np.array([[r2, r2, 0.0]], dtype=np.float32),    # true normal: 45 degrees off-axis
        patch=np.array([0], dtype=np.int32),
        area=np.array([1.0], dtype=np.float32),
    )
    surfaces = surfaces_from_override(ov)

    model = RadiationModel(
        d,
        RadiationConfig(skip_svf=True, n_reflection_steps=0, surface_reflections=False),
        surfaces=surfaces,
    )

    # Domain.lad is always a Taichi field (never None -- Domain.__init__
    # allocates it unconditionally), so compute_shortwave_radiation always
    # takes the canopy branch (compute_direct_with_canopy). With lad left at
    # its zero default and an unobstructed ray, that kernel's
    # ray_canopy_absorption returns full transmissivity (1.0), so
    # canopy_transmissivity == 1.0 here too -- same factor the no-canopy
    # branch would give. Either way sw_in_dir = sw_direct * cos_incidence * 1.0.
    model.solar_calc.sun_direction[None] = (r2, r2, 0.0)
    model.solar_calc.cos_zenith[None] = 0.7  # > min_stable_coszen (0.0262)
    model.solar_calc.sun_up[None] = 1

    model.compute_shortwave_radiation(sw_direct=1000.0, sw_diffuse=0.0)

    surfinswdir = float(model._surfinswdir.to_numpy()[0])
    axis_normal_value = 1000.0 * r2  # ~= 707.1, what the removed direction-switch would give

    assert surfinswdir == pytest.approx(1000.0, abs=1e-2)
    assert surfinswdir != pytest.approx(axis_normal_value, abs=1.0)
