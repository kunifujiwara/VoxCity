"""With no table, every surface must look exactly as it did before.

The risk this guards is the SVF rewrite: the analytic vffrac path is kept and
must still be the one that runs, and the direct-beam kernel's read of
surfaces.normal is only bit-identical if the stored values are exactly the old
axis constants.
"""
import numpy as np
import pytest

ti = pytest.importorskip("taichi")


@pytest.fixture(scope="module")
def _ti():
    from voxcity.simulator_gpu.init_taichi import ensure_initialized
    ensure_initialized()


def _domain_with_block(n=6, nz=4):
    from voxcity.simulator_gpu.solar.domain import Domain
    d = Domain(nx=n, ny=n, nz=nz, dx=1.0, dy=1.0, dz=1.0,
               origin_lat=35.0, origin_lon=139.0)
    solid = np.zeros((n, n, nz), dtype=np.int32)
    solid[2:4, 2:4, 0:2] = 1
    d.is_solid.from_numpy(solid)
    return d


def test_occupancy_built_surfaces_report_no_patch(_ti):
    """No patch -> the DDA skip is disabled and the analytic SVF path runs."""
    from voxcity.simulator_gpu.solar.domain import extract_surfaces_from_domain

    s = extract_surfaces_from_domain(_domain_with_block())
    assert s.count > 0
    assert (s.patch_id.to_numpy()[:s.count] == -1).all()


def test_stored_normals_are_exactly_the_six_axis_vectors(_ti):
    from voxcity.simulator_gpu.solar.domain import extract_surfaces_from_domain

    s = extract_surfaces_from_domain(_domain_with_block())
    n = s.normal.to_numpy()[:s.count]
    axes = {(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)}
    for row in n:
        assert tuple(int(round(v)) for v in row) in axes
        assert np.isclose(np.linalg.norm(row), 1.0, atol=1e-6)


def test_model_without_table_has_all_minus_one_cell_patch(_ti):
    from voxcity.simulator_gpu.solar.radiation import RadiationModel, RadiationConfig

    model = RadiationModel(_domain_with_block(), RadiationConfig(skip_svf=True))
    assert (model.cell_patch.to_numpy() == -1).all()


# ---------------------------------------------------------------------------
# End-to-end numerical identity, pinned against the pre-feature code.
#
# Reference values below were produced by running the identical script
# (build this same 6x6x4 domain with a 2x2x2 solid block, RadiationModel with
# no surface table, compute_svf(), update_solar_position(day_of_year=172,
# second_of_day=3*3600) -- a real daytime sun position, cos_zenith ~ 0.978 --
# then compute_shortwave_radiation(sw_direct=800.0, sw_diffuse=100.0)) in two
# places: this branch (feat/polygon-normal-surfaces, commit cc5627d) and a
# temporary `git worktree add` checkout of e4a5422 (merge-base with main,
# i.e. the code before this feature existed -- the branch point). The
# worktree was removed after the comparison (`git worktree remove`); nothing
# about the test relies on git or a worktree at run time.
#
# Both runs used env `voxcity`, RadiationConfig(n_azimuth=40, n_elevation=10,
# n_reflection_steps=0, surface_reflections=False, cache_svf_matrix=False,
# canopy_radiation=False, canopy_reflections=False, canopy_to_canopy=False).
#
# Result: 20/20 surfaces compared, in identical extraction order (position +
# direction arrays were byte-for-byte equal), and svf / shadow_factor /
# sw_in_direct / sw_in_diffuse were all max-abs-diff == 0.0 -- bit-identical,
# not merely close. So the reference below is pinned with exact equality
# (np.array_equal), not pytest.approx.
_REF_POSITION = [
    [2, 2, 1], [2, 3, 1], [3, 2, 1], [3, 3, 1],
    [2, 3, 0], [2, 3, 1], [3, 3, 0], [3, 3, 1],
    [2, 2, 0], [2, 2, 1], [3, 2, 0], [3, 2, 1],
    [3, 2, 0], [3, 2, 1], [3, 3, 0], [3, 3, 1],
    [2, 2, 0], [2, 2, 1], [2, 3, 0], [2, 3, 1],
]
_REF_DIRECTION = [0, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
_REF_SVF = [
    1.0, 1.0, 1.0, 1.0,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
]
_REF_SHADOW_FACTOR = [
    0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0,
]
_REF_SW_IN_DIRECT = [
    782.3353271484375, 782.3353271484375, 782.3353271484375, 782.3353271484375,
    0.0, 0.0, 0.0, 0.0,
    51.196163177490234, 51.196163177490234, 51.196163177490234, 51.196163177490234,
    0.0, 0.0, 0.0, 0.0,
    159.15521240234375, 159.15521240234375, 159.15521240234375, 159.15521240234375,
]
_REF_SW_IN_DIFFUSE = [
    100.1646728515625, 100.1646728515625, 100.1646728515625, 100.1646728515625,
    50.0, 50.0, 50.0, 50.0,
    49.991336822509766, 49.991336822509766, 49.991336822509766, 49.991336822509766,
    50.0, 50.0, 50.0, 50.0,
    49.96978759765625, 49.96978759765625, 49.96978759765625, 49.96978759765625,
]


def _canonical_order(position, direction):
    """Row order keyed by (direction, cell) rather than extraction order.

    extract_surfaces_from_domain populates surfaces from a parallel Taichi
    kernel with an atomically-allocated surface index, so which row a given
    (position, direction) lands on is a scheduling detail, not part of the
    contract -- it was observed to differ between running this test file in
    isolation and running it inside the full suite (same code, same domain,
    different row order). (position, direction) is unique per row in this
    fixture (checked once below), so sorting by it gives a stable order to
    compare against regardless of how either run happened to schedule the
    kernel.
    """
    pos = np.asarray(position)
    direction = np.asarray(direction)
    return np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0], direction))


def test_full_pipeline_without_table_matches_pre_feature_code(_ti):
    """svf, shadow_factor, sw_in_direct, sw_in_diffuse vs. the branch point.

    See the block comment above for how the reference was produced and the
    result of that comparison (20 surfaces, max abs diff 0.0 on every
    quantity, once both sides are placed in the same canonical order -- see
    _canonical_order). Re-derive the same run here and pin it exactly.
    """
    from voxcity.simulator_gpu.solar.radiation import RadiationModel, RadiationConfig

    domain = _domain_with_block()
    config = RadiationConfig(
        n_azimuth=40, n_elevation=10,
        n_reflection_steps=0, surface_reflections=False, cache_svf_matrix=False,
        canopy_radiation=False, canopy_reflections=False, canopy_to_canopy=False,
    )
    model = RadiationModel(domain, config)
    model.compute_svf()
    model.update_solar_position(day_of_year=172, second_of_day=3 * 3600)
    model.compute_shortwave_radiation(sw_direct=800.0, sw_diffuse=100.0)

    fluxes = model.get_surface_fluxes()

    assert model.n_surfaces == len(_REF_POSITION)

    ref_position = np.array(_REF_POSITION, dtype=np.int32)
    ref_direction = np.array(_REF_DIRECTION, dtype=np.int32)

    order_actual = _canonical_order(fluxes["position"], fluxes["direction"])
    order_ref = _canonical_order(ref_position, ref_direction)

    # (position, direction) must be a unique key on both sides, or the
    # canonical order above is not well defined.
    actual_keys = {tuple(p) + (int(d),) for p, d in zip(fluxes["position"], fluxes["direction"])}
    assert len(actual_keys) == model.n_surfaces

    assert np.array_equal(fluxes["position"][order_actual], ref_position[order_ref])
    assert np.array_equal(fluxes["direction"][order_actual], ref_direction[order_ref])

    diffs = {}
    for name, ref in (
        ("svf", _REF_SVF),
        ("shadow_factor", _REF_SHADOW_FACTOR),
        ("sw_in_direct", _REF_SW_IN_DIRECT),
        ("sw_in_diffuse", _REF_SW_IN_DIFFUSE),
    ):
        ref_arr = np.array(ref, dtype=np.float32)[order_ref]
        actual_arr = fluxes[name][order_actual]
        diffs[name] = float(np.max(np.abs(actual_arr.astype(np.float64) - ref_arr.astype(np.float64))))

    print(f"surfaces compared: {model.n_surfaces}; max abs diff per quantity: {diffs}")

    for name, d in diffs.items():
        # Bit-identical was observed when this reference was produced (see
        # comment above). If a future run differs only in the last float32
        # ulp, report the magnitude here rather than silently loosening this
        # to pytest.approx -- see the module docstring's guidance.
        assert d == 0.0, f"{name} differs from the pre-feature reference by {d}"
