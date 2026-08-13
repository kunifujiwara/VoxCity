"""Regression coverage for `_map_mesh_faces_to_surfaces`.

The bug (found on a real 100x100x103 city with override surfaces): the
mapping bucketed both mesh faces and surfaces by `np.rint(normal)`, and
queried each bucket's KD-tree with no distance cap.

- A surface whose true normal doesn't round to a unit axis (e.g. a 45 degree
  wall rounds to (1, 1, 0)) lands in a bucket no voxel-mesh face ever has --
  it can never be matched.
- Because there's no distance cap, mesh faces in an affected bucket still get
  matched to *something* in that bucket, however far away -- measured up to
  28.3 m on the real city, vs. ~0.47 m for a correct match.

The fix buckets by the surface's `direction` field (0-5, always a voxel-face
axis -- see domain.py's IUP..IWEST -- even for override surfaces, where it's
the exposed face the ray leaves from, not the true polygon normal) instead of
by rounding the surface's own normal, and caps the nearest-centroid distance
so an out-of-range match becomes -1 (which the caller already renders as
NaN) instead of a distant guess.
"""
import math

import numpy as np
import pytest

ti = pytest.importorskip("taichi")
trimesh = pytest.importorskip("trimesh")


@pytest.fixture(scope="module")
def _ti():
    from voxcity.simulator_gpu.init_taichi import ensure_initialized
    ensure_initialized()


def _scene(points):
    from voxcity.simulator.common.coordinates import uv_domain_points_to_scene
    return uv_domain_points_to_scene(np.asarray(points, dtype=np.float64))


# ---------------------------------------------------------------------------
# The regression: a non-axis (override) surface normal must still be found,
# by its `direction`, and a mesh face must never be matched to a surface
# implausibly far away just because it shares a rounded-normal bucket.
# ---------------------------------------------------------------------------

def test_non_axis_normal_surface_is_matched_by_direction_not_far_decoy(_ti):
    from voxcity.simulator_gpu.solar.domain import surfaces_from_override
    from voxcity.simulator_gpu.solar.surface_override import SurfaceOverride
    from voxcity.simulator_gpu.solar.integration.building import _map_mesh_faces_to_surfaces

    r2 = float(np.sqrt(0.5))
    # Row 0: a 45-degree wall. face=INORTH(2) is the voxel face it was cut
    # from (the correct join key); its true normal is (r2, r2, 0), which
    # rounds to (1, 1, 0) -- not a unit axis, per the reported defect.
    # Row 1: an ordinary axis-aligned INORTH surface far away (27 m in scene
    # x), a stand-in for "whatever else happens to share the old rint bucket".
    table = SurfaceOverride(
        cell=np.array([[2, 2, 1], [2, 2, 1]], dtype=np.int32),
        face=np.array([2, 2], dtype=np.int8),  # INORTH, INORTH
        origin=np.array([[2.5, 3.0, 1.5], [2.5, 30.0, 1.5]], dtype=np.float32),
        normal=np.array([[r2, r2, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        patch=np.array([5, 9], dtype=np.int32),
        area=np.array([1.41, 1.0], dtype=np.float32),
    )
    surfaces = surfaces_from_override(table, default_albedo=0.2)
    assert surfaces.count == 2

    surf_centers_scene = _scene(surfaces.center.to_numpy()[:2])
    surf_directions = surfaces.direction.to_numpy()[:2]
    bldg_indices = np.arange(2)

    # A single mesh face at the sloped wall's exposed voxel face: INORTH's
    # scene-space normal key is (1, 0, 0) (domain (0,1,0) with the u/v<->x/y
    # swap uv_domain_points_to_scene applies). Its center sits close to row
    # 0's true center, not row 1's.
    mesh_face_centers = _scene(np.array([[2.5, 3.0, 1.5]]))
    mesh_face_normals = np.array([[1.0, 0.0, 0.0]])

    result = _map_mesh_faces_to_surfaces(
        mesh_face_centers, mesh_face_normals,
        surf_centers_scene, surf_directions, bldg_indices,
        max_match_distance=2.0,
    )

    assert result.shape == (1,)
    assert result[0] == 0, "must match the true (non-axis-normal) surface, not the far decoy"

    matched_center = surf_centers_scene[result[0]]
    dist = float(np.linalg.norm(mesh_face_centers[0] - matched_center))
    assert dist <= 2.0, f"match distance {dist} exceeds the cap"
    print(f"non-axis regression: matched surface 0, distance={dist:.4f} m (decoy was 27 m away)")


# ---------------------------------------------------------------------------
# The no-match case: nothing within the cap must yield -1, not a distant
# same-direction surface.
# ---------------------------------------------------------------------------

def test_face_with_no_nearby_surface_gets_minus_one(_ti):
    from voxcity.simulator_gpu.solar.integration.building import _map_mesh_faces_to_surfaces

    # One IWEST-facing (scene normal (0, -1, 0)) surface, 5 m from the mesh
    # face -- well past a 2.0 m cap.
    surf_centers_scene = np.array([[2.5, -2.5, 1.5]])
    surf_directions = np.array([5])  # IWEST
    bldg_indices = np.array([0])

    mesh_face_centers = np.array([[2.5, 2.5, 1.5]])
    mesh_face_normals = np.array([[0.0, -1.0, 0.0]])

    result = _map_mesh_faces_to_surfaces(
        mesh_face_centers, mesh_face_normals,
        surf_centers_scene, surf_directions, bldg_indices,
        max_match_distance=2.0,
    )

    assert result.tolist() == [-1]


def test_face_with_nearby_surface_within_cap_still_matches(_ti):
    """Sanity check on the same geometry as above, cap wide enough to allow it."""
    from voxcity.simulator_gpu.solar.integration.building import _map_mesh_faces_to_surfaces

    surf_centers_scene = np.array([[2.5, -2.5, 1.5]])
    surf_directions = np.array([5])
    bldg_indices = np.array([0])

    mesh_face_centers = np.array([[2.5, 2.5, 1.5]])
    mesh_face_normals = np.array([[0.0, -1.0, 0.0]])

    result = _map_mesh_faces_to_surfaces(
        mesh_face_centers, mesh_face_normals,
        surf_centers_scene, surf_directions, bldg_indices,
        max_match_distance=10.0,
    )

    assert result.tolist() == [0]


# ---------------------------------------------------------------------------
# No-op proof: on occupancy-built surfaces (every normal already a unit
# axis), old rint-bucketing and new direction-bucketing must return the
# identical index array.
# ---------------------------------------------------------------------------

def _old_map_mesh_faces_to_surfaces(mesh_face_centers, mesh_face_normals,
                                     surface_centers_scene, surface_normals_scene,
                                     bldg_indices):
    """Verbatim copy of the pre-fix implementation (rint-keyed, no distance
    cap), kept only so the no-op test can compare against it directly."""
    from scipy.spatial import cKDTree

    mesh_normals_key = np.rint(mesh_face_normals).astype(np.int8)
    surface_normals_key = np.rint(surface_normals_scene).astype(np.int8)
    result = np.empty(len(mesh_face_centers), dtype=np.int64)

    for normal_key in np.unique(mesh_normals_key, axis=0):
        face_mask = np.all(mesh_normals_key == normal_key, axis=1)
        surface_mask = np.all(surface_normals_key[bldg_indices] == normal_key, axis=1)
        candidate_indices = bldg_indices[surface_mask]

        if candidate_indices.size == 0:
            result[face_mask] = -1
            continue

        tree = cKDTree(surface_centers_scene[candidate_indices])
        _, nearest_idx = tree.query(mesh_face_centers[face_mask], k=1)
        result[face_mask] = candidate_indices[nearest_idx]

    return result


def _domain_with_block(n=6, nz=4):
    from voxcity.simulator_gpu.solar.domain import Domain
    d = Domain(nx=n, ny=n, nz=nz, dx=1.0, dy=1.0, dz=1.0,
               origin_lat=35.0, origin_lon=139.0)
    solid = np.zeros((n, n, nz), dtype=np.int32)
    solid[2:4, 2:4, 0:2] = 1
    d.is_solid.from_numpy(solid)
    return d


def test_occupancy_built_surfaces_map_identically_old_vs_new(_ti):
    """Build a real domain + a real voxel mesh over the same block, extract
    occupancy-built surfaces (every normal already a unit axis, per
    test_flag_off_identity.py), and confirm the new direction-keyed,
    distance-capped mapping returns exactly the same index array as the old
    rint-keyed, uncapped one.
    """
    from voxcity.simulator_gpu.solar.domain import extract_surfaces_from_domain
    from voxcity.simulator_gpu.solar.integration.building import _map_mesh_faces_to_surfaces
    from voxcity.geoprocessor.mesh import create_voxel_mesh

    n, nz, meshsize = 6, 4, 1.0
    domain = _domain_with_block(n=n, nz=nz)
    surfaces = extract_surfaces_from_domain(domain)
    count = surfaces.count
    assert count > 0

    voxel_array = np.zeros((n, n, nz), dtype=np.int32)
    voxel_array[2:4, 2:4, 0:2] = -3  # VOXCITY_BUILDING_CODE
    building_id_grid = np.zeros((n, n), dtype=np.int32)

    mesh = create_voxel_mesh(voxel_array, class_id=-3, meshsize=meshsize,
                              building_id_grid=building_id_grid, mesh_type='open_air')
    assert mesh is not None
    n_faces = len(mesh.faces)
    assert n_faces > 0

    surf_centers_scene = _scene(surfaces.center.to_numpy()[:count])
    surf_normals_scene = _scene(surfaces.normal.to_numpy()[:count])
    surf_directions = surfaces.direction.to_numpy()[:count]
    bldg_indices = np.arange(count)

    mesh_face_centers = mesh.triangles_center
    mesh_face_normals = mesh.face_normals

    old_result = _old_map_mesh_faces_to_surfaces(
        mesh_face_centers, mesh_face_normals,
        surf_centers_scene, surf_normals_scene, bldg_indices,
    )
    new_result = _map_mesh_faces_to_surfaces(
        mesh_face_centers, mesh_face_normals,
        surf_centers_scene, surf_directions, bldg_indices,
        max_match_distance=2.0 * meshsize,
    )

    n_unmatched = int((old_result < 0).sum())
    print(f"no-op proof: {count} surfaces, {n_faces} mesh faces compared "
          f"({n_unmatched} unmatched on both sides, e.g. the block's ground-contact "
          f"underside, which extract_surfaces_from_domain never emits a surface for)")
    assert np.array_equal(old_result, new_result)


# ---------------------------------------------------------------------------
# Exporting the mapping (and the normal it selected) alongside the values.
#
# get_building_solar_irradiance returns an axis-aligned voxel-staircase mesh
# with per-face irradiance in `metadata`. A consumer that projects those
# values onto a polygon city model has to re-derive which surface produced
# each value, and can only do so by axis bucket -- so a value computed for a
# 20-degree-tilted facade lands on a horizontal roof polygon, and the roof
# then reports irradiance above the physical ceiling for a horizontal
# surface. The join is already known here (`mesh_to_surface_idx`), so it is
# exported: the surface index per face, and that surface's own normal in the
# mesh's own (scene) frame.
#
# Only when an override table is active: without one every surface normal is
# already the mesh face's own axis normal, so the export would carry no
# information, and the flag-off output must stay byte-identical.
# ---------------------------------------------------------------------------

# Both in the table's own frame, i.e. uv-domain (u=row, v=col, z) -- see
# surface_override.SURFACE_TABLE_CONVENTION. Deliberately chosen with u != v
# so that the u/v <-> x/y swap into scene coordinates is observable: if the
# export dropped the swap these would come back unchanged and the assertions
# below would still see a unit, finite, correctly-indexed normal.
_ROOF_NORMAL_UV = (math.sin(math.radians(20.0)), 0.0, math.cos(math.radians(20.0)))
_WALL_NORMAL_UV = (math.cos(math.radians(15.0)), math.sin(math.radians(15.0)), 0.0)

# A second table over the identical geometry, differing only in the normals it
# declares -- and differing in both components on both rows, so serving either
# run the other's normals is unmistakable.
_ALT_ROOF_NORMAL_UV = (0.0, math.sin(math.radians(40.0)), math.cos(math.radians(40.0)))
_ALT_WALL_NORMAL_UV = (math.sin(math.radians(55.0)), math.cos(math.radians(55.0)), 0.0)


def _override_table(roof_normal, wall_normal):
    """A two-row table on the building column of the city below: a tilted
    "roof" on the column's IUP face and an azimuthally rotated "wall" on one
    IEAST face.

    Only two of the five voxel-face directions the staircase mesh actually
    has are covered, so the remaining faces must come back unmapped -- that
    is what keeps the NaN/-1 half of the contract from being vacuous.
    """
    from voxcity.simulator_gpu.solar.surface_override import SurfaceOverride

    return SurfaceOverride(
        cell=np.array([[2, 2, 3], [2, 2, 2]], dtype=np.int32),
        face=np.array([0, 4], dtype=np.int8),          # IUP, IEAST
        origin=np.array([[2.5, 2.5, 4.0], [3.0, 2.5, 2.5]], dtype=np.float32),
        normal=np.array([roof_normal, wall_normal], dtype=np.float32),
        patch=np.array([0, 1], dtype=np.int32),
        area=np.array([1.0, 1.0], dtype=np.float32),
    )


def _override_city():
    """A 6x6x6 city with one building column at (row, col) = (2, 2), plus the
    default override table over that column."""
    from tests.simulator._roof_helpers import make_voxcity_with_building

    voxcity = make_voxcity_with_building(nx=6, ny=6, nz=6, bh=3)
    return voxcity, _override_table(_ROOF_NORMAL_UV, _WALL_NORMAL_UV)


@pytest.fixture
def small_city_with_override(_ti):
    from voxcity.simulator_gpu.solar.integration import caching

    caching.clear_building_radiation_model_cache()
    yield _override_city()
    caching.clear_building_radiation_model_cache()


@pytest.fixture
def small_city_no_override(_ti):
    from voxcity.simulator_gpu.solar.integration import caching

    caching.clear_building_radiation_model_cache()
    yield _override_city()[0]
    caching.clear_building_radiation_model_cache()


def _run(voxcity, **kwargs):
    from voxcity.simulator_gpu.solar.integration.building import (
        get_building_solar_irradiance,
    )

    return get_building_solar_irradiance(
        voxcity, azimuth_degrees_ori=210.0, elevation_degrees=20.0,
        direct_normal_irradiance=900.0, diffuse_irradiance=100.0,
        **kwargs)


def test_override_run_exports_used_normals(small_city_with_override):
    """When surface_override is active, the returned mesh carries the normal
    the solver actually used for each face, in scene coordinates."""
    voxcity, table = small_city_with_override
    mesh = _run(voxcity, surface_override=table)

    n = len(mesh.faces)
    normals = mesh.metadata["surface_override_normals"]
    index = mesh.metadata["surface_override_index"]

    assert normals.shape == (n, 3) and normals.dtype == np.float64
    assert index.shape == (n,) and index.dtype == np.int64

    finite = np.isfinite(normals).all(axis=1)
    assert finite.any(), "no mesh face mapped to an override surface at all"
    assert (~finite).any(), (
        "every face mapped, so the NaN/-1 assertions below are vacuous -- the "
        "table deliberately leaves three of the mesh's face directions uncovered")
    assert (index[finite] >= 0).all()
    assert (index[~finite] == -1).all()
    assert np.allclose(np.linalg.norm(normals[finite], axis=1), 1.0, atol=1e-5)

    # Scene frame, checked against the table's own normals through the
    # exported mapping. surfaces_from_override copies one surface per table
    # row in order, so the surface index is the table row index.
    expected = _scene(np.asarray(table.normal, dtype=np.float64))
    np.testing.assert_allclose(normals[finite], expected[index[finite]],
                               rtol=0, atol=1e-7)

    # ...and pinned explicitly on the tilted roof, whose u and v components
    # differ, so this fails if the u/v <-> x/y swap is ever dropped.
    roof_faces = np.flatnonzero(finite & (index == 0))
    wall_faces = np.flatnonzero(finite & (index == 1))
    assert roof_faces.size and wall_faces.size, (
        f"expected both table rows to be reached; got roof={roof_faces.size} "
        f"wall={wall_faces.size}")
    np.testing.assert_allclose(
        normals[roof_faces[0]],
        [_ROOF_NORMAL_UV[1], _ROOF_NORMAL_UV[0], _ROOF_NORMAL_UV[2]],
        rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        normals[wall_faces[0]],
        [_WALL_NORMAL_UV[1], _WALL_NORMAL_UV[0], _WALL_NORMAL_UV[2]],
        rtol=0, atol=1e-7)

    print(f"used-normal export: {n} faces, {int(finite.sum())} mapped "
          f"({roof_faces.size} to the tilted roof, {wall_faces.size} to the "
          f"rotated wall), {int((~finite).sum())} unmapped")


def test_override_export_survives_the_mesh_map_cache_hit(small_city_with_override):
    """The second call reuses the cached mesh->surface mapping and the cached
    export derived from it -- the branch that loads no surface arrays at all.
    The export must still be there, and identical: nothing about the run
    changed."""
    voxcity, table = small_city_with_override

    first = _run(voxcity, surface_override=table)
    first_normals = first.metadata["surface_override_normals"].copy()
    first_index = first.metadata["surface_override_index"].copy()

    second = _run(voxcity, surface_override=table)

    np.testing.assert_array_equal(second.metadata["surface_override_index"],
                                  first_index)
    np.testing.assert_allclose(second.metadata["surface_override_normals"],
                               first_normals, rtol=0, atol=0, equal_nan=True)


def test_a_new_override_table_is_not_served_the_old_normals(small_city_with_override):
    """A run under a second table reports that table's own normals, not the
    first's -- which the identical geometry would happily have reused. (This
    pins the cache-*replacement* mechanism only: a new override signature
    swaps in a fresh cache object with mesh_used_normals=None, so it says
    nothing about the read guard itself. That is the next test's job.)"""
    voxcity, table = small_city_with_override

    first = _run(voxcity, surface_override=table)
    first_normals = first.metadata["surface_override_normals"].copy()

    alt = _override_table(_ALT_ROOF_NORMAL_UV, _ALT_WALL_NORMAL_UV)
    second = _run(voxcity, surface_override=alt)
    normals = second.metadata["surface_override_normals"]
    index = second.metadata["surface_override_index"]

    finite = np.isfinite(normals).all(axis=1)
    assert finite.any()
    expected = _scene(np.asarray(alt.normal, dtype=np.float64))
    np.testing.assert_allclose(normals[finite], expected[index[finite]],
                               rtol=0, atol=1e-7)
    # ...and emphatically not the first table's, which the identical geometry
    # would happily have reused.
    assert not np.allclose(normals[finite], first_normals[finite], atol=1e-3)


def test_a_changed_mesh_is_not_served_the_old_normals(small_city_with_override):
    """`cache_matches_mesh` in the export's read guard. The override signature
    is not the only thing that can move: a warm refresh keeps the same cache
    object while the building mesh is rebuilt, so a geometry change that
    preserves the face count would otherwise be served the stale normals off
    the surviving cache. Drop that term from the guard and this test reads
    back the zeros planted below."""
    from voxcity.simulator_gpu.solar.integration import caching

    voxcity, table = small_city_with_override
    _run(voxcity, surface_override=table)

    cache = caching.get_building_radiation_model_cache()
    assert cache is not None and cache.mesh_used_normals is not None
    # Same face count, so the length check in the guard still passes; only
    # the signature says "different mesh".
    cache.mesh_used_normals = np.zeros_like(cache.mesh_used_normals)
    cache.mesh_geometry_signature = ("not", "this", "mesh")

    again = _run(voxcity, surface_override=table)
    normals = again.metadata["surface_override_normals"]

    assert np.isfinite(normals).any()
    assert not np.allclose(np.nan_to_num(normals), 0.0), (
        "served the stale all-zero normals off the surviving cache object")


def test_a_changed_mesh_is_not_served_the_old_face_mapping(small_city_with_override):
    """The sibling guard on `mesh_to_surface_idx`, which has the identical
    shape and the identical hole -- same warm-refresh scenario, and a stale
    mapping would silently re-route every value to the wrong surface."""
    from voxcity.simulator_gpu.solar.integration import caching

    voxcity, table = small_city_with_override
    first = _run(voxcity, surface_override=table)
    assert (first.metadata["surface_override_index"] >= 0).any()

    cache = caching.get_building_radiation_model_cache()
    assert cache is not None and cache.mesh_to_surface_idx is not None
    # "Nothing matched anything" -- same length, unmistakably not the truth.
    cache.mesh_to_surface_idx = np.full_like(cache.mesh_to_surface_idx, -1)
    cache.mesh_geometry_signature = ("not", "this", "mesh")

    again = _run(voxcity, surface_override=table)
    assert (again.metadata["surface_override_index"] >= 0).any(), (
        "served the stale all-unmatched mapping off the surviving cache object")


def test_the_exported_arrays_are_not_views_on_the_cache(small_city_with_override):
    """Both exported arrays are handed out as copies. On the cache-hit path
    the originals are the cache's own buffers, so a consumer writing into what
    it was handed would corrupt every later timestep -- and the two calls that
    make the copies (`.copy()`, `.astype()`) both have no-copy spellings that
    look like harmless tidy-ups."""
    voxcity, table = small_city_with_override

    first = _run(voxcity, surface_override=table)
    baseline_normals = first.metadata["surface_override_normals"].copy()
    baseline_index = first.metadata["surface_override_index"].copy()

    # A consumer scribbles on what it was handed.
    first.metadata["surface_override_normals"][:] = 0.0
    first.metadata["surface_override_index"][:] = -99

    second = _run(voxcity, surface_override=table)

    np.testing.assert_array_equal(second.metadata["surface_override_index"],
                                  baseline_index)
    np.testing.assert_allclose(second.metadata["surface_override_normals"],
                               baseline_normals, rtol=0, atol=0, equal_nan=True)


def test_no_override_no_keys(small_city_no_override):
    """Flag-off: the keys must be ABSENT, not present-and-empty, so that a
    consumer can use their presence as the signal that an override ran."""
    mesh = _run(small_city_no_override)

    assert "surface_override_normals" not in mesh.metadata
    assert "surface_override_index" not in mesh.metadata


# ---------------------------------------------------------------------------
# ...and the same export off the two accumulating entry points.
#
# get_cumulative_building_solar_irradiance and get_building_sunlight_hours
# both return `building_svf_mesh.copy()`, taken BEFORE their loop, and
# accumulate scalars into it from per-timestep meshes. The export above lands
# on those per-timestep meshes and so never reached the returned one -- a
# consumer calling either of these got values with no way to tell which normal
# produced them, which is the whole defect. The mapping is constant across
# timesteps (same mesh, same surfaces), so carrying one timestep's copy is
# exact rather than an approximation.
# ---------------------------------------------------------------------------

# Local noon-ish at the site below, so every timestep is sunlit and carries
# DNI -- both loop modes need at least one live iteration to capture from.
_SITE = dict(lon=139.0, lat=35.0, tz=9.0)
_SUMMER_DAY = ("06-21 10:00:00", "06-21 12:00:00")


def _weather_df():
    import pandas as pd

    return pd.DataFrame(
        {"DNI": [800.0, 850.0, 900.0], "DHI": [100.0, 110.0, 120.0]},
        index=pd.date_range("2020-06-21 10:00:00", periods=3, freq="h"))


def _fresh_building_mesh(voxcity):
    """A staircase mesh that has never been through
    get_building_solar_irradiance.

    That function writes its metadata onto the mesh object it is handed, and
    both accumulating functions return `building_svf_mesh.copy()` -- so a
    reused mesh arrives already carrying an export, and a test asserting the
    result has one would no longer be able to tell the pass-through from the
    copy. `_drop_override_export` now clears that on the way in, so this is
    belt-and-braces rather than the only thing standing between these tests
    and vacuity; `_mesh_carrying_a_stale_export` below is the deliberate
    inverse, for pinning the clear itself.
    """
    from voxcity.geoprocessor.mesh import create_voxel_mesh
    from voxcity.simulator_gpu.solar.integration.caching import (
        BUILDING_SURFACE_CLASSES,
    )

    mesh = create_voxel_mesh(
        voxcity.voxels.classes,
        BUILDING_SURFACE_CLASSES,
        voxcity.voxels.meta.meshsize,
        building_id_grid=voxcity.buildings.ids,
        mesh_type='open_air',
    )
    assert mesh is not None and len(mesh.faces) > 0
    assert "surface_override_normals" not in (mesh.metadata or {})
    # No pre-computed SVF either, so the sky-patch branch takes its
    # get_building_solar_irradiance diffuse call rather than the svf shortcut.
    assert "svf" not in (mesh.metadata or {})
    return mesh


def _assert_carries_the_export(result, voxcity, table):
    """The result carries the same export a single-timestep run produces."""
    n = len(result.faces)
    normals = result.metadata["surface_override_normals"]
    index = result.metadata["surface_override_index"]

    assert normals.shape == (n, 3)
    assert index.shape == (n,)
    finite = np.isfinite(normals).all(axis=1)
    assert finite.any(), "carried, but nothing in it is finite"
    assert (index[finite] >= 0).all()

    reference = _run(voxcity, building_svf_mesh=_fresh_building_mesh(voxcity),
                     surface_override=table)
    np.testing.assert_array_equal(index,
                                  reference.metadata["surface_override_index"])
    np.testing.assert_allclose(normals,
                               reference.metadata["surface_override_normals"],
                               rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("use_sky_patches", [False, True])
def test_cumulative_run_carries_the_export(small_city_with_override, use_sky_patches):
    """Parametrized over both of this function's loops: the per-timestep loop
    and the sky-patch loop, which are separate capture sites."""
    from voxcity.simulator_gpu.solar.integration.building import (
        get_cumulative_building_solar_irradiance,
    )

    voxcity, table = small_city_with_override
    result = get_cumulative_building_solar_irradiance(
        voxcity, _fresh_building_mesh(voxcity), _weather_df(),
        use_sky_patches=use_sky_patches, surface_override=table, **_SITE)

    _assert_carries_the_export(result, voxcity, table)


def test_cumulative_sky_patch_run_with_precomputed_svf_carries_the_export(
        small_city_with_override):
    """The sky-patch branch's patch loop, isolated. With SVF already on the
    mesh the diffuse base is arithmetic and the branch's *other*
    get_building_solar_irradiance call never happens, so only the patch loop
    can supply the export -- without this, the test above would keep passing
    on the diffuse call alone."""
    from voxcity.simulator_gpu.solar.integration.building import (
        get_cumulative_building_solar_irradiance,
    )

    voxcity, table = small_city_with_override
    mesh = _fresh_building_mesh(voxcity)
    mesh.metadata["svf"] = np.full(len(mesh.faces), 0.5, dtype=np.float64)

    result = get_cumulative_building_solar_irradiance(
        voxcity, mesh, _weather_df(),
        use_sky_patches=True, surface_override=table, **_SITE)

    _assert_carries_the_export(result, voxcity, table)


def test_a_later_timestep_that_lost_the_export_does_not_clear_it(
        small_city_with_override, monkeypatch):
    """`if override_export is None`: capture once, from the first result that
    carries it. Re-capturing unconditionally looks equivalent -- the export is
    invariant across timesteps -- right up until one result doesn't carry it,
    and then the whole run silently returns without the keys."""
    from voxcity.simulator_gpu.solar.integration import building as B

    voxcity, table = small_city_with_override
    real = B.get_building_solar_irradiance
    calls = {"n": 0}

    def only_the_first_carries_it(*args, **kwargs):
        mesh = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] > 1 and mesh is not None:
            for key in ("surface_override_normals", "surface_override_index"):
                mesh.metadata.pop(key, None)
        return mesh

    monkeypatch.setattr(B, "get_building_solar_irradiance",
                        only_the_first_carries_it)
    result = B.get_cumulative_building_solar_irradiance(
        voxcity, _fresh_building_mesh(voxcity), _weather_df(),
        use_sky_patches=False, surface_override=table, **_SITE)
    # The reference run below must see the real function again.
    monkeypatch.undo()

    assert calls["n"] > 1, "one timestep only -- the guard was never exercised"
    _assert_carries_the_export(result, voxcity, table)


def test_cumulative_overcast_sky_patch_run_carries_the_export(small_city_with_override):
    """The sky-patch branch's third capture site. With no direct beam anywhere
    in the period there are no active patches, so its patch loop never runs and
    the only get_building_solar_irradiance call left is the one that builds the
    diffuse base -- which has to carry the export or an overcast period returns
    values with no normals attached."""
    from voxcity.simulator_gpu.solar.integration.building import (
        get_cumulative_building_solar_irradiance,
    )

    voxcity, table = small_city_with_override
    overcast = _weather_df()
    overcast["DNI"] = 0.0

    result = get_cumulative_building_solar_irradiance(
        voxcity, _fresh_building_mesh(voxcity), overcast,
        use_sky_patches=True, surface_override=table, **_SITE)

    # Nothing direct got accumulated, i.e. the patch loop really was empty.
    assert np.nanmax(np.abs(result.metadata["cumulative_direct"])) == 0.0
    _assert_carries_the_export(result, voxcity, table)


@pytest.mark.parametrize("use_sky_patches", [False, True])
def test_sunlight_run_carries_the_export(small_city_with_override, use_sky_patches):
    """Same, for the sunlight-hours function's own two loops."""
    from voxcity.simulator_gpu.solar.integration.building import (
        get_building_sunlight_hours,
    )

    voxcity, table = small_city_with_override
    result = get_building_sunlight_hours(
        voxcity, building_svf_mesh=_fresh_building_mesh(voxcity), mode='DSH',
        period_start=_SUMMER_DAY[0], period_end=_SUMMER_DAY[1],
        use_sky_patches=use_sky_patches, surface_override=table, **_SITE)

    _assert_carries_the_export(result, voxcity, table)


def _mesh_carrying_a_stale_export(voxcity):
    """A mesh that has already been through an override run, i.e. carrying
    that run's export in its own metadata.

    Both accumulating functions start from `building_svf_mesh.copy()`, which
    copies metadata too -- so without an explicit clear a no-override run
    hands the caller a previous run's normals under the key that is supposed
    to mean "this run had an override". Poisoning with an obviously bogus
    value keeps the assertions below from being vacuous the way a fresh mesh
    would leave them.
    """
    mesh = _fresh_building_mesh(voxcity)
    n = len(mesh.faces)
    mesh.metadata["surface_override_normals"] = np.zeros((n, 3), dtype=np.float64)
    mesh.metadata["surface_override_index"] = np.zeros(n, dtype=np.int64)
    return mesh


def test_no_override_clears_an_inherited_export(small_city_no_override):
    """get_building_solar_irradiance seeds its metadata from the input mesh's
    own dict and writes the result back onto that same mesh object, so a
    caller reusing one mesh across an override run and a plain one would keep
    reading the override run's normals. Absence has to keep meaning "no
    override ran"."""
    voxcity = small_city_no_override
    mesh = _mesh_carrying_a_stale_export(voxcity)

    result = _run(voxcity, building_svf_mesh=mesh)

    assert "surface_override_normals" not in result.metadata
    assert "surface_override_index" not in result.metadata


@pytest.mark.parametrize("use_sky_patches", [False, True])
def test_cumulative_no_override_no_keys(small_city_no_override, use_sky_patches):
    """Flag-off through the accumulating path too: absence stays the signal,
    even when the input mesh arrives carrying an earlier run's export."""
    from voxcity.simulator_gpu.solar.integration.building import (
        get_cumulative_building_solar_irradiance,
    )

    voxcity = small_city_no_override
    result = get_cumulative_building_solar_irradiance(
        voxcity, _mesh_carrying_a_stale_export(voxcity), _weather_df(),
        use_sky_patches=use_sky_patches, **_SITE)

    assert "surface_override_normals" not in result.metadata
    assert "surface_override_index" not in result.metadata


@pytest.mark.parametrize("use_sky_patches", [False, True])
def test_sunlight_no_override_no_keys(small_city_no_override, use_sky_patches):
    """Same as the cumulative case, through both of this function's loops."""
    from voxcity.simulator_gpu.solar.integration.building import (
        get_building_sunlight_hours,
    )

    voxcity = small_city_no_override
    result = get_building_sunlight_hours(
        voxcity, building_svf_mesh=_mesh_carrying_a_stale_export(voxcity),
        mode='DSH', period_start=_SUMMER_DAY[0], period_end=_SUMMER_DAY[1],
        use_sky_patches=use_sky_patches, **_SITE)

    assert "surface_override_normals" not in result.metadata
    assert "surface_override_index" not in result.metadata


def test_sunlight_no_sunshine_early_return_no_keys(small_city_no_override):
    """A period with no sunshine at all takes an early return of its own,
    several hundred lines before the normal one. It is easy to reason that
    nothing was captured there -- no timestep ran -- and conclude the result
    is clean, but that result is `building_svf_mesh.copy()`, whose metadata
    was inherited rather than built. What the run captured says nothing about
    what the result already carried; only an entry-side clear covers every
    exit, including this one."""
    from voxcity.simulator_gpu.solar.integration.building import (
        get_building_sunlight_hours,
    )

    voxcity = small_city_no_override
    result = get_building_sunlight_hours(
        voxcity, building_svf_mesh=_mesh_carrying_a_stale_export(voxcity),
        mode='DSH', period_start="06-21 00:00:00", period_end="06-21 02:00:00",
        **_SITE)

    # The early return really is the path under test.
    assert result.metadata["potential_sunlight_hours"] == 0.0
    assert "surface_override_normals" not in result.metadata
    assert "surface_override_index" not in result.metadata
