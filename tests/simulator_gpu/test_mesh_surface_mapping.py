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
