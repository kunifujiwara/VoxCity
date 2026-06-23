import math

import numpy as np
import pytest

from voxcity.importer.transform import (
    grid_geom_from_voxcity,
    build_placement_transform,
    _domain_rotation_deg,
)
from tests.importer.conftest import make_flat_voxcity
from voxcity.models import VoxCity, VoxelGrid, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid, GridMetadata

GROUND_CODE = -1


def _apply(M, pts):
    pts = np.asarray(pts, dtype=float)
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    return (homog @ M.T)[:, :3]


def test_grid_geom_has_expected_keys():
    vc = make_flat_voxcity(nx=20, ny=20, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    for key in ("origin", "u_vec", "v_vec", "adj_mesh", "grid_size", "meshsize_m"):
        assert key in geom


def test_anchor_maps_to_origin_cell_axis_aligned():
    """rotation=0, units=m, anchor at grid origin corner, model origin anchored.

    Model point (0,0,0) -> voxel index ~ (0, 0, 1): the +1 is the ground offset.
    """
    vc = make_flat_voxcity(nx=20, ny=20, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    anchor_lonlat = geom["origin"]  # grid SW corner
    M = build_placement_transform(
        vc,
        anchor_lonlat=(float(anchor_lonlat[0]), float(anchor_lonlat[1])),
        anchor_elevation=0.0,
        anchor_model_point=(0.0, 0.0, 0.0),
        rotation=0.0,
        move=(0.0, 0.0, 0.0),
        units="m",
    )
    out = _apply(M, [[0.0, 0.0, 0.0]])[0]
    assert out[0] == pytest.approx(0.0, abs=1e-6)   # i (u/north)
    assert out[1] == pytest.approx(0.0, abs=1e-6)   # j (v/east)
    assert out[2] == pytest.approx(1.0, abs=1e-6)   # k (ground offset)


def test_axis_mapping_x_to_v_y_to_u():
    """+X (east) advances j; +Y (north) advances i; meshsize scales."""
    vc = make_flat_voxcity(nx=20, ny=20, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    anchor_lonlat = geom["origin"]
    M = build_placement_transform(
        vc, anchor_lonlat=(float(anchor_lonlat[0]), float(anchor_lonlat[1])),
        anchor_elevation=0.0, anchor_model_point=(0.0, 0.0, 0.0),
        rotation=0.0, move=(0.0, 0.0, 0.0), units="m",
    )
    out = _apply(M, [[3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 4.0]])
    # +X=3 -> j=3
    assert out[0][1] == pytest.approx(3.0, abs=1e-4)
    assert out[0][0] == pytest.approx(0.0, abs=1e-4)
    # +Y=5 -> i=5
    assert out[1][0] == pytest.approx(5.0, abs=1e-4)
    assert out[1][1] == pytest.approx(0.0, abs=1e-4)
    # +Z=4 -> k=4+1
    assert out[2][2] == pytest.approx(5.0, abs=1e-4)


def test_units_feet_scale():
    vc = make_flat_voxcity(nx=50, ny=50, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    anchor_lonlat = geom["origin"]
    M = build_placement_transform(
        vc, anchor_lonlat=(float(anchor_lonlat[0]), float(anchor_lonlat[1])),
        anchor_elevation=0.0, anchor_model_point=(0.0, 0.0, 0.0),
        rotation=0.0, move=(0.0, 0.0, 0.0), units="ft",
    )
    # 10 ft along +X = 3.048 m -> j ~ 3.048 voxels (meshsize 1)
    out = _apply(M, [[10.0, 0.0, 0.0]])[0]
    assert out[1] == pytest.approx(3.048, abs=1e-3)


def test_rotation_90_maps_x_to_north():
    """rotation=90 deg. Convention: positive rotation rotates the model so that
    its +X axis ends up pointing +u (north). Verify +X(east) -> +i.
    """
    vc = make_flat_voxcity(nx=20, ny=20, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    anchor_lonlat = geom["origin"]
    M = build_placement_transform(
        vc, anchor_lonlat=(float(anchor_lonlat[0]), float(anchor_lonlat[1])),
        anchor_elevation=0.0, anchor_model_point=(0.0, 0.0, 0.0),
        rotation=90.0, move=(0.0, 0.0, 0.0), units="m",
    )
    out = _apply(M, [[4.0, 0.0, 0.0]])[0]
    assert out[0] == pytest.approx(4.0, abs=1e-4)   # now along i (north)
    assert out[1] == pytest.approx(0.0, abs=1e-4)


def test_move_offsets_in_voxels():
    vc = make_flat_voxcity(nx=20, ny=20, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    anchor_lonlat = geom["origin"]
    M = build_placement_transform(
        vc, anchor_lonlat=(float(anchor_lonlat[0]), float(anchor_lonlat[1])),
        anchor_elevation=0.0, anchor_model_point=(0.0, 0.0, 0.0),
        rotation=0.0, move=(2.0, 3.0, 4.0), units="m",  # (east_m, north_m, up_m)
    )
    out = _apply(M, [[0.0, 0.0, 0.0]])[0]
    assert out[1] == pytest.approx(2.0, abs=1e-4)   # east move -> j
    assert out[0] == pytest.approx(3.0, abs=1e-4)   # north move -> i
    assert out[2] == pytest.approx(5.0, abs=1e-4)   # up move 4 + ground offset 1


def _make_rotated_voxcity(bearing_deg, nx=20, ny=20, nz=10, meshsize=1.0):
    """Build a minimal flat-DEM VoxCity whose domain u-axis (side_1) has the
    given compass bearing (degrees, clockwise from true north) instead of
    being axis-aligned.

    Mirrors make_flat_voxcity's construction but rotates the rectangle's
    side_1/side_2 directions by `bearing_deg` in the local (east, north)
    plane, using the same flat-earth (lon/lat-per-metre) approximation near
    the equator that make_flat_voxcity relies on.
    """
    lon0, lat0 = 0.0, 0.0
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0  # ~valid near the equator, matches conftest's approximation

    side_1_len = meshsize * nx
    side_2_len = meshsize * ny
    phi = math.radians(bearing_deg)

    # (east, north) unit directions of the domain's u (side_1) and v (side_2)
    # axes, consistent with _domain_rotation_deg's convention: a u-axis with
    # bearing `phi` has (east, north) components (sin(phi), cos(phi)); v is
    # perpendicular, (cos(phi), -sin(phi)).
    u_e, u_n = math.sin(phi), math.cos(phi)
    v_e, v_n = math.cos(phi), -math.sin(phi)

    def en_to_lonlat(e_m, n_m):
        return (lon0 + e_m / m_per_deg_lon, lat0 + n_m / m_per_deg_lat)

    v0 = (lon0, lat0)
    v1 = en_to_lonlat(u_e * side_1_len, u_n * side_1_len)              # v0 + side_1
    v3 = en_to_lonlat(v_e * side_2_len, v_n * side_2_len)              # v0 + side_2
    v2 = en_to_lonlat(u_e * side_1_len + v_e * side_2_len,
                       u_n * side_1_len + v_n * side_2_len)            # v0 + side_1 + side_2
    rectangle_vertices = [v0, v1, v2, v3]

    lons = [p[0] for p in rectangle_vertices]
    lats = [p[1] for p in rectangle_vertices]
    meta = GridMetadata(
        crs="EPSG:4326",
        bounds=(min(lons), min(lats), max(lons), max(lats)),
        meshsize=meshsize,
    )

    classes = np.zeros((nx, ny, nz), dtype=np.int8)
    classes[:, :, 0] = GROUND_CODE

    heights = np.zeros((nx, ny), dtype=float)
    ids = np.zeros((nx, ny), dtype=np.int32)
    min_heights = np.empty((nx, ny), dtype=object)
    for i in range(nx):
        for j in range(ny):
            min_heights[i, j] = []
    dem = np.zeros((nx, ny), dtype=float)
    lc = np.zeros((nx, ny), dtype=np.int32)
    canopy = np.zeros((nx, ny), dtype=float)

    return VoxCity(
        voxels=VoxelGrid(classes=classes, meta=meta),
        buildings=BuildingGrid(heights=heights, min_heights=min_heights, ids=ids, meta=meta),
        land_cover=LandCoverGrid(classes=lc, meta=meta),
        dem=DemGrid(elevation=dem, meta=meta),
        tree_canopy=CanopyGrid(top=canopy, bottom=None, meta=meta),
        extras={"rectangle_vertices": rectangle_vertices},
    )


def test_rotation_with_rotated_domain():
    """domain_rotation correction must actually be applied (not a no-op).

    Build a domain whose u-axis is rotated 30 deg clockwise from true north
    (NOT axis-aligned). Set model `rotation` to the *opposite* of that
    bearing (-30 deg): per the rotation step's documented basis-vector
    identity (u = e*sin(phi) + n*cos(phi), v = e*cos(phi) - n*sin(phi)),
    rotating the model's +Y by theta=-phi before that projection exactly
    cancels phi, so the point lands back on the domain's own u-axis: i > 0,
    j ~ 0. If the domain_rotation (phi) correction were silently skipped (a
    no-op), the point would instead pick up a large spurious j component
    (model +Y rotated by -30 deg alone gives u=4.33, v=2.5 -- not (5, 0)).
    """
    vc = _make_rotated_voxcity(bearing_deg=30.0, nx=20, ny=20, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    assert _domain_rotation_deg(geom) == pytest.approx(30.0, abs=1e-6)

    anchor_lonlat = geom["origin"]
    M = build_placement_transform(
        vc, anchor_lonlat=(float(anchor_lonlat[0]), float(anchor_lonlat[1])),
        anchor_elevation=0.0, anchor_model_point=(0.0, 0.0, 0.0),
        rotation=-30.0, move=(0.0, 0.0, 0.0), units="m",
    )
    out = _apply(M, [[0.0, 5.0, 0.0]])[0]   # model +Y, 5 m
    assert out[0] == pytest.approx(5.0, abs=1e-4)   # i: lands on the domain's own u-axis
    assert out[1] == pytest.approx(0.0, abs=1e-4)   # j: ~0, not a spurious east component


def test_missing_rectangle_vertices_raises():
    """grid_geom_from_voxcity must raise a clear ValueError when extras has
    no usable rectangle_vertices, since that error is the only signal a
    caller gets that georeferencing is impossible.
    """
    vc = make_flat_voxcity(nx=20, ny=20, meshsize=1.0)
    vc.extras = {}
    with pytest.raises(ValueError, match="rectangle_vertices"):
        grid_geom_from_voxcity(vc)
