import numpy as np
import pytest

from voxcity.importer.transform import grid_geom_from_voxcity, build_placement_transform
from tests.importer.conftest import make_flat_voxcity


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
