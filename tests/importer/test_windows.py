"""Tests for window glass-skin stamping."""
import logging

import numpy as np
import pytest
import trimesh

from voxcity.importer.windows import stamp_windows, _surface_cells
from tests.importer.conftest import make_flat_voxcity

GLASS_CODE = -16
BUILDING_CODE = -3
IDENTITY = np.eye(4)


def _vertical_pane(x0, x1, y, z0, z1):
    """A planar quad in the plane y=const, spanning x in [x0,x1], z in [z0,z1]."""
    verts = np.array(
        [[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], dtype=float
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _wall_voxcity():
    """Flat model with a solid building wall slab at j=4, i in [2,8), z in [1,9)."""
    vc = make_flat_voxcity(nx=12, ny=12, nz=14, meshsize=1.0)
    vc.voxels.classes[2:8, 4, 1:9] = BUILDING_CODE
    return vc


def test_window_recolors_coincident_building_cells():
    vc = _wall_voxcity()
    n_building_before = int(np.sum(vc.voxels.classes == BUILDING_CODE))
    pane = _vertical_pane(3.0, 6.5, 4.5, 2.0, 7.0)  # within the wall slab
    n = stamp_windows(vc, [("Windows", pane)], IDENTITY)
    assert n > 0
    glass = vc.voxels.classes == GLASS_CODE
    assert glass.any()
    # all recolored cells are at the wall plane j=4
    assert np.all(np.where(glass)[1] == 4)
    # glass cells came out of the building set; no NEW occupancy was created
    assert int(np.sum(vc.voxels.classes == BUILDING_CODE)) == n_building_before - n


def test_window_recolor_matches_pane_footprint_no_inflation():
    """The glass skin must match the pane's lateral footprint, not a dilated
    halo. Regression: the previous isotropic dilation grew windows ~1 voxel per
    side in the facade plane, roughly doubling small windows."""
    vc = _wall_voxcity()
    pane = _vertical_pane(3.0, 6.5, 4.5, 2.0, 7.0)
    raw = _surface_cells(pane, IDENTITY, vc.voxels.classes.shape)
    n = stamp_windows(vc, [("Windows", pane)], IDENTITY)
    glass = np.argwhere(vc.voxels.classes == GLASS_CODE)

    assert n > 0
    # All glass stays at the wall plane (depth axis j=4); no lateral spread.
    assert np.all(glass[:, 1] == 4)
    # In-facade-plane extent (i, k) equals the raw pane footprint exactly --
    # no +1 halo on any side.
    assert glass[:, 0].min() == raw[:, 0].min()
    assert glass[:, 0].max() == raw[:, 0].max()
    assert glass[:, 2].min() == raw[:, 2].min()
    assert glass[:, 2].max() == raw[:, 2].max()


def _mullion_window(x0, x1, z0, z1, y):
    """A window modeled as a frame + central cross of thin bars, like a real OBJ
    mullioned window (the glass detail is sub-voxel)."""
    xm, zm = (x0 + x1) / 2.0, (z0 + z1) / 2.0
    bars = [_vertical_pane(x - 0.2, x + 0.2, y, z0, z1) for x in (x0, xm, x1)]
    bars += [_vertical_pane(x0, x1, y, z - 0.2, z + 0.2) for z in (z0, zm, z1)]
    return trimesh.util.concatenate(bars)


def test_mullioned_window_fills_opening_to_solid_pane():
    """A mullioned window (thin frame bars) must voxelize to a SOLID pane filling
    the opening, not the bars themselves (which render as sparse strips).
    Regression for imported windows showing as vertical strips."""
    vc = _wall_voxcity()  # wall slab at j=4, i in [2,8), z in [1,9)
    win = _mullion_window(3.0, 6.0, 2.0, 7.0, 4.5)
    n = stamp_windows(vc, [("Windows", win)], IDENTITY)
    glass = vc.voxels.classes == GLASS_CODE

    assert n > 0
    # All glass stays at the wall plane.
    assert np.all(np.where(glass)[1] == 4)
    # The window's bounding box on the wall is SOLIDLY filled -- no strip gaps.
    gi = np.argwhere(glass)
    ilo, ihi = gi[:, 0].min(), gi[:, 0].max()
    klo, khi = gi[:, 2].min(), gi[:, 2].max()
    assert glass[ilo:ihi + 1, 4, klo:khi + 1].all(), (
        "window opening not solidly filled (mullion gaps / strips remain)"
    )
    # glass is a single depth layer (no inward smearing)
    assert set(np.argwhere(glass)[:, 1].tolist()) == {4}


def test_window_far_from_building_is_skipped(caplog, propagate_voxcity_logs):
    vc = _wall_voxcity()
    pane = _vertical_pane(3.0, 6.0, 10.5, 2.0, 7.0)  # j=10, far from the wall at j=4
    with caplog.at_level(logging.INFO, logger="voxcity"):
        n = stamp_windows(vc, [("Windows", pane)], IDENTITY)
    assert n == 0
    assert not (vc.voxels.classes == GLASS_CODE).any()
    assert "no building cell" in caplog.text


def test_window_does_not_change_building_metadata():
    vc = _wall_voxcity()
    vc.buildings.ids[2:8, 4] = 7
    vc.buildings.heights[2:8, 4] = 8.0
    ids_before = vc.buildings.ids.copy()
    heights_before = vc.buildings.heights.copy()
    pane = _vertical_pane(3.0, 6.5, 4.5, 2.0, 7.0)
    stamp_windows(vc, [("Windows", pane)], IDENTITY)
    assert np.array_equal(vc.buildings.ids, ids_before)
    assert np.array_equal(vc.buildings.heights, heights_before)


def test_no_window_groups_returns_zero():
    vc = _wall_voxcity()
    assert stamp_windows(vc, [], IDENTITY) == 0


def _thick_wall_voxcity(jlo=4, jhi=6):
    """Flat model with a wall slab THICKER than 1 voxel: j in [jlo, jhi],
    i in [2,8), z in [1,9). Default 3 voxels thick (j=4,5,6)."""
    vc = make_flat_voxcity(nx=12, ny=12, nz=14, meshsize=1.0)
    vc.voxels.classes[2:8, jlo:jhi + 1, 1:9] = BUILDING_CODE
    return vc


def test_window_recolors_only_outward_skin_of_thick_wall():
    """A pane at the OUTER face of a 3-voxel-thick wall recolors exactly the
    outward layer (j=4), never the interior layers (j=5,6). Depth==1 alone is
    insufficient; this asserts the correct face is chosen."""
    vc = _thick_wall_voxcity(4, 6)            # wall j in {4,5,6}
    pane = _vertical_pane(3.0, 6.5, 4.0, 2.0, 7.0)  # at the outer face (small j)
    n = stamp_windows(vc, [("Windows", pane)], IDENTITY)
    glass = np.argwhere(vc.voxels.classes == GLASS_CODE)

    assert n > 0
    js = set(glass[:, 1].tolist())
    assert js == {4}, f"glass must sit only on the outward face j=4, got {sorted(js)}"


def test_window_skin_is_one_deep_and_full_footprint():
    """Across the footprint, glass is exactly one voxel deep per (i,z) column and
    covers the filled opening's (i,z) bounding box."""
    vc = _thick_wall_voxcity(4, 6)
    pane = _vertical_pane(3.0, 6.5, 4.0, 2.0, 7.0)
    raw = _surface_cells(pane, IDENTITY, vc.voxels.classes.shape)
    stamp_windows(vc, [("Windows", pane)], IDENTITY)
    glass = vc.voxels.classes == GLASS_CODE
    gi = np.argwhere(glass)

    from collections import Counter
    per_col = Counter((int(i), int(z)) for i, _j, z in gi)
    assert per_col and all(c == 1 for c in per_col.values()), "glass must be 1 voxel deep per column"
    assert gi[:, 0].min() == raw[:, 0].min() and gi[:, 0].max() == raw[:, 0].max()
    assert gi[:, 2].min() == raw[:, 2].min() and gi[:, 2].max() == raw[:, 2].max()


def test_centered_pane_in_even_wall_stays_one_layer():
    """Pane centered in an even-thickness wall (air vote ties on both sides):
    the mean tiebreaker still yields a single outer layer (either face OK)."""
    vc = _thick_wall_voxcity(4, 5)                 # 2-thick wall j in {4,5}
    pane = _vertical_pane(3.0, 6.5, 4.5, 2.0, 7.0)  # centered between j=4 and j=5
    n = stamp_windows(vc, [("Windows", pane)], IDENTITY)
    glass = np.argwhere(vc.voxels.classes == GLASS_CODE)
    assert n > 0
    js = set(glass[:, 1].tolist())
    assert len(js) == 1 and js.issubset({4, 5}), f"expected one outer layer, got {sorted(js)}"


def _rot_z(angle_deg, cx, cy):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.eye(4)
    R[0, 0], R[0, 1] = c, -s
    R[1, 0], R[1, 1] = s, c
    pre = np.eye(4); pre[0, 3] = -cx; pre[1, 3] = -cy
    post = np.eye(4); post[0, 3] = cx; post[1, 3] = cy
    return post @ R @ pre


@pytest.mark.parametrize("angle", [15, 30, 45])
def test_rotated_facade_window_gap_free(angle):
    from scipy import ndimage
    vc = make_flat_voxcity(nx=24, ny=24, nz=14, meshsize=1.0)
    T = _rot_z(angle, 12.0, 12.0)
    # Bake a thick-ish wall from a rotated quad's surface cells, dilated once.
    wall = _vertical_pane(6.0, 16.0, 12.0, 1.0, 9.0)
    wcells = _surface_cells(wall, T, vc.voxels.classes.shape)
    m = np.zeros(vc.voxels.classes.shape, dtype=bool)
    m[wcells[:, 0], wcells[:, 1], wcells[:, 2]] = True
    m = ndimage.binary_dilation(m, ndimage.generate_binary_structure(3, 1))
    vc.voxels.classes[m] = BUILDING_CODE
    win = _mullion_window(8.0, 14.0, 2.0, 8.0, 12.0)

    n = stamp_windows(vc, [("Windows", win)], T)
    assert n > 0
    glass = vc.voxels.classes == GLASS_CODE
    gi = np.argwhere(glass)
    # Per z-row, glass along the facade run axis must be contiguous (no strips/gaps).
    for z in sorted(set(gi[:, 2].tolist())):
        sel = gi[gi[:, 2] == z]
        spread_i = sel[:, 0].max() - sel[:, 0].min()
        spread_j = sel[:, 1].max() - sel[:, 1].min()
        run_axis = 0 if spread_i >= spread_j else 1
        vals = np.sort(sel[:, run_axis])
        assert np.all(np.diff(vals) <= 1), f"strips/gaps at z={z} angle={angle}"


def _x_pane(y0, y1, z0, z1, x):
    """A planar quad in the plane x=const, spanning y in [y0,y1], z in [z0,z1]
    (a window on an x-facing facade)."""
    verts = np.array(
        [[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]], dtype=float
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def test_window_not_dropped_by_a_second_building_along_the_normal():
    """A window on building A's facade must recolor A's own wall, even when
    building B lies further along the window's normal line. Regression: selecting
    the GLOBALLY outermost building voxel per column picked B's far face, which
    the proximity gate then dropped -> the window vanished."""
    vc = make_flat_voxcity(nx=24, ny=24, nz=14, meshsize=1.0)
    vc.voxels.classes[4:8, 4:10, 1:9] = BUILDING_CODE     # box A: i in [4,8)
    vc.voxels.classes[14:18, 4:10, 1:9] = BUILDING_CODE   # box B further along +x
    win = _x_pane(4.5, 9.5, 2.0, 7.0, 8.0)                # pane on A's +x face (i=8)

    n = stamp_windows(vc, [("Windows", win)], IDENTITY)
    glass = np.argwhere(vc.voxels.classes == GLASS_CODE)

    assert n > 0, "window must not be dropped because of a second building"
    ivals = set(glass[:, 0].tolist())
    # Glass sits on A's own outer face (i=7), never on B (i in {14..17}).
    assert ivals == {7}, f"glass must be on building A's wall (i=7), got {sorted(ivals)}"

