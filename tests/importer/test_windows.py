"""Tests for window glass-skin stamping."""
import logging

import numpy as np
import trimesh

from voxcity.importer.windows import stamp_windows
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
