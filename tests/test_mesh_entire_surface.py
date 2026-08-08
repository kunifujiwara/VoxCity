"""create_voxel_mesh with tree canopy (-2) included in the surface class set.

Solar/open_air mode treats a tree neighbour as "open" (canopy is a
see-through occluder), so a face is emitted toward it. When trees are
themselves being meshed as surfaces -- the entire-surface landmark mode, a
consuming application's mode; the library only sees the class set -- that
rule would emit internal tree-to-tree seams and building faces buried
against canopy. With -2 in the class set, faces must be emitted only toward
air (out-of-bounds still counts as a boundary).

Shared derivation for every case below. The grid is (u=2, v=1, z=1): two
adjacent voxels in a box exactly their own size. So each voxel has 6 sides,
5 of which face out of bounds -- and out-of-bounds always counts as exposed
under every rule. The only side in question is the shared u interface. Hence

    quads = 5 * (meshed voxels) + (interface sides counted as exposed)

and triangles = 2 * quads, since every quad is split into two.
"""
import numpy as np

from voxcity.geoprocessor.mesh import create_voxel_mesh


def _mesh(grid, class_ids):
    # building_id_grid present so solar mode's id/class tracking is on,
    # matching how get_surface_view_factor calls it.
    return create_voxel_mesh(
        grid,
        class_ids,
        meshsize=1.0,
        building_id_grid=np.zeros(grid.shape[:2], dtype=np.int32),
        mesh_type="open_air",
    )


def test_building_next_to_tree_with_trees_in_set_is_skin_only():
    # One building voxel touching one tree voxel, both in the set.
    # Both are meshed and the interface is internal on both sides:
    # 5 * 2 + 0 = 10 quads = 20 triangles.
    # Without the fix the building's side toward the canopy counts as exposed
    # (the tree's side toward the building never does -- -3 is neither air nor
    # canopy), so exactly one extra quad appears: 11 quads = 22 triangles.
    grid = np.zeros((2, 1, 1), dtype=np.int32)
    grid[0, 0, 0] = -3
    grid[1, 0, 0] = -2
    mesh = _mesh(grid, (-3, -2))
    assert len(mesh.faces) == 20


def test_tree_next_to_tree_with_trees_in_set_has_no_seams():
    # Two adjacent tree voxels; -3 kept in the class set because solar mode
    # only activates when the set intersects BUILDING_SURFACE_CLASSES, which
    # is how the entire-surface caller always invokes it.
    # Both are meshed and the interface is internal: 5 * 2 + 0 = 10 quads
    # = 20 triangles. Without the fix each tree sees canopy across the
    # interface and emits a seam face toward the other, so both interface
    # sides count: 12 quads = 24 triangles.
    grid = np.zeros((2, 1, 1), dtype=np.int32)
    grid[0, 0, 0] = -2
    grid[1, 0, 0] = -2
    mesh = _mesh(grid, (-3, -2))
    assert len(mesh.faces) == 20


def test_building_only_set_still_treats_tree_neighbor_as_open():
    # Regression guard for the existing building-surface modes: with -2 NOT in
    # the class set, the tree voxel is not meshed at all, so only the building
    # voxel contributes -- and its side toward the canopy must still count as
    # exposed: 5 * 1 + 1 = 6 quads = 12 triangles. The fix must not change
    # this, so this case passes both before and after it.
    grid = np.zeros((2, 1, 1), dtype=np.int32)
    grid[0, 0, 0] = -3
    grid[1, 0, 0] = -2
    mesh = _mesh(grid, (-3,))
    assert len(mesh.faces) == 12
