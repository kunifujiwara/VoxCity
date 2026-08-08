"""create_voxel_mesh with tree canopy (-2) included in the surface class set.

Solar/open_air mode treats a tree neighbor as "open" (canopy is a
see-through occluder), so a face is emitted toward it. When trees are
themselves being meshed as surfaces -- the entire-surface landmark mode --
that rule would emit internal tree-to-tree seams and building faces buried
against canopy. With -2 in the class set, faces must be emitted only toward
air (out-of-bounds still counts as a boundary).
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
    # (u=2, v=1, z=1): one building voxel touching one tree voxel.
    # Exterior skin = 2 voxels * 6 sides - 2 shared = 10 quads = 20 triangles.
    # Without the fix the building->tree interface face is also emitted (22).
    grid = np.zeros((2, 1, 1), dtype=np.int32)
    grid[0, 0, 0] = -3
    grid[1, 0, 0] = -2
    mesh = _mesh(grid, (-3, -2))
    assert len(mesh.faces) == 20


def test_tree_next_to_tree_with_trees_in_set_has_no_seams():
    # Two adjacent tree voxels; -3 kept in the class set because solar mode
    # only activates when the set intersects BUILDING_SURFACE_CLASSES, which
    # is how the entire-surface caller always invokes it.
    # Without the fix each tree emits a face toward the other (24 triangles).
    grid = np.zeros((2, 1, 1), dtype=np.int32)
    grid[0, 0, 0] = -2
    grid[1, 0, 0] = -2
    mesh = _mesh(grid, (-3, -2))
    assert len(mesh.faces) == 20


def test_building_only_set_still_treats_tree_neighbor_as_open():
    # Regression guard for the existing building-surface modes: with -2 NOT
    # in the class set, the building face toward the tree must still be
    # emitted (5 air/oob sides + 1 tree side = 6 quads = 12 triangles), and
    # the tree voxel itself must not be meshed.
    grid = np.zeros((2, 1, 1), dtype=np.int32)
    grid[0, 0, 0] = -3
    grid[1, 0, 0] = -2
    mesh = _mesh(grid, (-3,))
    assert len(mesh.faces) == 12
