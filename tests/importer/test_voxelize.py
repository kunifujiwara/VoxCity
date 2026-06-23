"""Tests for the column z-ray mesh voxelizer."""
import logging

import numpy as np
import trimesh

from voxcity.importer.voxelize import voxelize_mesh


def test_unit_cube_fills_expected_cells():
    """A 1x1x1 box occupying model-space [0,1]^3, under an identity transform,
    must voxelize to exactly the single cell (0, 0, 0)."""
    box = trimesh.creation.box(extents=(1, 1, 1))
    box.apply_translation([0.5, 0.5, 0.5])  # trimesh box is centered -> shift to [0,1]^3

    transform = np.eye(4)
    result = voxelize_mesh(box, transform, grid_shape=(10, 10, 10))

    expected = np.array([[0, 0, 0]], dtype=np.int64)
    assert result.dtype == np.int64
    assert result.shape == (1, 3)
    np.testing.assert_array_equal(result, expected)


def test_l_shape_leaves_notch_empty():
    """An L-shaped solid (2x2x1 footprint plus a 1x1x1 leg extension, with the
    opposite corner notch empty) must voxelize so the two legs are filled but
    the notch cell is not."""
    # Big box: i in [0,2), j in [0,2), k in [0,1)
    big = trimesh.creation.box(extents=(2, 2, 1))
    big.apply_translation([1.0, 1.0, 0.5])
    # Leg: i in [2,3), j in [0,1), k in [0,1) -- extends the +i edge, leaving
    # the (i in [2,3), j in [1,2)) corner as the notch.
    leg = trimesh.creation.box(extents=(1, 1, 1))
    leg.apply_translation([2.5, 0.5, 0.5])

    l_shape = trimesh.boolean.union([big, leg])
    assert l_shape.is_watertight

    transform = np.eye(4)
    result = voxelize_mesh(l_shape, transform, grid_shape=(10, 10, 10))
    occupied = {tuple(row) for row in result.tolist()}

    # Notch: must be empty.
    assert (2, 1, 0) not in occupied
    # Both legs: must be filled.
    assert (0, 0, 0) in occupied
    assert (1, 1, 0) in occupied
    assert (2, 0, 0) in occupied


def test_out_of_bounds_clipped(caplog, propagate_voxcity_logs):
    """A mesh partially outside grid_shape must be clipped to in-bounds
    indices only, with a warning logged describing the clipping.

    The column (i, j) candidates are intersected with grid bounds before
    ray-casting (per the voxelizer's column-selection step), so the
    realistic out-of-bounds case is the *vertical* (k) extent of a ray-hit
    span exceeding nz -- e.g. a tall box whose k-span runs from -2 to 3 in a
    grid with nz=2.
    """
    box = trimesh.creation.box(extents=(1, 1, 5))
    box.apply_translation([0.5, 0.5, 0.5])  # i, j in [0,1]; k in [-2, 3]

    transform = np.eye(4)
    grid_shape = (2, 2, 2)

    with caplog.at_level(logging.WARNING, logger="voxcity"):
        result = voxelize_mesh(box, transform, grid_shape=grid_shape)

    assert result.shape[0] > 0
    assert np.all(result[:, 0] >= 0) and np.all(result[:, 0] < grid_shape[0])
    assert np.all(result[:, 1] >= 0) and np.all(result[:, 1] < grid_shape[1])
    assert np.all(result[:, 2] >= 0) and np.all(result[:, 2] < grid_shape[2])

    assert "clip" in caplog.text.lower()


def test_open_box_fallback_fills_solid(caplog, propagate_voxcity_logs):
    """A mesh with an odd hit count per column (here: a box with an extra
    internal plate face splitting one face into 3 ray crossings) must trigger
    the odd-hit-count warning and fall back to filling the span from the
    first to the last hit."""
    # Box spans z in [0, 2]; an internal plate at z=1 (same footprint) makes
    # a straight-up ray cross 3 faces (bottom, plate, top) -- an odd count.
    box = trimesh.creation.box(extents=(1, 1, 2))
    box.apply_translation([0.5, 0.5, 1.0])

    plate_verts = np.array(
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float
    )
    plate_faces = np.array([[0, 1, 2], [0, 2, 3]])
    plate = trimesh.Trimesh(vertices=plate_verts, faces=plate_faces, process=False)

    mesh = trimesh.util.concatenate([box, plate])
    assert not mesh.is_watertight

    transform = np.eye(4)
    with caplog.at_level(logging.WARNING, logger="voxcity"):
        result = voxelize_mesh(mesh, transform, grid_shape=(10, 10, 10))

    occupied = {tuple(row) for row in result.tolist()}
    # Fallback span: floor(0+0.5)=0 to floor(2+0.5)=2 -> k in {0, 1}.
    assert (0, 0, 0) in occupied
    assert (0, 0, 1) in occupied

    assert "odd" in caplog.text.lower()
