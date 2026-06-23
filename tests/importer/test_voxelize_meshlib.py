import numpy as np
import pytest
import trimesh

meshlib = pytest.importorskip("meshlib")

from voxcity.importer.voxelize import voxelize_mesh, voxelize_mesh_meshlib


def _box(min_corner, size):
    m = trimesh.creation.box(extents=size)
    m.apply_translation(np.array(min_corner) + np.array(size) / 2.0)
    return m


def test_meshlib_matches_trimesh_on_cube():
    mesh = _box((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))
    a = set(map(tuple, voxelize_mesh(mesh, np.eye(4), (10, 10, 10))))
    b = set(map(tuple, voxelize_mesh_meshlib(mesh, np.eye(4), (10, 10, 10))))
    # interiors should agree; allow boundary differences of a thin shell
    assert (1, 1, 1) in b
    assert len(a ^ b) <= len(a) * 0.25
