"""Tests for target_selectors restriction in the building-surface sims."""

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from voxcity.geoprocessor.surface_meta import resolve_target_face_mask


@dataclass
class _Mesh:
    vertices: Any
    faces: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


def _mesh_with_meta(face_meta):
    n = len(face_meta)
    verts = np.zeros((n * 3, 3), dtype=np.float32)
    faces = np.zeros((n, 3), dtype=np.int32)
    for i in range(n):
        verts[i * 3 : i * 3 + 3] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        faces[i] = [i * 3, i * 3 + 1, i * 3 + 2]
    mesh = _Mesh(vertices=verts, faces=faces)
    mesh.metadata = {"surface_face_meta": face_meta, "surface_face_meta_version": 1}
    return mesh


def _meta_value(meta, key, default=None):
    if isinstance(meta, dict):
        return meta.get(key, default)
    return getattr(meta, key, default)


def test_resolve_target_face_mask_unions_selectors():
    mesh = _mesh_with_meta(
        [
            {"face_key": "f0", "building_id": 1, "surface_kind": "wall", "orientation": "S"},
            {"face_key": "f1", "building_id": 1, "surface_kind": "roof"},
            {"face_key": "f2", "building_id": 2, "surface_kind": "wall", "orientation": "N"},
        ]
    )
    selectors = [
        {"building_id": 1, "mode": "wall_orientation", "orientation": "S"},
        {"building_id": 2, "mode": "whole"},
    ]

    mask = resolve_target_face_mask(mesh, selectors)

    np.testing.assert_array_equal(mask, [True, False, True])


def test_resolve_target_face_mask_attaches_meta_if_missing():
    """If the mesh has no surface_face_meta yet, the helper classifies it first."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = _Mesh(vertices=verts, faces=faces)
    mesh.metadata = {"building_id": np.array([7], dtype=int)}

    mask = resolve_target_face_mask(mesh, [{"building_id": 7, "mode": "whole"}])

    assert mask.shape == (1,)
    attached_meta = _meta_value(mesh.metadata, "surface_face_meta")
    assert attached_meta is not None
    assert len(attached_meta) == 1
    assert int(_meta_value(attached_meta[0], "building_id")) == 7
    np.testing.assert_array_equal(mask, [True])
