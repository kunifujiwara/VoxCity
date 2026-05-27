"""Tests for target_selectors restriction in the building-surface sims."""

from dataclasses import dataclass, field
from typing import Any, Dict
from types import SimpleNamespace

import numpy as np
import pytest

from voxcity.geoprocessor.surface_meta import resolve_target_face_mask
from voxcity.simulator_gpu.visibility import integration as visibility_integration


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


def _tiny_voxcity_two_buildings():
    classes = np.zeros((7, 4, 3), dtype=np.int32)
    classes[1:3, 1:3, 0:2] = -3
    classes[4:6, 1:3, 0:2] = -3

    building_ids = np.zeros(classes.shape[:2], dtype=np.int32)
    building_ids[1:3, 1:3] = 1
    building_ids[4:6, 1:3] = 2

    return SimpleNamespace(
        voxels=SimpleNamespace(
            classes=classes,
            meta=SimpleNamespace(meshsize=1.0),
        ),
        buildings=SimpleNamespace(ids=building_ids),
    )


def _building_2_whole_selector():
    return [{"building_id": 2, "mode": "whole"}]


@pytest.mark.gpu
def test_view_restriction_matches_full_on_target_faces():
    voxcity = _tiny_voxcity_two_buildings()
    kwargs = {"N_azimuth": 12, "N_elevation": 4, "ray_sampling": "grid"}

    full_mesh = visibility_integration.get_surface_view_factor(voxcity, mode="sky", **kwargs)
    restricted_mesh = visibility_integration.get_surface_view_factor(
        voxcity,
        mode="sky",
        target_selectors=_building_2_whole_selector(),
        **kwargs,
    )

    full_values = full_mesh.metadata["view_factor_values"]
    restricted_values = restricted_mesh.metadata["view_factor_values"]
    target_mask = resolve_target_face_mask(restricted_mesh, _building_2_whole_selector())

    assert target_mask.any()
    assert restricted_values.shape == full_values.shape
    np.testing.assert_allclose(restricted_values[target_mask], full_values[target_mask])
    assert np.all(np.isnan(restricted_values[~target_mask]))


@pytest.mark.gpu
def test_view_target_selectors_none_is_unchanged():
    voxcity = _tiny_voxcity_two_buildings()
    kwargs = {"N_azimuth": 12, "N_elevation": 4, "ray_sampling": "grid"}

    default_mesh = visibility_integration.get_surface_view_factor(
        voxcity,
        mode="sky",
        **kwargs,
    )
    none_mesh = visibility_integration.get_surface_view_factor(
        voxcity,
        mode="sky",
        target_selectors=None,
        **kwargs,
    )

    default_values = default_mesh.metadata["view_factor_values"]
    none_values = none_mesh.metadata["view_factor_values"]
    target_mask = resolve_target_face_mask(none_mesh, _building_2_whole_selector())

    assert none_values.shape == default_values.shape
    np.testing.assert_allclose(none_values, default_values, equal_nan=True)
    assert target_mask.any()
    assert np.all(np.isfinite(none_values[target_mask]))


def test_view_empty_target_returns_all_nan(monkeypatch):
    voxcity = _tiny_voxcity_two_buildings()

    class _FakeSurfaceViewFactorCalculator:
        def __init__(self, *args, **kwargs):
            pass

        def compute_surface_view_factor(self, *, face_centers, **kwargs):
            return np.full(len(face_centers), 0.5, dtype=np.float32)

    monkeypatch.setattr(visibility_integration, "_get_or_create_domain", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        visibility_integration,
        "SurfaceViewFactorCalculator",
        _FakeSurfaceViewFactorCalculator,
    )

    mesh = visibility_integration.get_surface_view_factor(
        voxcity,
        mode="sky",
        target_selectors=[{"building_id": 999, "mode": "whole"}],
    )

    values = mesh.metadata["view_factor_values"]
    assert values.shape == (len(mesh.faces),)
    assert np.all(np.isnan(values))
