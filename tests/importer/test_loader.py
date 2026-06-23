"""Tests for the OBJ group loader and role-based building selection."""
import logging

import trimesh

from voxcity.importer.loader import (
    classify_roles,
    load_obj_groups,
    select_building_groups,
)


def test_load_groups_returns_named_meshes(tmp_path):
    """A multi-geometry OBJ (exported from a trimesh Scene with named
    geometries) must load back as a list of (name, mesh) pairs preserving
    those names."""
    box_a = trimesh.creation.box(extents=(1, 1, 1))
    box_b = trimesh.creation.box(extents=(2, 2, 2))
    box_b.apply_translation([5.0, 0.0, 0.0])

    scene = trimesh.Scene()
    scene.add_geometry(box_a, node_name="group_a", geom_name="group_a")
    scene.add_geometry(box_b, node_name="group_b", geom_name="group_b")

    path = tmp_path / "multi.obj"
    scene.export(str(path))

    groups = load_obj_groups(path)
    names = {name for name, _mesh in groups}

    assert names == {"group_a", "group_b"}
    for _name, mesh in groups:
        assert hasattr(mesh, "vertices")
        assert hasattr(mesh, "faces")
        assert len(mesh.vertices) > 0


def test_load_single_box_obj_falls_back_to_single_group(box_obj_factory):
    """A simple single-box OBJ (no named scene structure) must still load,
    whether trimesh hands back a Scene or a bare Trimesh."""
    path = box_obj_factory()

    groups = load_obj_groups(path)

    assert len(groups) >= 1
    for name, mesh in groups:
        assert isinstance(name, str)
        assert hasattr(mesh, "vertices")
        assert hasattr(mesh, "faces")


def test_classify_roles_defaults_to_building():
    roles = classify_roles(["a", "b", "c"])
    assert roles == {"a": "building", "b": "building", "c": "building"}


def test_classify_roles_applies_mapping():
    roles = classify_roles(["a", "glass_01", "b"], roles={"glass_01": "window"})
    assert roles == {"a": "building", "glass_01": "window", "b": "building"}


def test_select_building_groups_skips_non_building(caplog, propagate_voxcity_logs):
    box = trimesh.creation.box(extents=(1, 1, 1))
    groups = [
        ("wall_1", box),
        ("glass_01", box),
        ("wall_2", box),
    ]

    with caplog.at_level(logging.INFO, logger="voxcity"):
        result = select_building_groups(groups, roles={"glass_01": "window"})

    names = [name for name, _mesh in result]
    assert names == ["wall_1", "wall_2"]
    assert "glass_01" in caplog.text
    assert "window" in caplog.text


def test_missing_file_raises(tmp_path):
    missing = tmp_path / "nonexistent.obj"
    try:
        load_obj_groups(missing)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
