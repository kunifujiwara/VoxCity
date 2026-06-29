"""Tests for the OBJ group loader and role-based building selection."""
import logging

import pytest
import trimesh

from voxcity.importer.loader import (
    DEFAULT_WINDOW_KEYWORDS,
    classify_roles,
    group_material_name,
    load_obj_groups,
    select_building_groups,
    select_groups_by_role,
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


def test_select_building_groups_skips_non_building():
    box = trimesh.creation.box(extents=(1, 1, 1))
    groups = [
        ("wall_1", box),
        ("glass_01", box),
        ("wall_2", box),
    ]

    result = select_building_groups(groups, roles={"glass_01": "window"})

    names = [name for name, _mesh in result]
    assert names == ["wall_1", "wall_2"]


def test_missing_file_raises(tmp_path):
    missing = tmp_path / "nonexistent.obj"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_obj_groups(missing)
    assert str(missing) in str(exc_info.value)


def test_directory_path_raises_file_not_found_error(tmp_path):
    """A directory path must not slip past the existence check and reach
    trimesh's loader (which would raise an opaque ValueError instead of
    the documented FileNotFoundError)."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_obj_groups(tmp_path)
    assert str(tmp_path) in str(exc_info.value)


def test_classify_roles_auto_detects_window_by_name():
    roles = classify_roles(["WallA", "Windows_South", "Glazing_2"])
    assert roles == {"WallA": "building", "Windows_South": "window", "Glazing_2": "window"}


def test_classify_roles_auto_detects_window_by_material():
    # generic group name, but the assigned material is glass
    roles = classify_roles(
        ["panelA", "panelB"],
        material_names={"panelA": "Glass_Clear", "panelB": "Concrete"},
    )
    assert roles == {"panelA": "window", "panelB": "building"}


def test_classify_roles_explicit_override_beats_auto():
    # force a glass-named group to stay building, and a plain group to window
    roles = classify_roles(
        ["Glass_Wall", "plain"],
        roles={"Glass_Wall": "building", "plain": "window"},
    )
    assert roles == {"Glass_Wall": "building", "plain": "window"}


def test_classify_roles_auto_window_can_be_disabled():
    roles = classify_roles(["Windows_1"], auto_window=False)
    assert roles == {"Windows_1": "building"}


def test_group_material_name_reads_texture_material(tmp_path):
    mtl = tmp_path / "m.mtl"
    mtl.write_text("newmtl GlassMat\nKd 0.2 0.4 0.9\n")
    obj = tmp_path / "m.obj"
    obj.write_text(
        "mtllib m.mtl\no Pane\nv 0 0 0\nv 1 0 0\nv 1 0 1\nv 0 0 1\n"
        "usemtl GlassMat\nf 1 2 3\nf 1 3 4\n"
    )
    groups = load_obj_groups(obj)
    assert any(group_material_name(mesh) == "GlassMat" for _name, mesh in groups)


def test_select_groups_by_role_buckets_building_and_window():
    box = trimesh.creation.box(extents=(1, 1, 1))
    groups = [("WallA", box), ("Windows_1", box), ("WallB", box)]
    buckets = select_groups_by_role(groups)
    assert [n for n, _ in buckets["building"]] == ["WallA", "WallB"]
    assert [n for n, _ in buckets["window"]] == ["Windows_1"]


def test_select_groups_by_role_drops_skip(caplog, propagate_voxcity_logs):
    box = trimesh.creation.box(extents=(1, 1, 1))
    groups = [("WallA", box), ("ctx", box)]
    with caplog.at_level(logging.INFO, logger="voxcity"):
        buckets = select_groups_by_role(groups, roles={"ctx": "skip"})
    assert [n for n, _ in buckets["building"]] == ["WallA"]
    assert buckets["window"] == []
    assert "ctx" in caplog.text


def test_default_window_keywords_are_english():
    assert DEFAULT_WINDOW_KEYWORDS == ("window", "glass", "glazing")


def _write_material_only_obj(tmp_path):
    """An OBJ with two materials (one named 'Glass') and NO o/g named groups,
    plus its .mtl so the material names survive the load. Mirrors a default
    Rhino export that separates geometry by material rather than by object."""
    mtl = tmp_path / "mm.mtl"
    mtl.write_text("newmtl Wall\nKd 0.6 0.6 0.6\nnewmtl Glass\nKd 0.2 0.4 0.9\n")
    obj = tmp_path / "mm.obj"
    obj.write_text(
        "mtllib mm.mtl\n"
        "usemtl Wall\n"
        "v 0 0 0\nv 1 0 0\nv 1 0 1\nv 0 0 1\n"
        "f 1 2 3 4\n"
        "usemtl Glass\n"
        "v 0 0.001 0.2\nv 1 0.001 0.2\nv 1 0.001 0.8\nv 0 0.001 0.8\n"
        "f 5 6 7 8\n"
    )
    return obj


def test_material_only_obj_splits_by_material(tmp_path):
    """A material-authored OBJ with no named groups must split into one group
    per material, named by material, so a 'Glass' material becomes its own
    group that window auto-detection can see."""
    obj = _write_material_only_obj(tmp_path)
    groups = load_obj_groups(obj)
    names = {name for name, _mesh in groups}
    assert names == {"Wall", "Glass"}


def test_material_only_obj_window_is_auto_detected(tmp_path):
    obj = _write_material_only_obj(tmp_path)
    buckets = select_groups_by_role(load_obj_groups(obj))
    assert [n for n, _ in buckets["window"]] == ["Glass"]
    assert [n for n, _ in buckets["building"]] == ["Wall"]


def test_single_material_obj_keeps_fallback_single_group(tmp_path):
    """A single-material OBJ with no named groups must stay one group with the
    fallback name -- material splitting only kicks in when it yields 2+ groups,
    so existing single-building imports are unchanged."""
    mtl = tmp_path / "one.mtl"
    mtl.write_text("newmtl Wall\nKd 0.6 0.6 0.6\n")
    obj = tmp_path / "one.obj"
    obj.write_text(
        "mtllib one.mtl\nusemtl Wall\n"
        "v 0 0 0\nv 1 0 0\nv 1 0 1\nv 0 0 1\nf 1 2 3 4\n"
    )
    groups = load_obj_groups(obj)
    assert len(groups) == 1
    assert groups[0][0] == "imported_building_1"
