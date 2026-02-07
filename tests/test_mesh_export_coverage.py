"""
Tests for voxcity.geoprocessor.mesh export functions and split_vertices_manual.
Targets: export_mesh_files, split_vertices_manual, save_obj_from_colored_mesh.
"""

import os
import numpy as np
import pytest
import trimesh

from voxcity.geoprocessor.mesh import (
    export_meshes,
    split_vertices_manual,
    save_obj_from_colored_mesh,
)


def _make_simple_mesh(color=None):
    """Create a simple triangle mesh."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if color is not None:
        face_colors = np.array([color, color], dtype=np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=face_colors)
    return mesh


class TestSplitVerticesManual:
    def test_splits_shared_vertices(self):
        mesh = _make_simple_mesh(color=[255, 0, 0, 255])
        split_mesh = split_vertices_manual(mesh)
        # Original has 4 vertices shared between 2 faces
        # After split, each face has its own 3 vertices = 6 total
        assert len(split_mesh.vertices) == 6
        assert len(split_mesh.faces) == 2

    def test_preserves_face_colors(self):
        colors = np.array([[255, 0, 0, 255], [0, 255, 0, 255]], dtype=np.uint8)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=colors)
        split_mesh = split_vertices_manual(mesh)
        assert len(split_mesh.faces) == 2

    def test_single_face(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh, face_colors=np.array([[128, 128, 128, 255]], dtype=np.uint8)
        )
        split_mesh = split_vertices_manual(mesh)
        assert len(split_mesh.vertices) == 3
        assert len(split_mesh.faces) == 1


class TestExportMeshFiles:
    def test_exports_obj_and_stl(self, tmp_path):
        mesh = _make_simple_mesh(color=[200, 200, 200, 255])
        meshes = {-3: mesh}
        export_meshes(meshes, str(tmp_path), "test_export")
        assert os.path.exists(tmp_path / "test_export.obj")
        assert os.path.exists(tmp_path / "test_export_-3.stl")

    def test_multiple_classes(self, tmp_path):
        mesh1 = _make_simple_mesh(color=[255, 0, 0, 255])
        mesh2 = _make_simple_mesh(color=[0, 255, 0, 255])
        meshes = {-3: mesh1, -2: mesh2}
        export_meshes(meshes, str(tmp_path), "multi")
        assert os.path.exists(tmp_path / "multi.obj")
        assert os.path.exists(tmp_path / "multi_-3.stl")
        assert os.path.exists(tmp_path / "multi_-2.stl")


class TestSaveObjFromColoredMesh:
    def test_basic_export(self, tmp_path):
        mesh = _make_simple_mesh(color=[100, 150, 200, 255])
        meshes = {"building": mesh}
        obj_path, mtl_path = save_obj_from_colored_mesh(meshes, str(tmp_path), "test")
        assert os.path.exists(obj_path)
        assert os.path.exists(mtl_path)
        # Read OBJ and verify content
        with open(obj_path) as f:
            content = f.read()
        assert "mtllib" in content
        assert "v " in content
        assert "f " in content

    def test_no_face_colors(self, tmp_path):
        """Mesh without face colors should use default grey."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        meshes = {"default": mesh}
        obj_path, mtl_path = save_obj_from_colored_mesh(meshes, str(tmp_path), "nocolor")
        assert os.path.exists(obj_path)
        # MTL should have at least one material with default grey
        with open(mtl_path) as f:
            mtl_content = f.read()
        assert "newmtl" in mtl_content

    def test_multiple_materials(self, tmp_path):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        colors = np.array([[255, 0, 0, 255], [0, 0, 255, 255]], dtype=np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=colors)
        meshes = {"colored": mesh}
        obj_path, mtl_path = save_obj_from_colored_mesh(meshes, str(tmp_path), "multi_mat")
        with open(mtl_path) as f:
            mtl_content = f.read()
        # Should have at least 2 materials
        assert mtl_content.count("newmtl") >= 2

    def test_float_face_colors(self, tmp_path):
        """Face colors as float [0..1] should be handled."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        colors = np.array([[0.5, 0.3, 0.8, 1.0]], dtype=np.float64)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=colors)
        meshes = {"floatcolor": mesh}
        obj_path, _ = save_obj_from_colored_mesh(meshes, str(tmp_path), "float_colors")
        assert os.path.exists(obj_path)

    def test_empty_mesh_skipped(self, tmp_path):
        """Empty mesh should not crash."""
        mesh_empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
        mesh_valid = _make_simple_mesh(color=[100, 100, 100, 255])
        meshes = {"empty": mesh_empty, "valid": mesh_valid}
        obj_path, mtl_path = save_obj_from_colored_mesh(meshes, str(tmp_path), "empty_test")
        assert os.path.exists(obj_path)
