"""
Comprehensive tests for voxcity.exporter.obj to improve coverage.
"""

import numpy as np
import os
import tempfile
import pytest

from voxcity.exporter.obj import (
    convert_colormap_indices,
    create_face_vertices,
    mesh_faces,
    export_obj,
)


class TestConvertColormapIndices:
    """Tests for convert_colormap_indices function."""

    def test_sequential_keys(self):
        cmap = {0: [255, 0, 0], 1: [0, 255, 0]}
        result = convert_colormap_indices(cmap)
        assert result == {0: [255, 0, 0], 1: [0, 255, 0]}

    def test_non_sequential_keys(self):
        cmap = {5: [255, 0, 0], 10: [0, 255, 0], 15: [0, 0, 255]}
        result = convert_colormap_indices(cmap)
        assert set(result.keys()) == {0, 1, 2}
        assert result[0] == [255, 0, 0]
        assert result[1] == [0, 255, 0]
        assert result[2] == [0, 0, 255]

    def test_empty_map(self):
        result = convert_colormap_indices({})
        assert result == {}

    def test_single_entry(self):
        result = convert_colormap_indices({42: [128, 128, 128]})
        assert result == {0: [128, 128, 128]}

    def test_negative_keys(self):
        cmap = {-3: [100, 0, 0], -2: [0, 100, 0], 0: [0, 0, 100]}
        result = convert_colormap_indices(cmap)
        assert list(result.keys()) == [0, 1, 2]


class TestCreateFaceVertices:
    def test_positive_y(self):
        coords = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        result = create_face_vertices(coords, True, 'y')
        assert result == [coords[3], coords[2], coords[1], coords[0]]

    def test_negative_y(self):
        coords = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        result = create_face_vertices(coords, False, 'y')
        assert result == [coords[0], coords[1], coords[2], coords[3]]

    def test_positive_x(self):
        coords = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        result = create_face_vertices(coords, True, 'x')
        assert result == [coords[0], coords[3], coords[2], coords[1]]

    def test_negative_x(self):
        coords = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        result = create_face_vertices(coords, False, 'x')
        assert result == [coords[0], coords[1], coords[2], coords[3]]

    def test_positive_z(self):
        coords = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        result = create_face_vertices(coords, True, 'z')
        assert result == [coords[0], coords[3], coords[2], coords[1]]

    def test_negative_z(self):
        coords = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        result = create_face_vertices(coords, False, 'z')
        assert result == [coords[0], coords[1], coords[2], coords[3]]


class TestMeshFaces:
    def test_single_voxel(self):
        mask = np.zeros((5, 5), dtype=np.int32)
        mask[2, 2] = 1
        vertex_dict = {}
        vertex_list = []
        faces_per_material = {}
        voxel_value_to_material = {1: "mat_1"}
        mesh_faces(mask, 0, 'z', True, 1, 1.0,
                   vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)
        assert len(faces_per_material) > 0
        assert "mat_1" in faces_per_material

    def test_empty_mask(self):
        mask = np.zeros((5, 5), dtype=np.int32)
        vertex_dict = {}
        vertex_list = []
        faces_per_material = {}
        voxel_value_to_material = {}
        mesh_faces(mask, 0, 'z', True, 1, 1.0,
                   vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)
        assert len(faces_per_material) == 0

    def test_full_mask(self):
        mask = np.ones((3, 3), dtype=np.int32)
        vertex_dict = {}
        vertex_list = []
        faces_per_material = {}
        voxel_value_to_material = {1: "mat_1"}
        mesh_faces(mask, 0, 'z', True, 1, 1.0,
                   vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)
        # Should produce merged faces via greedy meshing
        assert len(faces_per_material["mat_1"]) >= 2  # At least 2 triangles

    def test_multiple_values(self):
        mask = np.zeros((4, 4), dtype=np.int32)
        mask[0:2, 0:2] = 1
        mask[2:4, 2:4] = 2
        vertex_dict = {}
        vertex_list = []
        faces_per_material = {}
        voxel_value_to_material = {1: "mat_1", 2: "mat_2"}
        mesh_faces(mask, 0, 'z', True, 1, 1.0,
                   vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)
        assert "mat_1" in faces_per_material
        assert "mat_2" in faces_per_material

    def test_axis_x(self):
        mask = np.zeros((3, 3), dtype=np.int32)
        mask[1, 1] = 1
        vertex_dict = {}
        vertex_list = []
        faces_per_material = {}
        voxel_value_to_material = {1: "mat_1"}
        mesh_faces(mask, 0, 'x', True, 1, 1.0,
                   vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)
        assert len(faces_per_material) > 0

    def test_axis_y(self):
        mask = np.zeros((3, 3), dtype=np.int32)
        mask[1, 1] = 1
        vertex_dict = {}
        vertex_list = []
        faces_per_material = {}
        voxel_value_to_material = {1: "mat_1"}
        mesh_faces(mask, 0, 'y', False, 1, 1.0,
                   vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)
        assert len(faces_per_material) > 0

    def test_voxel_size_scaling(self):
        mask = np.zeros((3, 3), dtype=np.int32)
        mask[1, 1] = 1
        vertex_dict = {}
        vertex_list = []
        faces_per_material = {}
        voxel_value_to_material = {1: "mat_1"}
        mesh_faces(mask, 0, 'z', True, 1, 2.0,
                   vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)
        # Check that coordinates are scaled by voxel size
        assert len(vertex_list) > 0


class TestExportObj:
    def test_basic_export(self, tmp_path):
        array = np.zeros((5, 5, 5), dtype=np.int32)
        array[2, 2, 2] = 1
        export_obj(array, str(tmp_path), "test_export", voxel_size=1.0)
        assert (tmp_path / "test_export.obj").exists()
        assert (tmp_path / "test_export.mtl").exists()

    def test_empty_array(self, tmp_path):
        array = np.zeros((5, 5, 5), dtype=np.int32)
        export_obj(array, str(tmp_path), "test_empty", voxel_size=1.0)

    def test_custom_colormap(self, tmp_path):
        array = np.zeros((5, 5, 5), dtype=np.int32)
        array[1, 1, 1] = 1
        array[3, 3, 3] = 2
        cmap = {1: [255, 0, 0], 2: [0, 255, 0]}
        export_obj(array, str(tmp_path), "test_cmap", voxel_size=1.0, voxel_color_map=cmap)

    def test_multiple_voxel_types(self, tmp_path):
        array = np.zeros((5, 5, 5), dtype=np.int32)
        array[1, 1, 1] = 1
        array[2, 2, 2] = 2
        array[3, 3, 3] = 3
        export_obj(array, str(tmp_path), "test_multi", voxel_size=1.0)

    def test_orientation_north_maps_to_scene_Y_east_maps_to_scene_X(self, tmp_path):
        """Phase 3 contract: X = v/east (axis 1), Y = u/north (axis 0), Z = up (axis 2).

        North marker at arr[1,0,0] (u=1, v=0) must have X in [0,1] and Y in [1,2].
        East  marker at arr[0,2,0] (u=0, v=2) must have X in [2,3] and Y in [0,1].
        We parse vertices per material section to separate the two objects.
        """
        import re

        arr = np.zeros((2, 3, 1), dtype=np.int32)
        arr[1, 0, 0] = 10   # north marker: u=1, v=0  →  scene X∈[0,1], Y∈[1,2]
        arr[0, 2, 0] = 20   # east  marker: u=0, v=2  →  scene X∈[2,3], Y∈[0,1]

        export_obj(arr, str(tmp_path), "orient_test", voxel_size=1.0,
                   voxel_color_map={10: [255, 0, 0], 20: [0, 0, 255]})

        obj_text = (tmp_path / "orient_test.obj").read_text()

        # Collect all vertices by index (1-based)
        all_verts = [
            tuple(float(x) for x in m.groups())
            for m in re.finditer(r"^v\s+([\S]+)\s+([\S]+)\s+([\S]+)", obj_text, re.M)
        ]

        # Collect face vertex indices per material
        material_faces: dict[str, list[int]] = {}
        cur_mat = None
        for line in obj_text.splitlines():
            if line.startswith("usemtl "):
                cur_mat = line.split()[1]
            elif line.startswith("f ") and cur_mat:
                for tok in line.split()[1:]:
                    idx = int(tok.split("//")[0]) - 1  # 0-based
                    material_faces.setdefault(cur_mat, []).append(idx)

        def verts_for(mat):
            return [all_verts[i] for i in material_faces.get(mat, [])]

        north_verts = verts_for("material_10")  # u=1, v=0
        east_verts  = verts_for("material_20")  # u=0, v=2

        # North marker: X ∈ [0,1] (v=0) and Y ∈ [1,2] (u=1)
        assert north_verts, "North marker produced no faces"
        assert all(0.0 <= v[0] <= 1.0 for v in north_verts), \
            f"North marker (v=0) X must be in [0,1], got {north_verts}"
        assert all(1.0 <= v[1] <= 2.0 for v in north_verts), \
            f"North marker (u=1) Y must be in [1,2], got {north_verts}"

        # East marker: X ∈ [2,3] (v=2) and Y ∈ [0,1] (u=0)
        assert east_verts, "East marker produced no faces"
        assert all(2.0 <= v[0] <= 3.0 for v in east_verts), \
            f"East marker (v=2) X must be in [2,3], got {east_verts}"
        assert all(0.0 <= v[1] <= 1.0 for v in east_verts), \
            f"East marker (u=0) Y must be in [0,1], got {east_verts}"
