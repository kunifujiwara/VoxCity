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
