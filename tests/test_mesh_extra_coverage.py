"""
Tests for mesh.py additional coverage:
  - create_voxel_mesh returning None (empty vertices)
  - create_colored_grid_mesh returning None (no data)
  - create_city_meshes edge cases (missing color, mesh None, ValueError)
  - save_obj_from_colored_mesh: int colors, RGB->RGBA, no face_colors default grey
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import trimesh


class TestCreateVoxelMeshEmptyReturn:
    """Cover create_voxel_mesh returning None when no voxels exist for the class."""

    def test_empty_voxel_array_returns_none(self):
        from voxcity.geoprocessor.mesh import create_voxel_mesh
        # All zeros - class_id=1 has no voxels
        voxel_array = np.zeros((3, 3, 3), dtype=np.int8)
        result = create_voxel_mesh(voxel_array, 1, meshsize=1.0)
        assert result is None


class TestCreateSimSurfaceMeshEmptyReturn:
    """Cover create_sim_surface_mesh returning None (line 361)."""

    def test_all_nan_returns_none(self):
        from voxcity.geoprocessor.mesh import create_sim_surface_mesh
        sim_grid = np.full((3, 3), np.nan)
        dem_grid = np.zeros((3, 3))
        result = create_sim_surface_mesh(sim_grid, dem_grid, meshsize=1.0)
        assert result is None


class TestCreateCityMeshesEdgeCases:
    """Cover create_city_meshes skip branches."""

    def test_class_with_no_voxels_skipped(self):
        """Class id present in dict but no voxels -> mesh is None -> continue."""
        from voxcity.geoprocessor.mesh import create_city_meshes
        voxel_array = np.zeros((3, 3, 3), dtype=np.int8)
        voxel_array[0, 0, 0] = -3  # building
        # Include class -2 (tree) in dict but no tree voxels exist
        vox_dict = {-3: [200, 200, 200], -2: [0, 255, 0]}
        meshes = create_city_meshes(voxel_array, vox_dict, meshsize=1.0)
        assert -3 in meshes
        assert -2 not in meshes  # no tree voxels -> skipped

    def test_class_without_color_skipped(self):
        """Class exists in voxels but not in color dict -> continue."""
        from voxcity.geoprocessor.mesh import create_city_meshes
        voxel_array = np.zeros((3, 3, 3), dtype=np.int8)
        voxel_array[0, 0, 0] = -3  # building
        voxel_array[1, 1, 1] = 2  # land cover 2
        # Color dict missing class 2
        vox_dict = {-3: [200, 200, 200]}
        meshes = create_city_meshes(voxel_array, vox_dict, meshsize=1.0)
        assert -3 in meshes
        assert 2 not in meshes  # no color for class 2


class TestSaveObjFromColoredMesh:
    """Cover save_obj_from_colored_mesh edge cases."""

    def test_int_colors_conversion(self, tmp_path):
        """Face colors with int dtype -> should be converted to uint8."""
        from voxcity.geoprocessor.mesh import save_obj_from_colored_mesh
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
            process=False,
        )
        # Set face_colors as int array (not uint8, not float)
        colors = np.array([[200, 100, 50, 255]], dtype=np.int32)
        mesh.visual.face_colors = colors

        save_obj_from_colored_mesh(
            {-3: mesh}, str(tmp_path), "test_int"
        )
        assert (tmp_path / "test_int.obj").exists()
        assert (tmp_path / "test_int.mtl").exists()

    def test_rgb_to_rgba_padding(self, tmp_path):
        """Face colors with 3 channels -> should be padded to RGBA."""
        from voxcity.geoprocessor.mesh import save_obj_from_colored_mesh
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
            process=False,
        )
        # Set face_colors as 3-channel RGB float
        colors = np.array([[0.8, 0.4, 0.2]], dtype=np.float64)
        mesh.visual.face_colors = colors

        save_obj_from_colored_mesh(
            {-3: mesh}, str(tmp_path), "test_rgb"
        )
        assert (tmp_path / "test_rgb.obj").exists()

    def test_no_face_colors_default_grey(self, tmp_path):
        """Mesh with no face_colors -> default grey material."""
        from voxcity.geoprocessor.mesh import save_obj_from_colored_mesh
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
            process=False,
        )
        # Remove face colors
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)

        save_obj_from_colored_mesh(
            {-3: mesh}, str(tmp_path), "test_grey"
        )
        assert (tmp_path / "test_grey.obj").exists()
        # Check MTL file contains a material
        mtl_content = (tmp_path / "test_grey.mtl").read_text()
        assert "newmtl" in mtl_content

    def test_quantization_with_max_materials(self, tmp_path):
        """Cover the color quantization branch (max_materials != None)."""
        from voxcity.geoprocessor.mesh import save_obj_from_colored_mesh

        # Create two meshes with different colors
        mesh1 = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            faces=[[0, 1, 2], [3, 4, 5]],
            process=False,
        )
        colors1 = np.array([[255, 0, 0, 255], [0, 255, 0, 255]], dtype=np.uint8)
        mesh1.visual.face_colors = colors1

        mesh2 = trimesh.Trimesh(
            vertices=[[2, 0, 0], [3, 0, 0], [2, 1, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0]],
            faces=[[0, 1, 2], [3, 4, 5]],
            process=False,
        )
        colors2 = np.array([[0, 0, 255, 255], [255, 255, 0, 255]], dtype=np.uint8)
        mesh2.visual.face_colors = colors2

        save_obj_from_colored_mesh(
            {-3: mesh1, -2: mesh2}, str(tmp_path), "test_quant",
            max_materials=2,
        )
        assert (tmp_path / "test_quant.obj").exists()
        assert (tmp_path / "test_quant.mtl").exists()
        # Should have at most 2 materials
        mtl_content = (tmp_path / "test_quant.mtl").read_text()
        assert mtl_content.count("newmtl") <= 2
