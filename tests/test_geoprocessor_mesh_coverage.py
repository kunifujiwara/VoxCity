"""
Comprehensive tests for voxcity.geoprocessor.mesh to improve coverage.
"""

import numpy as np
import pytest

from voxcity.geoprocessor.mesh import (
    create_voxel_mesh,
    create_sim_surface_mesh,
)


class TestCreateVoxelMesh:
    def test_empty_class(self):
        voxels = np.zeros((5, 5, 5), dtype=np.int32)
        mesh = create_voxel_mesh(voxels, class_id=1)
        assert mesh is None

    def test_single_voxel(self):
        voxels = np.zeros((5, 5, 5), dtype=np.int32)
        voxels[2, 2, 2] = 1
        mesh = create_voxel_mesh(voxels, class_id=1)
        assert mesh is not None
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_meshsize_scaling(self):
        voxels = np.zeros((5, 5, 5), dtype=np.int32)
        voxels[2, 2, 2] = 1
        mesh1 = create_voxel_mesh(voxels, class_id=1, meshsize=1.0)
        mesh2 = create_voxel_mesh(voxels, class_id=1, meshsize=2.0)
        # Larger meshsize should result in larger coordinates
        assert mesh2.vertices.max() > mesh1.vertices.max()

    def test_building_class_with_ids(self):
        voxels = np.zeros((5, 5, 5), dtype=np.int32)
        voxels[2, 2, 0:3] = -3  # Building
        building_id_grid = np.zeros((5, 5))
        building_id_grid[2, 2] = 42

        mesh = create_voxel_mesh(voxels, class_id=-3, building_id_grid=building_id_grid)
        assert mesh is not None
        assert 'building_id' in mesh.metadata

    def test_building_solar_mesh_type(self):
        voxels = np.zeros((5, 5, 5), dtype=np.int32)
        voxels[2, 2, 0:3] = -3  # Building
        building_id_grid = np.zeros((5, 5))
        building_id_grid[2, 2] = 1

        mesh = create_voxel_mesh(voxels, class_id=-3,
                                 building_id_grid=building_id_grid,
                                 mesh_type='building_solar')
        assert mesh is not None

    def test_tree_voxels(self):
        voxels = np.zeros((5, 5, 5), dtype=np.int32)
        voxels[2, 2, 2:4] = -2  # Tree
        mesh = create_voxel_mesh(voxels, class_id=-2)
        assert mesh is not None

    def test_cube(self):
        voxels = np.zeros((5, 5, 5), dtype=np.int32)
        voxels[1:4, 1:4, 1:4] = 1
        mesh = create_voxel_mesh(voxels, class_id=1)
        assert mesh is not None
        # A 3x3x3 cube should have 6 faces x 9 = 54 quads = 108 triangles
        assert len(mesh.faces) > 0

    def test_face_normals_stored(self):
        voxels = np.zeros((5, 5, 5), dtype=np.int32)
        voxels[2, 2, 2] = 1
        mesh = create_voxel_mesh(voxels, class_id=1)
        assert 'provided_face_normals' in mesh.metadata

    def test_multiple_classes_isolation(self):
        voxels = np.zeros((5, 5, 5), dtype=np.int32)
        voxels[1, 1, 1] = 1
        voxels[3, 3, 3] = 2
        mesh1 = create_voxel_mesh(voxels, class_id=1)
        mesh2 = create_voxel_mesh(voxels, class_id=2)
        assert mesh1 is not None
        assert mesh2 is not None


class TestCreateSimSurfaceMesh:
    def test_basic(self):
        sim = np.array([[0.5, 0.6], [0.4, 0.8]])
        dem = np.array([[10.0, 10.2], [9.8, 10.1]])
        mesh = create_sim_surface_mesh(sim, dem)
        assert mesh is not None
        assert len(mesh.vertices) > 0

    def test_all_nan(self):
        sim = np.full((3, 3), np.nan)
        dem = np.zeros((3, 3))
        mesh = create_sim_surface_mesh(sim, dem)
        assert mesh is None

    def test_partial_nan(self):
        sim = np.array([[0.5, np.nan], [np.nan, 0.8]])
        dem = np.zeros((2, 2))
        mesh = create_sim_surface_mesh(sim, dem)
        assert mesh is not None

    def test_custom_colormap(self):
        sim = np.array([[0.5, 0.6], [0.4, 0.8]])
        dem = np.zeros((2, 2))
        mesh = create_sim_surface_mesh(sim, dem, cmap_name='RdYlBu')
        assert mesh is not None

    def test_custom_vmin_vmax(self):
        sim = np.array([[0.5, 0.6], [0.4, 0.8]])
        dem = np.zeros((2, 2))
        mesh = create_sim_surface_mesh(sim, dem, vmin=0.0, vmax=1.0)
        assert mesh is not None

    def test_z_offset(self):
        sim = np.array([[1.0]])
        dem = np.array([[10.0]])
        mesh1 = create_sim_surface_mesh(sim, dem, z_offset=0.0)
        mesh2 = create_sim_surface_mesh(sim, dem, z_offset=5.0)
        assert mesh1 is not None
        assert mesh2 is not None
        # Different z_offsets should produce different z coordinates
        assert mesh1.vertices[:, 2].max() != mesh2.vertices[:, 2].max()

    def test_meshsize(self):
        sim = np.array([[1.0, 2.0], [3.0, 4.0]])
        dem = np.zeros((2, 2))
        mesh1 = create_sim_surface_mesh(sim, dem, meshsize=1.0)
        mesh2 = create_sim_surface_mesh(sim, dem, meshsize=5.0)
        assert mesh2.vertices[:, 0].max() > mesh1.vertices[:, 0].max()
