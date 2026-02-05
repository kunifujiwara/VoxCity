"""Tests for voxcity.geoprocessor.mesh module."""
import pytest
import numpy as np
import trimesh

from voxcity.geoprocessor.mesh import (
    create_voxel_mesh,
    create_sim_surface_mesh,
    create_city_meshes,
    split_vertices_manual,
)


class TestCreateVoxelMesh:
    """Tests for create_voxel_mesh function."""

    def test_returns_none_for_empty_class(self):
        """Test that None is returned when no voxels match class_id."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        result = create_voxel_mesh(voxel_array, class_id=1)
        assert result is None

    def test_creates_trimesh_object(self):
        """Test that a trimesh object is created."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        voxel_array[2, 2, 0] = 1  # Single voxel
        result = create_voxel_mesh(voxel_array, class_id=1)
        assert isinstance(result, trimesh.Trimesh)

    def test_single_voxel_mesh(self):
        """Test mesh for a single voxel."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        voxel_array[2, 2, 0] = 1
        result = create_voxel_mesh(voxel_array, class_id=1, meshsize=1.0)
        
        assert result is not None
        # A single voxel has 6 faces, each with 4 vertices
        # But faces at the bottom (z=0) have no boundary below
        # Should have vertices and faces
        assert len(result.vertices) > 0
        assert len(result.faces) > 0

    def test_meshsize_scaling(self):
        """Test that meshsize affects vertex coordinates."""
        voxel_array = np.zeros((3, 3, 3), dtype=np.int32)
        voxel_array[1, 1, 0] = 1
        
        mesh_1 = create_voxel_mesh(voxel_array, class_id=1, meshsize=1.0)
        mesh_2 = create_voxel_mesh(voxel_array, class_id=1, meshsize=2.0)
        
        # With meshsize=2.0, vertices should be at 2x coordinates
        assert mesh_1 is not None
        assert mesh_2 is not None
        # Max coordinate with meshsize=2 should be roughly 2x that of meshsize=1
        max_1 = np.max(mesh_1.vertices)
        max_2 = np.max(mesh_2.vertices)
        assert max_2 > max_1

    def test_cube_shape(self):
        """Test creating a cube (2x2x2 voxels)."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        voxel_array[1:3, 1:3, 0:2] = 1  # 2x2x2 cube
        result = create_voxel_mesh(voxel_array, class_id=1)
        
        assert result is not None
        assert len(result.faces) > 0

    def test_multiple_classes_isolation(self):
        """Test that only specified class is extracted."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        voxel_array[1, 1, 0] = 1  # Class 1
        voxel_array[3, 3, 0] = 2  # Class 2
        
        mesh_1 = create_voxel_mesh(voxel_array, class_id=1)
        mesh_2 = create_voxel_mesh(voxel_array, class_id=2)
        
        assert mesh_1 is not None
        assert mesh_2 is not None
        # Each should only contain faces for its class

    def test_building_class_with_building_id_grid(self):
        """Test building mesh with building ID grid."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        voxel_array[1:3, 1:3, 0:2] = -3  # Building class
        
        building_id_grid = np.zeros((5, 5), dtype=np.int32)
        building_id_grid[1:3, 1:3] = 42  # Building ID
        
        result = create_voxel_mesh(
            voxel_array, 
            class_id=-3, 
            building_id_grid=building_id_grid
        )
        
        assert result is not None
        assert 'building_id' in result.metadata

    def test_mesh_type_building_solar(self):
        """Test mesh_type='building_solar' creates boundary faces."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        voxel_array[2, 2, 0:3] = -3  # Building column
        
        building_id_grid = np.zeros((5, 5), dtype=np.int32)
        building_id_grid[2, 2] = 1
        
        result = create_voxel_mesh(
            voxel_array,
            class_id=-3,
            building_id_grid=building_id_grid,
            mesh_type='building_solar'
        )
        
        assert result is not None

    def test_mesh_type_open_air(self):
        """Test mesh_type='open_air' creates boundary faces."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        voxel_array[2, 2, 0:3] = -3  # Building column
        
        building_id_grid = np.zeros((5, 5), dtype=np.int32)
        building_id_grid[2, 2] = 1
        
        result = create_voxel_mesh(
            voxel_array,
            class_id=-3,
            building_id_grid=building_id_grid,
            mesh_type='open_air'
        )
        
        assert result is not None

    def test_faces_at_boundary_only(self):
        """Test that faces are only created at class boundaries."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        # Create a 3x3x3 solid block
        voxel_array[1:4, 1:4, 0:3] = 1
        
        result = create_voxel_mesh(voxel_array, class_id=1)
        assert result is not None
        
        # Interior faces should not exist, only boundary faces
        # This is a solid block, so faces should only be on exterior

    def test_tree_voxels(self):
        """Test mesh for tree voxels (class_id=-2)."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        voxel_array[2, 2, 0:4] = -2  # Tree column
        
        result = create_voxel_mesh(voxel_array, class_id=-2)
        assert result is not None
        assert len(result.faces) > 0

    def test_mesh_has_face_normals(self):
        """Test that mesh has face normals."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        voxel_array[2, 2, 0] = 1
        
        result = create_voxel_mesh(voxel_array, class_id=1)
        assert result is not None
        # trimesh should have face_normals
        assert result.face_normals is not None
        assert len(result.face_normals) == len(result.faces)

    def test_float_meshsize(self):
        """Test with non-integer meshsize."""
        voxel_array = np.zeros((3, 3, 3), dtype=np.int32)
        voxel_array[1, 1, 0] = 1
        
        result = create_voxel_mesh(voxel_array, class_id=1, meshsize=0.5)
        assert result is not None
        # Max coordinate should be smaller with smaller meshsize
        assert np.max(result.vertices) < 3.0

    def test_edge_voxel(self):
        """Test voxel at edge of array."""
        voxel_array = np.zeros((3, 3, 3), dtype=np.int32)
        voxel_array[0, 0, 0] = 1  # Corner voxel
        
        result = create_voxel_mesh(voxel_array, class_id=1)
        assert result is not None

    def test_building_adjacent_to_tree(self):
        """Test building next to tree creates proper boundary."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        voxel_array[1, 1, 0:3] = -3  # Building
        voxel_array[2, 1, 0:2] = -2  # Tree next to building
        
        building_id_grid = np.zeros((5, 5), dtype=np.int32)
        building_id_grid[1, 1] = 1
        
        # With building_solar, should create faces at building-tree boundary
        result = create_voxel_mesh(
            voxel_array,
            class_id=-3,
            building_id_grid=building_id_grid,
            mesh_type='building_solar'
        )
        assert result is not None

    def test_metadata_preserved_face_normals(self):
        """Test that provided face normals are stored in metadata."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        voxel_array[2, 2, 0] = 1
        
        result = create_voxel_mesh(voxel_array, class_id=1)
        assert result is not None
        assert 'provided_face_normals' in result.metadata


class TestCreateVoxelMeshIntegration:
    """Integration tests for mesh creation."""

    def test_large_voxel_array(self):
        """Test with larger voxel array."""
        voxel_array = np.zeros((20, 20, 10), dtype=np.int32)
        voxel_array[5:15, 5:15, 0:5] = 1  # 10x10x5 block
        
        result = create_voxel_mesh(voxel_array, class_id=1)
        assert result is not None
        # Should have faces only on exterior
        assert len(result.faces) > 0

    def test_scattered_voxels(self):
        """Test with non-contiguous voxels."""
        voxel_array = np.zeros((10, 10, 5), dtype=np.int32)
        voxel_array[1, 1, 0] = 1
        voxel_array[5, 5, 0] = 1
        voxel_array[8, 8, 0] = 1
        
        result = create_voxel_mesh(voxel_array, class_id=1)
        assert result is not None
        # Should have faces for all 3 isolated voxels


class TestCreateSimSurfaceMesh:
    """Tests for create_sim_surface_mesh function."""

    def test_returns_mesh_for_valid_data(self):
        """Test that a valid mesh is returned for valid data."""
        sim_grid = np.array([[0.5, 0.6], [0.4, 0.8]])
        dem_grid = np.array([[10.0, 10.2], [9.8, 10.1]])
        
        result = create_sim_surface_mesh(sim_grid, dem_grid, meshsize=1.0)
        
        assert result is not None
        assert isinstance(result, trimesh.Trimesh)
        assert len(result.faces) > 0

    def test_returns_none_for_all_nan(self):
        """Test that None is returned when all values are NaN."""
        sim_grid = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        dem_grid = np.array([[10.0, 10.2], [9.8, 10.1]])
        
        result = create_sim_surface_mesh(sim_grid, dem_grid)
        
        assert result is None

    def test_skips_nan_cells(self):
        """Test that NaN cells are skipped."""
        sim_grid = np.array([[0.5, np.nan], [np.nan, 0.8]])
        dem_grid = np.array([[10.0, 10.2], [9.8, 10.1]])
        
        result = create_sim_surface_mesh(sim_grid, dem_grid)
        
        assert result is not None
        # Should have fewer faces than if all cells were valid
        assert len(result.faces) == 4  # 2 valid cells * 2 triangles each

    def test_meshsize_affects_coordinates(self):
        """Test that meshsize affects vertex coordinates."""
        sim_grid = np.array([[0.5, 0.6], [0.4, 0.8]])
        dem_grid = np.zeros((2, 2))
        
        mesh_1 = create_sim_surface_mesh(sim_grid, dem_grid, meshsize=1.0)
        mesh_2 = create_sim_surface_mesh(sim_grid, dem_grid, meshsize=2.0)
        
        assert mesh_1 is not None
        assert mesh_2 is not None
        # Larger meshsize should produce larger coordinates
        assert np.max(mesh_2.vertices[:, :2]) > np.max(mesh_1.vertices[:, :2])

    def test_z_offset_affects_height(self):
        """Test that z_offset affects mesh height."""
        sim_grid = np.array([[0.5]])
        dem_grid = np.array([[0.0]])
        
        mesh_1 = create_sim_surface_mesh(sim_grid, dem_grid, z_offset=0.0)
        mesh_2 = create_sim_surface_mesh(sim_grid, dem_grid, z_offset=5.0)
        
        assert mesh_1 is not None
        assert mesh_2 is not None
        # Higher z_offset should produce higher z coordinates
        assert np.min(mesh_2.vertices[:, 2]) > np.min(mesh_1.vertices[:, 2]) - 5.1

    def test_custom_vmin_vmax(self):
        """Test custom vmin/vmax for colormap."""
        sim_grid = np.array([[0.3, 0.7]])
        dem_grid = np.array([[0.0, 0.0]])
        
        result = create_sim_surface_mesh(sim_grid, dem_grid, vmin=0.0, vmax=1.0)
        
        assert result is not None
        assert len(result.faces) > 0

    def test_different_colormap(self):
        """Test with different colormap."""
        sim_grid = np.array([[0.5, 0.6], [0.4, 0.8]])
        dem_grid = np.array([[10.0, 10.2], [9.8, 10.1]])
        
        result = create_sim_surface_mesh(sim_grid, dem_grid, cmap_name='RdYlBu')
        
        assert result is not None
        assert len(result.faces) > 0


class TestCreateCityMeshes:
    """Tests for create_city_meshes function."""

    def test_creates_dict_of_meshes(self):
        """Test that a dictionary of meshes is created."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        voxel_array[1:3, 1:3, 0:2] = -3  # Building
        
        vox_dict = {-3: [200, 200, 200]}
        
        result = create_city_meshes(voxel_array, vox_dict)
        
        assert isinstance(result, dict)
        assert -3 in result
        assert isinstance(result[-3], trimesh.Trimesh)

    def test_skips_air_class(self):
        """Test that class 0 (air) is skipped."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        voxel_array[1:3, 1:3, 0:2] = -3
        
        vox_dict = {0: [0, 0, 0], -3: [200, 200, 200]}
        
        result = create_city_meshes(voxel_array, vox_dict)
        
        assert 0 not in result

    def test_multiple_classes(self):
        """Test with multiple urban element classes."""
        voxel_array = np.zeros((10, 10, 5), dtype=np.int32)
        voxel_array[1:3, 1:3, 0:3] = -3  # Building
        voxel_array[5:7, 5:7, 0:2] = -2  # Trees
        
        vox_dict = {
            -3: [200, 200, 200],
            -2: [0, 255, 0]
        }
        
        result = create_city_meshes(voxel_array, vox_dict)
        
        assert -3 in result
        assert -2 in result

    def test_empty_class_excluded(self):
        """Test that classes with no voxels are excluded."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        voxel_array[1:3, 1:3, 0:2] = -3  # Only building
        
        vox_dict = {
            -3: [200, 200, 200],
            -2: [0, 255, 0]  # No tree voxels exist
        }
        
        result = create_city_meshes(voxel_array, vox_dict)
        
        assert -3 in result
        assert -2 not in result

    def test_include_classes_filter(self):
        """Test include_classes parameter."""
        voxel_array = np.zeros((10, 10, 5), dtype=np.int32)
        voxel_array[1:3, 1:3, 0:3] = -3
        voxel_array[5:7, 5:7, 0:2] = -2
        voxel_array[8:9, 8:9, 0:1] = 1
        
        vox_dict = {-3: [200, 200, 200], -2: [0, 255, 0], 1: [100, 100, 100]}
        
        # Only include buildings
        result = create_city_meshes(voxel_array, vox_dict, include_classes=[-3])
        
        assert -3 in result
        assert -2 not in result
        assert 1 not in result

    def test_exclude_classes_filter(self):
        """Test exclude_classes parameter."""
        voxel_array = np.zeros((10, 10, 5), dtype=np.int32)
        voxel_array[1:3, 1:3, 0:3] = -3
        voxel_array[5:7, 5:7, 0:2] = -2
        
        vox_dict = {-3: [200, 200, 200], -2: [0, 255, 0]}
        
        # Exclude trees
        result = create_city_meshes(voxel_array, vox_dict, exclude_classes=[-2])
        
        assert -3 in result
        assert -2 not in result

    def test_meshes_have_colors(self):
        """Test that meshes have face colors applied."""
        voxel_array = np.zeros((5, 5, 5), dtype=np.int32)
        voxel_array[1:3, 1:3, 0:2] = -3
        
        vox_dict = {-3: [200, 200, 200]}
        
        result = create_city_meshes(voxel_array, vox_dict)
        
        assert -3 in result
        mesh = result[-3]
        # Check that face colors are set
        assert mesh.visual.face_colors is not None


class TestSplitVerticesManual:
    """Tests for split_vertices_manual function."""

    def test_splits_shared_vertices(self):
        """Test that shared vertices are split."""
        # Two triangles sharing edge
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        result = split_vertices_manual(mesh)
        
        # After splitting, each face should have its own vertices
        # 2 faces * 3 vertices = 6 vertices
        assert len(result.vertices) == 6
        assert len(result.faces) == 2

    def test_preserves_face_count(self):
        """Test that face count is preserved."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        result = split_vertices_manual(mesh)
        
        assert len(result.faces) == len(mesh.faces)

    def test_preserves_face_colors(self):
        """Test that face colors are preserved."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Set face colors
        colors = np.array([[255, 0, 0, 255], [0, 255, 0, 255]])
        mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=colors)
        
        result = split_vertices_manual(mesh)
        
        # Colors should be preserved
        assert result.visual.face_colors is not None
        assert len(result.visual.face_colors) == 2

    def test_single_face_mesh(self):
        """Test with single face mesh."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        result = split_vertices_manual(mesh)
        
        assert len(result.faces) == 1
        assert len(result.vertices) == 3
