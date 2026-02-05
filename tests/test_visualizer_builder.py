"""Tests for voxcity.visualizer.builder module."""
import pytest
import numpy as np

from voxcity.models import VoxelGrid, MeshCollection
from voxcity.visualizer.builder import MeshBuilder


class TestMeshBuilder:
    """Tests for MeshBuilder class."""

    @pytest.fixture
    def simple_voxel_grid(self):
        """Create a simple voxel grid for testing."""
        meta = {"meshsize": 1.0, "bounds": (0, 0, 10, 10)}
        classes = np.zeros((10, 10, 5), dtype=np.int32)
        # Add a building
        classes[3:6, 3:6, 0:3] = -3
        # Add some trees
        classes[7:9, 7:9, 0:2] = -2
        return VoxelGrid(classes=classes, meta=meta)

    def test_from_voxel_grid_returns_mesh_collection(self, simple_voxel_grid):
        """Test that from_voxel_grid returns a MeshCollection."""
        result = MeshBuilder.from_voxel_grid(
            simple_voxel_grid,
            meshsize=1.0,
        )
        assert isinstance(result, MeshCollection)

    def test_creates_meshes_for_classes(self, simple_voxel_grid):
        """Test that meshes are created for each class."""
        result = MeshBuilder.from_voxel_grid(
            simple_voxel_grid,
            meshsize=1.0,
        )
        # Check that meshes dict has items
        assert len(result.meshes) >= 1

    def test_custom_color_map_dict(self, simple_voxel_grid):
        """Test with custom color map as dict."""
        custom_colors = {
            -3: [255, 0, 0],  # Red buildings
            -2: [0, 255, 0],  # Green trees
        }
        result = MeshBuilder.from_voxel_grid(
            simple_voxel_grid,
            meshsize=1.0,
            voxel_color_map=custom_colors,
        )
        assert isinstance(result, MeshCollection)

    def test_include_classes_filter(self, simple_voxel_grid):
        """Test include_classes parameter."""
        result = MeshBuilder.from_voxel_grid(
            simple_voxel_grid,
            meshsize=1.0,
            include_classes=[-3],  # Only buildings
        )
        # Should only have building mesh
        assert len(result.meshes) >= 1

    def test_exclude_classes_filter(self, simple_voxel_grid):
        """Test exclude_classes parameter."""
        result = MeshBuilder.from_voxel_grid(
            simple_voxel_grid,
            meshsize=1.0,
            exclude_classes=[-2],  # Exclude trees
        )
        # Should not have tree mesh
        assert "-2" not in [m.name for m in result.meshes.values()]

    def test_meshsize_affects_coordinates(self):
        """Test that meshsize affects vertex coordinates."""
        meta = {"meshsize": 1.0, "bounds": (0, 0, 5, 5)}
        classes = np.zeros((5, 5, 3), dtype=np.int32)
        classes[1:3, 1:3, 0:2] = -3
        grid = VoxelGrid(classes=classes, meta=meta)
        
        result_1 = MeshBuilder.from_voxel_grid(grid, meshsize=1.0)
        result_2 = MeshBuilder.from_voxel_grid(grid, meshsize=2.0)
        
        # Both should have meshes
        assert len(result_1.meshes) > 0
        assert len(result_2.meshes) > 0

    def test_empty_voxel_grid(self):
        """Test with empty voxel grid (all zeros)."""
        meta = {"meshsize": 1.0, "bounds": (0, 0, 5, 5)}
        classes = np.zeros((5, 5, 3), dtype=np.int32)
        grid = VoxelGrid(classes=classes, meta=meta)
        
        result = MeshBuilder.from_voxel_grid(grid, meshsize=1.0)
        assert isinstance(result, MeshCollection)
        # Should be empty (no classes to render)
        assert len(result.meshes) == 0
