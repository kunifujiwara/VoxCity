"""Extended tests for voxcity.utils.material module."""
import pytest
import numpy as np

from voxcity.utils.material import (
    get_material_dict,
    get_modulo_numbers,
    set_building_material_by_id,
)


class TestGetMaterialDict:
    def test_returns_dict(self):
        mat = get_material_dict()
        assert isinstance(mat, dict)

    def test_known_materials(self):
        mat = get_material_dict()
        assert "brick" in mat
        assert "glass" in mat
        assert "concrete" in mat
        assert "unknown" in mat
        assert "wood" in mat
        assert "metal" in mat
        assert "stone" in mat
        assert "plaster" in mat

    def test_all_negative_ids(self):
        """Material IDs should be negative to distinguish from other voxel types."""
        mat = get_material_dict()
        for material, id_val in mat.items():
            assert id_val < 0, f"{material} has non-negative ID {id_val}"

    def test_unique_ids(self):
        mat = get_material_dict()
        ids = list(mat.values())
        assert len(ids) == len(set(ids)), "Material IDs should be unique"

    def test_specific_values(self):
        mat = get_material_dict()
        assert mat["unknown"] == -3
        assert mat["brick"] == -11
        assert mat["glass"] == -16


class TestGetModuloNumbers:
    def test_very_sparse_windows(self):
        result = get_modulo_numbers(0.10)
        assert result == (2, 2, 2)

    def test_maximum_density(self):
        result = get_modulo_numbers(0.95)
        assert result == (1, 1, 1)

    def test_returns_tuple_of_three(self):
        for ratio in [0.1, 0.25, 0.5, 0.75, 0.9]:
            result = get_modulo_numbers(ratio)
            assert isinstance(result, tuple)
            assert len(result) == 3

    def test_all_values_positive(self):
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            result = get_modulo_numbers(ratio)
            assert all(v > 0 for v in result)

    def test_boundary_values(self):
        # Test around 0.125
        assert get_modulo_numbers(0.125) == (2, 2, 2)
        
        # Test above threshold
        result = get_modulo_numbers(0.25)
        assert result in [(2, 2, 1), (2, 1, 2), (1, 2, 2)]

    def test_medium_density(self):
        result = get_modulo_numbers(0.5)
        assert result in [(2, 1, 1), (1, 2, 1), (1, 1, 2)]


class TestSetBuildingMaterialById:
    @pytest.fixture
    def simple_grid(self):
        """Create a simple 3x3x5 voxel grid with a building at center."""
        voxel = np.full((3, 3, 5), fill_value=0, dtype=int)
        # Mark building voxels at center (1,1) with unknown material (-3)
        voxel[1, 1, :4] = -3  # Building from z=0 to z=3, z=4 is empty
        
        building_ids = np.zeros((3, 3), dtype=int)
        building_ids[1, 1] = 1  # Building ID 1 at center
        
        return voxel, building_ids

    def test_sets_material(self, simple_grid):
        voxel, building_ids = simple_grid
        brick_id = get_material_dict()["brick"]
        glass_id = get_material_dict()["glass"]
        
        result = set_building_material_by_id(
            voxel, building_ids, ids=[1], mark=brick_id,
            window_ratio=0.95, glass_id=glass_id
        )
        
        # Check that at least some cells have the material
        center_col = result[1, 1, :]
        assert brick_id in center_col or glass_id in center_col

    def test_top_floor_no_glass(self, simple_grid):
        """Top floor should never have glass."""
        voxel, building_ids = simple_grid
        brick_id = get_material_dict()["brick"]
        glass_id = get_material_dict()["glass"]
        
        result = set_building_material_by_id(
            voxel, building_ids, ids=[1], mark=brick_id,
            window_ratio=0.95, glass_id=glass_id
        )
        
        # Find maximum z with building material
        center_col = result[1, 1, :]
        building_mask = (center_col == brick_id) | (center_col == glass_id)
        if building_mask.any():
            max_z = np.where(building_mask)[0].max()
            # Top floor should be brick, not glass
            assert center_col[max_z] == brick_id

    def test_no_change_outside_building(self, simple_grid):
        voxel, building_ids = simple_grid
        brick_id = get_material_dict()["brick"]
        
        result = set_building_material_by_id(
            voxel, building_ids, ids=[1], mark=brick_id
        )
        
        # Cells outside the building should remain unchanged (0)
        assert result[0, 0, 0] == 0
        assert result[2, 2, 2] == 0

    def test_multiple_building_ids(self):
        voxel = np.full((4, 4, 5), fill_value=0, dtype=int)
        voxel[1, 1, :3] = -3  # Building 1
        voxel[2, 2, :3] = -3  # Building 2
        
        building_ids = np.zeros((4, 4), dtype=int)
        building_ids[1, 1] = 1
        building_ids[2, 2] = 2
        
        brick_id = get_material_dict()["brick"]
        
        # Only process building ID 1
        result = set_building_material_by_id(
            voxel, building_ids, ids=[1], mark=brick_id
        )
        
        # Building 2 should still be -3 (unprocessed)
        # Note: building_id_grid is flipped inside the function, so check consistently
        assert -3 in result[2, 2, :]

    def test_sparse_window_ratio(self, simple_grid):
        """With sparse window ratio, fewer windows should be created."""
        voxel, building_ids = simple_grid
        brick_id = get_material_dict()["brick"]
        glass_id = get_material_dict()["glass"]
        
        result = set_building_material_by_id(
            voxel, building_ids, ids=[1], mark=brick_id,
            window_ratio=0.1, glass_id=glass_id
        )
        
        # Should still work without errors
        assert result is not None
        assert result.shape == voxel.shape
