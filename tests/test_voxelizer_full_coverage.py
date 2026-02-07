"""
Tests for Voxelizer.generate_combined and generate_components.
Covers the full voxelization pipeline including _convert_land_cover,
_estimate_and_allocate, and generate_components.
"""

import numpy as np
import pytest

from voxcity.generator.voxelizer import Voxelizer, replace_nan_in_nested, GROUND_CODE, TREE_CODE, BUILDING_CODE


class TestVoxelizerGenerateCombined:
    """Test the full generate_combined method."""

    def _make_grids(self, rows=5, cols=5):
        building_height = np.zeros((rows, cols))
        building_min_height = np.empty((rows, cols), dtype=object)
        building_id = np.zeros((rows, cols), dtype=np.int32)
        land_cover = np.ones((rows, cols), dtype=np.int32) * 2
        dem = np.zeros((rows, cols))
        tree = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                building_min_height[i, j] = []
        return building_height, building_min_height, building_id, land_cover, dem, tree

    def test_basic_generation(self):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="OpenStreetMap")
        bh, bmin, bid, lc, dem, tree = self._make_grids()
        vg = voxelizer.generate_combined(bh, bmin, bid, lc, dem, tree)
        assert vg.ndim == 3
        assert vg.shape[0] == 5
        assert vg.shape[1] == 5

    def test_with_building(self):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="OpenStreetMap")
        bh, bmin, bid, lc, dem, tree = self._make_grids()
        bh[2, 2] = 10.0
        bid[2, 2] = 1
        bmin[2, 2] = [[0.0, 10.0]]
        vg = voxelizer.generate_combined(bh, bmin, bid, lc, dem, tree, print_class_info=False)
        # Building voxels should exist
        assert np.any(vg == BUILDING_CODE)

    def test_with_tree(self):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="OpenStreetMap")
        bh, bmin, bid, lc, dem, tree = self._make_grids()
        tree[1, 1] = 8.0
        vg = voxelizer.generate_combined(bh, bmin, bid, lc, dem, tree, print_class_info=False)
        assert np.any(vg == TREE_CODE)

    def test_with_canopy_bottom(self):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="OpenStreetMap")
        bh, bmin, bid, lc, dem, tree = self._make_grids()
        tree[1, 1] = 10.0
        canopy_bottom = np.zeros((5, 5))
        canopy_bottom[1, 1] = 4.0
        vg = voxelizer.generate_combined(bh, bmin, bid, lc, dem, tree,
                                          canopy_bottom_height_grid_ori=canopy_bottom,
                                          print_class_info=False)
        assert np.any(vg == TREE_CODE)

    def test_with_dem(self):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="OpenStreetMap")
        bh, bmin, bid, lc, dem, tree = self._make_grids()
        dem[:, :] = 5.0
        bh[2, 2] = 3.0  # add a small building to ensure max_height > 0
        bmin[2, 2] = [[0.0, 3.0]]
        vg = voxelizer.generate_combined(bh, bmin, bid, lc, dem, tree, print_class_info=False)
        # DEM raises the ground, so voxel grid should have more z layers
        assert vg.shape[2] > 1

    def test_non_osm_source(self):
        """Test with non-OpenStreetMap land cover source."""
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="ESRI 10m")
        bh, bmin, bid, lc, dem, tree = self._make_grids()
        lc[:, :] = 0  # First class index
        vg = voxelizer.generate_combined(bh, bmin, bid, lc, dem, tree, print_class_info=False)
        assert vg.ndim == 3

    def test_memory_limit(self):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="OpenStreetMap",
                              max_voxel_ram_mb=0.0001)
        bh, bmin, bid, lc, dem, tree = self._make_grids()
        bh[2, 2] = 100.0
        bmin[2, 2] = [[0.0, 100.0]]
        # Should print warning but not crash (exception is caught)
        vg = voxelizer.generate_combined(bh, bmin, bid, lc, dem, tree, print_class_info=False)
        assert vg.ndim == 3


class TestVoxelizerGenerateComponents:
    """Test the generate_components method (layered output)."""

    def _make_grids(self, rows=5, cols=5):
        building_height = np.zeros((rows, cols))
        land_cover = np.ones((rows, cols), dtype=np.int32) * 2
        dem = np.zeros((rows, cols))
        tree = np.zeros((rows, cols))
        return building_height, land_cover, dem, tree

    def test_basic_components(self):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="OpenStreetMap")
        bh, lc, dem, tree = self._make_grids()
        bh[2, 2] = 5.0
        tree[1, 1] = 3.0
        lc_vox, bld_vox, tree_vox, dem_vox, layered = voxelizer.generate_components(
            bh, lc, dem, tree, print_class_info=False
        )
        assert lc_vox.ndim == 3
        assert bld_vox.ndim == 3
        assert tree_vox.ndim == 3
        assert dem_vox.ndim == 3
        assert layered.ndim == 3

    def test_layered_interval(self):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="OpenStreetMap")
        bh, lc, dem, tree = self._make_grids()
        bh[2, 2] = 3.0
        _, _, _, _, layered = voxelizer.generate_components(
            bh, lc, dem, tree, layered_interval=10, print_class_info=False
        )
        # Layered grid z-dimension = 4 * layered_interval
        assert layered.shape[2] == 40

    def test_non_osm_source(self):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="ESRI 10m")
        bh, lc, dem, tree = self._make_grids()
        lc[:, :] = 0
        bh[2, 2] = 5.0  # need some height to avoid zero max_height
        result = voxelizer.generate_components(bh, lc, dem, tree, print_class_info=False)
        assert len(result) == 5

    def test_tree_voxels_present(self):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="OpenStreetMap")
        bh, lc, dem, tree = self._make_grids()
        tree[1, 1] = 6.0
        _, _, tree_vox, _, _ = voxelizer.generate_components(
            bh, lc, dem, tree, print_class_info=False
        )
        assert np.any(tree_vox == -2)

    def test_building_voxels_present(self):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="OpenStreetMap")
        bh, lc, dem, tree = self._make_grids()
        bh[3, 3] = 8.0
        _, bld_vox, _, _, _ = voxelizer.generate_components(
            bh, lc, dem, tree, print_class_info=False
        )
        assert np.any(bld_vox == -3)


class TestVoxelizerConvertLandCover:
    def test_osm_shifts_by_one(self):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="OpenStreetMap")
        lc = np.array([[0, 1], [2, 3]], dtype=np.int32)
        result = voxelizer._convert_land_cover(lc)
        np.testing.assert_array_equal(result, lc + 1)

    def test_esri_uses_convert(self):
        voxelizer = Voxelizer(voxel_size=1.0, land_cover_source="ESRI 10m")
        lc = np.array([[0, 0], [0, 0]], dtype=np.int32)
        result = voxelizer._convert_land_cover(lc)
        assert result.shape == lc.shape
