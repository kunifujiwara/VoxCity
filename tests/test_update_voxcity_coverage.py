"""Round 6 – cover update.py (lines 167-430): update_voxcity with building_gdf, tree_gdf, shape validation, regenerate_voxels."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from voxcity.models import (
    GridMetadata,
    BuildingGrid,
    LandCoverGrid,
    DemGrid,
    VoxelGrid,
    CanopyGrid,
    VoxCity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_city(shape=(4, 4)):
    """Build a minimal VoxCity with known grid data."""
    meta = GridMetadata(crs="EPSG:4326", bounds=(0, 0, 1, 1), meshsize=5.0)
    bh = np.zeros(shape, dtype=float)
    bmin = np.empty(shape, dtype=object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            bmin[i, j] = []
    bid = np.zeros(shape, dtype=int)
    lc = np.ones(shape, dtype=int)
    dem = np.zeros(shape, dtype=float)
    canopy_top = np.zeros(shape, dtype=float)
    canopy_bottom = np.zeros(shape, dtype=float)
    voxel_classes = np.zeros((*shape, 10), dtype=np.int8)

    city = VoxCity(
        voxels=VoxelGrid(classes=voxel_classes, meta=meta),
        buildings=BuildingGrid(heights=bh, min_heights=bmin, ids=bid, meta=meta),
        land_cover=LandCoverGrid(classes=lc, meta=meta),
        dem=DemGrid(elevation=dem, meta=meta),
        tree_canopy=CanopyGrid(top=canopy_top, bottom=canopy_bottom, meta=meta),
        extras={
            "rectangle_vertices": [(0, 0), (0, 1), (1, 1), (1, 0)],
            "selected_sources": {"land_cover_source": "OpenStreetMap"},
        },
    )
    return city


# ===========================================================================
# Tests for update_voxcity – building_gdf auto-generation path
# ===========================================================================

class TestUpdateWithBuildingGdf:
    """Cover lines 167-183 of update.py."""

    @patch("voxcity.generator.update.Voxelizer")
    @patch("voxcity.geoprocessor.raster.create_building_height_grid_from_gdf_polygon")
    def test_auto_generates_from_gdf(self, mock_create, mock_vox):
        from voxcity.generator.update import update_voxcity

        city = _make_city()
        shape = city.buildings.heights.shape

        new_bh = np.full(shape, 20.0)
        new_bmin = np.empty(shape, dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                new_bmin[i, j] = []
        new_bid = np.ones(shape, dtype=int)
        mock_create.return_value = (new_bh, new_bmin, new_bid, None)

        vox_instance = MagicMock()
        vox_instance.generate_combined.return_value = np.zeros((*shape, 10), dtype=np.int8)
        mock_vox.return_value = vox_instance

        building_gdf = MagicMock()
        result = update_voxcity(city, building_gdf=building_gdf)

        mock_create.assert_called_once()
        np.testing.assert_array_equal(result.buildings.heights, new_bh)

    @patch("voxcity.generator.update.Voxelizer")
    def test_no_rect_raises(self, mock_vox):
        from voxcity.generator.update import update_voxcity

        city = _make_city()
        city.extras = {}  # no rectangle_vertices

        with pytest.raises(ValueError, match="rectangle_vertices"):
            update_voxcity(city, building_gdf=MagicMock())


# ===========================================================================
# Tests for update_voxcity – tree_gdf auto-generation path
# ===========================================================================

class TestUpdateWithTreeGdf:
    """Cover lines 189-223 of update.py."""

    @patch("voxcity.generator.update.Voxelizer")
    @patch("voxcity.geoprocessor.raster.create_canopy_grids_from_tree_gdf")
    def test_replace_mode(self, mock_canopy, mock_vox):
        from voxcity.generator.update import update_voxcity

        city = _make_city()
        shape = city.buildings.heights.shape

        new_top = np.full(shape, 8.0)
        new_bottom = np.full(shape, 3.0)
        mock_canopy.return_value = (new_top, new_bottom)

        vox_instance = MagicMock()
        vox_instance.generate_combined.return_value = np.zeros((*shape, 10), dtype=np.int8)
        mock_vox.return_value = vox_instance

        tree_gdf = MagicMock()
        result = update_voxcity(city, tree_gdf=tree_gdf, tree_gdf_mode="replace")
        np.testing.assert_array_equal(result.tree_canopy.top, new_top)

    @patch("voxcity.generator.update.Voxelizer")
    @patch("voxcity.geoprocessor.raster.create_canopy_grids_from_tree_gdf")
    def test_add_mode(self, mock_canopy, mock_vox):
        from voxcity.generator.update import update_voxcity

        city = _make_city()
        shape = city.buildings.heights.shape
        # Existing canopy has some values
        city.tree_canopy = CanopyGrid(
            top=np.full(shape, 5.0),
            bottom=np.full(shape, 2.0),
            meta=city.buildings.meta,
        )

        new_top = np.full(shape, 3.0)
        new_bottom = np.full(shape, 1.0)
        mock_canopy.return_value = (new_top, new_bottom)

        vox_instance = MagicMock()
        vox_instance.generate_combined.return_value = np.zeros((*shape, 10), dtype=np.int8)
        mock_vox.return_value = vox_instance

        tree_gdf = MagicMock()
        result = update_voxcity(city, tree_gdf=tree_gdf, tree_gdf_mode="add")
        # Maximum of existing (5.0) and new (3.0) → 5.0
        np.testing.assert_array_equal(result.tree_canopy.top, np.full(shape, 5.0))

    @patch("voxcity.generator.update.Voxelizer")
    def test_invalid_tree_gdf_mode_raises(self, mock_vox):
        from voxcity.generator.update import update_voxcity

        city = _make_city()
        with pytest.raises(ValueError, match="Invalid tree_gdf_mode"):
            update_voxcity(city, tree_gdf=MagicMock(), tree_gdf_mode="merge")

    @patch("voxcity.generator.update.Voxelizer")
    def test_no_rect_for_tree_raises(self, mock_vox):
        from voxcity.generator.update import update_voxcity

        city = _make_city()
        city.extras = {}  # no rectangle_vertices
        with pytest.raises(ValueError, match="rectangle_vertices"):
            update_voxcity(city, tree_gdf=MagicMock())


# ===========================================================================
# Tests for shape validation
# ===========================================================================

class TestUpdateShapeValidation:
    """Cover lines 304-312 of update.py."""

    @patch("voxcity.generator.update.Voxelizer")
    def test_shape_mismatch_raises(self, mock_vox):
        from voxcity.generator.update import update_voxcity

        city = _make_city(shape=(4, 4))
        # Provide a DEM with wrong shape
        bad_dem = np.zeros((5, 5))
        with pytest.raises(ValueError, match="Grid shape mismatch"):
            update_voxcity(city, dem=bad_dem)


# ===========================================================================
# Tests for regenerate_voxels
# ===========================================================================

class TestRegenerateVoxels:
    """Cover lines 390-430 of update.py."""

    @patch("voxcity.generator.update.Voxelizer")
    def test_regenerate_inplace(self, mock_vox):
        from voxcity.generator.update import regenerate_voxels

        city = _make_city()
        shape = city.buildings.heights.shape

        vox_instance = MagicMock()
        vox_instance.generate_combined.return_value = np.ones((*shape, 10), dtype=np.int8) * 5
        mock_vox.return_value = vox_instance

        result = regenerate_voxels(city, inplace=True)
        assert result is city  # same object
        assert np.all(result.voxels.classes == 5)

    @patch("voxcity.generator.update.Voxelizer")
    def test_regenerate_new_object(self, mock_vox):
        from voxcity.generator.update import regenerate_voxels

        city = _make_city()
        shape = city.buildings.heights.shape

        vox_instance = MagicMock()
        vox_instance.generate_combined.return_value = np.ones((*shape, 10), dtype=np.int8)
        mock_vox.return_value = vox_instance

        result = regenerate_voxels(city, inplace=False)
        assert result is not city


# ===========================================================================
# Tests for update_voxcity – passing BuildingGrid / LandCoverGrid / DemGrid / CanopyGrid
# ===========================================================================

class TestUpdateWithModelObjects:
    """Cover lines 230-295 of update.py."""

    @patch("voxcity.generator.update.Voxelizer")
    def test_pass_building_grid_directly(self, mock_vox):
        from voxcity.generator.update import update_voxcity

        city = _make_city()
        shape = city.buildings.heights.shape

        new_bg = BuildingGrid(
            heights=np.full(shape, 30.0),
            min_heights=city.buildings.min_heights,
            ids=city.buildings.ids,
            meta=city.buildings.meta,
        )

        vox_instance = MagicMock()
        vox_instance.generate_combined.return_value = np.zeros((*shape, 10), dtype=np.int8)
        mock_vox.return_value = vox_instance

        result = update_voxcity(city, buildings=new_bg)
        np.testing.assert_array_equal(result.buildings.heights, np.full(shape, 30.0))

    @patch("voxcity.generator.update.Voxelizer")
    def test_pass_land_cover_grid_object(self, mock_vox):
        from voxcity.generator.update import update_voxcity

        city = _make_city()
        shape = city.buildings.heights.shape
        new_lc = LandCoverGrid(classes=np.full(shape, 3, dtype=int), meta=city.buildings.meta)

        vox_instance = MagicMock()
        vox_instance.generate_combined.return_value = np.zeros((*shape, 10), dtype=np.int8)
        mock_vox.return_value = vox_instance

        result = update_voxcity(city, land_cover=new_lc)
        np.testing.assert_array_equal(result.land_cover.classes, np.full(shape, 3))

    @patch("voxcity.generator.update.Voxelizer")
    def test_pass_dem_grid_object(self, mock_vox):
        from voxcity.generator.update import update_voxcity

        city = _make_city()
        shape = city.buildings.heights.shape
        new_dem = DemGrid(elevation=np.full(shape, 100.0), meta=city.buildings.meta)

        vox_instance = MagicMock()
        vox_instance.generate_combined.return_value = np.zeros((*shape, 10), dtype=np.int8)
        mock_vox.return_value = vox_instance

        result = update_voxcity(city, dem=new_dem)
        np.testing.assert_array_equal(result.dem.elevation, np.full(shape, 100.0))

    @patch("voxcity.generator.update.Voxelizer")
    def test_pass_canopy_grid_object(self, mock_vox):
        from voxcity.generator.update import update_voxcity

        city = _make_city()
        shape = city.buildings.heights.shape
        new_canopy = CanopyGrid(
            top=np.full(shape, 12.0),
            bottom=np.full(shape, 4.0),
            meta=city.buildings.meta,
        )

        vox_instance = MagicMock()
        vox_instance.generate_combined.return_value = np.zeros((*shape, 10), dtype=np.int8)
        mock_vox.return_value = vox_instance

        result = update_voxcity(city, tree_canopy=new_canopy)
        np.testing.assert_array_equal(result.tree_canopy.top, np.full(shape, 12.0))
        np.testing.assert_array_equal(result.tree_canopy.bottom, np.full(shape, 4.0))
