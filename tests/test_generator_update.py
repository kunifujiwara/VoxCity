"""Tests for voxcity.generator.update module."""
import pytest
import numpy as np

from voxcity.models import (
    GridMetadata,
    BuildingGrid,
    LandCoverGrid,
    DemGrid,
    VoxelGrid,
    CanopyGrid,
    VoxCity,
)
from voxcity.generator.update import update_voxcity, regenerate_voxels


class TestVoxCityFixtures:
    """Fixtures for VoxCity test objects."""
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample grid metadata."""
        return GridMetadata(
            crs="EPSG:4326",
            bounds=(139.7, 35.6, 139.71, 35.61),
            meshsize=1.0
        )
    
    @pytest.fixture
    def sample_building_grid(self, sample_metadata):
        """Create sample building grid."""
        shape = (10, 10)
        heights = np.zeros(shape, dtype=np.float64)
        heights[2:4, 2:4] = 20.0
        heights[6:8, 6:8] = 30.0
        
        min_heights = np.empty(shape, dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if heights[i, j] > 0:
                    min_heights[i, j] = [(0.0, heights[i, j])]
                else:
                    min_heights[i, j] = []
        
        ids = np.zeros(shape, dtype=np.int32)
        ids[2:4, 2:4] = 1
        ids[6:8, 6:8] = 2
        
        return BuildingGrid(
            heights=heights,
            min_heights=min_heights,
            ids=ids,
            meta=sample_metadata
        )
    
    @pytest.fixture
    def sample_land_cover_grid(self, sample_metadata):
        """Create sample land cover grid."""
        shape = (10, 10)
        classes = np.ones(shape, dtype=np.int8) * 5  # Default to grass
        classes[2:4, 2:4] = 13  # Building
        classes[6:8, 6:8] = 13  # Building
        return LandCoverGrid(classes=classes, meta=sample_metadata)
    
    @pytest.fixture
    def sample_dem_grid(self, sample_metadata):
        """Create sample DEM grid."""
        shape = (10, 10)
        elevation = np.zeros(shape, dtype=np.float32)
        return DemGrid(elevation=elevation, meta=sample_metadata)
    
    @pytest.fixture
    def sample_canopy_grid(self, sample_metadata):
        """Create sample canopy grid."""
        shape = (10, 10)
        top = np.zeros(shape, dtype=np.float32)
        top[4:6, 4:6] = 15.0  # Trees
        bottom = np.zeros(shape, dtype=np.float32)
        bottom[4:6, 4:6] = 5.0
        return CanopyGrid(top=top, bottom=bottom, meta=sample_metadata)
    
    @pytest.fixture
    def sample_voxel_grid(self, sample_metadata):
        """Create sample voxel grid."""
        voxel_shape = (10, 10, 50)  # x, y, z
        classes = np.zeros(voxel_shape, dtype=np.int8)
        return VoxelGrid(classes=classes, meta=sample_metadata)
    
    @pytest.fixture
    def sample_voxcity(self, sample_voxel_grid, sample_building_grid, 
                       sample_land_cover_grid, sample_dem_grid, sample_canopy_grid):
        """Create a complete sample VoxCity object."""
        return VoxCity(
            voxels=sample_voxel_grid,
            buildings=sample_building_grid,
            land_cover=sample_land_cover_grid,
            dem=sample_dem_grid,
            tree_canopy=sample_canopy_grid,
            extras={
                "rectangle_vertices": [
                    (139.7, 35.6),
                    (139.7, 35.61),
                    (139.71, 35.61),
                    (139.71, 35.6)
                ],
                "land_cover_source": "OpenStreetMap"
            }
        )


class TestUpdateVoxcityBuildingHeights(TestVoxCityFixtures):
    """Tests for updating building heights."""
    
    def test_update_building_heights(self, sample_voxcity):
        """Test updating building heights array."""
        new_heights = sample_voxcity.buildings.heights.copy()
        new_heights[2:4, 2:4] = 50.0  # Increase height
        
        result = update_voxcity(sample_voxcity, building_heights=new_heights)
        
        # Check heights were updated
        assert result.buildings.heights[2, 2] == 50.0
        # Other heights should remain
        assert result.buildings.heights[6, 6] == 30.0
        
        # Original should be unchanged (not inplace)
        assert sample_voxcity.buildings.heights[2, 2] == 20.0
    
    def test_update_building_heights_inplace(self, sample_voxcity):
        """Test inplace update of building heights."""
        original_id = id(sample_voxcity)
        new_heights = sample_voxcity.buildings.heights.copy()
        new_heights[2:4, 2:4] = 60.0
        
        result = update_voxcity(sample_voxcity, building_heights=new_heights, inplace=True)
        
        # Should return same object
        assert id(result) == original_id
        # Heights should be updated in the original object
        assert result.buildings.heights[2, 2] == 60.0


class TestUpdateVoxcityLandCover(TestVoxCityFixtures):
    """Tests for updating land cover."""
    
    def test_update_land_cover_array(self, sample_voxcity):
        """Test updating land cover with numpy array."""
        new_lc = sample_voxcity.land_cover.classes.copy()
        new_lc[0:2, 0:2] = 9  # Water
        
        result = update_voxcity(sample_voxcity, land_cover=new_lc)
        
        assert result.land_cover.classes[0, 0] == 9
        assert result.land_cover.classes[5, 5] == 5  # Original preserved
    
    def test_update_land_cover_grid(self, sample_voxcity, sample_metadata):
        """Test updating land cover with LandCoverGrid object."""
        new_classes = np.ones((10, 10), dtype=np.int8) * 7  # Forest
        new_lc = LandCoverGrid(classes=new_classes, meta=sample_metadata)
        
        result = update_voxcity(sample_voxcity, land_cover=new_lc)
        
        assert np.all(result.land_cover.classes == 7)


class TestUpdateVoxcityDem(TestVoxCityFixtures):
    """Tests for updating DEM."""
    
    def test_update_dem_array(self, sample_voxcity):
        """Test updating DEM with numpy array."""
        new_dem = sample_voxcity.dem.elevation.copy()
        new_dem[:] = 10.0  # Raise elevation
        
        result = update_voxcity(sample_voxcity, dem=new_dem)
        
        assert result.dem.elevation[0, 0] == 10.0
    
    def test_update_dem_grid(self, sample_voxcity, sample_metadata):
        """Test updating DEM with DemGrid object."""
        new_elev = np.ones((10, 10), dtype=np.float32) * 25.0
        new_dem = DemGrid(elevation=new_elev, meta=sample_metadata)
        
        result = update_voxcity(sample_voxcity, dem=new_dem)
        
        assert np.all(result.dem.elevation == 25.0)


class TestUpdateVoxcityCanopy(TestVoxCityFixtures):
    """Tests for updating tree canopy."""
    
    def test_update_canopy_top(self, sample_voxcity):
        """Test updating canopy top heights."""
        new_top = sample_voxcity.tree_canopy.top.copy()
        new_top[0:2, 0:2] = 20.0
        
        result = update_voxcity(sample_voxcity, canopy_top=new_top)
        
        assert result.tree_canopy.top[0, 0] == 20.0
    
    def test_update_canopy_bottom(self, sample_voxcity):
        """Test updating canopy bottom heights."""
        new_bottom = sample_voxcity.tree_canopy.bottom.copy()
        new_bottom[4:6, 4:6] = 8.0
        
        result = update_voxcity(sample_voxcity, canopy_bottom=new_bottom)
        
        assert result.tree_canopy.bottom[4, 4] == 8.0
    
    def test_update_canopy_grid(self, sample_voxcity, sample_metadata):
        """Test updating with CanopyGrid object."""
        new_top = np.ones((10, 10), dtype=np.float32) * 12.0
        new_bottom = np.ones((10, 10), dtype=np.float32) * 4.0
        new_canopy = CanopyGrid(top=new_top, bottom=new_bottom, meta=sample_metadata)
        
        result = update_voxcity(sample_voxcity, tree_canopy=new_canopy)
        
        assert np.all(result.tree_canopy.top == 12.0)
        assert np.all(result.tree_canopy.bottom == 4.0)


class TestUpdateVoxcityBuildingGrid(TestVoxCityFixtures):
    """Tests for updating with complete BuildingGrid."""
    
    def test_update_with_building_grid(self, sample_voxcity, sample_metadata):
        """Test updating with complete BuildingGrid object."""
        new_heights = np.zeros((10, 10), dtype=np.float64)
        new_heights[5:7, 5:7] = 100.0
        
        new_min_heights = np.empty((10, 10), dtype=object)
        for i in range(10):
            for j in range(10):
                if new_heights[i, j] > 0:
                    new_min_heights[i, j] = [(0.0, new_heights[i, j])]
                else:
                    new_min_heights[i, j] = []
        
        new_ids = np.zeros((10, 10), dtype=np.int32)
        new_ids[5:7, 5:7] = 999
        
        new_buildings = BuildingGrid(
            heights=new_heights,
            min_heights=new_min_heights,
            ids=new_ids,
            meta=sample_metadata
        )
        
        result = update_voxcity(sample_voxcity, buildings=new_buildings)
        
        assert result.buildings.heights[5, 5] == 100.0
        assert result.buildings.ids[5, 5] == 999
        # Old buildings should be gone
        assert result.buildings.heights[2, 2] == 0.0


class TestUpdateVoxcityMultiple(TestVoxCityFixtures):
    """Tests for updating multiple components at once."""
    
    def test_update_multiple_grids(self, sample_voxcity):
        """Test updating multiple grids simultaneously."""
        new_heights = sample_voxcity.buildings.heights.copy()
        new_heights[2:4, 2:4] = 45.0
        
        new_dem = sample_voxcity.dem.elevation.copy()
        new_dem[:] = 5.0
        
        new_lc = sample_voxcity.land_cover.classes.copy()
        new_lc[8:10, 8:10] = 9  # Water
        
        result = update_voxcity(
            sample_voxcity,
            building_heights=new_heights,
            dem=new_dem,
            land_cover=new_lc
        )
        
        assert result.buildings.heights[2, 2] == 45.0
        assert result.dem.elevation[0, 0] == 5.0
        assert result.land_cover.classes[9, 9] == 9


class TestUpdateVoxcityValidation(TestVoxCityFixtures):
    """Tests for validation in update_voxcity."""
    
    def test_shape_mismatch_raises_error(self, sample_voxcity):
        """Test that mismatched shapes raise ValueError."""
        wrong_shape_heights = np.zeros((5, 5), dtype=np.float64)  # Wrong shape
        
        with pytest.raises(ValueError, match="shape mismatch"):
            update_voxcity(sample_voxcity, building_heights=wrong_shape_heights)
    
    def test_invalid_tree_gdf_mode_raises_error(self, sample_voxcity):
        """Test that invalid tree_gdf_mode raises ValueError."""
        # This test requires a tree_gdf, so we skip if dependencies aren't met
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            
            tree_gdf = gpd.GeoDataFrame({
                'top_height': [10.0],
                'bottom_height': [3.0],
                'crown_diameter': [5.0],
                'geometry': [Point(139.705, 35.605)]
            }, crs="EPSG:4326")
            
            with pytest.raises(ValueError, match="Invalid tree_gdf_mode"):
                update_voxcity(sample_voxcity, tree_gdf=tree_gdf, tree_gdf_mode="invalid")
        except ImportError:
            pytest.skip("geopandas not available")


class TestRegenerateVoxels(TestVoxCityFixtures):
    """Tests for regenerate_voxels function."""
    
    def test_regenerate_voxels_basic(self, sample_voxcity):
        """Test basic voxel regeneration."""
        result = regenerate_voxels(sample_voxcity)
        
        # Should return a VoxCity object
        assert isinstance(result, VoxCity)
        
        # Should have same building data
        assert np.array_equal(result.buildings.heights, sample_voxcity.buildings.heights)
    
    def test_regenerate_voxels_inplace(self, sample_voxcity):
        """Test inplace voxel regeneration."""
        original_id = id(sample_voxcity)
        
        result = regenerate_voxels(sample_voxcity, inplace=True)
        
        assert id(result) == original_id
    
    def test_regenerate_with_custom_source(self, sample_voxcity):
        """Test regeneration with custom land cover source."""
        result = regenerate_voxels(
            sample_voxcity, 
            land_cover_source="Urbanwatch"
        )
        
        assert isinstance(result, VoxCity)


class TestRegenerateVoxelsIsAFunctionOfTheComponentGrids(TestVoxCityFixtures):
    """The rebuild claim the app's LOD2 routing stands on.

    ``regenerate_voxels`` rebuilds the voxel grid from the 2.5-D component
    grids alone — whatever detail the *existing* voxel grid carried does not
    survive. That is precisely why the app's nDSM pipeline routes LOD2
    (mesh-voxelized CityGML) models around this function and through
    ``voxcitygml.reapply_canopy`` instead
    (``app/backend/test_ndsm_lod2_geometry.py`` pins the routing with the real
    dataset; ``reapply_canopy``'s preservation is pinned in VoxCityGML's own
    ``tests/test_reapply_canopy.py``). This class pins the library half with a
    synthetic fixture: a stepped, LOD2-style roof stamped into the voxels of a
    model whose height grid says "flat 20 m block" must come back as a flat
    extrusion, identical to what an empty prior grid produces.
    """

    BUILDING_CODE = -3
    # The fixture's first footprint: rows 2..4, cols 2..4, heights grid 20 m.
    FOOT = (slice(2, 4), slice(2, 4))
    #: metres of building per footprint column in the stepped "LOD2" roof —
    #: deliberately all different, and none equal to the 20 m the height grid
    #: declares, so surviving detail cannot masquerade as the rebuilt slab.
    STEPS = {(2, 2): 19, (2, 3): 14, (3, 2): 9, (3, 3): 4}

    def _column_heights(self, classes):
        """Building-voxel count per column, meshsize 1 → metres."""
        return (np.asarray(classes) == self.BUILDING_CODE).sum(axis=2)

    def _with_stepped_roof(self, city):
        """Overwrite the fixture's voxels with a mesh-style stepped roof."""
        vox = np.zeros((10, 10, 50), dtype=np.int8)
        vox[:, :, 0] = -1  # ground
        for (i, j), h in self.STEPS.items():
            vox[i, j, 1:1 + h] = self.BUILDING_CODE
        city.voxels = VoxelGrid(classes=vox, meta=city.voxels.meta)
        return city

    def test_the_fixture_roof_is_genuinely_stepped(self, sample_voxcity):
        """Guard: four distinct column heights, or 'the steps are gone' below
        would be vacuous."""
        city = self._with_stepped_roof(sample_voxcity)
        cols = self._column_heights(city.voxels.classes)
        assert sorted(cols[self.FOOT].ravel().tolist()) == sorted(self.STEPS.values())
        assert len(set(self.STEPS.values())) == 4

    def test_regenerate_replaces_the_stepped_roof_with_a_flat_extrusion(self, sample_voxcity):
        city = self._with_stepped_roof(sample_voxcity)

        result = regenerate_voxels(city)

        cols = self._column_heights(result.voxels.classes)
        heights_of_building_columns = sorted(set(cols[cols > 0].ravel().tolist()))
        # Two flat blocks — 20 m and 30 m from the *height grids* — and not a
        # single one of the stepped 4/9/14/19 m columns.
        assert heights_of_building_columns == [20, 30], (
            f"building columns are {heights_of_building_columns} m tall; the "
            "component grids describe flat 20 m and 30 m extrusions, so any "
            "other value means detail from the prior voxel grid leaked "
            "through the rebuild — or the rebuild never happened")

    def test_the_result_is_identical_whatever_the_prior_voxel_grid_held(self):
        """The function-of-the-grids claim, asserted bit-exactly."""
        import copy

        base = _build_sample_city()
        stepped = self._with_stepped_roof(copy.deepcopy(base))
        empty = copy.deepcopy(base)
        empty.voxels = VoxelGrid(
            classes=np.zeros((10, 10, 50), dtype=np.int8), meta=base.voxels.meta
        )

        out_stepped = regenerate_voxels(stepped)
        out_empty = regenerate_voxels(empty)

        assert np.array_equal(
            np.asarray(out_stepped.voxels.classes),
            np.asarray(out_empty.voxels.classes),
        ), (
            "regenerate_voxels produced different voxels for the same "
            "component grids — the prior voxel grid leaked into the rebuild"
        )


def _build_sample_city():
    """Module-level twin of the ``sample_voxcity`` fixture, for tests that
    need two independent copies without fixture plumbing."""
    meta = GridMetadata(crs="EPSG:4326", bounds=(139.7, 35.6, 139.71, 35.61), meshsize=1.0)
    shape = (10, 10)
    heights = np.zeros(shape, dtype=np.float64)
    heights[2:4, 2:4] = 20.0
    heights[6:8, 6:8] = 30.0
    min_heights = np.empty(shape, dtype=object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            min_heights[i, j] = [(0.0, heights[i, j])] if heights[i, j] > 0 else []
    ids = np.zeros(shape, dtype=np.int32)
    ids[2:4, 2:4] = 1
    ids[6:8, 6:8] = 2
    classes = np.ones(shape, dtype=np.int8) * 5
    classes[2:4, 2:4] = 13
    classes[6:8, 6:8] = 13
    top = np.zeros(shape, dtype=np.float32)
    top[4:6, 4:6] = 15.0
    bottom = np.zeros(shape, dtype=np.float32)
    bottom[4:6, 4:6] = 5.0
    return VoxCity(
        voxels=VoxelGrid(classes=np.zeros((10, 10, 50), dtype=np.int8), meta=meta),
        buildings=BuildingGrid(heights=heights, min_heights=min_heights, ids=ids, meta=meta),
        land_cover=LandCoverGrid(classes=classes, meta=meta),
        dem=DemGrid(elevation=np.zeros(shape, dtype=np.float32), meta=meta),
        tree_canopy=CanopyGrid(top=top, bottom=bottom, meta=meta),
        extras={
            "rectangle_vertices": [
                (139.7, 35.6), (139.7, 35.61), (139.71, 35.61), (139.71, 35.6)
            ],
            "land_cover_source": "OpenStreetMap",
        },
    )


class TestExtrasHandling(TestVoxCityFixtures):
    """Tests for extras dictionary handling."""
    
    def test_extras_preserved(self, sample_voxcity):
        """Test that extras are preserved after update."""
        # Add custom extras
        sample_voxcity.extras["custom_key"] = "custom_value"
        
        new_heights = sample_voxcity.buildings.heights.copy()
        result = update_voxcity(sample_voxcity, building_heights=new_heights)
        
        assert "custom_key" in result.extras
        assert result.extras["custom_key"] == "custom_value"
    
    def test_canopy_stored_in_extras(self, sample_voxcity):
        """Test that canopy data is stored in extras."""
        new_top = np.ones((10, 10), dtype=np.float32) * 18.0
        
        result = update_voxcity(sample_voxcity, canopy_top=new_top)
        
        assert "canopy_top" in result.extras
        assert np.array_equal(result.extras["canopy_top"], new_top)
