"""Tests for pipeline.py: strategy patterns, factory methods, and VoxCityPipeline.run logic."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from voxcity.generator.pipeline import (
    PipelineConfig,
    VoxCityPipeline,
    LandCoverSourceFactory,
    BuildingSourceFactory,
    CanopySourceFactory,
    DemSourceFactory,
    FlatDemStrategy,
    StaticCanopyStrategy,
    SourceCanopyStrategy,
    SourceDemStrategy,
    DefaultLandCoverStrategy,
    DefaultBuildingSourceStrategy,
)


def _rect():
    return [(139.7, 35.6), (139.7, 35.7), (139.8, 35.7), (139.8, 35.6)]


# ---------------------------------------------------------------------------
# DemSourceFactory - additional branch coverage
# ---------------------------------------------------------------------------
class TestDemSourceFactoryBranches:
    def test_none_source_returns_flat(self):
        strat = DemSourceFactory.create(None)
        assert isinstance(strat, FlatDemStrategy)

    def test_empty_string_returns_flat(self):
        strat = DemSourceFactory.create("")
        assert isinstance(strat, FlatDemStrategy)

    def test_none_string_returns_flat(self):
        strat = DemSourceFactory.create("none")
        assert isinstance(strat, FlatDemStrategy)

    def test_null_string_returns_flat(self):
        strat = DemSourceFactory.create("null")
        assert isinstance(strat, FlatDemStrategy)

    def test_flat_string_returns_flat(self):
        strat = DemSourceFactory.create("Flat")
        assert isinstance(strat, FlatDemStrategy)

    def test_other_source_returns_source(self):
        strat = DemSourceFactory.create("OpenTopography")
        assert isinstance(strat, SourceDemStrategy)


class TestFlatDemStrategyBranches:
    def test_with_land_cover_grid(self):
        strat = FlatDemStrategy()
        lc = np.ones((10, 10), dtype=float)
        result = strat.build_grid(_rect(), 1.0, lc, "output")
        assert result.shape == (10, 10)
        assert np.all(result == 0)

    def test_with_none_land_cover_grid(self):
        strat = FlatDemStrategy()
        with patch("voxcity.geoprocessor.raster.core.compute_grid_shape", return_value=(5, 5)):
            result = strat.build_grid(_rect(), 1.0, None, "output")
        assert result.shape == (5, 5)
        assert np.all(result == 0)


class TestSourceDemStrategy:
    @patch("voxcity.generator.pipeline.get_dem_grid")
    def test_success(self, mock_get_dem):
        mock_get_dem.return_value = np.ones((10, 10))
        strat = SourceDemStrategy("OpenTopography")
        result = strat.build_grid(_rect(), 1.0, None, "output")
        assert result.shape == (10, 10)

    def test_terrain_gdf_path(self):
        strat = SourceDemStrategy("OpenTopography")
        mock_gdf = MagicMock()
        with patch("voxcity.geoprocessor.raster.create_dem_grid_from_gdf_polygon") as mock_create:
            mock_create.return_value = np.ones((8, 8))
            result = strat.build_grid(_rect(), 1.0, None, "output", terrain_gdf=mock_gdf)
        assert result.shape == (8, 8)

    @patch("voxcity.generator.pipeline.get_dem_grid", side_effect=RuntimeError("API error"))
    def test_fallback_on_error_with_lc(self, mock_get_dem):
        strat = SourceDemStrategy("FailingSource")
        lc = np.ones((10, 10), dtype=float)
        result = strat.build_grid(_rect(), 1.0, lc, "output")
        assert result.shape == (10, 10)
        assert np.all(result == 0)

    @patch("voxcity.generator.pipeline.get_dem_grid", side_effect=RuntimeError("API error"))
    def test_fallback_on_error_no_lc(self, mock_get_dem):
        strat = SourceDemStrategy("FailingSource")
        with patch("voxcity.geoprocessor.raster.core.compute_grid_shape", return_value=(5, 5)):
            result = strat.build_grid(_rect(), 1.0, None, "output")
        assert result.shape == (5, 5)


# ---------------------------------------------------------------------------
# CanopySourceFactory
# ---------------------------------------------------------------------------
class TestCanopySourceFactoryBranches:
    def test_static_creates_static_strategy(self):
        cfg = PipelineConfig(rectangle_vertices=_rect(), meshsize=1.0, land_cover_source="OpenStreetMap")
        strat = CanopySourceFactory.create("Static", cfg)
        assert isinstance(strat, StaticCanopyStrategy)

    def test_osm_source(self):
        cfg = PipelineConfig(rectangle_vertices=_rect(), meshsize=1.0, land_cover_source="OpenStreetMap")
        strat = CanopySourceFactory.create("OpenStreetMap", cfg)
        from voxcity.generator.pipeline import OSMCanopyStrategy
        assert isinstance(strat, OSMCanopyStrategy)

    def test_other_source(self):
        cfg = PipelineConfig(rectangle_vertices=_rect(), meshsize=1.0, land_cover_source="OpenStreetMap")
        strat = CanopySourceFactory.create("GEE", cfg)
        assert isinstance(strat, SourceCanopyStrategy)


class TestStaticCanopyStrategyBranches:
    def test_basic(self):
        cfg = PipelineConfig(
            rectangle_vertices=_rect(), meshsize=1.0,
            land_cover_source="OpenStreetMap",
            static_tree_height=15.0,
            trunk_height_ratio=0.5,
        )
        strat = StaticCanopyStrategy(cfg)
        # Land cover grid with tree index
        lc = np.zeros((5, 5), dtype=int)
        from voxcity.utils.lc import get_land_cover_classes
        classes = get_land_cover_classes("OpenStreetMap")
        class_to_int = {name: i for i, name in enumerate(classes.values())}
        tree_labels = ["Tree", "Trees", "Tree Canopy"]
        tree_idx = [class_to_int[label] for label in tree_labels if label in class_to_int]
        if tree_idx:
            lc[2, 2] = tree_idx[0]
        
        top, bottom = strat.build_grids(_rect(), 1.0, lc, "output")
        assert top.shape == (5, 5)
        if tree_idx:
            assert top[2, 2] == 15.0
            assert bottom[2, 2] == 7.5  # 15.0 * 0.5

    def test_default_height(self):
        cfg = PipelineConfig(
            rectangle_vertices=_rect(), meshsize=1.0,
            land_cover_source="OpenStreetMap",
            static_tree_height=None,
            trunk_height_ratio=None,
        )
        strat = StaticCanopyStrategy(cfg)
        lc = np.zeros((3, 3), dtype=int)
        top, bottom = strat.build_grids(_rect(), 1.0, lc, "output")
        assert top.shape == (3, 3)


# ---------------------------------------------------------------------------
# DefaultLandCoverStrategy / DefaultBuildingSourceStrategy 
# ---------------------------------------------------------------------------
class TestDefaultStrategies:
    def test_land_cover_factory(self):
        strat = LandCoverSourceFactory.create("OpenStreetMap")
        assert isinstance(strat, DefaultLandCoverStrategy)
        assert strat.source == "OpenStreetMap"

    def test_building_factory(self):
        strat = BuildingSourceFactory.create("OpenStreetMap")
        assert isinstance(strat, DefaultBuildingSourceStrategy)
        assert strat.source == "OpenStreetMap"


# ---------------------------------------------------------------------------
# VoxCityPipeline.assemble_voxcity
# ---------------------------------------------------------------------------
class TestVoxCityPipelineAssemble:
    def test_assemble_creates_voxcity(self):
        pipeline = VoxCityPipeline(
            rectangle_vertices=_rect(),
            meshsize=1.0,
        )
        vox_grid = np.zeros((5, 5, 10), dtype=np.int8)
        vox_grid[:, :, 0] = 1
        bh = np.zeros((5, 5))
        bmin = np.empty((5, 5), dtype=object)
        bid = np.zeros((5, 5), dtype=int)
        lc = np.ones((5, 5), dtype=int)
        dem = np.zeros((5, 5))
        result = pipeline.assemble_voxcity(
            voxcity_grid=vox_grid,
            building_height_grid=bh,
            building_min_height_grid=bmin,
            building_id_grid=bid,
            land_cover_grid=lc,
            dem_grid=dem,
        )
        from voxcity.models import VoxCity
        assert isinstance(result, VoxCity)
        assert result.voxels.classes.shape == (5, 5, 10)

    def test_assemble_with_canopy(self):
        pipeline = VoxCityPipeline(rectangle_vertices=_rect(), meshsize=1.0)
        vox_grid = np.zeros((5, 5, 10), dtype=np.int8)
        bh = np.zeros((5, 5))
        bmin = np.empty((5, 5), dtype=object)
        bid = np.zeros((5, 5), dtype=int)
        lc = np.ones((5, 5), dtype=int)
        dem = np.zeros((5, 5))
        canopy_top = np.ones((5, 5)) * 10.0
        canopy_bottom = np.ones((5, 5)) * 5.0
        result = pipeline.assemble_voxcity(
            voxcity_grid=vox_grid,
            building_height_grid=bh,
            building_min_height_grid=bmin,
            building_id_grid=bid,
            land_cover_grid=lc,
            dem_grid=dem,
            canopy_height_top=canopy_top,
            canopy_height_bottom=canopy_bottom,
        )
        assert result.tree_canopy is not None
        np.testing.assert_allclose(result.tree_canopy.top, 10.0)

    def test_assemble_with_extras(self):
        pipeline = VoxCityPipeline(rectangle_vertices=_rect(), meshsize=1.0)
        vox_grid = np.zeros((3, 3, 5), dtype=np.int8)
        bh = np.zeros((3, 3))
        bmin = np.empty((3, 3), dtype=object)
        bid = np.zeros((3, 3), dtype=int)
        lc = np.ones((3, 3), dtype=int)
        dem = np.zeros((3, 3))
        result = pipeline.assemble_voxcity(
            voxcity_grid=vox_grid,
            building_height_grid=bh,
            building_min_height_grid=bmin,
            building_id_grid=bid,
            land_cover_grid=lc,
            dem_grid=dem,
            extras={"custom_key": "custom_value"},
        )
        assert result.extras["custom_key"] == "custom_value"
        assert result.extras["rectangle_vertices"] == _rect()
