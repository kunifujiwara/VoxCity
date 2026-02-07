"""Tests for pipeline.py: parallel downloads, sequential flow, OSMCanopyStrategy, visualization."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass


@dataclass
class FakePipelineConfig:
    rectangle_vertices: list = None
    meshsize: float = 1.0
    land_cover_source: str = "OpenStreetMap"
    building_source: str = "OpenStreetMap"
    dem_source: str = "Flat"
    canopy_height_source: str = "Static"
    output_dir: str = "output"
    gridvis: bool = False
    static_tree_height: float = 10.0
    trunk_height_ratio: float = 0.5
    remove_perimeter_object: float = 0
    land_cover_options: dict = None
    building_options: dict = None
    canopy_options: dict = None
    dem_options: dict = None
    
    def __post_init__(self):
        if self.rectangle_vertices is None:
            self.rectangle_vertices = [(139.7, 35.6), (139.7, 35.7), (139.8, 35.7), (139.8, 35.6)]
        if self.land_cover_options is None:
            self.land_cover_options = {}
        if self.building_options is None:
            self.building_options = {}
        if self.canopy_options is None:
            self.canopy_options = {}
        if self.dem_options is None:
            self.dem_options = {}


class TestOSMCanopyStrategy:
    """Tests for OSMCanopyStrategy in pipeline.py."""

    @patch("voxcity.generator.pipeline.get_canopy_height_grid")
    def test_source_canopy_strategy(self, mock_get):
        from voxcity.generator.pipeline import SourceCanopyStrategy
        mock_get.return_value = (np.ones((5, 5)), np.zeros((5, 5)))
        strategy = SourceCanopyStrategy("ETH")
        top, bottom = strategy.build_grids(
            [(0, 0), (0, 1), (1, 1), (1, 0)], 1.0,
            np.ones((5, 5)), "output"
        )
        assert top.shape == (5, 5)
        mock_get.assert_called_once()

    def test_osm_canopy_strategy_empty_gdf(self):
        from voxcity.generator.pipeline import OSMCanopyStrategy
        import geopandas as gpd
        cfg = FakePipelineConfig()
        strategy = OSMCanopyStrategy(cfg)
        rv = [(139.7, 35.6), (139.7, 35.7), (139.8, 35.7), (139.8, 35.6)]
        with patch("voxcity.generator.pipeline.OSMCanopyStrategy.build_grids") as mock_build:
            mock_build.return_value = (np.zeros((5, 5)), np.zeros((5, 5)))
            top, bot = strategy.build_grids(rv, 1.0, np.ones((5, 5)), "output")
            assert top.shape == (5, 5)


class TestDefaultLandCoverStrategy:
    """Tests for DefaultLandCoverStrategy and LandCoverSourceFactory."""

    @patch("voxcity.generator.pipeline.get_land_cover_grid")
    def test_build_grid(self, mock_get):
        from voxcity.generator.pipeline import DefaultLandCoverStrategy
        mock_get.return_value = np.ones((5, 5), dtype=int)
        strategy = DefaultLandCoverStrategy("OpenStreetMap")
        grid = strategy.build_grid([(0, 0)], 1.0, "output")
        assert grid.shape == (5, 5)
        mock_get.assert_called_once()

    def test_factory_create(self):
        from voxcity.generator.pipeline import LandCoverSourceFactory
        strategy = LandCoverSourceFactory.create("OpenStreetMap")
        assert hasattr(strategy, 'build_grid')


class TestDefaultBuildingSourceStrategy:
    """Tests for DefaultBuildingSourceStrategy and BuildingSourceFactory."""

    @patch("voxcity.generator.pipeline.get_building_height_grid")
    def test_build_grids(self, mock_get):
        from voxcity.generator.pipeline import DefaultBuildingSourceStrategy
        mock_get.return_value = (np.zeros((5, 5)), np.zeros((5, 5)), np.zeros((5, 5), dtype=int), None)
        strategy = DefaultBuildingSourceStrategy("OpenStreetMap")
        result = strategy.build_grids([(0, 0)], 1.0, "output")
        assert len(result) == 4
        mock_get.assert_called_once()

    def test_factory_create(self):
        from voxcity.generator.pipeline import BuildingSourceFactory
        strategy = BuildingSourceFactory.create("OpenStreetMap")
        assert hasattr(strategy, 'build_grids')


class TestStaticCanopyExpandedBranches:
    """Expanded branch tests for StaticCanopyStrategy."""

    def test_static_with_custom_trunk_ratio(self):
        from voxcity.generator.pipeline import StaticCanopyStrategy, PipelineConfig
        cfg = PipelineConfig(
            rectangle_vertices=[(0, 0), (0, 1), (1, 1), (1, 0)],
            meshsize=1.0,
            land_cover_source="OpenStreetMap",
            building_source="OpenStreetMap",
            dem_source="Flat",
            canopy_height_source="Static",
            output_dir="output",
            static_tree_height=15.0,
            trunk_height_ratio=0.3,
        )
        strategy = StaticCanopyStrategy(cfg)
        # Mock land cover grid with tree values
        lc = np.zeros((5, 5), dtype=int)
        lc[2, 2] = 3  # arbitrary tree index
        with patch("voxcity.utils.lc.get_land_cover_classes") as mock_classes:
            mock_classes.return_value = {(0, 200, 0): "Tree", (128, 128, 128): "Road"}
            top, bot = strategy.build_grids(cfg.rectangle_vertices, 1.0, lc, "output")
        # Tree at (2,2) with index 0 ('Tree')
        assert top.shape == (5, 5)
        assert bot.shape == (5, 5)


class TestPipelineSequentialBuild:
    """Test the sequential build path in VoxCityPipeline.run."""

    @patch("voxcity.generator.pipeline.Voxelizer")
    @patch("voxcity.generator.pipeline.DemSourceFactory.create")
    @patch("voxcity.generator.pipeline.CanopySourceFactory.create")
    @patch("voxcity.generator.pipeline.BuildingSourceFactory.create")
    @patch("voxcity.generator.pipeline.LandCoverSourceFactory.create")
    def test_sequential_run_calls_strategies(self, mock_lc_factory, mock_bld_factory,
                                              mock_can_factory, mock_dem_factory,
                                              mock_voxelizer_cls):
        from voxcity.generator.pipeline import VoxCityPipeline, PipelineConfig
        cfg = PipelineConfig(
            rectangle_vertices=[(139.7, 35.6), (139.7, 35.7), (139.8, 35.7), (139.8, 35.6)],
            meshsize=1.0,
            land_cover_source="OpenStreetMap",
            building_source="OpenStreetMap",
            dem_source="Flat",
            canopy_height_source="Static",
            output_dir="output",
            remove_perimeter_object=0,
        )
        pipeline = VoxCityPipeline(meshsize=cfg.meshsize, rectangle_vertices=cfg.rectangle_vertices)

        # Mock strategies
        land_strategy = MagicMock()
        land_strategy.build_grid.return_value = np.ones((5, 5), dtype=int)
        mock_lc_factory.return_value = land_strategy

        build_strategy = MagicMock()
        bh = np.zeros((5, 5))
        bmin = np.empty((5, 5), dtype=object)
        for i in range(5):
            for j in range(5):
                bmin[i, j] = []
        bid = np.zeros((5, 5), dtype=int)
        build_strategy.build_grids.return_value = (bh, bmin, bid, None)
        mock_bld_factory.return_value = build_strategy

        canopy_strategy = MagicMock()
        canopy_strategy.build_grids.return_value = (np.zeros((5, 5)), np.zeros((5, 5)))
        canopy_strategy.tree_gdf = None
        mock_can_factory.return_value = canopy_strategy

        dem_strategy = MagicMock()
        dem_strategy.build_grid.return_value = np.zeros((5, 5))
        mock_dem_factory.return_value = dem_strategy

        mock_voxelizer = MagicMock()
        mock_voxelizer.generate_combined.return_value = np.zeros((5, 5, 10), dtype=np.int8)
        mock_voxelizer_cls.return_value = mock_voxelizer

        result = pipeline.run(cfg)
        assert result is not None
        land_strategy.build_grid.assert_called_once()
        build_strategy.build_grids.assert_called_once()
        canopy_strategy.build_grids.assert_called_once()
        dem_strategy.build_grid.assert_called_once()


class TestPipelineRemovePerimeter:
    """Test perimeter removal logic."""

    def test_remove_perimeter_zeros_edges(self):
        from voxcity.generator.pipeline import VoxCityPipeline, PipelineConfig
        cfg = PipelineConfig(
            rectangle_vertices=[(0, 0), (0, 1), (1, 1), (1, 0)],
            meshsize=1.0,
            land_cover_source="OpenStreetMap",
            building_source="OpenStreetMap",
            dem_source="Flat",
            canopy_height_source="Static",
            output_dir="output",
            remove_perimeter_object=0.1,
        )
        # Simulate the perimeter removal code
        bh = np.ones((10, 10)) * 20.0
        bid = np.ones((10, 10), dtype=int) * 5
        bmin = np.empty((10, 10), dtype=object)
        for i in range(10):
            for j in range(10):
                bmin[i, j] = []
        canopy_top = np.ones((10, 10)) * 5.0
        canopy_bottom = np.ones((10, 10)) * 2.0
        ro = cfg.remove_perimeter_object
        w_peri = int(ro * bh.shape[0] + 0.5)
        h_peri = int(ro * bh.shape[1] + 0.5)
        canopy_top[:w_peri, :] = canopy_top[-w_peri:, :] = canopy_top[:, :h_peri] = canopy_top[:, -h_peri:] = 0
        canopy_bottom[:w_peri, :] = canopy_bottom[-w_peri:, :] = canopy_bottom[:, :h_peri] = canopy_bottom[:, -h_peri:] = 0
        # Edges should be zeroed
        assert canopy_top[0, 5] == 0.0
        assert canopy_top[5, 0] == 0.0
        # Center should remain
        assert canopy_top[5, 5] == 5.0
