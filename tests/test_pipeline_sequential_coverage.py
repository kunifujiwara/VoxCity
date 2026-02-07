"""
Tests for pipeline.py remaining coverage:
  - sequential mode (lines 121-158)
  - CanopySourceFactory / OSMCanopyStrategy / DemSourceStrategy
  - PipelineConfig / VoxCityPipeline.run misc branches
"""
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, PropertyMock


def _make_cfg(**overrides):
    """Create a PipelineConfig-like object."""
    defaults = dict(
        meshsize=1.0,
        rectangle_vertices=[(139.75, 35.68), (139.75, 35.69), (139.76, 35.69), (139.76, 35.68)],
        crs="EPSG:4326",
        land_cover_source="OpenStreetMap",
        building_source="OpenStreetMap",
        canopy_height_source="OpenStreetMap",
        dem_source="OpenStreetMap",
        output_dir="output",
        gridvis=False,
        parallel_download=False,
        remove_perimeter_object=None,
        trunk_height_ratio=0.3,
        land_cover_options={},
        building_options={},
        canopy_options={},
        dem_options={},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestPipelineSequentialMode:

    @patch("voxcity.generator.pipeline.Voxelizer")
    @patch("voxcity.generator.pipeline.DemSourceFactory")
    @patch("voxcity.generator.pipeline.CanopySourceFactory")
    @patch("voxcity.generator.pipeline.BuildingSourceFactory")
    @patch("voxcity.generator.pipeline.LandCoverSourceFactory")
    def test_sequential_run(self, mock_lc_f, mock_build_f, mock_canopy_f, mock_dem_f, mock_voxelizer):
        """Sequential mode (parallel_download=False) runs all strategies in order."""
        cfg = _make_cfg(parallel_download=False)

        # Land cover
        lc_strategy = MagicMock()
        lc_strategy.build_grid.return_value = np.ones((5, 5), dtype=np.int8)
        mock_lc_f.create.return_value = lc_strategy

        # Building
        build_strategy = MagicMock()
        bh = np.zeros((5, 5))
        bmin = np.empty((5, 5), dtype=object)
        for i in range(5):
            for j in range(5):
                bmin[i, j] = []
        bid = np.zeros((5, 5), dtype=int)
        build_strategy.build_grids.return_value = (bh, bmin, bid, None)
        mock_build_f.create.return_value = build_strategy

        # Canopy
        canopy_strategy = MagicMock()
        canopy_strategy.build_grids.return_value = (np.zeros((5, 5)), np.zeros((5, 5)))
        mock_canopy_f.create.return_value = canopy_strategy

        # DEM
        dem_strategy = MagicMock()
        dem_strategy.build_grid.return_value = np.zeros((5, 5))
        mock_dem_f.create.return_value = dem_strategy

        # Voxelizer
        voxelizer_inst = MagicMock()
        vox_grid = np.zeros((5, 5, 4), dtype=np.int8)
        voxelizer_inst.generate_combined.return_value = vox_grid
        mock_voxelizer.return_value = voxelizer_inst

        from voxcity.generator.pipeline import VoxCityPipeline
        pipeline = VoxCityPipeline(meshsize=1.0, rectangle_vertices=cfg.rectangle_vertices)
        
        # Mock get_last_effective_land_cover_source
        with patch("voxcity.generator.grids.get_last_effective_land_cover_source", return_value="OpenStreetMap"):
            result = pipeline.run(cfg)

        assert result is not None
        lc_strategy.build_grid.assert_called_once()
        build_strategy.build_grids.assert_called_once()
        canopy_strategy.build_grids.assert_called_once()
        dem_strategy.build_grid.assert_called_once()


class TestCanopySourceFactory:

    def test_static_strategy(self):
        from voxcity.generator.pipeline import CanopySourceFactory, StaticCanopyStrategy
        cfg = _make_cfg(canopy_height_source="Static")
        strategy = CanopySourceFactory.create("Static", cfg)
        assert isinstance(strategy, StaticCanopyStrategy)

    def test_osm_strategy(self):
        from voxcity.generator.pipeline import CanopySourceFactory, OSMCanopyStrategy
        cfg = _make_cfg(canopy_height_source="OpenStreetMap")
        strategy = CanopySourceFactory.create("OpenStreetMap", cfg)
        assert isinstance(strategy, OSMCanopyStrategy)

    def test_source_strategy(self):
        from voxcity.generator.pipeline import CanopySourceFactory, SourceCanopyStrategy
        cfg = _make_cfg(canopy_height_source="ETH")
        strategy = CanopySourceFactory.create("ETH", cfg)
        assert isinstance(strategy, SourceCanopyStrategy)


class TestRemovePerimeterObject:

    @patch("voxcity.generator.pipeline.Voxelizer")
    @patch("voxcity.generator.pipeline.DemSourceFactory")
    @patch("voxcity.generator.pipeline.CanopySourceFactory")
    @patch("voxcity.generator.pipeline.BuildingSourceFactory")
    @patch("voxcity.generator.pipeline.LandCoverSourceFactory")
    def test_perimeter_removal(self, mock_lc_f, mock_build_f, mock_canopy_f, mock_dem_f, mock_voxelizer):
        """remove_perimeter_object > 0 strips border cells."""
        cfg = _make_cfg(parallel_download=False, remove_perimeter_object=0.1)

        lc_strategy = MagicMock()
        lc_strategy.build_grid.return_value = np.ones((10, 10), dtype=np.int8)
        mock_lc_f.create.return_value = lc_strategy

        build_strategy = MagicMock()
        bh = np.full((10, 10), 5.0)
        bmin = np.empty((10, 10), dtype=object)
        for i in range(10):
            for j in range(10):
                bmin[i, j] = []
        bid = np.ones((10, 10), dtype=int)
        build_strategy.build_grids.return_value = (bh, bmin, bid, None)
        mock_build_f.create.return_value = build_strategy

        canopy_strategy = MagicMock()
        canopy_strategy.build_grids.return_value = (np.ones((10, 10)), np.ones((10, 10)))
        mock_canopy_f.create.return_value = canopy_strategy

        dem_strategy = MagicMock()
        dem_strategy.build_grid.return_value = np.zeros((10, 10))
        mock_dem_f.create.return_value = dem_strategy

        voxelizer_inst = MagicMock()
        voxelizer_inst.generate_combined.return_value = np.zeros((10, 10, 5), dtype=np.int8)
        mock_voxelizer.return_value = voxelizer_inst

        from voxcity.generator.pipeline import VoxCityPipeline
        pipeline = VoxCityPipeline(meshsize=1.0, rectangle_vertices=cfg.rectangle_vertices)
        
        with patch("voxcity.generator.grids.get_last_effective_land_cover_source", return_value="OpenStreetMap"):
            result = pipeline.run(cfg)

        assert result is not None
