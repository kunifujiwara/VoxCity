"""
Tests for pipeline.py OSMCanopyStrategy.build_grids (lines 542-584)
and misc pipeline branches.
"""
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import geopandas as gpd
from shapely.geometry import Point


def _make_cfg(**overrides):
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


class TestOSMCanopyStrategy:

    @patch("voxcity.geoprocessor.raster.create_canopy_grids_from_tree_gdf")
    @patch("voxcity.downloader.osm.load_tree_gdf_from_osm")
    def test_build_grids_with_trees(self, mock_load, mock_create):
        from voxcity.generator.pipeline import OSMCanopyStrategy
        cfg = _make_cfg()
        strategy = OSMCanopyStrategy(cfg)
        
        # Simulate tree data
        tree_gdf = gpd.GeoDataFrame({
            "geometry": [Point(139.755, 35.685), Point(139.758, 35.688)],
            "height": [10.0, 8.0],
        })
        mock_load.return_value = tree_gdf
        mock_create.return_value = (np.ones((5, 5)) * 10, np.ones((5, 5)) * 4)

        top, bottom = strategy.build_grids(
            cfg.rectangle_vertices, cfg.meshsize, np.ones((5, 5)), "output",
            gridvis=False,
        )
        assert top.shape == (5, 5)
        assert bottom.shape == (5, 5)
        mock_load.assert_called_once()
        mock_create.assert_called_once()

    @patch("voxcity.geoprocessor.raster.core.compute_grid_shape", return_value=(5, 5))
    @patch("voxcity.downloader.osm.load_tree_gdf_from_osm")
    def test_build_grids_no_trees(self, mock_load, mock_shape):
        from voxcity.generator.pipeline import OSMCanopyStrategy
        cfg = _make_cfg()
        strategy = OSMCanopyStrategy(cfg)

        mock_load.return_value = gpd.GeoDataFrame(columns=["geometry", "height"])

        top, bottom = strategy.build_grids(
            cfg.rectangle_vertices, cfg.meshsize, np.ones((5, 5)), "output",
        )
        assert np.all(top == 0)
        assert np.all(bottom == 0)

    @patch("voxcity.visualizer.grids.visualize_numerical_grid")
    @patch("voxcity.geoprocessor.raster.create_canopy_grids_from_tree_gdf")
    @patch("voxcity.downloader.osm.load_tree_gdf_from_osm")
    def test_build_grids_with_gridvis(self, mock_load, mock_create, mock_vis):
        from voxcity.generator.pipeline import OSMCanopyStrategy
        cfg = _make_cfg()
        strategy = OSMCanopyStrategy(cfg)

        tree_gdf = gpd.GeoDataFrame({
            "geometry": [Point(139.755, 35.685)],
            "height": [10.0],
        })
        mock_load.return_value = tree_gdf
        mock_create.return_value = (np.ones((5, 5)) * 10, np.ones((5, 5)) * 4)

        top, bottom = strategy.build_grids(
            cfg.rectangle_vertices, cfg.meshsize, np.ones((5, 5)), "output",
            gridvis=True,
        )
        mock_vis.assert_called_once()


class TestDemSourceFactory:

    def test_flat_dem(self):
        from voxcity.generator.pipeline import DemSourceFactory, FlatDemStrategy
        strategy = DemSourceFactory.create(None)
        assert isinstance(strategy, FlatDemStrategy)

    def test_flat_dem_string_none(self):
        from voxcity.generator.pipeline import DemSourceFactory, FlatDemStrategy
        strategy = DemSourceFactory.create("none")
        assert isinstance(strategy, FlatDemStrategy)

    def test_source_dem(self):
        from voxcity.generator.pipeline import DemSourceFactory, SourceDemStrategy
        strategy = DemSourceFactory.create("SRTM")
        assert isinstance(strategy, SourceDemStrategy)


class TestFlatDemStrategy:

    def test_with_land_cover(self):
        from voxcity.generator.pipeline import FlatDemStrategy
        strategy = FlatDemStrategy()
        lc = np.ones((5, 5))
        result = strategy.build_grid([], 1.0, lc, "output")
        assert result.shape == (5, 5)
        assert np.all(result == 0)

    @patch("voxcity.geoprocessor.raster.core.compute_grid_shape", return_value=(5, 5))
    def test_without_land_cover(self, mock_shape):
        from voxcity.generator.pipeline import FlatDemStrategy
        strategy = FlatDemStrategy()
        result = strategy.build_grid([], 1.0, None, "output")
        assert result.shape == (5, 5)
