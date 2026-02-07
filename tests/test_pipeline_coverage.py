"""
Comprehensive tests for voxcity.generator.pipeline to improve coverage.
Covers: VoxCityPipeline, factory classes, strategy constructors.
"""

import numpy as np
import pytest

from voxcity.models import PipelineConfig, VoxCity, VoxelGrid, GridMetadata, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid
from voxcity.generator.pipeline import (
    VoxCityPipeline,
    LandCoverSourceFactory,
    BuildingSourceFactory,
    CanopySourceFactory,
    DemSourceFactory,
    FlatDemStrategy,
    StaticCanopyStrategy,
)


# --- Pipeline init and helper methods ---

class TestVoxCityPipelineInit:
    def test_basic_init(self):
        verts = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        p = VoxCityPipeline(meshsize=10.0, rectangle_vertices=verts)
        assert p.meshsize == 10.0
        assert p.crs == "EPSG:4326"

    def test_bounds(self):
        verts = [(1.0, 2.0), (1.0, 4.0), (3.0, 4.0), (3.0, 2.0)]
        p = VoxCityPipeline(meshsize=10.0, rectangle_vertices=verts)
        west, south, east, north = p._bounds()
        assert west == 1.0
        assert south == 2.0
        assert east == 3.0
        assert north == 4.0

    def test_meta(self):
        verts = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        p = VoxCityPipeline(meshsize=5.0, rectangle_vertices=verts)
        meta = p._meta()
        assert meta.meshsize == 5.0
        assert meta.crs == "EPSG:4326"
        assert meta.bounds == (0.0, 0.0, 1.0, 1.0)


class TestVoxCityPipelineAssemble:
    def test_basic_assemble(self):
        verts = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        p = VoxCityPipeline(meshsize=10.0, rectangle_vertices=verts)
        shape2d = (5, 5)
        shape3d = (5, 5, 10)
        vc = p.assemble_voxcity(
            voxcity_grid=np.zeros(shape3d, dtype=np.int8),
            building_height_grid=np.zeros(shape2d),
            building_min_height_grid=np.zeros(shape2d),
            building_id_grid=np.zeros(shape2d, dtype=np.int32),
            land_cover_grid=np.zeros(shape2d, dtype=np.int32),
            dem_grid=np.zeros(shape2d),
        )
        assert isinstance(vc, VoxCity)
        assert vc.voxels is not None
        assert vc.buildings is not None
        assert vc.land_cover is not None
        assert vc.dem is not None
        assert 'rectangle_vertices' in vc.extras

    def test_assemble_with_canopy(self):
        verts = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        p = VoxCityPipeline(meshsize=10.0, rectangle_vertices=verts)
        shape2d = (3, 3)
        shape3d = (3, 3, 5)
        canopy_top = np.ones(shape2d) * 8.0
        canopy_bottom = np.ones(shape2d) * 3.0
        vc = p.assemble_voxcity(
            voxcity_grid=np.zeros(shape3d, dtype=np.int8),
            building_height_grid=np.zeros(shape2d),
            building_min_height_grid=np.zeros(shape2d),
            building_id_grid=np.zeros(shape2d, dtype=np.int32),
            land_cover_grid=np.zeros(shape2d, dtype=np.int32),
            dem_grid=np.zeros(shape2d),
            canopy_height_top=canopy_top,
            canopy_height_bottom=canopy_bottom,
        )
        assert 'canopy_top' in vc.extras
        np.testing.assert_array_equal(vc.extras['canopy_top'], canopy_top)


# --- Factory tests ---

class TestLandCoverSourceFactory:
    def test_osm(self):
        strategy = LandCoverSourceFactory.create("OpenStreetMap")
        assert strategy is not None

    def test_esri(self):
        strategy = LandCoverSourceFactory.create("Esri 10m Annual Land Use")
        assert strategy is not None

    def test_esa(self):
        strategy = LandCoverSourceFactory.create("ESA WorldCover")
        assert strategy is not None

    def test_dynamic_world(self):
        strategy = LandCoverSourceFactory.create("Dynamic World V1")
        assert strategy is not None

    def test_urbanwatch(self):
        strategy = LandCoverSourceFactory.create("UrbanWatch")
        assert strategy is not None

    def test_oemj(self):
        strategy = LandCoverSourceFactory.create("OEMJ")
        assert strategy is not None


class TestBuildingSourceFactory:
    def test_osm(self):
        strategy = BuildingSourceFactory.create("OpenStreetMap")
        assert strategy is not None

    def test_overture(self):
        strategy = BuildingSourceFactory.create("Overture")
        assert strategy is not None

    def test_default(self):
        strategy = BuildingSourceFactory.create("SomeOtherSource")
        assert strategy is not None


class TestCanopySourceFactory:
    VERTS = [(0, 0), (0, 1), (1, 1), (1, 0)]

    def test_static(self):
        cfg = PipelineConfig(rectangle_vertices=self.VERTS, meshsize=10.0,
                             land_cover_source="OpenStreetMap", building_source="OpenStreetMap",
                             canopy_height_source="Static")
        strategy = CanopySourceFactory.create("Static", cfg)
        assert isinstance(strategy, StaticCanopyStrategy)

    def test_osm_canopy(self):
        cfg = PipelineConfig(rectangle_vertices=self.VERTS, meshsize=10.0,
                             land_cover_source="OpenStreetMap", building_source="OpenStreetMap",
                             canopy_height_source="OpenStreetMap")
        strategy = CanopySourceFactory.create("OpenStreetMap", cfg)
        assert strategy is not None

    def test_source_canopy(self):
        cfg = PipelineConfig(rectangle_vertices=self.VERTS, meshsize=10.0,
                             land_cover_source="OpenStreetMap", building_source="OpenStreetMap")
        strategy = CanopySourceFactory.create("SomeSource", cfg)
        assert strategy is not None


class TestDemSourceFactory:
    def test_flat(self):
        strategy = DemSourceFactory.create("Flat")
        assert isinstance(strategy, FlatDemStrategy)

    def test_none(self):
        strategy = DemSourceFactory.create(None)
        assert isinstance(strategy, FlatDemStrategy)

    def test_none_string(self):
        strategy = DemSourceFactory.create("none")
        assert isinstance(strategy, FlatDemStrategy)

    def test_null_string(self):
        strategy = DemSourceFactory.create("null")
        assert isinstance(strategy, FlatDemStrategy)

    def test_source_dem(self):
        strategy = DemSourceFactory.create("OpenTopography")
        assert strategy is not None
        assert not isinstance(strategy, FlatDemStrategy)


class TestFlatDemStrategy:
    def test_from_land_cover(self):
        strategy = FlatDemStrategy()
        lc = np.ones((5, 5), dtype=np.int32)
        result = strategy.build_grid(
            rectangle_vertices=[(0, 0), (0, 1), (1, 1), (1, 0)],
            meshsize=10.0,
            land_cover_grid=lc,
            output_dir="/tmp"
        )
        assert result.shape == (5, 5)
        assert np.all(result == 0)

    def test_from_none_land_cover(self):
        strategy = FlatDemStrategy()
        result = strategy.build_grid(
            rectangle_vertices=[(139.74, 35.66), (139.74, 35.68), (139.76, 35.68), (139.76, 35.66)],
            meshsize=10.0,
            land_cover_grid=None,
            output_dir="/tmp"
        )
        assert result.shape[0] > 0
        assert result.shape[1] > 0
        assert np.all(result == 0)


class TestStaticCanopyStrategy:
    def test_basic(self):
        cfg = PipelineConfig(rectangle_vertices=[(0, 0), (0, 1), (1, 1), (1, 0)], meshsize=10.0,
                             land_cover_source="OpenStreetMap", building_source="OpenStreetMap",
                             canopy_height_source="Static", static_tree_height=15.0)
        strategy = StaticCanopyStrategy(cfg)
        # Create a small land cover grid with tree labels
        # OSM land cover: Tree class index depends on the mapping
        from voxcity.utils.lc import get_land_cover_classes
        classes = get_land_cover_classes("OpenStreetMap")
        class_to_int = {name: i for i, name in enumerate(classes.values())}
        tree_idx = class_to_int.get("Tree", class_to_int.get("Trees", 0))
        lc = np.zeros((5, 5), dtype=np.int32)
        lc[2, 2] = tree_idx
        top, bottom = strategy.build_grids(
            rectangle_vertices=[(0, 0), (0, 1), (1, 1), (1, 0)],
            meshsize=10.0,
            land_cover_grid=lc,
            output_dir="/tmp"
        )
        assert top.shape == (5, 5)
        assert bottom.shape == (5, 5)
        assert top[2, 2] == 15.0
        assert bottom[2, 2] > 0  # trunk height ratio * 15
        # Non-tree cells should be 0
        assert top[0, 0] == 0.0
