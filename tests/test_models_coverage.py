"""
Comprehensive tests for voxcity.models to improve coverage.
"""

import numpy as np
import pytest

from voxcity.models import (
    GridMetadata,
    BuildingGrid,
    LandCoverGrid,
    DemGrid,
    VoxelGrid,
    CanopyGrid,
    VoxCity,
    PipelineConfig,
    MeshModel,
    MeshCollection,
)


class TestGridMetadata:
    def test_creation(self):
        meta = GridMetadata(crs='EPSG:4326', bounds=(0, 0, 1, 1), meshsize=1.0)
        assert meta.crs == 'EPSG:4326'
        assert meta.bounds == (0, 0, 1, 1)
        assert meta.meshsize == 1.0

    def test_different_crs(self):
        meta = GridMetadata(crs='EPSG:3857', bounds=(-1, -1, 1, 1), meshsize=0.5)
        assert meta.crs == 'EPSG:3857'


class TestBuildingGrid:
    def test_creation(self):
        meta = GridMetadata(crs='EPSG:4326', bounds=(0, 0, 1, 1), meshsize=1.0)
        bg = BuildingGrid(
            heights=np.zeros((5, 5)),
            min_heights=np.empty((5, 5), dtype=object),
            ids=np.zeros((5, 5)),
            meta=meta,
        )
        assert bg.heights.shape == (5, 5)
        assert bg.meta.meshsize == 1.0

    def test_with_values(self):
        meta = GridMetadata(crs='EPSG:4326', bounds=(0, 0, 1, 1), meshsize=2.0)
        heights = np.array([[10.0, 20.0], [30.0, 0.0]])
        bg = BuildingGrid(
            heights=heights,
            min_heights=np.empty((2, 2), dtype=object),
            ids=np.array([[1, 2], [3, 0]]),
            meta=meta,
        )
        assert bg.heights[0, 0] == 10.0
        assert bg.ids[1, 1] == 0


class TestLandCoverGrid:
    def test_creation(self):
        meta = GridMetadata(crs='EPSG:4326', bounds=(0, 0, 1, 1), meshsize=1.0)
        lc = LandCoverGrid(classes=np.ones((3, 3), dtype=int), meta=meta)
        assert lc.classes.shape == (3, 3)
        assert np.all(lc.classes == 1)


class TestDemGrid:
    def test_creation(self):
        meta = GridMetadata(crs='EPSG:4326', bounds=(0, 0, 1, 1), meshsize=1.0)
        dem = DemGrid(elevation=np.zeros((3, 3)), meta=meta)
        assert dem.elevation.shape == (3, 3)


class TestVoxelGrid:
    def test_creation(self):
        meta = GridMetadata(crs='EPSG:4326', bounds=(0, 0, 1, 1), meshsize=1.0)
        vg = VoxelGrid(classes=np.zeros((3, 3, 10), dtype=np.int8), meta=meta)
        assert vg.classes.ndim == 3


class TestCanopyGrid:
    def test_with_bottom(self):
        meta = GridMetadata(crs='EPSG:4326', bounds=(0, 0, 1, 1), meshsize=1.0)
        cg = CanopyGrid(top=np.ones((3, 3)) * 5, bottom=np.ones((3, 3)) * 2, meta=meta)
        assert cg.bottom is not None
        np.testing.assert_array_equal(cg.bottom, np.ones((3, 3)) * 2)

    def test_without_bottom(self):
        meta = GridMetadata(crs='EPSG:4326', bounds=(0, 0, 1, 1), meshsize=1.0)
        cg = CanopyGrid(top=np.ones((3, 3)) * 5, meta=meta)
        assert cg.bottom is None


class TestVoxCity:
    def _make_voxcity(self):
        meta = GridMetadata(crs='EPSG:4326', bounds=(0, 0, 1, 1), meshsize=1.0)
        return VoxCity(
            voxels=VoxelGrid(classes=np.zeros((3, 3, 5), dtype=np.int8), meta=meta),
            buildings=BuildingGrid(
                heights=np.zeros((3, 3)),
                min_heights=np.empty((3, 3), dtype=object),
                ids=np.zeros((3, 3)),
                meta=meta,
            ),
            land_cover=LandCoverGrid(classes=np.ones((3, 3), dtype=int), meta=meta),
            dem=DemGrid(elevation=np.zeros((3, 3)), meta=meta),
            tree_canopy=CanopyGrid(top=np.zeros((3, 3)), meta=meta),
        )

    def test_creation(self):
        vc = self._make_voxcity()
        assert isinstance(vc, VoxCity)

    def test_extras_default(self):
        vc = self._make_voxcity()
        assert isinstance(vc.extras, dict)
        assert len(vc.extras) == 0

    def test_extras_custom(self):
        meta = GridMetadata(crs='EPSG:4326', bounds=(0, 0, 1, 1), meshsize=1.0)
        vc = VoxCity(
            voxels=VoxelGrid(classes=np.zeros((3, 3, 5), dtype=np.int8), meta=meta),
            buildings=BuildingGrid(
                heights=np.zeros((3, 3)),
                min_heights=np.empty((3, 3), dtype=object),
                ids=np.zeros((3, 3)),
                meta=meta,
            ),
            land_cover=LandCoverGrid(classes=np.ones((3, 3), dtype=int), meta=meta),
            dem=DemGrid(elevation=np.zeros((3, 3)), meta=meta),
            tree_canopy=CanopyGrid(top=np.zeros((3, 3)), meta=meta),
            extras={'key': 'value'},
        )
        assert vc.extras['key'] == 'value'


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig(
            rectangle_vertices=[(0, 0), (1, 0), (1, 1), (0, 1)],
            meshsize=1.0,
        )
        assert cfg.output_dir == "output"
        assert cfg.mapvis is False
        assert cfg.gridvis is True
        assert cfg.parallel_download is False
        assert cfg.building_source is None
        assert cfg.land_cover_source is None
        assert isinstance(cfg.land_cover_options, dict)
        assert isinstance(cfg.building_options, dict)

    def test_full_config(self):
        cfg = PipelineConfig(
            rectangle_vertices=[(0, 0), (1, 0), (1, 1), (0, 1)],
            meshsize=2.0,
            building_source='OSM',
            land_cover_source='ESA WorldCover',
            output_dir='/tmp/output',
            trunk_height_ratio=0.5,
            static_tree_height=10.0,
            remove_perimeter_object=5.0,
            mapvis=True,
            gridvis=False,
            parallel_download=True,
            io_options={'save_grid': True},
        )
        assert cfg.meshsize == 2.0
        assert cfg.building_source == 'OSM'
        assert cfg.trunk_height_ratio == 0.5
        assert cfg.parallel_download is True
        assert cfg.io_options == {'save_grid': True}


class TestMeshModel:
    def test_creation(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        mm = MeshModel(vertices=verts, faces=faces)
        assert mm.vertices.shape == (3, 3)
        assert mm.faces.shape == (1, 3)
        assert mm.colors is None
        assert mm.name is None

    def test_with_colors(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        colors = np.array([[255, 0, 0, 255]], dtype=np.uint8)
        mm = MeshModel(vertices=verts, faces=faces, colors=colors, name="test")
        assert mm.colors is not None
        assert mm.name == "test"


class TestMeshCollection:
    def test_empty(self):
        mc = MeshCollection()
        assert len(mc.meshes) == 0

    def test_add_get(self):
        mc = MeshCollection()
        verts = np.array([[0, 0, 0]], dtype=float)
        faces = np.array([[0, 0, 0]])
        mm = MeshModel(vertices=verts, faces=faces)
        mc.add("test", mm)
        assert mc.get("test") is mm
        assert mc.get("nonexistent") is None

    def test_iter(self):
        mc = MeshCollection()
        verts = np.array([[0, 0, 0]], dtype=float)
        faces = np.array([[0, 0, 0]])
        mc.add("a", MeshModel(vertices=verts, faces=faces))
        mc.add("b", MeshModel(vertices=verts, faces=faces))

        items = list(mc)
        assert len(items) == 2
        names = [name for name, _ in items]
        assert "a" in names
        assert "b" in names

    def test_items_property(self):
        mc = MeshCollection()
        verts = np.array([[0, 0, 0]], dtype=float)
        faces = np.array([[0, 0, 0]])
        mc.add("mesh1", MeshModel(vertices=verts, faces=faces))
        assert isinstance(mc.items, dict)
        assert "mesh1" in mc.items

    def test_overwrite(self):
        mc = MeshCollection()
        verts1 = np.array([[0, 0, 0]], dtype=float)
        verts2 = np.array([[1, 1, 1]], dtype=float)
        faces = np.array([[0, 0, 0]])
        mc.add("x", MeshModel(vertices=verts1, faces=faces))
        mc.add("x", MeshModel(vertices=verts2, faces=faces))
        assert mc.get("x").vertices[0, 0] == 1.0
