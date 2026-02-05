"""Tests for voxcity.models module - dataclasses and data structures."""
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
    PipelineConfig,
    MeshModel,
    MeshCollection,
)


class TestGridMetadata:
    def test_creation(self):
        meta = GridMetadata(crs="EPSG:4326", bounds=(0, 0, 100, 100), meshsize=1.0)
        assert meta.crs == "EPSG:4326"
        assert meta.bounds == (0, 0, 100, 100)
        assert meta.meshsize == 1.0

    def test_default_values_not_supported(self):
        # GridMetadata requires all fields
        with pytest.raises(TypeError):
            GridMetadata(crs="EPSG:4326")


class TestBuildingGrid:
    @pytest.fixture
    def building_grid(self):
        heights = np.array([[0.0, 10.0], [5.0, 15.0]])
        min_heights = np.array([[[], [(0, 10)]], [[(0, 5)], [(0, 15)]]], dtype=object)
        ids = np.array([[0, 1], [2, 3]])
        meta = GridMetadata(crs="EPSG:4326", bounds=(0, 0, 2, 2), meshsize=1.0)
        return BuildingGrid(heights=heights, min_heights=min_heights, ids=ids, meta=meta)

    def test_creation(self, building_grid):
        assert building_grid.heights.shape == (2, 2)
        assert building_grid.ids.shape == (2, 2)
        assert building_grid.meta.meshsize == 1.0

    def test_height_values(self, building_grid):
        assert building_grid.heights[0, 1] == 10.0
        assert building_grid.heights[1, 1] == 15.0


class TestLandCoverGrid:
    def test_creation(self):
        classes = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        meta = GridMetadata(crs="EPSG:4326", bounds=(0, 0, 2, 2), meshsize=1.0)
        lc = LandCoverGrid(classes=classes, meta=meta)
        assert lc.classes.shape == (2, 2)
        assert lc.classes[0, 0] == 1


class TestDemGrid:
    def test_creation(self):
        elevation = np.array([[10.0, 11.0], [12.0, 13.0]])
        meta = GridMetadata(crs="EPSG:4326", bounds=(0, 0, 2, 2), meshsize=1.0)
        dem = DemGrid(elevation=elevation, meta=meta)
        assert dem.elevation.shape == (2, 2)
        assert dem.elevation[1, 1] == 13.0


class TestVoxelGrid:
    def test_creation(self):
        classes = np.zeros((2, 2, 10), dtype=np.int8)
        classes[:, :, 0] = -1  # ground
        meta = GridMetadata(crs="EPSG:4326", bounds=(0, 0, 2, 2), meshsize=1.0)
        voxels = VoxelGrid(classes=classes, meta=meta)
        assert voxels.classes.shape == (2, 2, 10)
        assert voxels.classes[0, 0, 0] == -1


class TestCanopyGrid:
    def test_creation_with_top_only(self):
        top = np.array([[5.0, 0.0], [10.0, 8.0]])
        meta = GridMetadata(crs="EPSG:4326", bounds=(0, 0, 2, 2), meshsize=1.0)
        canopy = CanopyGrid(top=top, meta=meta)
        assert canopy.top.shape == (2, 2)
        assert canopy.bottom is None

    def test_creation_with_bottom(self):
        top = np.array([[5.0, 0.0], [10.0, 8.0]])
        bottom = np.array([[2.0, 0.0], [4.0, 3.0]])
        meta = GridMetadata(crs="EPSG:4326", bounds=(0, 0, 2, 2), meshsize=1.0)
        canopy = CanopyGrid(top=top, meta=meta, bottom=bottom)
        assert canopy.bottom is not None
        assert canopy.bottom[0, 0] == 2.0


class TestVoxCity:
    @pytest.fixture
    def voxcity_instance(self):
        meta = GridMetadata(crs="EPSG:4326", bounds=(0, 0, 2, 2), meshsize=1.0)
        voxels = VoxelGrid(classes=np.zeros((2, 2, 5), dtype=np.int8), meta=meta)
        buildings = BuildingGrid(
            heights=np.zeros((2, 2)),
            min_heights=np.array([[[], []], [[], []]], dtype=object),
            ids=np.zeros((2, 2), dtype=int),
            meta=meta,
        )
        land_cover = LandCoverGrid(classes=np.ones((2, 2), dtype=np.uint8), meta=meta)
        dem = DemGrid(elevation=np.zeros((2, 2)), meta=meta)
        tree_canopy = CanopyGrid(top=np.zeros((2, 2)), meta=meta)
        return VoxCity(
            voxels=voxels,
            buildings=buildings,
            land_cover=land_cover,
            dem=dem,
            tree_canopy=tree_canopy,
        )

    def test_creation(self, voxcity_instance):
        assert voxcity_instance.voxels is not None
        assert voxcity_instance.buildings is not None
        assert voxcity_instance.dem is not None

    def test_extras_default(self, voxcity_instance):
        assert isinstance(voxcity_instance.extras, dict)
        assert len(voxcity_instance.extras) == 0


class TestPipelineConfig:
    def test_creation_with_defaults(self):
        config = PipelineConfig(
            rectangle_vertices=[(0, 0), (0, 1), (1, 1), (1, 0)],
            meshsize=1.0,
        )
        assert config.meshsize == 1.0
        assert config.building_source is None
        assert config.output_dir == "output"
        assert config.parallel_download is False

    def test_creation_with_all_options(self):
        config = PipelineConfig(
            rectangle_vertices=[(0, 0), (0, 1), (1, 1), (1, 0)],
            meshsize=2.0,
            building_source="osm",
            land_cover_source="oemj",
            canopy_height_source="gee",
            dem_source="gee",
            output_dir="my_output",
            trunk_height_ratio=0.5,
            parallel_download=True,
        )
        assert config.building_source == "osm"
        assert config.parallel_download is True


class TestMeshModel:
    def test_creation(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = MeshModel(vertices=vertices, faces=faces)
        assert mesh.vertices.shape == (3, 3)
        assert mesh.faces.shape == (1, 3)
        assert mesh.colors is None
        assert mesh.name is None

    def test_creation_with_colors_and_name(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        colors = np.array([[255, 0, 0, 255]], dtype=np.uint8)
        mesh = MeshModel(vertices=vertices, faces=faces, colors=colors, name="test_mesh")
        assert mesh.colors is not None
        assert mesh.name == "test_mesh"


class TestMeshCollection:
    def test_creation_empty(self):
        collection = MeshCollection()
        assert len(collection.meshes) == 0

    def test_add_and_get(self):
        collection = MeshCollection()
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = MeshModel(vertices=vertices, faces=faces, name="triangle")
        
        collection.add("triangle", mesh)
        assert collection.get("triangle") is mesh
        assert collection.get("nonexistent") is None

    def test_iteration(self):
        collection = MeshCollection()
        vertices = np.array([[0, 0, 0]], dtype=np.float32)
        faces = np.array([[0, 0, 0]], dtype=np.int32)
        
        collection.add("mesh1", MeshModel(vertices=vertices, faces=faces))
        collection.add("mesh2", MeshModel(vertices=vertices, faces=faces))
        
        names = [name for name, _ in collection]
        assert "mesh1" in names
        assert "mesh2" in names

    def test_items_property(self):
        collection = MeshCollection()
        assert collection.items == collection.meshes
