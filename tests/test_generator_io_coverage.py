"""
Tests for voxcity.generator.io - save/load round-trip.
"""

import os
import tempfile
import numpy as np
import pytest

from voxcity.generator.io import (
    save_voxcity_data,
    load_voxcity,
    save_voxcity,
)
from voxcity.models import (
    GridMetadata,
    VoxelGrid,
    BuildingGrid,
    LandCoverGrid,
    DemGrid,
    CanopyGrid,
    VoxCity,
)


def _make_voxcity():
    """Create a minimal VoxCity instance for testing."""
    meta = GridMetadata(crs='EPSG:4326', bounds=(0.0, 0.0, 1.0, 1.0), meshsize=1.0)
    voxels = VoxelGrid(classes=np.zeros((3, 3, 5), dtype=np.int8), meta=meta)
    buildings = BuildingGrid(
        heights=np.zeros((3, 3)),
        min_heights=np.empty((3, 3), dtype=object),
        ids=np.zeros((3, 3)),
        meta=meta,
    )
    for i in range(3):
        for j in range(3):
            buildings.min_heights[i, j] = []
    land = LandCoverGrid(classes=np.ones((3, 3), dtype=int), meta=meta)
    dem = DemGrid(elevation=np.zeros((3, 3)), meta=meta)
    canopy = CanopyGrid(top=np.zeros((3, 3)), meta=meta)
    return VoxCity(voxels=voxels, buildings=buildings, land_cover=land, dem=dem, tree_canopy=canopy)


class TestSaveLoadVoxcity:
    def test_round_trip_v2(self):
        city = _make_voxcity()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.pkl')
            save_voxcity(path, city)
            loaded = load_voxcity(path)

            assert isinstance(loaded, VoxCity)
            assert loaded.voxels.meta.crs == 'EPSG:4326'
            assert loaded.voxels.meta.meshsize == 1.0
            np.testing.assert_array_equal(loaded.voxels.classes, city.voxels.classes)

    def test_legacy_format(self):
        """Test loading legacy dict format."""
        import pickle
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'legacy.pkl')
            data = {
                'voxcity_grid': np.zeros((3, 3, 5), dtype=np.int8),
                'building_height_grid': np.zeros((3, 3)),
                'building_min_height_grid': np.empty((3, 3), dtype=object),
                'building_id_grid': np.zeros((3, 3)),
                'canopy_height_grid': np.zeros((3, 3)),
                'land_cover_grid': np.ones((3, 3), dtype=int),
                'dem_grid': np.zeros((3, 3)),
                'building_gdf': None,
                'meshsize': 2.0,
                'rectangle_vertices': [(0, 0), (1, 0), (1, 1), (0, 1)],
            }
            for i in range(3):
                for j in range(3):
                    data['building_min_height_grid'][i, j] = []

            with open(path, 'wb') as f:
                pickle.dump(data, f)

            loaded = load_voxcity(path)
            assert isinstance(loaded, VoxCity)
            assert loaded.voxels.meta.meshsize == 2.0

    def test_save_non_voxcity_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'bad.pkl')
            with pytest.raises(TypeError):
                save_voxcity(path, {"not": "a VoxCity"})

    def test_save_creates_directories(self):
        city = _make_voxcity()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'sub', 'dir', 'test.pkl')
            save_voxcity(path, city)
            assert os.path.exists(path)

    def test_direct_voxcity_pickle(self):
        """Test loading a VoxCity object directly pickled."""
        import pickle
        city = _make_voxcity()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'direct.pkl')
            with open(path, 'wb') as f:
                pickle.dump(city, f)
            loaded = load_voxcity(path)
            assert isinstance(loaded, VoxCity)


class TestSaveVoxcityData:
    def test_legacy_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pkl')
            save_voxcity_data(
                output_path=path,
                voxcity_grid=np.zeros((3, 3, 5)),
                building_height_grid=np.zeros((3, 3)),
                building_min_height_grid=np.zeros((3, 3)),
                building_id_grid=np.zeros((3, 3)),
                canopy_height_grid=np.zeros((3, 3)),
                land_cover_grid=np.ones((3, 3)),
                dem_grid=np.zeros((3, 3)),
                building_gdf=None,
                meshsize=1.0,
                rectangle_vertices=[(0, 0), (1, 0), (1, 1), (0, 1)],
            )
            assert os.path.exists(path)
