"""Tests for voxcity.generator.io module - save/load functionality."""
import pytest
import numpy as np
import tempfile
import os
import pickle

from voxcity.models import VoxCity, GridMetadata, VoxelGrid, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid
from voxcity.generator.io import save_voxcity, load_voxcity, save_voxcity_data


@pytest.fixture
def sample_grid_metadata():
    """Create sample grid metadata."""
    return GridMetadata(
        crs='EPSG:4326',
        bounds=(139.0, 35.0, 139.1, 35.1),
        meshsize=1.0
    )


@pytest.fixture
def sample_voxcity(sample_grid_metadata):
    """Create a minimal VoxCity object for testing."""
    meta = sample_grid_metadata
    
    # Create small test grids
    voxel_data = np.zeros((10, 10, 5), dtype=np.int8)
    voxel_data[5, 5, 0:3] = 1  # Some building voxels
    
    building_heights = np.zeros((10, 10))
    building_heights[5, 5] = 15.0
    
    building_min_heights = np.zeros((10, 10))
    building_ids = np.zeros((10, 10), dtype=np.int32)
    building_ids[5, 5] = 1
    
    land_cover = np.ones((10, 10), dtype=np.int8)  # All grass
    land_cover[5, 5] = 0  # Building footprint
    
    dem = np.zeros((10, 10))
    
    canopy_top = np.zeros((10, 10))
    canopy_top[3, 3] = 10.0  # A tree
    
    voxels = VoxelGrid(classes=voxel_data, meta=meta)
    buildings = BuildingGrid(
        heights=building_heights,
        min_heights=building_min_heights,
        ids=building_ids,
        meta=meta
    )
    land = LandCoverGrid(classes=land_cover, meta=meta)
    dem_grid = DemGrid(elevation=dem, meta=meta)
    canopy = CanopyGrid(top=canopy_top, bottom=None, meta=meta)
    
    extras = {
        'rectangle_vertices': [(139.0, 35.0), (139.0, 35.1), (139.1, 35.1), (139.1, 35.0)],
    }
    
    return VoxCity(voxels=voxels, buildings=buildings, land_cover=land, dem=dem_grid, tree_canopy=canopy, extras=extras)


class TestSaveVoxcity:
    """Tests for save_voxcity function."""
    
    def test_save_voxcity_creates_file(self, sample_voxcity):
        """Test that save_voxcity creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_city.pkl')
            save_voxcity(output_path, sample_voxcity)
            
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
    
    def test_save_voxcity_creates_directory(self, sample_voxcity):
        """Test that save_voxcity creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, 'nested', 'dirs', 'test_city.pkl')
            save_voxcity(nested_path, sample_voxcity)
            
            assert os.path.exists(nested_path)
    
    def test_save_voxcity_rejects_non_voxcity(self):
        """Test that save_voxcity raises TypeError for non-VoxCity input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.pkl')
            
            with pytest.raises(TypeError, match="expects a VoxCity instance"):
                save_voxcity(output_path, {"not": "a VoxCity"})
    
    def test_save_voxcity_v2_format(self, sample_voxcity):
        """Test that save_voxcity uses v2 format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_city.pkl')
            save_voxcity(output_path, sample_voxcity)
            
            with open(output_path, 'rb') as f:
                data = pickle.load(f)
            
            assert isinstance(data, dict)
            assert data.get('__format__') == 'voxcity.v2'
            assert isinstance(data.get('voxcity'), VoxCity)


class TestLoadVoxcity:
    """Tests for load_voxcity function."""
    
    def test_load_voxcity_v2_format(self, sample_voxcity):
        """Test loading v2 format VoxCity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_city.pkl')
            save_voxcity(output_path, sample_voxcity)
            
            loaded = load_voxcity(output_path)
            
            assert isinstance(loaded, VoxCity)
            assert loaded.voxels.classes.shape == sample_voxcity.voxels.classes.shape
    
    def test_load_voxcity_raw_object(self, sample_voxcity):
        """Test loading raw VoxCity object (not wrapped in dict)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_city.pkl')
            
            # Save raw VoxCity object
            with open(output_path, 'wb') as f:
                pickle.dump(sample_voxcity, f)
            
            loaded = load_voxcity(output_path)
            
            assert isinstance(loaded, VoxCity)
    
    def test_load_voxcity_legacy_format(self):
        """Test loading legacy dict format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'legacy_city.pkl')
            
            # Create legacy format data
            legacy_data = {
                'voxcity_grid': np.zeros((10, 10, 5), dtype=np.int8),
                'building_height_grid': np.zeros((10, 10)),
                'building_min_height_grid': np.zeros((10, 10)),
                'building_id_grid': np.zeros((10, 10), dtype=np.int32),
                'canopy_height_grid': np.zeros((10, 10)),
                'land_cover_grid': np.ones((10, 10), dtype=np.int8),
                'dem_grid': np.zeros((10, 10)),
                'meshsize': 1.0,
                'rectangle_vertices': [(139.0, 35.0), (139.0, 35.1), (139.1, 35.1), (139.1, 35.0)],
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(legacy_data, f)
            
            loaded = load_voxcity(output_path)
            
            assert isinstance(loaded, VoxCity)
            assert loaded.voxels.classes.shape == (10, 10, 5)
    
    def test_load_voxcity_legacy_without_vertices(self):
        """Test loading legacy format without rectangle_vertices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'legacy_no_vertices.pkl')
            
            # Create legacy format without vertices
            legacy_data = {
                'voxcity_grid': np.zeros((10, 10, 5), dtype=np.int8),
                'building_height_grid': np.zeros((10, 10)),
                'building_min_height_grid': np.zeros((10, 10)),
                'building_id_grid': np.zeros((10, 10), dtype=np.int32),
                'canopy_height_grid': np.zeros((10, 10)),
                'land_cover_grid': np.ones((10, 10), dtype=np.int8),
                'dem_grid': np.zeros((10, 10)),
                'meshsize': 2.0,
                'rectangle_vertices': None,  # No vertices
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(legacy_data, f)
            
            loaded = load_voxcity(output_path)
            
            assert isinstance(loaded, VoxCity)
            # Bounds should be calculated from grid size and meshsize
            expected_bounds = (0.0, 0.0, 10 * 2.0, 10 * 2.0)
            assert loaded.voxels.meta.bounds == expected_bounds


class TestSaveVoxcityData:
    """Tests for save_voxcity_data (legacy function)."""
    
    def test_save_voxcity_data_creates_file(self):
        """Test that save_voxcity_data creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'legacy_data.pkl')
            
            # Create test data
            voxcity_grid = np.zeros((10, 10, 5), dtype=np.int8)
            building_height_grid = np.zeros((10, 10))
            building_min_height_grid = np.zeros((10, 10))
            building_id_grid = np.zeros((10, 10), dtype=np.int32)
            canopy_height_grid = np.zeros((10, 10))
            land_cover_grid = np.ones((10, 10), dtype=np.int8)
            dem_grid = np.zeros((10, 10))
            building_gdf = None
            meshsize = 1.0
            rectangle_vertices = [(139.0, 35.0), (139.0, 35.1), (139.1, 35.1), (139.1, 35.0)]
            
            save_voxcity_data(
                output_path, voxcity_grid, building_height_grid, building_min_height_grid,
                building_id_grid, canopy_height_grid, land_cover_grid, dem_grid,
                building_gdf, meshsize, rectangle_vertices
            )
            
            assert os.path.exists(output_path)
    
    def test_save_voxcity_data_contains_all_fields(self):
        """Test that save_voxcity_data saves all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'legacy_data.pkl')
            
            voxcity_grid = np.zeros((10, 10, 5), dtype=np.int8)
            building_height_grid = np.ones((10, 10)) * 15.0
            building_min_height_grid = np.zeros((10, 10))
            building_id_grid = np.arange(100, dtype=np.int32).reshape(10, 10)
            canopy_height_grid = np.ones((10, 10)) * 5.0
            land_cover_grid = np.ones((10, 10), dtype=np.int8)
            dem_grid = np.ones((10, 10)) * 10.0
            building_gdf = None
            meshsize = 2.5
            rectangle_vertices = [(139.0, 35.0), (139.0, 35.1), (139.1, 35.1), (139.1, 35.0)]
            
            save_voxcity_data(
                output_path, voxcity_grid, building_height_grid, building_min_height_grid,
                building_id_grid, canopy_height_grid, land_cover_grid, dem_grid,
                building_gdf, meshsize, rectangle_vertices
            )
            
            with open(output_path, 'rb') as f:
                data = pickle.load(f)
            
            assert 'voxcity_grid' in data
            assert 'building_height_grid' in data
            assert 'building_min_height_grid' in data
            assert 'building_id_grid' in data
            assert 'canopy_height_grid' in data
            assert 'land_cover_grid' in data
            assert 'dem_grid' in data
            assert 'building_gdf' in data
            assert 'meshsize' in data
            assert 'rectangle_vertices' in data
            
            np.testing.assert_array_equal(data['voxcity_grid'], voxcity_grid)
            assert data['meshsize'] == 2.5


class TestRoundTrip:
    """Tests for save/load roundtrip."""
    
    def test_roundtrip_preserves_voxels(self, sample_voxcity):
        """Test that voxel data is preserved through save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'roundtrip.pkl')
            
            save_voxcity(output_path, sample_voxcity)
            loaded = load_voxcity(output_path)
            
            np.testing.assert_array_equal(loaded.voxels.classes, sample_voxcity.voxels.classes)
    
    def test_roundtrip_preserves_building_data(self, sample_voxcity):
        """Test that building data is preserved through save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'roundtrip.pkl')
            
            save_voxcity(output_path, sample_voxcity)
            loaded = load_voxcity(output_path)
            
            np.testing.assert_array_equal(loaded.buildings.heights, sample_voxcity.buildings.heights)
            np.testing.assert_array_equal(loaded.buildings.ids, sample_voxcity.buildings.ids)
    
    def test_roundtrip_preserves_metadata(self, sample_voxcity):
        """Test that metadata is preserved through save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'roundtrip.pkl')
            
            save_voxcity(output_path, sample_voxcity)
            loaded = load_voxcity(output_path)
            
            assert loaded.voxels.meta.crs == sample_voxcity.voxels.meta.crs
            assert loaded.voxels.meta.meshsize == sample_voxcity.voxels.meta.meshsize
            assert loaded.voxels.meta.bounds == sample_voxcity.voxels.meta.bounds
    
    def test_roundtrip_preserves_extras(self, sample_voxcity):
        """Test that extras are preserved through save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'roundtrip.pkl')
            
            save_voxcity(output_path, sample_voxcity)
            loaded = load_voxcity(output_path)
            
            assert 'rectangle_vertices' in loaded.extras
            assert loaded.extras['rectangle_vertices'] == sample_voxcity.extras['rectangle_vertices']
