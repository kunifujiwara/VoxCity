import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from voxcity.exporter.obj import (
    convert_colormap_indices,
    create_face_vertices,
    mesh_faces,
    export_obj,
    grid_to_obj
)

from voxcity.exporter.cityles import (
    create_cityles_directories,
    export_topog,
    export_landuse,
    export_dem,
    export_vmap,
    export_lonlat,
    export_cityles
)

from voxcity.exporter.envimet import (
    export_inx,
    generate_edb_file
)

@pytest.fixture
def sample_voxel_grid():
    """Sample voxel grid for testing"""
    return np.random.randint(0, 5, (10, 10, 10), dtype=int)

@pytest.fixture
def sample_building_height_grid():
    """Sample building height grid for testing"""
    return np.random.uniform(0, 50, (10, 10))

@pytest.fixture
def sample_building_id_grid():
    """Sample building ID grid for testing"""
    return np.random.randint(0, 10, (10, 10), dtype=int)

@pytest.fixture
def sample_land_cover_grid():
    """Sample land cover grid for testing"""
    return np.random.randint(1, 5, (10, 10), dtype=int)

@pytest.fixture
def sample_canopy_height_grid():
    """Sample canopy height grid for testing"""
    return np.random.uniform(0, 20, (10, 10))

@pytest.fixture
def sample_dem_grid():
    """Sample DEM grid for testing"""
    return np.random.uniform(0, 100, (10, 10))

@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)

class TestOBJExporter:
    """Tests for OBJ format exporter functions"""
    
    def test_convert_colormap_indices(self):
        """Test colormap index conversion"""
        original_map = {1: 'red', 2: 'blue', 3: 'green'}
        result = convert_colormap_indices(original_map)
        assert isinstance(result, dict)
        assert len(result) == len(original_map)
    
    def test_create_face_vertices(self):
        """Test face vertex creation"""
        coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        positive_direction = True
        axis = 0
        
        result = create_face_vertices(coords, positive_direction, axis)
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 3  # 3D coordinates
    
    def test_mesh_faces(self):
        """Test face meshing"""
        mask = np.array([[True, False], [False, True]])
        layer_index = 0
        axis = 0
        positive_direction = True
        normal_idx = 0
        voxel_size_m = 1.0
        
        result = mesh_faces(
            mask, layer_index, axis, positive_direction, 
            normal_idx, voxel_size_m
        )
        assert isinstance(result, list)
    
    def test_export_obj(self, sample_voxel_grid, temp_output_dir):
        """Test OBJ export functionality"""
        file_name = "test_export"
        voxel_size = 1.0
        voxel_color_map = {1: 'red', 2: 'blue'}
        
        result = export_obj(
            sample_voxel_grid,
            temp_output_dir,
            file_name,
            voxel_size,
            voxel_color_map
        )
        
        assert result is not None
        # Check if files were created
        obj_file = temp_output_dir / f"{file_name}.obj"
        mtl_file = temp_output_dir / f"{file_name}.mtl"
        assert obj_file.exists()
        assert mtl_file.exists()
    
    def test_grid_to_obj(self, sample_building_height_grid, sample_dem_grid, temp_output_dir):
        """Test grid to OBJ conversion"""
        file_name = "test_grid"
        cell_size = 1.0
        offset = [0, 0, 0]
        
        result = grid_to_obj(
            sample_building_height_grid,
            sample_dem_grid,
            temp_output_dir,
            file_name,
            cell_size,
            offset
        )
        
        assert result is not None
        obj_file = temp_output_dir / f"{file_name}.obj"
        assert obj_file.exists()

class TestCityLESExporter:
    """Tests for CityLES format exporter functions"""
    
    def test_create_cityles_directories(self, temp_output_dir):
        """Test CityLES directory creation"""
        result = create_cityles_directories(temp_output_dir)
        assert result is not None
        
        # Check if subdirectories were created
        subdirs = ['constant', 'system', '0']
        for subdir in subdirs:
            assert (temp_output_dir / subdir).exists()
    
    def test_export_topog(self, sample_building_height_grid, sample_building_id_grid, temp_output_dir):
        """Test topography export"""
        output_path = temp_output_dir / "topog"
        meshsize = 1.0
        land_cover_source = 'Urbanwatch'
        
        result = export_topog(
            sample_building_height_grid,
            sample_building_id_grid,
            output_path,
            meshsize,
            land_cover_source
        )
        
        assert result is not None
        assert output_path.exists()
    
    def test_export_landuse(self, sample_land_cover_grid, temp_output_dir):
        """Test land use export"""
        output_path = temp_output_dir / "landuse"
        land_cover_source = 'Urbanwatch'
        
        result = export_landuse(
            sample_land_cover_grid,
            output_path,
            land_cover_source
        )
        
        assert result is not None
        assert output_path.exists()
    
    def test_export_dem(self, sample_dem_grid, temp_output_dir):
        """Test DEM export"""
        output_path = temp_output_dir / "dem"
        
        result = export_dem(sample_dem_grid, output_path)
        
        assert result is not None
        assert output_path.exists()
    
    def test_export_vmap(self, sample_canopy_height_grid, temp_output_dir):
        """Test vegetation map export"""
        output_path = temp_output_dir / "vmap"
        tree_base_ratio = 0.3
        tree_type = 'default'
        
        result = export_vmap(
            sample_canopy_height_grid,
            output_path,
            tree_base_ratio,
            tree_type
        )
        
        assert result is not None
        assert output_path.exists()
    
    def test_export_lonlat(self, temp_output_dir):
        """Test longitude/latitude export"""
        rectangle_vertices = [
            (139.7564, 35.6713),
            (139.7619, 35.6758)
        ]
        grid_shape = (10, 10)
        output_path = temp_output_dir / "lonlat"
        
        result = export_lonlat(
            rectangle_vertices,
            grid_shape,
            output_path
        )
        
        assert result is not None
        assert output_path.exists()
    
    def test_export_cityles(self, sample_building_height_grid, sample_building_id_grid,
                           sample_canopy_height_grid, sample_dem_grid, temp_output_dir):
        """Test complete CityLES export"""
        output_directory = temp_output_dir / "cityles"
        meshsize = 1.0
        land_cover_source = 'Urbanwatch'
        
        result = export_cityles(
            sample_building_height_grid,
            sample_building_id_grid,
            sample_canopy_height_grid,
            sample_dem_grid,
            output_directory,
            meshsize,
            land_cover_source
        )
        
        assert result is not None
        assert output_directory.exists()

class TestEnvimetExporter:
    """Tests for ENVI-met format exporter functions"""
    
    def test_export_inx(self, sample_building_height_grid, sample_building_id_grid,
                        sample_canopy_height_grid, sample_land_cover_grid, sample_dem_grid, temp_output_dir):
        """Test ENVI-met INX export functionality"""
        meshsize = 1.0
        land_cover_source = 'Urbanwatch'
        rectangle_vertices = [(0, 0), (1, 1)]
        
        result = export_inx(
            sample_building_height_grid,
            sample_building_id_grid,
            sample_canopy_height_grid,
            sample_land_cover_grid,
            sample_dem_grid,
            meshsize,
            land_cover_source,
            rectangle_vertices
        )
        
        assert result is not None
    
    def test_generate_edb_file(self):
        """Test EDB file generation"""
        result = generate_edb_file()
        
        assert result is not None
        assert isinstance(result, str)
