import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from voxcity.utils.visualization import (
    get_voxel_color_map,
    visualize_3d_voxel,
    visualize_3d_voxel_plotly,
    visualize_land_cover_grid,
    visualize_numerical_grid,
    visualize_landcover_grid_on_basemap,
    visualize_numerical_grid_on_basemap,
    visualize_numerical_gdf_on_basemap,
    visualize_point_gdf_on_basemap,
    create_multi_view_scene,
    visualize_voxcity_multi_view,
    visualize_voxcity_with_sim_meshes
)

@pytest.fixture
def sample_voxel_grid():
    """Sample voxel grid for testing"""
    return np.random.randint(0, 5, (10, 10, 10), dtype=int)

@pytest.fixture
def sample_land_cover_grid():
    """Sample land cover grid for testing"""
    return np.random.randint(1, 5, (10, 10), dtype=int)

@pytest.fixture
def sample_numerical_grid():
    """Sample numerical grid for testing"""
    return np.random.uniform(0, 100, (10, 10))

@pytest.fixture
def sample_rectangle_vertices():
    """Sample rectangle vertices for testing"""
    return [
        (139.7564216011559, 35.671290792464255),
        (139.7564216011559, 35.67579720669077),
        (139.76194439884412, 35.67579720669077),
        (139.76194439884412, 35.671290792464255)
    ]

@pytest.fixture
def sample_land_cover_classes():
    """Sample land cover classes for testing"""
    return {
        1: 'Building',
        2: 'Road',
        3: 'Vegetation',
        4: 'Water'
    }

class TestVoxelVisualization:
    """Tests for voxel visualization functions"""
    
    def test_get_voxel_color_map(self):
        """Test voxel color map generation"""
        color_scheme = 'default'
        result = get_voxel_color_map(color_scheme)
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_visualize_3d_voxel(self, sample_voxel_grid):
        """Test 3D voxel visualization"""
        voxel_color_map = 'default'
        voxel_size = 2.0
        
        result = visualize_3d_voxel(
            sample_voxel_grid,
            voxel_color_map,
            voxel_size
        )
        
        assert result is not None
    
    @patch('voxcity.utils.visualization.plotly.graph_objects.Figure')
    def test_visualize_3d_voxel_plotly(self, mock_figure, sample_voxel_grid):
        """Test 3D voxel visualization with Plotly"""
        voxel_color_map = 'default'
        voxel_size = 2.0
        
        result = visualize_3d_voxel_plotly(
            sample_voxel_grid,
            voxel_color_map,
            voxel_size
        )
        
        assert result is not None
        mock_figure.assert_called_once()

class TestGridVisualization:
    """Tests for grid visualization functions"""
    
    def test_visualize_land_cover_grid(self, sample_land_cover_grid, sample_land_cover_classes):
        """Test land cover grid visualization"""
        mesh_size = 1.0
        color_map = 'tab10'
        
        result = visualize_land_cover_grid(
            sample_land_cover_grid,
            mesh_size,
            color_map,
            sample_land_cover_classes
        )
        
        assert result is not None
    
    def test_visualize_numerical_grid(self, sample_numerical_grid):
        """Test numerical grid visualization"""
        mesh_size = 1.0
        title = "Test Grid"
        cmap = 'viridis'
        label = 'Value'
        
        result = visualize_numerical_grid(
            sample_numerical_grid,
            mesh_size,
            title,
            cmap,
            label
        )
        
        assert result is not None

class TestBasemapVisualization:
    """Tests for basemap visualization functions"""
    
    @patch('voxcity.utils.visualization.plt.subplots')
    def test_visualize_landcover_grid_on_basemap(self, mock_subplots, sample_land_cover_grid, sample_rectangle_vertices):
        """Test land cover grid visualization on basemap"""
        meshsize = 100
        source = 'Standard'
        alpha = 0.6
        figsize = (12, 8)
        
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        result = visualize_landcover_grid_on_basemap(
            sample_land_cover_grid,
            sample_rectangle_vertices,
            meshsize,
            source,
            alpha,
            figsize
        )
        
        assert result is not None
        mock_subplots.assert_called_once()
    
    @patch('voxcity.utils.visualization.plt.subplots')
    def test_visualize_numerical_grid_on_basemap(self, mock_subplots, sample_numerical_grid, sample_rectangle_vertices):
        """Test numerical grid visualization on basemap"""
        meshsize = 100
        value_name = "Height"
        cmap = 'viridis'
        
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        result = visualize_numerical_grid_on_basemap(
            sample_numerical_grid,
            sample_rectangle_vertices,
            meshsize,
            value_name,
            cmap
        )
        
        assert result is not None
        mock_subplots.assert_called_once()
    
    @patch('voxcity.utils.visualization.plt.subplots')
    def test_visualize_numerical_gdf_on_basemap(self, mock_subplots):
        """Test numerical GeoDataFrame visualization on basemap"""
        import geopandas as gpd
        from shapely.geometry import Point
        
        gdf = gpd.GeoDataFrame(
            [{"geometry": Point(139.7564, 35.6713), "value": 25.0}],
            crs='EPSG:4326'
        )
        value_name = "Height"
        cmap = 'viridis'
        
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        result = visualize_numerical_gdf_on_basemap(
            gdf,
            value_name,
            cmap
        )
        
        assert result is not None
        mock_subplots.assert_called_once()
    
    @patch('voxcity.utils.visualization.plt.subplots')
    def test_visualize_point_gdf_on_basemap(self, mock_subplots):
        """Test point GeoDataFrame visualization on basemap"""
        import geopandas as gpd
        from shapely.geometry import Point
        
        gdf = gpd.GeoDataFrame(
            [{"geometry": Point(139.7564, 35.6713), "value": 25.0}],
            crs='EPSG:4326'
        )
        value_name = "Height"
        
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        result = visualize_point_gdf_on_basemap(
            gdf,
            value_name
        )
        
        assert result is not None
        mock_subplots.assert_called_once()

class TestMultiViewVisualization:
    """Tests for multi-view visualization functions"""
    
    def test_create_multi_view_scene(self, temp_output_dir):
        """Test multi-view scene creation"""
        meshes = [MagicMock(), MagicMock()]
        output_directory = temp_output_dir / "multi_view"
        projection_type = "perspective"
        distance_factor = 1.0
        
        result = create_multi_view_scene(
            meshes,
            output_directory,
            projection_type,
            distance_factor
        )
        
        assert result is not None
        assert output_directory.exists()
    
    def test_visualize_voxcity_multi_view(self, sample_voxel_grid):
        """Test voxcity multi-view visualization"""
        meshsize = 1.0
        output_directory = "output"
        projection_type = "perspective"
        distance_factor = 1.0
        
        result = visualize_voxcity_multi_view(
            sample_voxel_grid,
            meshsize,
            output_directory,
            projection_type,
            distance_factor
        )
        
        assert result is not None
    
    def test_visualize_voxcity_with_sim_meshes(self, sample_voxel_grid):
        """Test voxcity visualization with simulation meshes"""
        meshsize = 1.0
        custom_meshes = [MagicMock(), MagicMock()]
        output_directory = "output"
        
        result = visualize_voxcity_with_sim_meshes(
            sample_voxel_grid,
            meshsize,
            custom_meshes,
            output_directory
        )
        
        assert result is not None
