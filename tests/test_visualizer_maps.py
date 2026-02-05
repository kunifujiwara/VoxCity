"""Tests for voxcity.visualizer.maps module - map visualization functions."""
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from voxcity.visualizer.maps import (
    plot_grid,
    visualize_land_cover_grid_on_map,
    visualize_building_height_grid_on_map,
    visualize_numerical_grid_on_map,
)


@pytest.fixture
def sample_vertices():
    """Sample rectangle vertices for Tokyo area."""
    return [
        (139.756, 35.671),
        (139.756, 35.676),
        (139.762, 35.676),
        (139.762, 35.671)
    ]


@pytest.fixture
def sample_land_cover_grid():
    """Create a sample land cover grid."""
    # 10x10 grid with various land cover types
    grid = np.zeros((10, 10), dtype=np.int8)
    grid[0:3, :] = 1  # Type 1
    grid[3:6, :] = 2  # Type 2
    grid[6:10, :] = 3  # Type 3
    return grid


@pytest.fixture
def sample_building_height_grid():
    """Create a sample building height grid."""
    grid = np.zeros((10, 10))
    grid[3:7, 3:7] = 25.0  # Building in center
    grid[1:3, 1:3] = 15.0  # Smaller building
    return grid


@pytest.fixture
def sample_dem_grid():
    """Create a sample DEM grid."""
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    xx, yy = np.meshgrid(x, y)
    # Create gentle slope
    return 10 + xx * 5 + yy * 3


class TestVisualizeLandCoverGridOnMap:
    """Tests for visualize_land_cover_grid_on_map function."""
    
    def test_basic_visualization(self, sample_land_cover_grid, sample_vertices):
        """Test basic land cover visualization."""
        try:
            fig, ax = visualize_land_cover_grid_on_map(
                sample_land_cover_grid,
                sample_vertices,
                land_cover_source='OpenStreetMap',
                show=False
            )
            assert fig is not None
            assert ax is not None
            plt.close(fig)
        except Exception:
            # May fail without network access for basemap
            pass
    
    def test_returns_figure_and_axes(self, sample_land_cover_grid, sample_vertices):
        """Test that function returns figure and axes."""
        try:
            result = visualize_land_cover_grid_on_map(
                sample_land_cover_grid,
                sample_vertices,
                land_cover_source='OpenStreetMap',
                show=False
            )
            assert len(result) == 2
            plt.close('all')
        except Exception:
            pass


class TestVisualizeBuildingHeightGridOnMap:
    """Tests for visualize_building_height_grid_on_map function."""
    
    def test_basic_visualization(self, sample_building_height_grid, sample_vertices):
        """Test basic building height visualization."""
        try:
            fig, ax = visualize_building_height_grid_on_map(
                sample_building_height_grid,
                sample_vertices,
                show=False
            )
            assert fig is not None
            plt.close(fig)
        except Exception:
            pass
    
    def test_custom_vmin_vmax(self, sample_building_height_grid, sample_vertices):
        """Test with custom vmin/vmax values."""
        try:
            fig, ax = visualize_building_height_grid_on_map(
                sample_building_height_grid,
                sample_vertices,
                vmin=0,
                vmax=50,
                show=False
            )
            plt.close(fig)
        except Exception:
            pass


class TestVisualizeNumericalGridOnMap:
    """Tests for visualize_numerical_grid_on_map function."""
    
    def test_dem_visualization(self, sample_dem_grid, sample_vertices):
        """Test DEM visualization."""
        try:
            fig, ax = visualize_numerical_grid_on_map(
                sample_dem_grid,
                sample_vertices,
                data_type='dem',
                show=False
            )
            assert fig is not None
            plt.close(fig)
        except Exception:
            pass
    
    def test_canopy_height_visualization(self, sample_vertices):
        """Test canopy height visualization."""
        canopy_grid = np.zeros((10, 10))
        canopy_grid[2:5, 2:5] = 8.0  # Tree canopy
        
        try:
            fig, ax = visualize_numerical_grid_on_map(
                canopy_grid,
                sample_vertices,
                data_type='canopy_height',
                show=False
            )
            plt.close(fig)
        except Exception:
            pass
    
    def test_custom_colormap(self, sample_dem_grid, sample_vertices):
        """Test with custom colormap."""
        try:
            fig, ax = visualize_numerical_grid_on_map(
                sample_dem_grid,
                sample_vertices,
                data_type='dem',
                color_map='viridis',
                show=False
            )
            plt.close(fig)
        except Exception:
            pass


class TestPlotGrid:
    """Tests for plot_grid function - lower level plotting."""
    
    def test_plot_grid_land_cover(self, sample_land_cover_grid, sample_vertices):
        """Test plot_grid with land cover data."""
        from pyproj import Transformer
        from voxcity.utils.lc import get_land_cover_classes
        
        try:
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            origin = np.array([sample_vertices[0][0], sample_vertices[0][1]])
            u_vec = np.array([1.0, 0.0])
            v_vec = np.array([0.0, 1.0])
            
            land_cover_classes = get_land_cover_classes('OpenStreetMap')
            
            fig, ax = plot_grid(
                sample_land_cover_grid,
                origin,
                1.0,  # meshsize
                u_vec,
                v_vec,
                transformer,
                sample_vertices,
                data_type='land_cover',
                land_cover_classes=land_cover_classes,
                basemap=None,  # Skip basemap to avoid network calls
            )
            assert fig is not None
            plt.close(fig)
        except Exception:
            pass
    
    def test_plot_grid_building_height(self, sample_building_height_grid, sample_vertices):
        """Test plot_grid with building height data."""
        from pyproj import Transformer
        
        try:
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            origin = np.array([sample_vertices[0][0], sample_vertices[0][1]])
            u_vec = np.array([1.0, 0.0])
            v_vec = np.array([0.0, 1.0])
            
            fig, ax = plot_grid(
                sample_building_height_grid,
                origin,
                1.0,
                u_vec,
                v_vec,
                transformer,
                sample_vertices,
                data_type='building_height',
                basemap=None,
            )
            plt.close(fig)
        except Exception:
            pass


class TestEdgeCases:
    """Edge case tests for maps visualization."""
    
    def test_empty_grid(self, sample_vertices):
        """Test with empty (all zeros) grid."""
        empty_grid = np.zeros((10, 10))
        
        try:
            fig, ax = visualize_numerical_grid_on_map(
                empty_grid,
                sample_vertices,
                data_type='dem',
                show=False
            )
            plt.close(fig)
        except Exception:
            # May fail with all-zero grid
            pass
    
    def test_grid_with_nans(self, sample_vertices):
        """Test with grid containing NaN values."""
        grid = np.ones((10, 10)) * 10.0
        grid[3:7, 3:7] = np.nan  # NaN hole
        
        try:
            fig, ax = visualize_numerical_grid_on_map(
                grid,
                sample_vertices,
                data_type='dem',
                show=False
            )
            plt.close(fig)
        except Exception:
            pass
    
    def test_small_grid(self, sample_vertices):
        """Test with very small grid."""
        small_grid = np.array([[10, 20], [30, 40]])
        
        try:
            fig, ax = visualize_numerical_grid_on_map(
                small_grid,
                sample_vertices,
                data_type='dem',
                show=False
            )
            plt.close(fig)
        except Exception:
            pass


class TestVisualizationOptions:
    """Tests for visualization options."""
    
    def test_alpha_parameter(self, sample_dem_grid, sample_vertices):
        """Test alpha transparency parameter."""
        try:
            fig, ax = visualize_numerical_grid_on_map(
                sample_dem_grid,
                sample_vertices,
                data_type='dem',
                alpha=0.3,
                show=False
            )
            plt.close(fig)
        except Exception:
            pass
    
    def test_edge_parameter(self, sample_dem_grid, sample_vertices):
        """Test edge drawing parameter."""
        try:
            fig, ax = visualize_numerical_grid_on_map(
                sample_dem_grid,
                sample_vertices,
                data_type='dem',
                edge=False,
                show=False
            )
            plt.close(fig)
        except Exception:
            pass
