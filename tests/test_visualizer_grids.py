"""Tests for voxcity.visualizer.grids module."""
import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from unittest.mock import patch, MagicMock

# Note: Visualization tests use mocking to avoid actual rendering
# which requires a display and can be slow


class TestVisualizeLandcoverGridOnBasemap:
    """Tests for visualize_landcover_grid_on_basemap function."""
    
    @pytest.fixture
    def sample_grid(self):
        """Create sample land cover grid."""
        return np.array([
            [1, 1, 2, 2],
            [1, 3, 3, 2],
            [5, 5, 5, 9],
            [5, 13, 13, 9]
        ], dtype=np.int8)
    
    @pytest.fixture
    def sample_vertices(self):
        """Create sample rectangle vertices."""
        return [
            (139.7, 35.6),
            (139.7, 35.61),
            (139.71, 35.61),
            (139.71, 35.6)
        ]
    
    @patch('matplotlib.pyplot.show')
    @patch('contextily.add_basemap')
    def test_visualize_landcover_basic(self, mock_basemap, mock_show, sample_grid, sample_vertices):
        """Test basic land cover visualization."""
        from voxcity.visualizer.grids import visualize_landcover_grid_on_basemap
        
        # Should not raise
        visualize_landcover_grid_on_basemap(
            sample_grid,
            sample_vertices,
            meshsize=1.0
        )
        
        # Show should be called
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('contextily.add_basemap')
    def test_visualize_landcover_custom_params(self, mock_basemap, mock_show, sample_grid, sample_vertices):
        """Test land cover visualization with custom parameters."""
        from voxcity.visualizer.grids import visualize_landcover_grid_on_basemap
        
        visualize_landcover_grid_on_basemap(
            sample_grid,
            sample_vertices,
            meshsize=2.0,
            source='Standard',
            alpha=0.8,
            figsize=(10, 10),
            show_edge=True,
            edge_color='red',
            edge_width=1.0
        )
        
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('contextily.add_basemap')
    def test_visualize_landcover_different_basemaps(self, mock_basemap, mock_show, sample_grid, sample_vertices):
        """Test land cover visualization with different basemap styles."""
        from voxcity.visualizer.grids import visualize_landcover_grid_on_basemap
        
        basemap_styles = ['CartoDB dark', 'CartoDB light', 'CartoDB voyager']
        
        for style in basemap_styles:
            visualize_landcover_grid_on_basemap(
                sample_grid,
                sample_vertices,
                meshsize=1.0,
                basemap=style
            )


class TestVisualizeNumericalGridOnBasemap:
    """Tests for visualize_numerical_grid_on_basemap function."""
    
    @pytest.fixture
    def sample_numerical_grid(self):
        """Create sample numerical grid."""
        return np.array([
            [0.0, 5.0, 10.0],
            [15.0, 20.0, 25.0],
            [30.0, 35.0, 40.0]
        ], dtype=np.float32)
    
    @pytest.fixture
    def sample_vertices(self):
        """Create sample rectangle vertices."""
        return [
            (139.7, 35.6),
            (139.7, 35.61),
            (139.71, 35.61),
            (139.71, 35.6)
        ]
    
    @patch('matplotlib.pyplot.show')
    @patch('contextily.add_basemap')
    def test_visualize_numerical_basic(self, mock_basemap, mock_show, sample_numerical_grid, sample_vertices):
        """Test basic numerical grid visualization."""
        from voxcity.visualizer.grids import visualize_numerical_grid_on_basemap
        
        visualize_numerical_grid_on_basemap(
            sample_numerical_grid,
            sample_vertices,
            meshsize=1.0
        )
        
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('contextily.add_basemap')
    def test_visualize_numerical_custom_cmap(self, mock_basemap, mock_show, sample_numerical_grid, sample_vertices):
        """Test numerical grid visualization with custom colormap."""
        from voxcity.visualizer.grids import visualize_numerical_grid_on_basemap
        
        cmaps = ['plasma', 'inferno', 'magma', 'cividis']
        
        for cmap in cmaps:
            visualize_numerical_grid_on_basemap(
                sample_numerical_grid,
                sample_vertices,
                meshsize=1.0,
                cmap=cmap
            )
    
    @patch('matplotlib.pyplot.show')
    @patch('contextily.add_basemap')
    def test_visualize_numerical_vmin_vmax(self, mock_basemap, mock_show, sample_numerical_grid, sample_vertices):
        """Test numerical grid visualization with custom vmin/vmax."""
        from voxcity.visualizer.grids import visualize_numerical_grid_on_basemap
        
        visualize_numerical_grid_on_basemap(
            sample_numerical_grid,
            sample_vertices,
            meshsize=1.0,
            vmin=10,
            vmax=30
        )
        
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('contextily.add_basemap')
    def test_visualize_numerical_with_edges(self, mock_basemap, mock_show, sample_numerical_grid, sample_vertices):
        """Test numerical grid visualization with edges shown."""
        from voxcity.visualizer.grids import visualize_numerical_grid_on_basemap
        
        visualize_numerical_grid_on_basemap(
            sample_numerical_grid,
            sample_vertices,
            meshsize=1.0,
            show_edge=True,
            edge_color='white',
            edge_width=0.3
        )
        
        mock_show.assert_called_once()


class TestVisualizeNumericalGdfOnBasemap:
    """Tests for visualize_numerical_gdf_on_basemap function."""
    
    @pytest.fixture
    def sample_gdf(self):
        """Create sample GeoDataFrame."""
        return gpd.GeoDataFrame({
            'value': [10.0, 20.0, 30.0, 40.0],
            'geometry': [
                Polygon([(139.7, 35.6), (139.705, 35.6), (139.705, 35.605), (139.7, 35.605)]),
                Polygon([(139.705, 35.6), (139.71, 35.6), (139.71, 35.605), (139.705, 35.605)]),
                Polygon([(139.7, 35.605), (139.705, 35.605), (139.705, 35.61), (139.7, 35.61)]),
                Polygon([(139.705, 35.605), (139.71, 35.605), (139.71, 35.61), (139.705, 35.61)]),
            ]
        }, crs="EPSG:4326")
    
    @patch('matplotlib.pyplot.show')
    @patch('contextily.add_basemap')
    def test_visualize_gdf_basic(self, mock_basemap, mock_show, sample_gdf):
        """Test basic GDF visualization."""
        from voxcity.visualizer.grids import visualize_numerical_gdf_on_basemap
        
        visualize_numerical_gdf_on_basemap(sample_gdf)
        
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('contextily.add_basemap')
    def test_visualize_gdf_custom_value_name(self, mock_basemap, mock_show):
        """Test GDF visualization with custom value column."""
        from voxcity.visualizer.grids import visualize_numerical_gdf_on_basemap
        
        gdf = gpd.GeoDataFrame({
            'height': [10.0, 20.0],
            'geometry': [
                Polygon([(139.7, 35.6), (139.705, 35.6), (139.705, 35.605), (139.7, 35.605)]),
                Polygon([(139.705, 35.6), (139.71, 35.6), (139.71, 35.605), (139.705, 35.605)]),
            ]
        }, crs="EPSG:4326")
        
        visualize_numerical_gdf_on_basemap(gdf, value_name='height')
        
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('contextily.add_basemap')
    def test_visualize_gdf_no_crs_with_input_crs(self, mock_basemap, mock_show):
        """Test GDF visualization when CRS is not set but provided."""
        from voxcity.visualizer.grids import visualize_numerical_gdf_on_basemap
        
        gdf = gpd.GeoDataFrame({
            'value': [10.0],
            'geometry': [Polygon([(139.7, 35.6), (139.705, 35.6), (139.705, 35.605), (139.7, 35.605)])]
        })  # No CRS set
        
        visualize_numerical_gdf_on_basemap(gdf, input_crs="EPSG:4326")
        
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('contextily.add_basemap')
    def test_visualize_gdf_auto_detect_crs(self, mock_basemap, mock_show):
        """Test GDF visualization with auto-detected CRS from bounds."""
        from voxcity.visualizer.grids import visualize_numerical_gdf_on_basemap
        
        # Create GDF with coordinates that look like WGS84
        gdf = gpd.GeoDataFrame({
            'value': [10.0],
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        })  # No CRS, but bounds suggest lat/lon
        
        visualize_numerical_gdf_on_basemap(gdf)
        
        mock_show.assert_called_once()


class TestVisualizePointGdfOnBasemap:
    """Tests for visualize_point_gdf_on_basemap function."""
    
    @pytest.fixture
    def sample_point_gdf(self):
        """Create sample point GeoDataFrame."""
        return gpd.GeoDataFrame({
            'value': [10.0, 20.0, 30.0, 40.0, 50.0],
            'geometry': [
                Point(139.70, 35.60),
                Point(139.71, 35.61),
                Point(139.72, 35.60),
                Point(139.70, 35.62),
                Point(139.71, 35.59),
            ]
        }, crs="EPSG:4326")
    
    @patch('matplotlib.pyplot.show')
    @patch('contextily.add_basemap')
    def test_visualize_point_basic(self, mock_basemap, mock_show, sample_point_gdf):
        """Test basic point GDF visualization."""
        from voxcity.visualizer.grids import visualize_point_gdf_on_basemap
        
        visualize_point_gdf_on_basemap(sample_point_gdf)
        
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('contextily.add_basemap')
    def test_visualize_point_custom_params(self, mock_basemap, mock_show, sample_point_gdf):
        """Test point visualization with custom parameters."""
        from voxcity.visualizer.grids import visualize_point_gdf_on_basemap
        
        visualize_point_gdf_on_basemap(
            sample_point_gdf,
            value_name='value',
            colormap='plasma',
            markersize=50,
            alpha=0.5,
            figsize=(10, 8)
        )


class TestVisualizerEdgeCases:
    """Edge case tests for visualizer functions."""
    
    @patch('matplotlib.pyplot.show')
    @patch('contextily.add_basemap')
    def test_visualize_empty_grid(self, mock_basemap, mock_show):
        """Test visualization with minimal grid."""
        from voxcity.visualizer.grids import visualize_numerical_grid_on_basemap
        
        small_grid = np.array([[0.0]], dtype=np.float32)
        vertices = [(0, 0), (0, 0.001), (0.001, 0.001), (0.001, 0)]
        
        visualize_numerical_grid_on_basemap(small_grid, vertices, meshsize=1.0)
    
    @patch('matplotlib.pyplot.show')
    @patch('contextily.add_basemap')
    def test_visualize_with_nan_values(self, mock_basemap, mock_show):
        """Test visualization with NaN values in grid."""
        from voxcity.visualizer.grids import visualize_numerical_grid_on_basemap
        
        grid_with_nan = np.array([
            [1.0, np.nan, 3.0],
            [np.nan, 5.0, np.nan],
            [7.0, np.nan, 9.0]
        ], dtype=np.float32)
        
        vertices = [(0, 0), (0, 0.003), (0.003, 0.003), (0.003, 0)]
        
        visualize_numerical_grid_on_basemap(grid_with_nan, vertices, meshsize=1.0)
    
    @patch('matplotlib.pyplot.show')
    @patch('contextily.add_basemap')
    def test_visualize_with_negative_values(self, mock_basemap, mock_show):
        """Test visualization with negative values."""
        from voxcity.visualizer.grids import visualize_numerical_grid_on_basemap
        
        grid = np.array([
            [-10.0, -5.0],
            [0.0, 5.0]
        ], dtype=np.float32)
        
        vertices = [(0, 0), (0, 0.002), (0.002, 0.002), (0.002, 0)]
        
        visualize_numerical_grid_on_basemap(grid, vertices, meshsize=1.0)
