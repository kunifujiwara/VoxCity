"""
Tests for voxcity.visualizer.__init__ module.
Tests GPU renderer import handling.
"""

import pytest


class TestVisualizerInit:
    """Tests for visualizer package initialization."""

    def test_main_exports_available(self):
        """Test that main visualization exports are available."""
        from voxcity.visualizer import (
            MeshBuilder,
            PyVistaRenderer,
            create_multi_view_scene,
            visualize_voxcity_plotly,
            visualize_voxcity,
            get_voxel_color_map,
        )
        
        assert MeshBuilder is not None
        assert PyVistaRenderer is not None
        assert callable(create_multi_view_scene)
        assert callable(visualize_voxcity_plotly)
        assert callable(visualize_voxcity)
        assert callable(get_voxel_color_map)

    def test_grid_visualization_exports(self):
        """Test that grid visualization exports are available."""
        from voxcity.visualizer import (
            visualize_landcover_grid_on_basemap,
            visualize_numerical_grid_on_basemap,
            visualize_numerical_gdf_on_basemap,
            visualize_point_gdf_on_basemap,
        )
        
        assert callable(visualize_landcover_grid_on_basemap)
        assert callable(visualize_numerical_grid_on_basemap)
        assert callable(visualize_numerical_gdf_on_basemap)
        assert callable(visualize_point_gdf_on_basemap)

    def test_maps_visualization_exports(self):
        """Test that maps visualization exports are available."""
        from voxcity.visualizer import (
            plot_grid,
            visualize_land_cover_grid_on_map,
            visualize_building_height_grid_on_map,
            visualize_numerical_grid_on_map,
        )
        
        assert callable(plot_grid)
        assert callable(visualize_land_cover_grid_on_map)
        assert callable(visualize_building_height_grid_on_map)
        assert callable(visualize_numerical_grid_on_map)

    def test_gpu_renderer_attributes_exist(self):
        """Test that GPU renderer attributes exist (may be None if taichi not installed)."""
        from voxcity import visualizer
        
        # These attributes should exist regardless of taichi availability
        assert hasattr(visualizer, 'GPURenderer')
        assert hasattr(visualizer, 'TaichiRenderer')
        assert hasattr(visualizer, 'visualize_voxcity_gpu')
        assert hasattr(visualizer, '_HAS_GPU_RENDERER')

    def test_all_exports_in_all(self):
        """Test that __all__ contains expected exports."""
        from voxcity import visualizer
        
        expected_exports = [
            "MeshBuilder",
            "PyVistaRenderer",
            "create_multi_view_scene",
            "visualize_voxcity_plotly",
            "visualize_voxcity",
            "get_voxel_color_map",
            "visualize_landcover_grid_on_basemap",
            "visualize_numerical_grid_on_basemap",
            "visualize_numerical_gdf_on_basemap",
            "visualize_point_gdf_on_basemap",
            "plot_grid",
            "visualize_land_cover_grid_on_map",
            "visualize_building_height_grid_on_map",
            "visualize_numerical_grid_on_map",
            "GPURenderer",
            "TaichiRenderer",
            "visualize_voxcity_gpu",
        ]
        
        for name in expected_exports:
            assert name in visualizer.__all__, f"{name} not in __all__"
