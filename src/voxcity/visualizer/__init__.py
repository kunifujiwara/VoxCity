from .builder import MeshBuilder
from .renderer import PyVistaRenderer, create_multi_view_scene, visualize_voxcity_plotly, visualize_voxcity
from .palette import get_voxel_color_map
from .grids import visualize_landcover_grid_on_basemap, visualize_numerical_grid_on_basemap, visualize_numerical_gdf_on_basemap, visualize_point_gdf_on_basemap
from .maps import plot_grid, visualize_land_cover_grid_on_map, visualize_building_height_grid_on_map, visualize_numerical_grid_on_map

# GPU-accelerated renderer (optional dependency: taichi)
try:
    from .renderer_gpu import GPURenderer, TaichiRenderer, visualize_voxcity_gpu
    _HAS_GPU_RENDERER = True
except ImportError:
    GPURenderer = None  # type: ignore
    TaichiRenderer = None  # type: ignore
    visualize_voxcity_gpu = None  # type: ignore
    _HAS_GPU_RENDERER = False

__all__ = [
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
    # GPU renderer (available if taichi is installed)
    "GPURenderer",
    "TaichiRenderer",
    "visualize_voxcity_gpu",
]


