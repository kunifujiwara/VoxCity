"""Visualization subpackage.

All exports resolve lazily (PEP 562): importing this package costs nothing
until an attribute is touched, so packages that import ``voxcity.visualizer``
submodules (e.g. the generator) do not pay for pyvista/plotly/taichi.

GPU renderer attributes (``GPURenderer`` etc.) resolve to ``None`` when
taichi is not installed, preserving the previous try/except behavior.
"""

import importlib
import importlib.util

_SUBMODULES = {"builder", "renderer", "palette", "grids", "maps", "renderer_gpu"}

_ATTR_TO_MODULE = {
    "MeshBuilder": ".builder",
    "PyVistaRenderer": ".renderer",
    "create_multi_view_scene": ".renderer",
    "visualize_voxcity_plotly": ".renderer",
    "visualize_voxcity": ".renderer",
    "get_voxel_color_map": ".palette",
    "visualize_landcover_grid_on_basemap": ".grids",
    "visualize_numerical_grid_on_basemap": ".grids",
    "visualize_numerical_gdf_on_basemap": ".grids",
    "visualize_point_gdf_on_basemap": ".grids",
    "plot_grid": ".maps",
    "visualize_land_cover_grid_on_map": ".maps",
    "visualize_building_height_grid_on_map": ".maps",
    "visualize_numerical_grid_on_map": ".maps",
}

# GPU renderer exports: None when taichi is missing (legacy contract)
_GPU_ATTRS = {"GPURenderer", "TaichiRenderer", "visualize_voxcity_gpu"}

__all__ = sorted(_ATTR_TO_MODULE) + sorted(_GPU_ATTRS)


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _GPU_ATTRS:
        try:
            module = importlib.import_module(".renderer_gpu", __name__)
        except ImportError:
            return None
        return getattr(module, name)
    if name == "_HAS_GPU_RENDERER":
        return importlib.util.find_spec("taichi") is not None
    module_name = _ATTR_TO_MODULE.get(name)
    if module_name is not None:
        module = importlib.import_module(module_name, __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        set(globals()) | _SUBMODULES | set(_ATTR_TO_MODULE) | _GPU_ATTRS
        | {"_HAS_GPU_RENDERER"}
    )
