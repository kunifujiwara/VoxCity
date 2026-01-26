"""
Raster processing package.

Orientation contract:
- All public functions accept and return 2D grids using the canonical internal
  orientation "north_up": row 0 is the northern/top row.
- Where data sources use south_up, conversions are performed internally; outputs
  are always north_up unless explicitly documented otherwise.
- Columns increase eastward (col 0 = west/left), indices increase to the east.
"""

# Re-export public APIs from submodules
from .core import (
    apply_operation,
    translate_array,
    group_and_label_cells,
    process_grid_optimized,
    process_grid,
    calculate_grid_size,
    create_coordinate_mesh,
    create_cell_polygon,
    compute_grid_shape,
)

from .landcover import (
    tree_height_grid_from_land_cover,
    create_land_cover_grid_from_geotiff_polygon,
    create_land_cover_grid_from_gdf_polygon,
)

from .raster import (
    create_height_grid_from_geotiff_polygon,
    create_dem_grid_from_geotiff_polygon,
)

from .terrain import (
    create_dem_grid_from_gdf_polygon,
    create_dem_grid_from_gdf_kdtree,
)

from .buildings import (
    create_building_height_grid_from_gdf_polygon,
    create_building_height_grid_from_open_building_temporal_polygon,
)

from .export import (
    grid_to_geodataframe,
    grid_to_point_geodataframe,
)

from .canopy import (
    create_vegetation_height_grid_from_gdf_polygon,
    create_canopy_grids_from_tree_gdf,
)

__all__ = [
    # core
    "apply_operation",
    "translate_array",
    "group_and_label_cells",
    "process_grid_optimized",
    "process_grid",
    "calculate_grid_size",
    "create_coordinate_mesh",
    "create_cell_polygon",
    "compute_grid_shape",
    # landcover
    "tree_height_grid_from_land_cover",
    "create_land_cover_grid_from_geotiff_polygon",
    "create_land_cover_grid_from_gdf_polygon",
    # raster
    "create_height_grid_from_geotiff_polygon",
    "create_dem_grid_from_geotiff_polygon",
    # buildings
    "create_building_height_grid_from_gdf_polygon",
    "create_building_height_grid_from_open_building_temporal_polygon",
    # export
    "grid_to_geodataframe",
    "grid_to_point_geodataframe",
    # vegetation/terrain/trees
    "create_vegetation_height_grid_from_gdf_polygon",
    "create_dem_grid_from_gdf_polygon",
    "create_dem_grid_from_gdf_kdtree",
    "create_canopy_grids_from_tree_gdf",
]




