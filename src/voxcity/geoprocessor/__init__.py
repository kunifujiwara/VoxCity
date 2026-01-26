from . import (
    draw,
    utils,
    network,
    mesh,
    raster,
    conversion,
    io,
    heights,
    selection,
    overlap,
    merge_utils,
    citygml,
)

# Re-export frequently used functions at package level for convenience
from .conversion import (
    filter_and_convert_gdf_to_geojson,
    geojson_to_gdf,
    gdf_to_geojson_dicts,
)
from .io import (
    get_geojson_from_gpkg,
    get_gdf_from_gpkg,
    load_gdf_from_multiple_gz,
    swap_coordinates,
    save_geojson,
)
from .heights import (
    extract_building_heights_from_gdf,
    extract_building_heights_from_geotiff,
    complement_building_heights_from_gdf,
)
from .selection import (
    filter_buildings,
    find_building_containing_point,
    get_buildings_in_drawn_polygon,
)
from .overlap import (
    process_building_footprints_by_overlap,
)
from .merge_utils import (
    merge_gdfs_with_id_conflict_resolution,
)

__all__ = [
    # submodules
    "draw",
    "utils",
    "network",
    "mesh",
    "raster",
    "conversion",
    "io",
    "heights",
    "selection",
    "overlap",
    "merge_utils",
    "citygml",
    # functions
    "filter_and_convert_gdf_to_geojson",
    "geojson_to_gdf",
    "gdf_to_geojson_dicts",
    "get_geojson_from_gpkg",
    "get_gdf_from_gpkg",
    "load_gdf_from_multiple_gz",
    "swap_coordinates",
    "save_geojson",
    "extract_building_heights_from_gdf",
    "extract_building_heights_from_geotiff",
    "complement_building_heights_from_gdf",
    "filter_buildings",
    "find_building_containing_point",
    "get_buildings_in_drawn_polygon",
    "process_building_footprints_by_overlap",
    "merge_gdfs_with_id_conflict_resolution",
]
