"""Geoprocessing subpackage.

Submodules and convenience re-exports resolve lazily (PEP 562), so importing
this package — or any single submodule — does not pay for heavy optional
dependencies pulled in by sibling modules (ipyleaflet via ``draw``, osmnx via
``network``, Earth Engine via ``raster``).
"""

import importlib

_SUBMODULES = {
    "draw", "utils", "network", "mesh", "raster", "conversion", "io",
    "heights", "selection", "overlap", "merge_utils", "surface_meta",
    "geojson",
}

# Convenience re-exports: attribute name -> defining submodule
_ATTR_TO_MODULE = {
    "filter_and_convert_gdf_to_geojson": ".conversion",
    "geojson_to_gdf": ".conversion",
    "gdf_to_geojson_dicts": ".conversion",
    "get_gdf_from_gpkg": ".io",
    "load_gdf_from_multiple_gz": ".io",
    "extract_building_heights_from_gdf": ".heights",
    "extract_building_heights_from_geotiff": ".heights",
    "complement_building_heights_from_gdf": ".heights",
    "filter_buildings": ".selection",
    "find_building_containing_point": ".selection",
    "get_buildings_in_drawn_polygon": ".selection",
    "process_building_footprints_by_overlap": ".overlap",
    "merge_gdfs_with_id_conflict_resolution": ".merge_utils",
    "build_building_geojson": ".geojson",
    "build_canopy_geojson": ".geojson",
    "build_lc_geojson": ".geojson",
    "get_lc_source_colors": ".geojson",
    "attach_surface_face_meta": ".surface_meta",
    "compute_face_areas": ".surface_meta",
    "surface_zone_mask": ".surface_meta",
    "classify_surface_faces": ".surface_meta",
    "make_surface_face_key": ".surface_meta",
    "classify_surface_kind": ".surface_meta",
    "wall_orientation": ".surface_meta",
    "SELECTABLE_KINDS": ".surface_meta",
}

__all__ = sorted(_SUBMODULES) + sorted(_ATTR_TO_MODULE)


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    module_name = _ATTR_TO_MODULE.get(name)
    if module_name is not None:
        module = importlib.import_module(module_name, __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES | set(_ATTR_TO_MODULE))
