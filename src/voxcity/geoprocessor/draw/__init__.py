"""
Draw subpackage — interactive map drawing and editing tools for VoxCity.

This package provides functions for:
- Drawing and rotating rectangles on interactive maps
- City-centred map initialisation with fixed-dimension rectangles
- Building footprint visualisation and polygon drawing
- Interactive building, tree, and land-cover editors

Submodules
----------
rectangle   : Rectangle drawing/rotation utilities.
polygon     : Building display and polygon vertex extraction.
edit_building : Interactive building height/footprint editor.
edit_tree     : Interactive tree canopy editor.
edit_landcover: Interactive land-cover class editor.
_common       : Shared helpers, constants, GeoJSON builders (internal).
"""

# Rectangle utilities
from .rectangle import (
    rotate_rectangle,
    draw_rectangle_map,
    draw_rectangle_map_cityname,
    center_location_map_cityname,
)

# Polygon utilities
from .polygon import (
    display_buildings_and_draw_polygon,
    get_polygon_vertices,
)

# Editors
from .edit_building import edit_building, create_building_editor
from .edit_tree import edit_tree, create_tree_editor
from .edit_landcover import edit_landcover

# Backward-compat constant
from ._common import _LC_COLORS_BY_NAME

__all__ = [
    "rotate_rectangle",
    "draw_rectangle_map",
    "draw_rectangle_map_cityname",
    "center_location_map_cityname",
    "display_buildings_and_draw_polygon",
    "get_polygon_vertices",
    "edit_building",
    "create_building_editor",
    "edit_tree",
    "create_tree_editor",
    "edit_landcover",
    "_LC_COLORS_BY_NAME",
]
