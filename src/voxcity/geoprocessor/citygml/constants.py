"""
CityGML Constants
=================

Defines constants, namespaces, and mappings used throughout the citygml subpackage.
These are shared across parsers, voxelizers, and visualization modules.
"""

import numpy as np

# =============================================================================
# VoxCity Voxel Semantic Codes
# =============================================================================
# These codes match VoxCity's standard voxel semantics for interoperability

GROUND_CODE = -1
"""Ground/terrain voxel code."""

TREE_CODE = -2
"""Vegetation/tree voxel code."""

BUILDING_CODE = -3
"""Building voxel code."""

BRIDGE_CODE = -4
"""Bridge voxel code."""

CITY_FURNITURE_CODE = -5
"""City furniture (benches, lamps, etc.) voxel code."""


# =============================================================================
# CityGML Namespaces
# =============================================================================

NAMESPACES = {
    """Standard CityGML namespaces (German/European format, versions 1.0 and 2.0)."""
    'core': 'http://www.opengis.net/citygml/1.0',
    'core2': 'http://www.opengis.net/citygml/2.0',
    'bldg': 'http://www.opengis.net/citygml/building/1.0',
    'bldg2': 'http://www.opengis.net/citygml/building/2.0',
    'gml': 'http://www.opengis.net/gml',
    'grp': 'http://www.opengis.net/citygml/cityobjectgroup/1.0',
    'gen': 'http://www.opengis.net/citygml/generics/1.0',
    'uro': 'http://www.kantei.go.jp/jp/singi/tiiki/toshisaisei/itoshisaisei/iur/uro/1.4',
    'xlink': 'http://www.w3.org/1999/xlink',
}

PLATEAU_NAMESPACES = {
    """Japanese PLATEAU CityGML namespaces."""
    'core': 'http://www.opengis.net/citygml/2.0',
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
    'gml': 'http://www.opengis.net/gml',
    'dem': 'http://www.opengis.net/citygml/relief/2.0',
    'veg': 'http://www.opengis.net/citygml/vegetation/2.0',
    'brid': 'http://www.opengis.net/citygml/bridge/2.0',
    'frn': 'http://www.opengis.net/citygml/cityfurniture/2.0',
    'tran': 'http://www.opengis.net/citygml/transportation/2.0',
    'luse': 'http://www.opengis.net/citygml/landuse/2.0',
    'uro': 'https://www.geospatial.jp/iur/uro/3.1',
    'xlink': 'http://www.w3.org/1999/xlink',
}


# =============================================================================
# PLATEAU Land Use to VoxCity Land Cover Mapping
# =============================================================================

# VoxCity Standard Classes (1-based indices):
#   1: Bareland, 2: Rangeland, 3: Shrub, 4: Agriculture land, 5: Tree,
#   6: Moss and lichen, 7: Wet land, 8: Mangrove, 9: Water, 10: Snow and ice,
#   11: Developed space, 12: Road, 13: Building, 14: No Data

PLATEAU_LANDUSE_TO_VOXCITY = {
    # Natural/Rural land uses
    '201': 4,   # Agriculture (田, 畑, 果樹園, 採草地) -> Agriculture land
    '202': 5,   # Forest (山林) -> Tree
    '203': 9,   # Water bodies (河川水面, 湖沼, 溜池, 用水路, 海水面) -> Water
    '204': 9,   # Other natural land (荒地, 低湿地, 河原, 海浜, 干潟) -> Water
    '205': 11,  # Residential land (住宅用地) -> Developed space
    
    # Urban/Built-up land uses
    '211': 13,  # Commercial/Business (商業用地) -> Building
    '212': 13,  # Industrial (工業用地) -> Building
    '213': 13,  # Residential/Commercial (住居併用施設用地) -> Building
    '214': 12,  # Road (道路用地) -> Road
    '215': 12,  # Transportation facilities (交通施設用地) -> Road
    '216': 11,  # Public open space (公共空地 - 公園, 緑地) -> Developed space
    '217': 11,  # Other public facility (その他公的施設用地) -> Developed space
    '218': 2,   # Golf course (ゴルフ場) -> Rangeland (grass)
    '219': 13,  # Public facilities (公共施設用地) -> Building
    '220': 11,  # Solar power facilities -> Developed space
    '221': 11,  # Flat parking (平面駐車場) -> Developed space
    '222': 11,  # Other urban (その他利用 - 倉庫街) -> Developed space
    '223': 1,   # Vacant land (未利用地, 空地) -> Bareland
    '224': 14,  # Unknown (不明) -> No Data
    
    # Special zones
    '231': 11,  # Planned development (可住地) -> Developed space
    '251': 1,   # Unusable area -> Bareland
    '252': 9,   # Sea area (海地) -> Water
    '260': 11,  # Residential/Commercial mixed -> Developed space
    '261': 12,  # Road/Transportation mixed -> Road
    '262': 11,  # Open space mixed -> Developed space
    '263': 11,  # Other mixed -> Developed space
}


# =============================================================================
# VoxCity Color Map for Visualization
# =============================================================================

VOXCITY_COLOR_MAP = {
    # Special codes (negative)
    GROUND_CODE: (0.6, 0.4, 0.2),          # Brown for ground
    TREE_CODE: (0.2, 0.6, 0.2),            # Green for trees
    BUILDING_CODE: (0.5, 0.5, 0.5),        # Gray for buildings
    BRIDGE_CODE: (0.7, 0.7, 0.5),          # Tan for bridges
    CITY_FURNITURE_CODE: (0.4, 0.4, 0.6),  # Blue-gray for furniture
    
    # Land cover classes (positive, 1-based)
    1: (0.8, 0.7, 0.6),   # Bareland
    2: (0.6, 0.8, 0.4),   # Rangeland
    3: (0.4, 0.6, 0.3),   # Shrub
    4: (0.9, 0.8, 0.3),   # Agriculture
    5: (0.1, 0.5, 0.1),   # Tree
    6: (0.5, 0.7, 0.5),   # Moss/lichen
    7: (0.3, 0.5, 0.6),   # Wetland
    8: (0.2, 0.4, 0.3),   # Mangrove
    9: (0.2, 0.4, 0.8),   # Water
    10: (0.9, 0.95, 1.0), # Snow/ice
    11: (0.6, 0.6, 0.6),  # Developed space
    12: (0.3, 0.3, 0.3),  # Road
    13: (0.5, 0.5, 0.5),  # Building
    14: (0.7, 0.7, 0.7),  # No data
}

VOXCITY_CLASS_NAMES = {
    # Special codes
    GROUND_CODE: "Ground",
    TREE_CODE: "Tree",
    BUILDING_CODE: "Building",
    BRIDGE_CODE: "Bridge",
    CITY_FURNITURE_CODE: "City Furniture",
    
    # Land cover classes
    0: "Empty/Air",
    1: "Bareland",
    2: "Rangeland",
    3: "Shrub",
    4: "Agriculture",
    5: "Tree",
    6: "Moss/Lichen",
    7: "Wetland",
    8: "Mangrove",
    9: "Water",
    10: "Snow/Ice",
    11: "Developed Space",
    12: "Road",
    13: "Building",
    14: "No Data",
}


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_VOXEL_SIZE = 1.0
"""Default voxel size in meters."""

DEFAULT_GROUND_THICKNESS = 1
"""Default number of voxel layers for ground."""

DEFAULT_TRUNK_HEIGHT_RATIO = 11.76 / 19.98
"""Default ratio of trunk height to total tree height (~0.59)."""

MAX_RAY_INTERSECTIONS = 100
"""Maximum number of ray-triangle intersections to track during voxelization."""
