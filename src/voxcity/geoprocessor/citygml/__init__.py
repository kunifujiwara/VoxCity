"""
VoxCity CityGML Subpackage
==========================

This subpackage provides comprehensive functionality for parsing and voxelizing
CityGML LOD2 building data from various sources:

- **German CityGML**: Standard European CityGML format (ETRS89_UTM coordinates)
- **Japanese PLATEAU**: Japan's 3D city model dataset (JGD2011/EPSG:6697 coordinates)

The subpackage integrates with VoxCity's existing pipeline for seamless 3D city generation.

Main Components
---------------
- parsers: CityGML and PLATEAU file parsers
- voxelizer: 3D voxelization from triangulated meshes
- models: Data structures for buildings, terrain, vegetation, etc.
- utils: Coordinate transformations, mesh code utilities, constants

Usage Examples
--------------
Basic German CityGML voxelization::

    from voxcity.geoprocessor.citygml import CityGMLVoxelizer

    voxelizer = CityGMLVoxelizer(voxel_size=1.0)
    voxelizer.load_dtm("path/to/dtm.tif")  # Optional terrain
    voxelizer.parse_citygml("path/to/building.gml")
    voxel_grid = voxelizer.voxelize()

Japanese PLATEAU voxelization::

    from voxcity.geoprocessor.citygml import PLATEAUVoxelizer

    voxelizer = PLATEAUVoxelizer(voxel_size=1.0)
    voxelizer.parse_plateau_directory("path/to/plateau_data")
    voxel_grid = voxelizer.voxelize(
        include_vegetation=True,
        include_bridges=True
    )

Integration with VoxCity pipeline::

    from voxcity.geoprocessor.citygml import PLATEAUParser
    from voxcity.generator import Voxelizer

    # Parse PLATEAU data
    parser = PLATEAUParser()
    parser.parse_plateau_directory("path/to/plateau_data")
    
    # Use with standard VoxCity pipeline
    # ...
"""

# Core parsers
from .parsers import (
    CityGMLParser,
    PLATEAUParser,
    LOD1CityGMLParser,
    load_citygml,
    load_plateau_citygml,
    load_lod1_citygml,
    parse_pos_list,
    triangulate_polygon,
    compute_triangle_normal,
    detect_citygml_format,  # Format detection function
)

# Voxelizers
from .voxelizer import (
    CityGMLVoxelizer,
    PLATEAUVoxelizer,
    GenericCityGMLVoxelizer,
    parse_citygml_subset,
    # CityGML pipeline helpers
    resolve_citygml_path,
    voxelize_buildings_citygml,
    voxelize_trees_citygml,
    voxelize_terrain_citygml,
    apply_citygml_post_processing,
    merge_lod2_voxels,
    # Optimized versions (with caching support)
    voxelize_buildings_citygml_optimized,
    voxelize_trees_citygml_optimized,
    voxelize_terrain_citygml_optimized,
)

# Data models
from .models import (
    Building,
    PLATEAUBuilding,
    PLATEAUVegetation,
    PLATEAUBridge,
    PLATEAUCityFurniture,
    PLATEAULandUse,
    TerrainTriangle,
)

# Utilities
from .utils import (
    CoordinateTransformer,
    decode_mesh_code,
    decode_2nd_level_mesh,
    get_mesh_code_from_filename,
    get_voxcity_landcover_code,
)

# Constants
from .constants import (
    # VoxCity voxel semantic codes
    GROUND_CODE,
    TREE_CODE,
    BUILDING_CODE,
    BRIDGE_CODE,
    CITY_FURNITURE_CODE,
    # CityGML namespaces
    NAMESPACES,
    PLATEAU_NAMESPACES,
    # Color maps
    VOXCITY_COLOR_MAP,
    VOXCITY_CLASS_NAMES,
    # Land use mapping
    PLATEAU_LANDUSE_TO_VOXCITY,
)

# Note: For visualization, use voxcity.visualizer module which provides
# comprehensive visualization capabilities including:
#   - visualize_voxcity_plotly() for interactive 3D views
#   - visualize_voxcity() for general voxel visualization
#   - PyVistaRenderer and GPURenderer for high-quality rendering

__all__ = [
    # Parsers
    "CityGMLParser",
    "PLATEAUParser",
    "LOD1CityGMLParser",
    "load_citygml",
    "load_plateau_citygml",
    "load_lod1_citygml",
    "parse_pos_list",
    "triangulate_polygon",
    "compute_triangle_normal",
    # Voxelizers
    "CityGMLVoxelizer",
    "PLATEAUVoxelizer",
    "GenericCityGMLVoxelizer",
    "parse_citygml_subset",
    # CityGML pipeline helpers
    "detect_citygml_format",
    "resolve_citygml_path",
    "voxelize_buildings_citygml",
    "voxelize_trees_citygml",
    "voxelize_terrain_citygml",
    "apply_citygml_post_processing",
    "merge_lod2_voxels",
    # Optimized versions (with caching support)
    "voxelize_buildings_citygml_optimized",
    "voxelize_trees_citygml_optimized",
    "voxelize_terrain_citygml_optimized",
    # Models
    "Building",
    "PLATEAUBuilding",
    "PLATEAUVegetation",
    "PLATEAUBridge",
    "PLATEAUCityFurniture",
    "PLATEAULandUse",
    "TerrainTriangle",
    # Utilities
    "CoordinateTransformer",
    "decode_mesh_code",
    "decode_2nd_level_mesh",
    "get_mesh_code_from_filename",
    "get_voxcity_landcover_code",
    # Constants
    "GROUND_CODE",
    "TREE_CODE",
    "BUILDING_CODE",
    "BRIDGE_CODE",
    "CITY_FURNITURE_CODE",
    "NAMESPACES",
    "PLATEAU_NAMESPACES",
    "VOXCITY_COLOR_MAP",
    "VOXCITY_CLASS_NAMES",
    "PLATEAU_LANDUSE_TO_VOXCITY",
]
