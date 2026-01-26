"""VoxCity generator subpackage.

This package organizes the voxel city generation pipeline into focused modules
while preserving the original public API under `voxcity.generator`.

Orientation contract:
- All 2D grids use north_up orientation (row 0 = north/top; columns increase eastward).
- 3D indexing follows (row, col, z) = (north→south, west→east, ground→up).
"""

from .api import get_voxcity, get_voxcity_CityGML, auto_select_data_sources
from .grids import (
    get_land_cover_grid,
    get_building_height_grid,
    get_canopy_height_grid,
    get_dem_grid,
)
from .voxelizer import (
    Voxelizer,
    GROUND_CODE,
    TREE_CODE,
    BUILDING_CODE,
)
from .pipeline import VoxCityPipeline
from .io import save_voxcity, load_voxcity
from .update import update_voxcity, regenerate_voxels

# CityGML subpackage for LOD2 parsing and voxelization
from ..geoprocessor import citygml

__all__ = [
    "get_voxcity",
    "auto_select_data_sources",
    "get_voxcity_CityGML",
    "get_land_cover_grid",
    "get_building_height_grid",
    "get_canopy_height_grid",
    "get_dem_grid",
    "Voxelizer",
    "GROUND_CODE",
    "TREE_CODE",
    "BUILDING_CODE",
    "VoxCityPipeline",
    "save_voxcity",
    "load_voxcity",
    "update_voxcity",
    "regenerate_voxels",
    # CityGML subpackage
    "citygml",
]


