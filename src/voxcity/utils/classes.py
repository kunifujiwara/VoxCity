"""
VoxCity Class Definitions

This module provides standard class definitions for voxel grid semantics
and land cover classification used throughout VoxCity.

Voxel Grid Semantics:
    The 3D voxel grid uses integer codes to represent different urban elements.
    Negative values represent structural elements, while positive values 
    represent land cover classes at the ground surface layer.

Land Cover Classes:
    Land cover is standardized to a 1-based indexing system (1-14) for 
    consistency across different data sources. This standard classification
    is used in the voxel representation and exports.
"""

import numpy as np
from typing import Dict, Optional


# =============================================================================
# Voxel Semantic Codes
# =============================================================================

VOXEL_CODES = {
    -3: "Building",
    -2: "Tree Canopy",
    -1: "Ground/Subsurface",
    # Positive values (>=1) represent land cover classes at ground surface
}

VOXEL_CODE_DESCRIPTIONS = """
Voxel Grid Semantic Codes:
  -3 : Building volume
  -2 : Tree canopy (vegetation)
  -1 : Ground/Subsurface
  >=1: Land cover class at ground surface (see Land Cover Classes)
"""


# =============================================================================
# Standard Land Cover Classes (1-based indices)
# =============================================================================

LAND_COVER_CLASSES = {
    1: "Bareland",
    2: "Rangeland",
    3: "Shrub",
    4: "Agriculture land",
    5: "Tree",
    6: "Moss and lichen",
    7: "Wet land",
    8: "Mangrove",
    9: "Water",
    10: "Snow and ice",
    11: "Developed space",
    12: "Road",
    13: "Building",
    14: "No Data",
}

LAND_COVER_DESCRIPTIONS = """
VoxCity Standard Land Cover Classes (1-based indices, used in voxel grids):
--------------------------------------------------
   1: Bareland           - Bare soil, rocks, desert
   2: Rangeland          - Grassland, pasture
   3: Shrub              - Shrubland, bushes
   4: Agriculture land   - Cropland, farmland
   5: Tree               - Forest, tree cover
   6: Moss and lichen    - Moss, lichen cover
   7: Wet land           - Wetland, marsh
   8: Mangrove           - Mangrove forest
   9: Water              - Water bodies
  10: Snow and ice       - Snow, ice, glaciers
  11: Developed space    - Urban areas, parking
  12: Road               - Roads, paved surfaces
  13: Building           - Building footprints
  14: No Data            - Missing or invalid data
--------------------------------------------------
Note: Source-specific land cover classes are converted to these
      standard classes during voxelization.
"""


# =============================================================================
# Print Helper Functions
# =============================================================================

def print_voxel_codes() -> None:
    """Print voxel semantic codes to console."""
    print(VOXEL_CODE_DESCRIPTIONS)


def print_land_cover_classes() -> None:
    """Print standard land cover class definitions to console."""
    print(LAND_COVER_DESCRIPTIONS)


def print_class_definitions() -> None:
    """Print both voxel codes and land cover class definitions."""
    print("\n" + "=" * 60)
    print("VoxCity Class Definitions")
    print("=" * 60)
    print(VOXEL_CODE_DESCRIPTIONS)
    print(LAND_COVER_DESCRIPTIONS)
    print("=" * 60 + "\n")


def get_land_cover_name(index: int) -> str:
    """
    Get the land cover class name for a given index.
    
    Args:
        index: Land cover class index (1-14)
        
    Returns:
        Class name string, or "Unknown" if index is invalid
    """
    return LAND_COVER_CLASSES.get(index, "Unknown")


def get_voxel_code_name(code: int) -> str:
    """
    Get the semantic name for a voxel code.
    
    Args:
        code: Voxel code (negative for structures, positive for land cover)
        
    Returns:
        Semantic name string
    """
    if code in VOXEL_CODES:
        return VOXEL_CODES[code]
    elif code >= 1:
        return f"Land Cover: {get_land_cover_name(code)}"
    elif code == 0:
        return "Empty/Air"
    else:
        return "Unknown"


def summarize_voxel_grid(voxel_grid: np.ndarray, print_output: bool = True) -> Dict[int, int]:
    """
    Summarize the contents of a voxel grid.
    
    Args:
        voxel_grid: 3D numpy array of voxel codes
        print_output: Whether to print the summary
        
    Returns:
        Dictionary mapping voxel codes to counts
    """
    unique, counts = np.unique(voxel_grid, return_counts=True)
    summary = dict(zip(unique.tolist(), counts.tolist()))
    
    if print_output:
        print("\nVoxel Grid Summary:")
        print("-" * 40)
        for code in sorted(summary.keys()):
            name = get_voxel_code_name(code)
            count = summary[code]
            percentage = 100.0 * count / voxel_grid.size
            print(f"  {code:4d}: {name:25s} - {count:,} voxels ({percentage:.2f}%)")
        print("-" * 40)
    
    return summary


def summarize_land_cover_grid(land_cover_grid: np.ndarray, print_output: bool = True) -> Dict[int, int]:
    """
    Summarize the contents of a land cover grid.
    
    Args:
        land_cover_grid: 2D numpy array of land cover class indices
        print_output: Whether to print the summary
        
    Returns:
        Dictionary mapping class indices to counts
    """
    unique, counts = np.unique(land_cover_grid, return_counts=True)
    summary = dict(zip(unique.tolist(), counts.tolist()))
    
    if print_output:
        print("\nLand Cover Grid Summary:")
        print("-" * 40)
        for idx in sorted(summary.keys()):
            name = get_land_cover_name(idx) if idx >= 1 else f"Source-specific ({idx})"
            count = summary[idx]
            percentage = 100.0 * count / land_cover_grid.size
            print(f"  {idx:4d}: {name:25s} - {count:,} cells ({percentage:.2f}%)")
        print("-" * 40)
    
    return summary
