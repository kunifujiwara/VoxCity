"""
CityLES export module for VoxCity
Exports VoxCity grid data to CityLES input file format
Updated 2025/08/05 with corrected land use and building material codes
Integrated with VoxCity land cover utilities

Notes:
- This module expects raw land cover grids as produced per-source by VoxCity, not
  standardized/converted indices. Supported sources:
  'OpenStreetMap', 'Urbanwatch', 'OpenEarthMapJapan', 'ESA WorldCover',
  'ESRI 10m Annual Land Cover', 'Dynamic World V1'.
"""

import os
import numpy as np
from pathlib import Path


# VoxCity standard land cover classes after conversion
# Based on convert_land_cover function output
VOXCITY_STANDARD_CLASSES = {
    0: 'Bareland',
    1: 'Rangeland',
    2: 'Shrub',
    3: 'Agriculture land',
    4: 'Tree',
    5: 'Moss and lichen',
    6: 'Wet land',
    7: 'Mangrove',
    8: 'Water',
    9: 'Snow and ice',
    10: 'Developed space',
    11: 'Road',
    12: 'Building',
    13: 'No Data'
}

## Source-specific class name to CityLES land use mappings
# CityLES land use codes (updated to match provided definitions):
# 1: High reflective ASPHALT, 2: High reflective ASPHALT without AH,
# 3: CONCRETE (proxy of jari), 4: CONCRETE building,
# 5: Slate roof (Ordinal wooden house), 6: PADDY,
# 7: Dryland Cropland and Pasture, 8: Barren or Sparsely Vegetated,
# 9: WATER, 10: Grassland, 11: CONCRETE (proxy of block),
# 12: ASPHALT without AH, 13: ASPHALT,
# 14-17: Deciduous Broadleaf Forest

# OpenStreetMap / Standard (mapped to updated CityLES landuse codes)
OSM_CLASS_TO_CITYLES = {
    'Bareland': 8,
    'Rangeland': 10,
    'Shrub': 10,
    'Moss and lichen': 10,
    'Agriculture land': 7,
    'Tree': 14,
    'Wet land': 6,
    'Mangroves': 14,
    'Water': 9,
    'Snow and ice': 8,
    'Developed space': 4,
    'Road': 13,
    'Building': 4,
    'No Data': 10
}

# Urbanwatch
URBANWATCH_CLASS_TO_CITYLES = {
    'Building': 4,
    'Road': 13,
    'Parking Lot': 13,
    'Tree Canopy': 14,
    'Grass/Shrub': 10,
    'Agriculture': 7,
    'Water': 9,
    'Barren': 8,
    'Unknown': 10,
    'Sea': 9
}

# OpenEarthMapJapan
OEMJ_CLASS_TO_CITYLES = {
    'Bareland': 8,
    'Rangeland': 10,
    'Developed space': 4,
    'Road': 13,
    'Tree': 14,
    'Water': 9,
    'Agriculture land': 7,
    'Building': 4
}

# ESA WorldCover
ESA_CLASS_TO_CITYLES = {
    'Trees': 14,
    'Shrubland': 10,
    'Grassland': 10,
    'Cropland': 7,
    'Built-up': 4,
    'Barren / sparse vegetation': 8,
    'Snow and ice': 8,
    'Open water': 9,
    'Herbaceous wetland': 6,
    'Mangroves': 14,
    'Moss and lichen': 10
}

# ESRI 10m Annual Land Cover
ESRI_CLASS_TO_CITYLES = {
    'No Data': 10,
    'Water': 9,
    'Trees': 14,
    'Grass': 10,
    'Flooded Vegetation': 6,
    'Crops': 7,
    'Scrub/Shrub': 10,
    'Built Area': 4,
    'Bare Ground': 8,
    'Snow/Ice': 8,
    'Clouds': 10
}

# Dynamic World V1
DYNAMIC_WORLD_CLASS_TO_CITYLES = {
    'Water': 9,
    'Trees': 14,
    'Grass': 10,
    'Flooded Vegetation': 6,
    'Crops': 7,
    'Shrub and Scrub': 10,
    'Built': 4,
    'Bare': 8,
    'Snow and Ice': 8
}

# Building material mapping based on corrected definitions (101-117)
BUILDING_MATERIAL_MAPPING = {
    'building': 104,         # CONCRETE building
    'concrete': 104,         # CONCRETE building
    'residential': 105,      # Slate roof (Ordinal wooden house)
    'wooden': 105,           # Slate roof (Ordinal wooden house)
    'commercial': 104,       # CONCRETE building
    'industrial': 104,       # CONCRETE building
    'default': 104           # Default to CONCRETE building
}

# Helper to convert landuse code (1-17) to building material code (101-117)
def landuse_to_building_material_code(landuse_code: int) -> int:
    """Map landuse code to building-material code with required adjustments.

    The general rule is 100 + landuse_code, except for ASPHALT classes where
    landuse 12 (ASPHALT without AH) maps to 113 and landuse 13 (ASPHALT) maps to 112.
    """
    # if landuse_code == 12:
    #     return 113
    # if landuse_code == 13:
    #     return 112
    return 100 + int(landuse_code)

# Tree type mapping for vmap.txt
TREE_TYPE_MAPPING = {
    'deciduous': 101,        # Leaf
    'evergreen': 101,        # Leaf (simplified)
    'leaf': 101,             # Leaf
    'shade': 102,            # Shade
    'default': 101           # Default to leaf
}


def create_cityles_directories(output_directory):
    """Create necessary directories for CityLES output"""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _get_source_name_mapping(land_cover_source):
    """Return the class-name-to-CityLES mapping dictionary for the given source."""
    if land_cover_source == 'OpenStreetMap' or land_cover_source == 'Standard':
        return OSM_CLASS_TO_CITYLES
    if land_cover_source == 'Urbanwatch':
        return URBANWATCH_CLASS_TO_CITYLES
    if land_cover_source == 'OpenEarthMapJapan':
        return OEMJ_CLASS_TO_CITYLES
    if land_cover_source == 'ESA WorldCover':
        return ESA_CLASS_TO_CITYLES
    if land_cover_source == 'ESRI 10m Annual Land Cover':
        return ESRI_CLASS_TO_CITYLES
    if land_cover_source == 'Dynamic World V1':
        return DYNAMIC_WORLD_CLASS_TO_CITYLES
    # Default fallback
    return OSM_CLASS_TO_CITYLES


def _build_index_to_cityles_map(land_cover_source):
    """Build mapping: raw per-source index -> CityLES code, using source class order."""
    try:
        from voxcity.utils.lc import get_land_cover_classes
        class_dict = get_land_cover_classes(land_cover_source)
        class_names = list(class_dict.values())
    except Exception:
        # Fallback: no class list; return empty so default is used
        class_names = []

    name_to_code = _get_source_name_mapping(land_cover_source)
    index_to_code = {}
    for idx, class_name in enumerate(class_names):
        index_to_code[idx] = name_to_code.get(class_name, 4)
    return index_to_code, class_names


def export_topog(building_height_grid, building_id_grid, output_path, 
                 building_material='default', cityles_landuse_grid=None):
    """
    Export topog.txt file for CityLES
    
    Parameters:
    -----------
    building_height_grid : numpy.ndarray
        2D array of building heights
    building_id_grid : numpy.ndarray
        2D array of building IDs
    output_path : Path
        Output directory path
    building_material : str
        Building material type for mapping
    """
    filename = output_path / 'topog.txt'
    
    ny, nx = building_height_grid.shape
    material_code = BUILDING_MATERIAL_MAPPING.get(building_material, 
                                                  BUILDING_MATERIAL_MAPPING['default'])
    
    # Count only cells with building height > 0
    building_mask = building_height_grid > 0
    n_buildings = int(np.count_nonzero(building_mask))
    
    with open(filename, 'w') as f:
        # Write number of buildings
        f.write(f"{n_buildings}\n")
        
        # Write data for ALL grid points (buildings and non-buildings)
        for j in range(ny):
            for i in range(nx):
                # CityLES uses 1-based indexing
                i_1based = i + 1
                j_1based = j + 1
                height = float(building_height_grid[j, i])
                # Decide material code per cell
                if cityles_landuse_grid is not None:
                    cell_lu = int(cityles_landuse_grid[j, i])
                    material_code_cell = landuse_to_building_material_code(cell_lu)
                else:
                    if height > 0:
                        material_code_cell = material_code
                    else:
                        material_code_cell = 102
                # Format: i j height material_code depth1 depth2 changed_material
                f.write(f"{i_1based} {j_1based} {height:.1f} {material_code_cell} 0.0 0.0 102\n")


def export_landuse(land_cover_grid, output_path, land_cover_source=None):
    """
    Export landuse.txt file for CityLES
    
    Parameters:
    -----------
    land_cover_grid : numpy.ndarray
        2D array of land cover values (may be raw or converted)
    output_path : Path
        Output directory path
    land_cover_source : str, optional
        Source of land cover data
    """
    filename = output_path / 'landuse.txt'
    
    ny, nx = land_cover_grid.shape

    # Build per-source index mapping
    index_to_code, class_names = _build_index_to_cityles_map(land_cover_source)

    print(f"Land cover source: {land_cover_source} (raw indices)")

    # Create mapping statistics
    mapping_stats = {}
    # Prepare grid to return
    cityles_landuse_grid = np.zeros((ny, nx), dtype=int)

    with open(filename, 'w') as f:
        # Write in row-major order (j varies first, then i)
        for j in range(ny):
            for i in range(nx):
                idx = int(land_cover_grid[j, i])
                cityles_code = index_to_code.get(idx, 4)
                f.write(f"{cityles_code}\n")

                cityles_landuse_grid[j, i] = cityles_code

                # Track mapping statistics
                if idx not in mapping_stats:
                    mapping_stats[idx] = {'cityles_code': cityles_code, 'count': 0}
                mapping_stats[idx]['count'] += 1

    # Print mapping summary
    print("\nLand cover mapping summary (by source class):")
    total = ny * nx
    for idx in sorted(mapping_stats.keys()):
        stats = mapping_stats[idx]
        percentage = (stats['count'] / total) * 100
        class_name = class_names[idx] if 0 <= idx < len(class_names) else 'Unknown'
        print(f"  {idx}: {class_name} -> CityLES {stats['cityles_code']}: "
              f"{stats['count']} cells ({percentage:.1f}%)")
    
    return cityles_landuse_grid


def export_dem(dem_grid, output_path):
    """
    Export dem.txt file for CityLES
    
    Parameters:
    -----------
    dem_grid : numpy.ndarray
        2D array of elevation values
    output_path : Path
        Output directory path
    """
    filename = output_path / 'dem.txt'
    
    ny, nx = dem_grid.shape
    
    with open(filename, 'w') as f:
        for j in range(ny):
            for i in range(nx):
                # CityLES uses 1-based indexing
                i_1based = i + 1
                j_1based = j + 1
                elevation = float(dem_grid[j, i])
                # Clamp negative elevations to 0.0 meters
                if elevation < 0.0:
                    elevation = 0.0
                f.write(f"{i_1based} {j_1based} {elevation:.1f}\n")


def export_vmap(canopy_height_grid, output_path, tree_base_ratio=0.3, tree_type='default', building_height_grid=None, canopy_bottom_height_grid=None):
    """
    Export vmap.txt file for CityLES
    
    Parameters:
    -----------
    canopy_height_grid : numpy.ndarray
        2D array of canopy heights
    output_path : Path
        Output directory path
    tree_base_ratio : float
        Ratio of tree base height to total canopy height
    tree_type : str
        Tree type for mapping
    """
    filename = output_path / 'vmap.txt'
    
    ny, nx = canopy_height_grid.shape
    tree_code = TREE_TYPE_MAPPING.get(tree_type, TREE_TYPE_MAPPING['default'])
    
    # If building heights are provided, remove trees where buildings exist
    if building_height_grid is not None:
        effective_canopy = np.where(building_height_grid > 0, 0.0, canopy_height_grid)
    else:
        effective_canopy = canopy_height_grid
    
    # Count only cells with canopy height > 0
    vegetation_mask = effective_canopy > 0
    n_trees = int(np.count_nonzero(vegetation_mask))
    
    with open(filename, 'w') as f:
        # Write number of trees
        f.write(f"{n_trees}\n")
        
        # Write data for ALL grid points (vegetation and non-vegetation)
        for j in range(ny):
            for i in range(nx):
                # CityLES uses 1-based indexing
                i_1based = i + 1
                j_1based = j + 1
                total_height = float(effective_canopy[j, i])
                if canopy_bottom_height_grid is not None:
                    lower_height = float(np.clip(canopy_bottom_height_grid[j, i], 0.0, total_height))
                else:
                    lower_height = total_height * tree_base_ratio
                upper_height = total_height
                # Format: i j lower_height upper_height tree_type
                f.write(f"{i_1based} {j_1based} {lower_height:.1f} {upper_height:.1f} {tree_code}\n")


def export_lonlat(rectangle_vertices, grid_shape, output_path):
    """
    Export lonlat.txt file for CityLES
    
    Parameters:
    -----------
    rectangle_vertices : list of tuples
        List of (lon, lat) vertices defining the area
    grid_shape : tuple
        Shape of the grid (ny, nx)
    output_path : Path
        Output directory path
    """
    filename = output_path / 'lonlat.txt'
    
    ny, nx = grid_shape
    
    # Extract bounds from vertices
    lons = [v[0] for v in rectangle_vertices]
    lats = [v[1] for v in rectangle_vertices]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    # Create coordinate grids
    lon_vals = np.linspace(min_lon, max_lon, nx)
    lat_vals = np.linspace(min_lat, max_lat, ny)
    
    with open(filename, 'w') as f:
        for j in range(ny):
            for i in range(nx):
                # CityLES uses 1-based indexing
                i_1based = i + 1
                j_1based = j + 1
                lon = lon_vals[i]
                lat = lat_vals[j]
                
                # Note: Format is i j longitude latitude (not latitude longitude)
                f.write(f"{i_1based} {j_1based} {lon:.7f} {lat:.8f}\n")


def export_cityles(building_height_grid, building_id_grid, canopy_height_grid,
                   land_cover_grid, dem_grid, meshsize, land_cover_source,
                   rectangle_vertices, output_directory="output/cityles",
                   building_material='default', tree_type='default',
                   tree_base_ratio=0.3, canopy_bottom_height_grid=None, **kwargs):
    """
    Export VoxCity data to CityLES format
    
    Parameters:
    -----------
    building_height_grid : numpy.ndarray
        2D array of building heights
    building_id_grid : numpy.ndarray
        2D array of building IDs
    canopy_height_grid : numpy.ndarray
        2D array of canopy heights
    land_cover_grid : numpy.ndarray
        2D array of land cover values (may be raw or VoxCity standard)
    dem_grid : numpy.ndarray
        2D array of elevation values
    meshsize : float
        Grid cell size in meters
    land_cover_source : str
        Source of land cover data (e.g., 'ESRI 10m Annual Land Cover', 'ESA WorldCover')
    rectangle_vertices : list of tuples
        List of (lon, lat) vertices defining the area
    output_directory : str
        Output directory path
    building_material : str
        Building material type for mapping
    tree_type : str
        Tree type for mapping
    tree_base_ratio : float
        Ratio of tree base height to total canopy height
    **kwargs : dict
        Additional parameters (for compatibility)
    
    Returns:
    --------
    str : Path to output directory
    """
    # Create output directory
    output_path = create_cityles_directories(output_directory)
    
    print(f"Exporting CityLES files to: {output_path}")
    print(f"Land cover source: {land_cover_source}")
    
    # Export individual files
    print("\nExporting landuse.txt...")
    cityles_landuse_grid = export_landuse(land_cover_grid, output_path, land_cover_source)

    print("\nExporting topog.txt...")
    export_topog(
        building_height_grid,
        building_id_grid,
        output_path,
        building_material,
        cityles_landuse_grid=cityles_landuse_grid,
    )
    
    print("\nExporting dem.txt...")
    export_dem(dem_grid, output_path)
    
    print("\nExporting vmap.txt...")
    export_vmap(canopy_height_grid, output_path, tree_base_ratio, tree_type, building_height_grid=building_height_grid, canopy_bottom_height_grid=canopy_bottom_height_grid)
    
    print("\nExporting lonlat.txt...")
    export_lonlat(rectangle_vertices, building_height_grid.shape, output_path)
    
    # Create metadata file for reference
    metadata_file = output_path / 'cityles_metadata.txt'
    with open(metadata_file, 'w') as f:
        f.write("CityLES Export Metadata\n")
        f.write("====================\n")
        f.write(f"Export date: 2025/08/05\n")
        f.write(f"Grid shape: {building_height_grid.shape}\n")
        f.write(f"Mesh size: {meshsize} m\n")
        f.write(f"Land cover source: {land_cover_source}\n")
        f.write(f"Building material: {building_material}\n")
        f.write(f"Tree type: {tree_type}\n")
        f.write(f"Bounds: {rectangle_vertices}\n")
        f.write(f"Buildings: {np.sum(building_height_grid > 0)}\n")
        # Trees count after removing overlaps with buildings
        trees_count = int(np.sum(np.where(building_height_grid > 0, 0.0, canopy_height_grid) > 0))
        f.write(f"Trees: {trees_count}\n")
        
        # Add land use value ranges
        f.write(f"\nLand cover value range: {land_cover_grid.min()} - {land_cover_grid.max()}\n")
        unique_values = np.unique(land_cover_grid)
        f.write(f"Unique land cover values: {unique_values}\n")
    
    print(f"\nCityLES export completed successfully!")
    return str(output_path)


# Helper function to apply VoxCity's convert_land_cover if needed
def ensure_converted_land_cover(land_cover_grid, land_cover_source):
    """
    Ensure land cover grid uses VoxCity standard indices
    
    This function checks if the land cover data needs conversion and applies
    VoxCity's convert_land_cover function if necessary.
    
    Parameters:
    -----------
    land_cover_grid : numpy.ndarray
        2D array of land cover values
    land_cover_source : str
        Source of land cover data
        
    Returns:
    --------
    numpy.ndarray : Land cover grid with VoxCity standard indices (0-13)
    """
    # Import VoxCity's convert function if available
    try:
        from voxcity.utils.lc import convert_land_cover
        
        # Apply conversion
        converted_grid = convert_land_cover(land_cover_grid, land_cover_source)
        print(f"Applied VoxCity land cover conversion for {land_cover_source}")
        return converted_grid
    except ImportError:
        print("Warning: Could not import VoxCity land cover utilities. Using direct mapping.")
        return land_cover_grid