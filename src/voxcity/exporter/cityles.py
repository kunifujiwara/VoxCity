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
from ..models import VoxCity


# VoxCity standard land cover classes after conversion
# Based on convert_land_cover function output (1-based indices)
VOXCITY_STANDARD_CLASSES = {
    1: 'Bareland',
    2: 'Rangeland',
    3: 'Shrub',
    4: 'Agriculture land',
    5: 'Tree',
    6: 'Moss and lichen',
    7: 'Wet land',
    8: 'Mangrove',
    9: 'Water',
    10: 'Snow and ice',
    11: 'Developed space',
    12: 'Road',
    13: 'Building',
    14: 'No Data'
}

## Source-specific class name to CityLES land use mappings
# CityLES land use codes: 1=Water, 2=Rice Paddy, 3=Crops, 4=Grassland, 5=Deciduous Broadleaf Forest,
# 9=Bare Land, 10=Building, 16=Asphalt (road), etc.

# OpenStreetMap / Standard
OSM_CLASS_TO_CITYLES = {
    'Bareland': 9,
    'Rangeland': 4,
    'Shrub': 4,
    'Moss and lichen': 4,
    'Agriculture land': 3,
    'Tree': 5,
    'Wet land': 2,
    'Mangroves': 5,
    'Water': 1,
    'Snow and ice': 9,
    'Developed space': 10,
    'Road': 16,
    'Building': 10,
    'No Data': 4
}

# Urbanwatch
URBANWATCH_CLASS_TO_CITYLES = {
    'Building': 10,
    'Road': 16,
    'Parking Lot': 16,
    'Tree Canopy': 5,
    'Grass/Shrub': 4,
    'Agriculture': 3,
    'Water': 1,
    'Barren': 9,
    'Unknown': 4,
    'Sea': 1
}

# OpenEarthMapJapan
OEMJ_CLASS_TO_CITYLES = {
    'Bareland': 9,
    'Rangeland': 4,
    'Developed space': 10,
    'Road': 16,
    'Tree': 5,
    'Water': 1,
    'Agriculture land': 3,
    'Building': 10
}

# ESA WorldCover
ESA_CLASS_TO_CITYLES = {
    'Trees': 5,
    'Shrubland': 4,
    'Grassland': 4,
    'Cropland': 3,
    'Built-up': 10,
    'Barren / sparse vegetation': 9,
    'Snow and ice': 9,
    'Open water': 1,
    'Herbaceous wetland': 2,
    'Mangroves': 5,
    'Moss and lichen': 9
}

# ESRI 10m Annual Land Cover
ESRI_CLASS_TO_CITYLES = {
    'No Data': 4,
    'Water': 1,
    'Trees': 5,
    'Grass': 4,
    'Flooded Vegetation': 2,
    'Crops': 3,
    'Scrub/Shrub': 4,
    'Built Area': 10,
    'Bare Ground': 9,
    'Snow/Ice': 9,
    'Clouds': 4
}

# Dynamic World V1
DYNAMIC_WORLD_CLASS_TO_CITYLES = {
    'Water': 1,
    'Trees': 5,
    'Grass': 4,
    'Flooded Vegetation': 2,
    'Crops': 3,
    'Shrub and Scrub': 4,
    'Built': 10,
    'Bare': 9,
    'Snow and Ice': 9
}

# Building material mapping based on corrected documentation
BUILDING_MATERIAL_MAPPING = {
    'building': 110,         # Building (general)
    'concrete': 110,         # Building (concrete)
    'residential': 111,      # Old wooden house
    'wooden': 111,           # Old wooden house
    'commercial': 110,       # Building (commercial)
    'industrial': 110,       # Building (industrial)
    'default': 110           # Default to general building
}

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


def _resolve_under_tree_code(under_tree_class_name, under_tree_cityles_code, land_cover_source):
    """Resolve the CityLES land-use code used under tree canopy.

    Priority:
    1) Explicit numeric code if provided
    2) Class name using the source-specific mapping
    3) Class name using the standard (OSM) mapping
    4) Default to 9 (Bare Land)
    """
    if under_tree_cityles_code is not None:
        try:
            return int(under_tree_cityles_code)
        except Exception:
            pass
    name_to_code = _get_source_name_mapping(land_cover_source)
    code = name_to_code.get(under_tree_class_name)
    if code is None:
        code = OSM_CLASS_TO_CITYLES.get(under_tree_class_name, 9)
    return code


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
                    material_code_cell = cell_lu + 100
                else:
                    if height > 0:
                        material_code_cell = material_code
                    else:
                        material_code_cell = 102
                # Format: i j height material_code depth1 depth2 changed_material
                f.write(f"{i_1based} {j_1based} {height:.1f} {material_code_cell} 0.0 0.0 102\n")


def export_landuse(land_cover_grid, output_path, land_cover_source=None,
                   canopy_height_grid=None, building_height_grid=None,
                   under_tree_class_name='Bareland', under_tree_cityles_code=None):
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
    canopy_height_grid : numpy.ndarray, optional
        2D array of canopy heights; if provided, cells with canopy (>0) will be
        assigned the ground class under the canopy instead of a tree class.
    building_height_grid : numpy.ndarray, optional
        2D array of building heights; if provided, canopy overrides will not be
        applied where buildings exist (height > 0).
    under_tree_class_name : str, optional
        Name of ground land-cover class to use under tree canopy. Defaults to 'Bareland'.
    under_tree_cityles_code : int, optional
        Explicit CityLES land-use code to use under canopy; if provided it takes
        precedence over under_tree_class_name.
    """
    filename = output_path / 'landuse.txt'
    
    ny, nx = land_cover_grid.shape

    # Build per-source index mapping
    index_to_code, class_names = _build_index_to_cityles_map(land_cover_source)

    print(f"Land cover source: {land_cover_source} (raw indices)")

    # Resolve the CityLES code to use under tree canopy
    under_tree_code = _resolve_under_tree_code(
        under_tree_class_name, under_tree_cityles_code, land_cover_source
    )

    # Create mapping statistics: per raw index, count per resulting CityLES code
    mapping_stats = {}
    # Prepare grid to return
    cityles_landuse_grid = np.zeros((ny, nx), dtype=int)

    with open(filename, 'w') as f:
        # Write in row-major order (j varies first, then i)
        for j in range(ny):
            for i in range(nx):
                idx = int(land_cover_grid[j, i])
                cityles_code = index_to_code.get(idx, 4)

                # If a canopy grid is provided, override tree canopy cells to the
                # specified ground class, optionally skipping where buildings exist.
                if canopy_height_grid is not None:
                    has_canopy = float(canopy_height_grid[j, i]) > 0.0
                    has_building = False
                    if building_height_grid is not None:
                        has_building = float(building_height_grid[j, i]) > 0.0
                    if has_canopy and not has_building:
                        cityles_code = under_tree_code

                # Override land use to Building (10) for cells with building height
                if building_height_grid is not None:
                    if float(building_height_grid[j, i]) > 0.0:
                        cityles_code = 10

                f.write(f"{cityles_code}\n")

                cityles_landuse_grid[j, i] = cityles_code

                # Track mapping statistics
                if idx not in mapping_stats:
                    mapping_stats[idx] = {}
                mapping_stats[idx][cityles_code] = mapping_stats[idx].get(cityles_code, 0) + 1

    # Print mapping summary
    print("\nLand cover mapping summary (by source class):")
    total = ny * nx
    for idx in sorted(mapping_stats.keys()):
        class_name = class_names[idx] if 0 <= idx < len(class_names) else 'Unknown'
        for code, count in sorted(mapping_stats[idx].items()):
            percentage = (count / total) * 100
            print(f"  {idx}: {class_name} -> CityLES {code}: {count} cells ({percentage:.1f}%)")
    
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


def export_vmap(canopy_height_grid, output_path, trunk_height_ratio=0.3, tree_type='default', building_height_grid=None, canopy_bottom_height_grid=None):
    """
    Export vmap.txt file for CityLES
    
    Parameters:
    -----------
    canopy_height_grid : numpy.ndarray
        2D array of canopy heights
    output_path : Path
        Output directory path
    trunk_height_ratio : float
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
                    lower_height = total_height * trunk_height_ratio
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


def export_cityles(city: VoxCity,
                   output_directory: str = "output/cityles",
                   building_material: str = 'default',
                   tree_type: str = 'default',
                   trunk_height_ratio: float = 0.3,
                   canopy_bottom_height_grid=None,
                   under_tree_class_name: str = 'Bareland',
                   under_tree_cityles_code=None,
                   land_cover_source: str | None = None,
                   **kwargs):
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
    trunk_height_ratio : float
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
    # Resolve data from VoxCity
    building_height_grid = city.buildings.heights
    building_id_grid = city.buildings.ids if city.buildings.ids is not None else np.zeros_like(building_height_grid, dtype=int)
    canopy_height_grid = city.tree_canopy.top if city.tree_canopy is not None else np.zeros_like(city.land_cover.classes, dtype=float)
    land_cover_grid = city.land_cover.classes
    dem_grid = city.dem.elevation
    meshsize = float(city.voxels.meta.meshsize)
    rectangle_vertices = city.extras.get("rectangle_vertices") or [(0.0, 0.0)] * 4
    land_cover_source = land_cover_source or city.extras.get("land_cover_source", "Standard")

    print(f"Land cover source: {land_cover_source}")
    
    # Export individual files
    print("\nExporting landuse.txt...")
    cityles_landuse_grid = export_landuse(
        land_cover_grid,
        output_path,
        land_cover_source,
        canopy_height_grid=canopy_height_grid,
        building_height_grid=building_height_grid,
        under_tree_class_name=under_tree_class_name,
        under_tree_cityles_code=under_tree_cityles_code,
    )

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
    export_vmap(canopy_height_grid, output_path, trunk_height_ratio, tree_type, building_height_grid=building_height_grid, canopy_bottom_height_grid=canopy_bottom_height_grid)
    
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
        # Under-tree land-use selection
        under_tree_code = _resolve_under_tree_code(
            under_tree_class_name, under_tree_cityles_code, land_cover_source
        )
        f.write(f"Under-tree land use: {under_tree_class_name} (CityLES {under_tree_code})\n")
        
        # Add land use value ranges
        f.write(f"\nLand cover value range: {land_cover_grid.min()} - {land_cover_grid.max()}\n")
        unique_values = np.unique(land_cover_grid)
        f.write(f"Unique land cover values: {unique_values}\n")
    
    print(f"\nCityLES export completed successfully!")
    return str(output_path)


class CityLesExporter:
    """Exporter adapter to write a VoxCity model to CityLES text files."""

    def export(self, obj, output_directory: str, base_filename: str, **kwargs):
        if not isinstance(obj, VoxCity):
            raise TypeError("CityLesExporter expects a VoxCity instance")
        city: VoxCity = obj
        # CityLES writes multiple files; use output_directory/base_filename as folder/name
        out_dir = os.path.join(output_directory, base_filename)
        os.makedirs(out_dir, exist_ok=True)
        export_cityles(
            city,
            output_directory=out_dir,
            **kwargs,
        )
        return out_dir


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