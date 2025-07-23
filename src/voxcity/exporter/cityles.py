"""
CityLES Exporter Module for VoxCity

This module provides functionality to export VoxCity grid data to the CityLES input file format.
CityLES is a large-eddy simulation (LES) model for urban environments, requiring specific input files
describing land use, building geometry, vegetation, and terrain.

Key Features:
    - Converts VoxCity grids to CityLES-compatible input files (topog.txt, landuse.txt, dem.txt, vmap.txt, lonlat.txt)
    - Handles land cover, building heights, canopy heights, and digital elevation models
    - Supports flexible mapping from land cover and building types to CityLES codes
    - Generates all required text files and metadata for CityLES runs

Main Functions:
    - export_cityles: Main function to export all required CityLES input files
    - export_topog: Exports building geometry (topog.txt)
    - export_landuse: Exports land use grid (landuse.txt)
    - export_dem: Exports digital elevation model (dem.txt)
    - export_vmap: Exports vegetation map (vmap.txt)
    - export_lonlat: Exports longitude/latitude grid (lonlat.txt)

Dependencies:
    - numpy: For array operations
    - pathlib: For file and directory management
    - os: For file system operations
"""

import os
import numpy as np
from pathlib import Path


# Land cover to CityLES land use mapping
# Based on common land cover classifications to CityLES codes
LANDCOVER_TO_CITYLES_LANDUSE = {
    # Built-up areas
    'building': 4,           # Concrete building
    'road': 2,              # High reflective asphalt without AH
    'parking': 2,           # High reflective asphalt without AH
    'pavement': 11,         # Concrete (proxy of block)
    
    # Vegetation
    'grass': 10,            # Grassland
    'forest': 16,           # Deciduous broadleaf forest
    'tree': 16,             # Deciduous broadleaf forest
    'agriculture': 7,       # Dryland cropland and pasture
    'cropland': 7,          # Dryland cropland and pasture
    'paddy': 6,             # Paddy
    
    # Water and bare land
    'water': 9,             # Water
    'bare_soil': 8,         # Barren or sparsely vegetated
    'sand': 8,              # Barren or sparsely vegetated
    
    # Default
    'default': 10           # Grassland as default
}

# Building material mapping
# Maps building types to CityLES building attribute codes
BUILDING_MATERIAL_MAPPING = {
    'concrete': 104,        # Concrete building
    'residential': 105,     # Slate roof (ordinal wooden house)
    'commercial': 104,      # Concrete building
    'industrial': 104,      # Concrete building
    'default': 104          # Default to concrete building
}

# Tree type mapping
TREE_TYPE_MAPPING = {
    'deciduous': 101,       # Leaf
    'evergreen': 101,       # Leaf (simplified)
    'default': 101          # Default to leaf
}


def create_cityles_directories(output_directory):
    """Create necessary directories for CityLES output"""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def get_land_use_code(land_cover_value, land_cover_source=None):
    """
    Convert land cover value to CityLES land use code
    
    Parameters:
    -----------
    land_cover_value : int or str
        Land cover value from VoxCity
    land_cover_source : str, optional
        Source of land cover data (e.g., 'esri', 'esa', 'osm')
    
    Returns:
    --------
    int : CityLES land use code (1-17)
    """
    # If using numeric codes, you might need source-specific mappings
    # This is a simplified example
    if isinstance(land_cover_value, str):
        return LANDCOVER_TO_CITYLES_LANDUSE.get(land_cover_value.lower(), 
                                                LANDCOVER_TO_CITYLES_LANDUSE['default'])
    
    # Example mapping for ESRI land cover (adjust based on actual data source)
    if land_cover_source == 'esri':
        esri_mapping = {
            1: 9,    # Water -> Water
            2: 16,   # Trees -> Deciduous broadleaf forest  
            4: 8,    # Flooded vegetation -> Barren
            5: 10,   # Crops -> Grassland (simplified)
            7: 4,    # Built Area -> Concrete building
            8: 8,    # Bare ground -> Barren
            9: 3,    # Snow/Ice -> Concrete (proxy of jari)
            10: 9,   # Clouds -> Water (simplified)
            11: 10   # Rangeland -> Grassland
        }
        return esri_mapping.get(land_cover_value, LANDCOVER_TO_CITYLES_LANDUSE['default'])
    
    # Default mapping
    return LANDCOVER_TO_CITYLES_LANDUSE['default']


def export_topog(building_height_grid, building_id_grid, output_path, 
                 building_material='default'):
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
    
    # Get building positions (where height > 0)
    building_positions = np.argwhere(building_height_grid > 0)
    n_buildings = len(building_positions)
    
    material_code = BUILDING_MATERIAL_MAPPING.get(building_material, 
                                                  BUILDING_MATERIAL_MAPPING['default'])
    
    with open(filename, 'w') as f:
        # Write number of buildings
        f.write(f"{n_buildings}\n")
        
        # Write building data
        for idx, (j, i) in enumerate(building_positions):
            # CityLES uses 1-based indexing
            i_1based = i + 1
            j_1based = j + 1
            height = building_height_grid[j, i]
            
            # Format: i j height material_code depth1 depth2 changed_material
            f.write(f"{i_1based} {j_1based} {height:.1f} {material_code} 0.0 0.0 102\n")


def export_landuse(land_cover_grid, output_path, land_cover_source=None):
    """
    Export landuse.txt file for CityLES
    
    Parameters:
    -----------
    land_cover_grid : numpy.ndarray
        2D array of land cover values
    output_path : Path
        Output directory path
    land_cover_source : str, optional
        Source of land cover data
    """
    filename = output_path / 'landuse.txt'
    
    # Flatten the grid and convert to CityLES codes
    flat_grid = land_cover_grid.flatten()
    
    with open(filename, 'w') as f:
        for value in flat_grid:
            cityles_code = get_land_use_code(value, land_cover_source)
            f.write(f"{cityles_code}\n")


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
                elevation = dem_grid[j, i]
                
                f.write(f"{i_1based} {j_1based} {elevation:.1f}\n")


def export_vmap(canopy_height_grid, output_path, tree_base_ratio=0.3, tree_type='default'):
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
    
    # Get tree positions (where canopy height > 0)
    tree_positions = np.argwhere(canopy_height_grid > 0)
    n_trees = len(tree_positions)
    
    tree_code = TREE_TYPE_MAPPING.get(tree_type, TREE_TYPE_MAPPING['default'])
    
    with open(filename, 'w') as f:
        # Write number of trees
        f.write(f"{n_trees}\n")
        
        # Write tree data
        for idx, (j, i) in enumerate(tree_positions):
            # CityLES uses 1-based indexing
            i_1based = i + 1
            j_1based = j + 1
            total_height = canopy_height_grid[j, i]
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
                
                f.write(f"{i_1based} {j_1based} {lon:.7f} {lat:.8f}\n")


def export_cityles(building_height_grid, building_id_grid, canopy_height_grid,
                   land_cover_grid, dem_grid, meshsize, land_cover_source,
                   rectangle_vertices, output_directory="output/cityles",
                   building_material='default', tree_type='default',
                   tree_base_ratio=0.3, **kwargs):
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
        2D array of land cover values
    dem_grid : numpy.ndarray
        2D array of elevation values
    meshsize : float
        Grid cell size in meters
    land_cover_source : str
        Source of land cover data (e.g., 'esri', 'esa', 'osm')
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
    
    # Export individual files
    print("Exporting topog.txt...")
    export_topog(building_height_grid, building_id_grid, output_path, building_material)
    
    print("Exporting landuse.txt...")
    export_landuse(land_cover_grid, output_path, land_cover_source)
    
    print("Exporting dem.txt...")
    export_dem(dem_grid, output_path)
    
    print("Exporting vmap.txt...")
    export_vmap(canopy_height_grid, output_path, tree_base_ratio, tree_type)
    
    print("Exporting lonlat.txt...")
    export_lonlat(rectangle_vertices, building_height_grid.shape, output_path)
    
    # Create metadata file for reference
    metadata_file = output_path / 'cityles_metadata.txt'
    with open(metadata_file, 'w') as f:
        f.write("CityLES Export Metadata\n")
        f.write("====================\n")
        f.write(f"Grid shape: {building_height_grid.shape}\n")
        f.write(f"Mesh size: {meshsize} m\n")
        f.write(f"Land cover source: {land_cover_source}\n")
        f.write(f"Building material: {building_material}\n")
        f.write(f"Tree type: {tree_type}\n")
        f.write(f"Bounds: {rectangle_vertices}\n")
    
    print(f"CityLES export completed successfully!")
    return str(output_path) 