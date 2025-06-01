"""
Material utilities for VoxelCity voxel grid processing.

This module provides functions for setting building materials and window patterns
in 3D voxel grids based on building IDs, material types, and window ratios.
The main functionality includes:
- Material ID mapping and retrieval
- Window pattern generation based on configurable ratios
- Building material assignment from GeoDataFrame data
"""

import numpy as np

def get_material_dict():
    """
    Returns a dictionary mapping material names to their corresponding ID values.
    
    The material IDs use negative values to distinguish them from other voxel types.
    Each material has a unique negative ID that can be used for material-based
    rendering and analysis.
    
    Returns:
        dict: Dictionary with material names as keys and negative integer IDs as values.
              Available materials: unknown, brick, wood, concrete, metal, stone, glass, plaster
    """
    return {
        "unknown": -3,
        "brick": -11,  
        "wood": -12,  
        "concrete": -13,  
        "metal": -14,  
        "stone": -15,  
        "glass": -16,  
        "plaster": -17,  
    }

def get_modulo_numbers(window_ratio):
    """
    Determines the appropriate modulo numbers for x, y, z based on window_ratio.
    
    This function creates different window patterns by returning modulo values that
    control the spacing of windows in the x, y, and z dimensions. Lower window_ratio
    values result in sparser window patterns (higher modulo values), while higher
    ratios create denser patterns.
    
    The function uses hash-based selection for certain ratios to introduce variety
    in window patterns for buildings with similar window ratios.
    
    Parameters:
        window_ratio (float): Value between 0 and 1.0 representing window density
    
    Returns:
        tuple: (x_mod, y_mod, z_mod) - modulo numbers for each dimension
               Higher values = sparser windows, lower values = denser windows
    """
    # Very sparse windows - every 2nd position in all dimensions
    if window_ratio <= 0.125 + 0.0625:  # around 0.125
        return (2, 2, 2)
    # Medium-sparse windows - vary pattern across dimensions
    elif window_ratio <= 0.25 + 0.125:  # around 0.25
        combinations = [(2, 2, 1), (2, 1, 2), (1, 2, 2)]
        # Use hash for consistent but varied selection
        return combinations[hash(str(window_ratio)) % len(combinations)]
    # Medium density windows - two dimensions sparse, one dense
    elif window_ratio <= 0.5 + 0.125:  # around 0.5
        combinations = [(2, 1, 1), (1, 2, 1), (1, 1, 2)]
        return combinations[hash(str(window_ratio)) % len(combinations)]
    # Dense windows - similar pattern to medium density
    elif window_ratio <= 0.75 + 0.125:  # around 0.75
        combinations = [(2, 1, 1), (1, 2, 1), (1, 1, 2)]
        return combinations[hash(str(window_ratio)) % len(combinations)]
    # Maximum density - windows at every position
    else:  # above 0.875
        return (1, 1, 1)

def set_building_material_by_id(voxelcity_grid, building_id_grid_ori, ids, mark, window_ratio=0.125, glass_id=-16):
    """
    Marks cells in voxelcity_grid based on building IDs and window ratio.
    Never sets glass_id to cells with maximum z index.
    
    This function processes buildings by:
    1. Finding all positions matching the specified building IDs
    2. Setting the base material for all building voxels
    3. Creating window patterns based on window_ratio and modulo calculations
    4. Ensuring the top floor (maximum z) never gets windows (glass_id)
    
    The window pattern is determined by the modulo values returned from get_modulo_numbers(),
    which creates different densities and arrangements of windows based on the window_ratio.
    
    Parameters:
        voxelcity_grid (numpy.ndarray): 3D numpy array representing the voxel grid
        building_id_grid_ori (numpy.ndarray): 2D numpy array containing building IDs
        ids (list/array): Building IDs to process
        mark (int): Material ID value to set for building cells
        window_ratio (float): Value between 0 and 1.0 determining window density:
            ~0.125: sparse windows (2,2,2)
            ~0.25: medium-sparse windows (2,2,1), (2,1,2), or (1,2,2)
            ~0.5: medium windows (2,1,1), (1,2,1), or (1,1,2)
            ~0.75: dense windows (2,1,1), (1,2,1), or (1,1,2)
            >0.875: maximum density (1,1,1)
        glass_id (int): Material ID for glass/window cells (default: -16)
    
    Returns:
        numpy.ndarray: Modified voxelcity_grid with building materials and windows applied
    """
    # Flip the building ID grid vertically to match coordinate system
    building_id_grid = np.flipud(building_id_grid_ori.copy())
    
    # Get modulo numbers based on window_ratio for pattern generation
    x_mod, y_mod, z_mod = get_modulo_numbers(window_ratio)
    
    # Find all positions where building IDs match the specified IDs
    building_positions = np.where(np.isin(building_id_grid, ids))
    
    # Process each building position
    for i in range(len(building_positions[0])):
        x, y = building_positions[0][i], building_positions[1][i]
        
        # Set base building material for all voxels at this x,y position
        # Only modify voxels that are currently marked as "unknown" (-3)
        z_mask = voxelcity_grid[x, y, :] == -3
        voxelcity_grid[x, y, z_mask] = mark
        
        # Apply window pattern if position meets modulo conditions
        if x % x_mod == 0 and y % y_mod == 0:
            # Find all z positions with the building material
            z_mask = voxelcity_grid[x, y, :] == mark
            if np.any(z_mask):
                # Get z indices and find the maximum (top floor)
                z_indices = np.where(z_mask)[0]
                max_z_index = np.max(z_indices)
                
                # Create base mask excluding the top floor
                # This ensures the roof never gets windows
                base_mask = z_mask.copy()
                base_mask[max_z_index] = False
                
                # Create window pattern based on z modulo
                pattern_mask = np.zeros_like(z_mask)
                valid_z_indices = z_indices[z_indices != max_z_index]  # Exclude max_z_index
                if len(valid_z_indices) > 0:
                    # Apply z modulo pattern to create vertical window spacing
                    pattern_mask[valid_z_indices[valid_z_indices % z_mod == 0]] = True
                
                # For higher window ratios, add additional window pattern
                # This creates denser window arrangements for buildings with more windows
                if 0.625 < window_ratio <= 0.875 and len(valid_z_indices) > 0:
                    additional_pattern = np.zeros_like(z_mask)
                    additional_pattern[valid_z_indices[valid_z_indices % (z_mod + 1) == 0]] = True
                    # Combine patterns using logical OR to increase window density
                    pattern_mask = np.logical_or(pattern_mask, additional_pattern)
                
                # Combine base mask (excluding top floor) with pattern mask
                final_glass_mask = np.logical_and(base_mask, pattern_mask)
                
                # Set glass material for all positions matching the final mask
                voxelcity_grid[x, y, final_glass_mask] = glass_id
    
    return voxelcity_grid

def set_building_material_by_gdf(voxelcity_grid_ori, building_id_grid, gdf_buildings, material_id_dict=None):
    """
    Sets building materials based on a GeoDataFrame containing building information.
    
    This function iterates through a GeoDataFrame of building data and applies
    materials and window patterns to the corresponding buildings in the voxel grid.
    It handles missing material information by defaulting to 'unknown' material.
    
    Parameters:
        voxelcity_grid_ori (numpy.ndarray): 3D numpy array of the original voxel grid
        building_id_grid (numpy.ndarray): 2D numpy array containing building IDs
        gdf_buildings (GeoDataFrame): Building information with required columns:
                      'building_id': Unique identifier for each building
                      'surface_material': Material type (brick, wood, concrete, etc.)
                      'window_ratio': Float between 0-1 for window density
        material_id_dict (dict, optional): Dictionary mapping material names to IDs.
                                         If None, uses default from get_material_dict()
    
    Returns:
        numpy.ndarray: Modified voxelcity_grid with all building materials and windows applied
    """
    # Create a copy to avoid modifying the original grid
    voxelcity_grid = voxelcity_grid_ori.copy()
    
    # Use default material dictionary if none provided
    if material_id_dict == None:
        material_id_dict = get_material_dict()

    # Process each building in the GeoDataFrame
    for index, row in gdf_buildings.iterrows():
        # Extract building properties from the current row
        osmid = row['building_id']
        surface_material = row['surface_material']
        window_ratio = row['window_ratio']
        
        # Handle missing surface material data
        if surface_material is None:
            surface_material = 'unknown'
        
        # Apply material and window pattern to this building
        set_building_material_by_id(voxelcity_grid, building_id_grid, osmid, 
                                  material_id_dict[surface_material], 
                                  window_ratio=window_ratio, 
                                  glass_id=material_id_dict['glass'])
    
    return voxelcity_grid