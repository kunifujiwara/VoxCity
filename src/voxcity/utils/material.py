import numpy as np

def get_material_dict():
    """
    Returns a dictionary mapping material names to their corresponding ID values.
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
    
    Parameters:
    window_ratio: float between 0 and 1.0
    
    Returns:
    tuple (x_mod, y_mod, z_mod): modulo numbers for each dimension
    """
    if window_ratio <= 0.125 + 0.0625:  # around 0.125
        return (2, 2, 2)
    elif window_ratio <= 0.25 + 0.125:  # around 0.25
        combinations = [(2, 2, 1), (2, 1, 2), (1, 2, 2)]
        return combinations[hash(str(window_ratio)) % len(combinations)]
    elif window_ratio <= 0.5 + 0.125:  # around 0.5
        combinations = [(2, 1, 1), (1, 2, 1), (1, 1, 2)]
        return combinations[hash(str(window_ratio)) % len(combinations)]
    elif window_ratio <= 0.75 + 0.125:  # around 0.75
        combinations = [(2, 1, 1), (1, 2, 1), (1, 1, 2)]
        return combinations[hash(str(window_ratio)) % len(combinations)]
    else:  # above 0.875
        return (1, 1, 1)

def set_building_material_by_id(voxelcity_grid, building_id_grid_ori, ids, mark, window_ratio=0.125, glass_id=-16):
    """
    Marks cells in voxelcity_grid based on building IDs and window ratio.
    Never sets glass_id to cells with maximum z index.
    
    Parameters:
    voxelcity_grid: 3D numpy array
    building_id_grid_ori: 2D numpy array containing building IDs
    ids: list/array of building IDs to check
    mark: value to set for marked cells
    window_ratio: float between 0 and 1.0, determines window density:
        ~0.125: sparse windows (2,2,2)
        ~0.25: medium-sparse windows (2,2,1), (2,1,2), or (1,2,2)
        ~0.5: medium windows (2,1,1), (1,2,1), or (1,1,2)
        ~0.75: dense windows (2,1,1), (1,2,1), or (1,1,2)
        >0.875: maximum density (1,1,1)
    glass_id: value to set for glass cells (default: -16)
    
    Returns:
    Modified voxelcity_grid
    """
    building_id_grid = np.flipud(building_id_grid_ori.copy())
    
    # Get modulo numbers based on window_ratio
    x_mod, y_mod, z_mod = get_modulo_numbers(window_ratio)
    
    # Get positions where building IDs match
    building_positions = np.where(np.isin(building_id_grid, ids))
    
    # Loop through each position that matches building IDs
    for i in range(len(building_positions[0])):
        x, y = building_positions[0][i], building_positions[1][i]
        z_mask = voxelcity_grid[x, y, :] == -3
        voxelcity_grid[x, y, z_mask] = mark
        
        # Check if x and y meet the modulo conditions
        if x % x_mod == 0 and y % y_mod == 0:
            z_mask = voxelcity_grid[x, y, :] == mark
            if np.any(z_mask):
                # Find the maximum z index where z_mask is True
                z_indices = np.where(z_mask)[0]
                max_z_index = np.max(z_indices)
                
                # Create base mask excluding maximum z index
                base_mask = z_mask.copy()
                base_mask[max_z_index] = False
                
                # Create pattern mask based on z modulo
                pattern_mask = np.zeros_like(z_mask)
                valid_z_indices = z_indices[z_indices != max_z_index]  # Exclude max_z_index
                if len(valid_z_indices) > 0:
                    pattern_mask[valid_z_indices[valid_z_indices % z_mod == 0]] = True
                
                # For window_ratio around 0.75, add additional pattern
                if 0.625 < window_ratio <= 0.875 and len(valid_z_indices) > 0:
                    additional_pattern = np.zeros_like(z_mask)
                    additional_pattern[valid_z_indices[valid_z_indices % (z_mod + 1) == 0]] = True
                    pattern_mask = np.logical_or(pattern_mask, additional_pattern)
                
                # Final mask combines base_mask and pattern_mask
                final_glass_mask = np.logical_and(base_mask, pattern_mask)
                
                # Set glass_id for all positions in the final mask
                voxelcity_grid[x, y, final_glass_mask] = glass_id
    
    return voxelcity_grid

def set_building_material_by_gdf(voxelcity_grid_ori, building_id_grid, gdf_buildings, material_id_dict=None):
    """
    Sets building materials based on a GeoDataFrame containing building information.
    
    Parameters:
    voxelcity_grid_ori: 3D numpy array of the original voxel grid
    building_id_grid: 2D numpy array containing building IDs
    gdf_buildings: GeoDataFrame containing building information with columns:
                  'building_id', 'surface_material', 'window_ratio'
    material_id_dict: Dictionary mapping material names to their IDs (optional)
    
    Returns:
    Modified voxelcity_grid
    """
    voxelcity_grid = voxelcity_grid_ori.copy()
    if material_id_dict == None:
        material_id_dict = get_material_dict()

    for index, row in gdf_buildings.iterrows():
        # Access properties
        osmid = row['building_id']
        surface_material = row['surface_material']
        window_ratio = row['window_ratio']
        if surface_material is None:
            surface_material = 'unknown'            
        set_building_material_by_id(voxelcity_grid, building_id_grid, osmid, 
                                  material_id_dict[surface_material], 
                                  window_ratio=window_ratio, 
                                  glass_id=material_id_dict['glass'])
    
    return voxelcity_grid