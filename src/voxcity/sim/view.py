"""Functions for computing and visualizing various view indices in a voxel city model.

This module provides functionality to compute and visualize:
- Green View Index (GVI): Measures visibility of green elements like trees and vegetation
- Sky View Index (SVI): Measures visibility of open sky from street level 
- Sky View Factor (SVF): Measures the ratio of visible sky hemisphere to total hemisphere
- Landmark Visibility: Measures visibility of specified landmark buildings from different locations

The module uses optimized ray tracing techniques with Numba JIT compilation for efficient computation.
Key features:
- Generic ray tracing framework that can be customized for different view indices
- Parallel processing for fast computation of view maps
- Tree transmittance modeling using Beer-Lambert law
- Visualization tools including matplotlib plots and OBJ exports
- Support for both inclusion and exclusion based visibility checks

The module provides several key functions:
- trace_ray_generic(): Core ray tracing function that handles tree transmittance
- compute_vi_generic(): Computes view indices by casting rays in specified directions
- compute_vi_map_generic(): Generates 2D maps of view indices
- get_view_index(): High-level function to compute various view indices
- compute_landmark_visibility(): Computes visibility of landmark buildings
- get_sky_view_factor_map(): Computes sky view factor maps

The module uses a voxel-based representation where:
- Empty space is represented by 0
- Trees are represented by -2 
- Buildings are represented by -3
- Other values can be used for different features

Tree transmittance is modeled using the Beer-Lambert law with configurable parameters:
- tree_k: Static extinction coefficient (default 0.6)
- tree_lad: Leaf area density in m^-1 (default 1.0)

Additional implementation details:
- Uses DDA (Digital Differential Analyzer) algorithm for efficient ray traversal
- Handles edge cases like zero-length rays and division by zero
- Supports early exit optimizations for performance
- Provides flexible observer placement rules
- Includes comprehensive error checking and validation
- Allows customization of visualization parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numba import njit, prange

from ..file.geojson import find_building_containing_point, get_buildings_in_drawn_polygon
from ..file.obj import grid_to_obj, export_obj

@njit
def calculate_transmittance(length, tree_k=0.6, tree_lad=1.0):
    """Calculate tree transmittance using the Beer-Lambert law.
    
    Uses the Beer-Lambert law to model light attenuation through tree canopy:
    transmittance = exp(-k * LAD * L)
    where:
    - k is the extinction coefficient
    - LAD is the leaf area density
    - L is the path length through the canopy
    
    Args:
        length (float): Path length through tree voxel in meters
        tree_k (float): Static extinction coefficient (default: 0.6)
            Controls overall light attenuation strength
        tree_lad (float): Leaf area density in m^-1 (default: 1.0)
            Higher values = denser foliage = more attenuation
    
    Returns:
        float: Transmittance value between 0 and 1
            1.0 = fully transparent
            0.0 = fully opaque
    """
    return np.exp(-tree_k * tree_lad * length)

@njit
def trace_ray_generic(voxel_data, origin, direction, hit_values, meshsize, tree_k, tree_lad, inclusion_mode=True):
    """Trace a ray through a voxel grid and check for hits with specified values.
    
    Uses DDA algorithm to efficiently traverse voxels along ray path.
    Handles tree transmittance using Beer-Lambert law.
    
    The DDA algorithm:
    1. Initializes ray at origin voxel
    2. Calculates distances to next voxel boundaries in each direction
    3. Steps to next voxel by choosing smallest distance
    4. Repeats until hit or out of bounds
    
    Tree transmittance:
    - When ray passes through tree voxels (-2), transmittance is accumulated
    - Uses Beer-Lambert law with configurable extinction coefficient and leaf area density
    - Ray is considered blocked if cumulative transmittance falls below 0.01
    
    Args:
        voxel_data (ndarray): 3D array of voxel values
        origin (ndarray): Starting point (x,y,z) of ray in voxel coordinates
        direction (ndarray): Direction vector of ray (will be normalized)
        hit_values (tuple): Values to check for hits
        meshsize (float): Size of each voxel in meters
        tree_k (float): Tree extinction coefficient
        tree_lad (float): Leaf area density in m^-1
        inclusion_mode (bool): If True, hit_values are hits. If False, hit_values are allowed values.
    
    Returns:
        tuple: (hit_detected, transmittance_value)
            hit_detected (bool): Whether ray hit a target voxel
            transmittance_value (float): Cumulative transmittance through trees
    """
    nx, ny, nz = voxel_data.shape
    x0, y0, z0 = origin
    dx, dy, dz = direction

    # Normalize direction vector
    length = np.sqrt(dx*dx + dy*dy + dz*dz)
    if length == 0.0:
        return False, 1.0
    dx /= length
    dy /= length
    dz /= length

    # Initialize ray position
    x, y, z = x0 + 0.5, y0 + 0.5, z0 + 0.5
    i, j, k = int(x0), int(y0), int(z0)

    # Calculate step directions and initial distances
    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1
    step_z = 1 if dz >= 0 else -1

    # Calculate DDA parameters with safety checks
    EPSILON = 1e-10  # Small value to prevent division by zero
    
    if abs(dx) > EPSILON:
        t_max_x = ((i + (step_x > 0)) - x) / dx
        t_delta_x = abs(1 / dx)
    else:
        t_max_x = np.inf
        t_delta_x = np.inf

    if abs(dy) > EPSILON:
        t_max_y = ((j + (step_y > 0)) - y) / dy
        t_delta_y = abs(1 / dy)
    else:
        t_max_y = np.inf
        t_delta_y = np.inf

    if abs(dz) > EPSILON:
        t_max_z = ((k + (step_z > 0)) - z) / dz
        t_delta_z = abs(1 / dz)
    else:
        t_max_z = np.inf
        t_delta_z = np.inf

    # Track cumulative values
    cumulative_transmittance = 1.0
    cumulative_hit_contribution = 0.0
    last_t = 0.0

    # Main ray traversal loop
    while (0 <= i < nx) and (0 <= j < ny) and (0 <= k < nz):
        voxel_value = voxel_data[i, j, k]
        
        # Find next intersection
        t_next = min(t_max_x, t_max_y, t_max_z)
        
        # Calculate segment length in current voxel
        segment_length = (t_next - last_t) * meshsize
        
        # Handle tree voxels (value -2)
        if voxel_value == -2:
            transmittance = calculate_transmittance(segment_length, tree_k, tree_lad)
            cumulative_transmittance *= transmittance
            
            # If transmittance becomes too low, consider it a hit
            if cumulative_transmittance < 0.01:
                return True, cumulative_transmittance

        # Check for hits with other objects
        if inclusion_mode:
            for hv in hit_values:
                if voxel_value == hv:
                    return True, cumulative_transmittance
        else:
            in_set = False
            for hv in hit_values:
                if voxel_value == hv:
                    in_set = True
                    break
            if not in_set and voxel_value != -2:  # Exclude trees from regular hits
                return True, cumulative_transmittance

        # Update for next iteration
        last_t = t_next
        
        # Move to next voxel
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                t_max_x += t_delta_x
                i += step_x
            else:
                t_max_z += t_delta_z
                k += step_z
        else:
            if t_max_y < t_max_z:
                t_max_y += t_delta_y
                j += step_y
            else:
                t_max_z += t_delta_z
                k += step_z

    return False, cumulative_transmittance

@njit
def compute_vi_generic(observer_location, voxel_data, ray_directions, hit_values, meshsize, tree_k, tree_lad, inclusion_mode=True):
    """Compute view index accounting for tree transmittance.
    
    Casts rays in specified directions and computes visibility index based on hits and transmittance.
    The view index is the ratio of visible rays to total rays cast, where:
    - For inclusion mode: Counts hits with target values
    - For exclusion mode: Counts rays that don't hit obstacles
    Tree transmittance is handled specially:
    - In inclusion mode with trees as targets: Uses (1 - transmittance) as contribution
    - In exclusion mode: Uses transmittance value directly
    
    Args:
        observer_location (ndarray): Observer position (x,y,z) in voxel coordinates
        voxel_data (ndarray): 3D array of voxel values
        ray_directions (ndarray): Array of direction vectors for rays
        hit_values (tuple): Values to check for hits
        meshsize (float): Size of each voxel in meters
        tree_k (float): Tree extinction coefficient
        tree_lad (float): Leaf area density in m^-1
        inclusion_mode (bool): If True, hit_values are hits. If False, hit_values are allowed values.
    
    Returns:
        float: View index value between 0 and 1
            0.0 = no visibility in any direction
            1.0 = full visibility in all directions
    """
    total_rays = ray_directions.shape[0]
    visibility_sum = 0.0

    for idx in range(total_rays):
        direction = ray_directions[idx]
        hit, value = trace_ray_generic(voxel_data, observer_location, direction, 
                                     hit_values, meshsize, tree_k, tree_lad, inclusion_mode)
        
        if inclusion_mode:
            if hit:
                if -2 in hit_values:
                    # For trees in hit_values, use the hit contribution (1 - transmittance)
                    visibility_sum += (1.0 - value) if value < 1.0 else 1.0
                else:
                    visibility_sum += 1.0
        else:
            if not hit:
                # For exclusion mode, use transmittance value directly
                visibility_sum += value

    return visibility_sum / total_rays

@njit(parallel=True)
def compute_vi_map_generic(voxel_data, ray_directions, view_height_voxel, hit_values, 
                          meshsize, tree_k, tree_lad, inclusion_mode=True):
    """Compute view index map incorporating tree transmittance.
    
    Places observers at valid locations and computes view index for each position.
    Valid observer locations are:
    - Empty voxels (0) or tree voxels (-2)
    - Above non-empty, non-tree voxels
    - Not above water (7,8,9) or negative values
    
    The function processes each x,y position in parallel for efficiency.
    
    Args:
        voxel_data (ndarray): 3D array of voxel values
        ray_directions (ndarray): Array of direction vectors for rays
        view_height_voxel (int): Observer height in voxel units
        hit_values (tuple): Values to check for hits
        meshsize (float): Size of each voxel in meters
        tree_k (float): Tree extinction coefficient
        tree_lad (float): Leaf area density in m^-1
        inclusion_mode (bool): If True, hit_values are hits. If False, hit_values are allowed values.
    
    Returns:
        ndarray: 2D array of view index values
            NaN = invalid observer location
            0.0-1.0 = view index value
    """
    nx, ny, nz = voxel_data.shape
    vi_map = np.full((nx, ny), np.nan)

    for x in prange(nx):
        for y in range(ny):
            found_observer = False
            for z in range(1, nz):
                # Check for valid observer location
                if voxel_data[x, y, z] in (0, -2) and voxel_data[x, y, z - 1] not in (0, -2):
                    # Skip invalid ground types
                    if (voxel_data[x, y, z - 1] in (7, 8, 9)) or (voxel_data[x, y, z - 1] < 0):
                        vi_map[x, y] = np.nan
                        found_observer = True
                        break
                    else:
                        # Place observer and compute view index
                        observer_location = np.array([x, y, z + view_height_voxel], dtype=np.float64)
                        vi_value = compute_vi_generic(observer_location, voxel_data, ray_directions, 
                                                    hit_values, meshsize, tree_k, tree_lad, inclusion_mode)
                        vi_map[x, y] = vi_value
                        found_observer = True
                        break
            if not found_observer:
                vi_map[x, y] = np.nan

    return np.flipud(vi_map)

def get_view_index(voxel_data, meshsize, mode=None, hit_values=None, inclusion_mode=True, **kwargs):
    """Calculate and visualize a generic view index for a voxel city model.

    This is a high-level function that provides a flexible interface for computing
    various view indices. It handles:
    - Mode presets for common indices (green, sky)
    - Ray direction generation
    - Tree transmittance parameters
    - Visualization
    - Optional OBJ export

    Args:
        voxel_data (ndarray): 3D array of voxel values.
        meshsize (float): Size of each voxel in meters.
        mode (str): Predefined mode. Options: 'green', 'sky', or None.
            If 'green': GVI mode - measures visibility of vegetation
            If 'sky': SVI mode - measures visibility of open sky
            If None: Custom mode requiring hit_values parameter
        hit_values (tuple): Voxel values considered as hits (if inclusion_mode=True)
                            or allowed values (if inclusion_mode=False), if mode is None.
        inclusion_mode (bool): 
            True = voxel_value in hit_values is success.
            False = voxel_value not in hit_values is success.
        **kwargs: Additional arguments:
            - view_point_height (float): Observer height in meters (default: 1.5)
            - colormap (str): Matplotlib colormap name (default: 'viridis')
            - obj_export (bool): Export as OBJ (default: False)
            - output_directory (str): Directory for OBJ output
            - output_file_name (str): Base filename for OBJ output
            - num_colors (int): Number of discrete colors for OBJ export
            - alpha (float): Transparency value for OBJ export
            - vmin (float): Minimum value for color mapping
            - vmax (float): Maximum value for color mapping
            - N_azimuth (int): Number of azimuth angles for ray directions
            - N_elevation (int): Number of elevation angles for ray directions
            - elevation_min_degrees (float): Minimum elevation angle in degrees
            - elevation_max_degrees (float): Maximum elevation angle in degrees
            - tree_k (float): Tree extinction coefficient (default: 0.5)
            - tree_lad (float): Leaf area density in m^-1 (default: 1.0)

    Returns:
        ndarray: 2D array of computed view index values.
    """
    # Handle mode presets
    if mode == 'green':
        # GVI defaults - detect vegetation and trees
        hit_values = (-2, 2, 5, 7)
        inclusion_mode = True
    elif mode == 'sky':
        # SVI defaults - detect open sky
        hit_values = (0,)
        inclusion_mode = False
    else:
        # For custom mode, user must specify hit_values
        if hit_values is None:
            raise ValueError("For custom mode, you must provide hit_values.")

    # Get parameters from kwargs with defaults
    view_point_height = kwargs.get("view_point_height", 1.5)
    view_height_voxel = int(view_point_height / meshsize)
    colormap = kwargs.get("colormap", 'viridis')
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", 1.0)
    N_azimuth = kwargs.get("N_azimuth", 60)
    N_elevation = kwargs.get("N_elevation", 10)
    elevation_min_degrees = kwargs.get("elevation_min_degrees", -30)
    elevation_max_degrees = kwargs.get("elevation_max_degrees", 30)
    
    # Tree transmittance parameters
    tree_k = kwargs.get("tree_k", 0.5)
    tree_lad = kwargs.get("tree_lad", 1.0)

    # Generate ray directions using spherical coordinates
    azimuth_angles = np.linspace(0, 2 * np.pi, N_azimuth, endpoint=False)
    elevation_angles = np.deg2rad(np.linspace(elevation_min_degrees, elevation_max_degrees, N_elevation))

    ray_directions = []
    for elevation in elevation_angles:
        cos_elev = np.cos(elevation)
        sin_elev = np.sin(elevation)
        for azimuth in azimuth_angles:
            dx = cos_elev * np.cos(azimuth)
            dy = cos_elev * np.sin(azimuth)
            dz = sin_elev
            ray_directions.append([dx, dy, dz])
    ray_directions = np.array(ray_directions, dtype=np.float64)

    # Compute the view index map with transmittance parameters
    vi_map = compute_vi_map_generic(voxel_data, ray_directions, view_height_voxel, 
                                  hit_values, meshsize, tree_k, tree_lad, inclusion_mode)

    # Plot results
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap(colormap).copy()
    cmap.set_bad(color='lightgray')
    plt.figure(figsize=(10, 8))
    plt.imshow(vi_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='View Index')
    plt.axis('off')
    plt.show()

    # Optional OBJ export
    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        dem_grid = kwargs.get("dem_grid", np.zeros_like(vi_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "view_index")
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        grid_to_obj(
            vi_map,
            dem_grid,
            output_dir,
            output_file_name,
            meshsize,
            view_point_height,
            colormap_name=colormap,
            num_colors=num_colors,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax
        )

    return vi_map

def mark_building_by_id(voxcity_grid_ori, building_id_grid_ori, ids, mark):
    """Mark specific buildings in the voxel grid with a given value.

    Used to identify landmark buildings for visibility analysis.
    Flips building ID grid vertically to match voxel grid orientation.

    Args:
        voxcity_grid (ndarray): 3D array of voxel values
        building_id_grid_ori (ndarray): 2D array of building IDs
        ids (list): List of building IDs to mark
        mark (int): Value to mark the buildings with
    """

    voxcity_grid = voxcity_grid_ori.copy()

    # Flip building ID grid vertically to match voxel grid orientation
    building_id_grid = np.flipud(building_id_grid_ori.copy())

    # Get x,y positions from building_id_grid where landmarks are
    positions = np.where(np.isin(building_id_grid, ids))

    # Loop through each x,y position and mark building voxels
    for i in range(len(positions[0])):
        x, y = positions[0][i], positions[1][i]
        # Replace building voxels (-3) with mark value at this x,y position
        z_mask = voxcity_grid[x, y, :] == -3
        voxcity_grid[x, y, z_mask] = mark
    
    return voxcity_grid

@njit
def trace_ray_to_target(voxel_data, origin, target, opaque_values):
    """Trace a ray from origin to target through voxel data.

    Uses DDA algorithm to efficiently traverse voxels along ray path.
    Checks for any opaque voxels blocking the line of sight.

    Args:
        voxel_data (ndarray): 3D array of voxel values
        origin (tuple): Starting point (x,y,z) in voxel coordinates
        target (tuple): End point (x,y,z) in voxel coordinates
        opaque_values (ndarray): Array of voxel values that block the ray

    Returns:
        bool: True if target is visible from origin, False otherwise
    """
    nx, ny, nz = voxel_data.shape
    x0, y0, z0 = origin
    x1, y1, z1 = target
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0

    # Normalize direction vector
    length = np.sqrt(dx*dx + dy*dy + dz*dz)
    if length == 0.0:
        return True  # Origin and target are at the same location
    dx /= length
    dy /= length
    dz /= length

    # Initialize ray position at center of starting voxel
    x, y, z = x0 + 0.5, y0 + 0.5, z0 + 0.5
    i, j, k = int(x0), int(y0), int(z0)

    # Determine step direction for each axis
    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1
    step_z = 1 if dz >= 0 else -1

    # Calculate distances to next voxel boundaries and step sizes
    # Handle cases where direction components are zero
    if dx != 0:
        t_max_x = ((i + (step_x > 0)) - x) / dx
        t_delta_x = abs(1 / dx)
    else:
        t_max_x = np.inf
        t_delta_x = np.inf

    if dy != 0:
        t_max_y = ((j + (step_y > 0)) - y) / dy
        t_delta_y = abs(1 / dy)
    else:
        t_max_y = np.inf
        t_delta_y = np.inf

    if dz != 0:
        t_max_z = ((k + (step_z > 0)) - z) / dz
        t_delta_z = abs(1 / dz)
    else:
        t_max_z = np.inf
        t_delta_z = np.inf

    # Main ray traversal loop
    while True:
        # Check if current voxel is within bounds and opaque
        if (0 <= i < nx) and (0 <= j < ny) and (0 <= k < nz):
            voxel_value = voxel_data[i, j, k]
            if voxel_value in opaque_values:
                return False  # Ray is blocked
        else:
            return False  # Out of bounds

        # Check if we've reached target voxel
        if i == int(x1) and j == int(y1) and k == int(z1):
            return True  # Ray has reached the target

        # Move to next voxel using DDA algorithm
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                t_max = t_max_x
                t_max_x += t_delta_x
                i += step_x
            else:
                t_max = t_max_z
                t_max_z += t_delta_z
                k += step_z
        else:
            if t_max_y < t_max_z:
                t_max = t_max_y
                t_max_y += t_delta_y
                j += step_y
            else:
                t_max = t_max_z
                t_max_z += t_delta_z
                k += step_z

@njit
def compute_visibility_to_all_landmarks(observer_location, landmark_positions, voxel_data, opaque_values):
    """Check if any landmark is visible from the observer location.

    Traces rays to each landmark position until finding one that's visible.
    Uses optimized ray tracing with early exit on first visible landmark.

    Args:
        observer_location (ndarray): Observer position (x,y,z) in voxel coordinates
        landmark_positions (ndarray): Array of landmark positions
        voxel_data (ndarray): 3D array of voxel values
        opaque_values (ndarray): Array of voxel values that block visibility

    Returns:
        int: 1 if any landmark is visible, 0 if none are visible
    """
    # Check visibility to each landmark until one is found visible
    for idx in range(landmark_positions.shape[0]):
        target = landmark_positions[idx].astype(np.float64)
        is_visible = trace_ray_to_target(voxel_data, observer_location, target, opaque_values)
        if is_visible:
            return 1  # Return as soon as one landmark is visible
    return 0  # No landmarks were visible

@njit(parallel=True)
def compute_visibility_map(voxel_data, landmark_positions, opaque_values, view_height_voxel):
    """Compute visibility map for landmarks in the voxel grid.

    Places observers at valid locations (empty voxels above ground, excluding building
    roofs and vegetation) and checks visibility to any landmark.

    The function processes each x,y position in parallel for efficiency.
    Valid observer locations are:
    - Empty voxels (0) or tree voxels (-2)
    - Above non-empty, non-tree voxels
    - Not above water (7,8,9) or negative values

    Args:
        voxel_data (ndarray): 3D array of voxel values
        landmark_positions (ndarray): Array of landmark positions
        opaque_values (ndarray): Array of voxel values that block visibility
        view_height_voxel (int): Height offset for observer in voxels

    Returns:
        ndarray: 2D array of visibility values
            NaN = invalid observer location
            0 = no landmarks visible
            1 = at least one landmark visible
    """
    nx, ny, nz = voxel_data.shape
    visibility_map = np.full((nx, ny), np.nan)

    # Process each x,y position in parallel
    for x in prange(nx):
        for y in range(ny):
            found_observer = False
            # Find lowest empty voxel above ground
            for z in range(1, nz):
                if voxel_data[x, y, z] == 0 and voxel_data[x, y, z - 1] != 0:
                    # Skip if standing on building or vegetation
                    if (voxel_data[x, y, z - 1] in (7, 8, 9)) or (voxel_data[x, y, z - 1] < 0):
                        visibility_map[x, y] = np.nan
                        found_observer = True
                        break
                    else:
                        # Place observer and check visibility
                        observer_location = np.array([x, y, z+view_height_voxel], dtype=np.float64)
                        visible = compute_visibility_to_all_landmarks(observer_location, landmark_positions, voxel_data, opaque_values)
                        visibility_map[x, y] = visible
                        found_observer = True
                        break
            if not found_observer:
                visibility_map[x, y] = np.nan

    return visibility_map

def compute_landmark_visibility(voxel_data, target_value=-30, view_height_voxel=0, colormap='viridis'):
    """Compute and visualize landmark visibility in a voxel grid.

    Places observers at valid locations and checks visibility to any landmark voxel.
    Generates a binary visibility map and visualization.

    The function:
    1. Identifies all landmark voxels (target_value)
    2. Determines which voxel values block visibility
    3. Computes visibility from each valid observer location
    4. Generates visualization with legend

    Args:
        voxel_data (ndarray): 3D array of voxel values
        target_value (int, optional): Value used to identify landmark voxels. Defaults to -30.
        view_height_voxel (int, optional): Height offset for observer in voxels. Defaults to 0.
        colormap (str, optional): Matplotlib colormap name. Defaults to 'viridis'.

    Returns:
        ndarray: 2D array of visibility values (0 or 1) with y-axis flipped
            NaN = invalid observer location
            0 = no landmarks visible
            1 = at least one landmark visible

    Raises:
        ValueError: If no landmark voxels are found with the specified target_value
    """
    # Find positions of all landmark voxels
    landmark_positions = np.argwhere(voxel_data == target_value)

    if landmark_positions.shape[0] == 0:
        raise ValueError(f"No landmark with value {target_value} found in the voxel data.")

    # Define which voxel values block visibility
    unique_values = np.unique(voxel_data)
    opaque_values = np.array([v for v in unique_values if v != 0 and v != target_value], dtype=np.int32)

    # Compute visibility map
    visibility_map = compute_visibility_map(voxel_data, landmark_positions, opaque_values, view_height_voxel)

    # Set up visualization
    cmap = plt.cm.get_cmap(colormap, 2).copy()
    cmap.set_bad(color='lightgray')

    # Create main plot
    plt.figure(figsize=(10, 8))
    plt.imshow(np.flipud(visibility_map), origin='lower', cmap=cmap, vmin=0, vmax=1)

    # Create and add legend
    visible_patch = mpatches.Patch(color=cmap(1.0), label='Visible (1)')
    not_visible_patch = mpatches.Patch(color=cmap(0.0), label='Not Visible (0)')
    plt.legend(handles=[visible_patch, not_visible_patch], 
            loc='center left',
            bbox_to_anchor=(1.0, 0.5))
    plt.axis('off')
    plt.show()

    return np.flipud(visibility_map)

def get_landmark_visibility_map(voxcity_grid_ori, building_id_grid, building_geojson, meshsize, **kwargs):
    """Generate a visibility map for landmark buildings in a voxel city.

    Places observers at valid locations and checks visibility to any part of the
    specified landmark buildings. Can identify landmarks either by ID or by finding
    buildings within a specified rectangle.

    Args:
        voxcity_grid (ndarray): 3D array representing the voxel city
        building_id_grid (ndarray): 3D array mapping voxels to building IDs
        building_geojson (dict): GeoJSON data containing building features
        meshsize (float): Size of each voxel in meters
        **kwargs: Additional keyword arguments
            view_point_height (float): Height of observer viewpoint in meters
            colormap (str): Matplotlib colormap name
            landmark_building_ids (list): List of building IDs to mark as landmarks
            rectangle_vertices (list): List of (lat,lon) coordinates defining rectangle
            obj_export (bool): Whether to export visibility map as OBJ file
            dem_grid (ndarray): Digital elevation model grid for OBJ export
            output_directory (str): Directory for OBJ file output
            output_file_name (str): Base filename for OBJ output
            alpha (float): Alpha transparency value for OBJ export
            vmin (float): Minimum value for color mapping
            vmax (float): Maximum value for color mapping

    Returns:
        ndarray: 2D array of visibility values for landmark buildings
    """
    # Convert observer height from meters to voxel units
    view_point_height = kwargs.get("view_point_height", 1.5)
    view_height_voxel = int(view_point_height / meshsize)

    colormap = kwargs.get("colormap", 'viridis')

    # Get landmark building IDs either directly or by finding buildings in rectangle
    features = building_geojson
    landmark_ids = kwargs.get('landmark_building_ids', None)
    landmark_polygon = kwargs.get('landmark_polygon', None)
    if landmark_ids is None:
        if landmark_polygon is not None:
            landmark_ids = get_buildings_in_drawn_polygon(building_geojson, landmark_polygon, operation='within')
        else:
            rectangle_vertices = kwargs.get("rectangle_vertices", None)
            if rectangle_vertices is None:
                print("Cannot set landmark buildings. You need to input either of rectangle_vertices or landmark_ids.")
                return None
                
            # Calculate center point of rectangle
            lons = [coord[0] for coord in rectangle_vertices]
            lats = [coord[1] for coord in rectangle_vertices]
            center_lon = (min(lons) + max(lons)) / 2
            center_lat = (min(lats) + max(lats)) / 2
            target_point = (center_lon, center_lat)
            
            # Find buildings at center point
            landmark_ids = find_building_containing_point(features, target_point)

    # Mark landmark buildings in voxel grid with special value
    target_value = -30
    voxcity_grid = mark_building_by_id(voxcity_grid_ori, building_id_grid, landmark_ids, target_value)
    
    # Compute visibility map
    landmark_vis_map = compute_landmark_visibility(voxcity_grid, target_value=target_value, view_height_voxel=view_height_voxel, colormap=colormap)

    # Handle optional OBJ export
    obj_export = kwargs.get("obj_export")
    if obj_export == True:
        dem_grid = kwargs.get("dem_grid", np.zeros_like(landmark_vis_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "landmark_visibility")        
        num_colors = 2
        alpha = kwargs.get("alpha", 1.0)
        vmin = kwargs.get("vmin", 0.0)
        vmax = kwargs.get("vmax", 1.0)
        
        # Export visibility map and voxel city as OBJ files
        grid_to_obj(
            landmark_vis_map,
            dem_grid,
            output_dir,
            output_file_name,
            meshsize,
            view_point_height,
            colormap_name=colormap,
            num_colors=num_colors,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax
        )
        output_file_name_vox = 'voxcity_' + output_file_name
        export_obj(voxcity_grid, output_dir, output_file_name_vox, meshsize)

    return landmark_vis_map, voxcity_grid

def get_sky_view_factor_map(voxel_data, meshsize, show_plot=False, **kwargs):
    """
    Compute and visualize the Sky View Factor (SVF) for each valid observer cell in the voxel grid.

    Args:
        voxel_data (ndarray): 3D array of voxel values.
        meshsize (float): Size of each voxel in meters.
        show_plot (bool): Whether to display the plot.
        **kwargs: Additional parameters.

    Returns:
        ndarray: 2D array of SVF values at each cell (x, y).
    """
    # Default parameters
    view_point_height = kwargs.get("view_point_height", 1.5)
    view_height_voxel = int(view_point_height / meshsize)
    colormap = kwargs.get("colormap", 'BuPu_r')
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", 1.0)
    N_azimuth = kwargs.get("N_azimuth", 60)
    N_elevation = kwargs.get("N_elevation", 10)
    elevation_min_degrees = kwargs.get("elevation_min_degrees", 0)
    elevation_max_degrees = kwargs.get("elevation_max_degrees", 90)

    # Get tree transmittance parameters
    tree_k = kwargs.get("tree_k", 0.6)  # Static extinction coefficient
    tree_lad = kwargs.get("tree_lad", 1.0)  # Leaf area density in m^-1

    # Define hit_values and inclusion_mode for sky detection
    hit_values = (0,)
    inclusion_mode = False

    # Generate ray directions over the specified hemisphere
    azimuth_angles = np.linspace(0, 2 * np.pi, N_azimuth, endpoint=False)
    elevation_angles = np.deg2rad(np.linspace(elevation_min_degrees, elevation_max_degrees, N_elevation))

    ray_directions = []
    for elevation in elevation_angles:
        cos_elev = np.cos(elevation)
        sin_elev = np.sin(elevation)
        for azimuth in azimuth_angles:
            dx = cos_elev * np.cos(azimuth)
            dy = cos_elev * np.sin(azimuth)
            dz = sin_elev
            ray_directions.append([dx, dy, dz])
    ray_directions = np.array(ray_directions, dtype=np.float64)

    # Compute the SVF map using the compute function
    vi_map = compute_vi_map_generic(voxel_data, ray_directions, view_height_voxel, 
                                  hit_values, meshsize, tree_k, tree_lad, inclusion_mode)

    # Plot results if requested
    if show_plot:
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap(colormap).copy()
        cmap.set_bad(color='lightgray')
        plt.figure(figsize=(10, 8))
        # plt.title("Sky View Factor Map")
        plt.imshow(vi_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Sky View Factor')
        plt.axis('off')
        plt.show()

    # Optional OBJ export
    obj_export = kwargs.get("obj_export", False)
    if obj_export:        
        dem_grid = kwargs.get("dem_grid", np.zeros_like(vi_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "sky_view_factor")
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        grid_to_obj(
            vi_map,
            dem_grid,
            output_dir,
            output_file_name,
            meshsize,
            view_point_height,
            colormap_name=colormap,
            num_colors=num_colors,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax
        )

    return vi_map