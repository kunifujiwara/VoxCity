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
import time
import trimesh

from ..geoprocessor.polygon import find_building_containing_point, get_buildings_in_drawn_polygon
from ..geoprocessor.mesh import create_voxel_mesh
from ..exporter.obj import grid_to_obj, export_obj

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
        segment_length = max(0.0, segment_length) 
        
        # Handle tree voxels (value -2)
        if voxel_value == -2:
            transmittance = calculate_transmittance(segment_length, tree_k, tree_lad)
            cumulative_transmittance *= transmittance

            # if segment_length < 0:
            #     print(f"segment_length = {segment_length}, transmittance = {transmittance}, cumulative_transmittance = {cumulative_transmittance}")
            
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
                    visibility_sum += value if value < 1.0 else 1.0
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

def get_landmark_visibility_map(voxcity_grid_ori, building_id_grid, building_gdf, meshsize, **kwargs):
    """Generate a visibility map for landmark buildings in a voxel city.

    Places observers at valid locations and checks visibility to any part of the
    specified landmark buildings. Can identify landmarks either by ID or by finding
    buildings within a specified rectangle.

    Args:
        voxcity_grid (ndarray): 3D array representing the voxel city
        building_id_grid (ndarray): 3D array mapping voxels to building IDs
        building_gdf (GeoDataFrame): GeoDataFrame containing building features
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
    landmark_ids = kwargs.get('landmark_building_ids', None)
    landmark_polygon = kwargs.get('landmark_polygon', None)
    if landmark_ids is None:
        if landmark_polygon is not None:
            landmark_ids = get_buildings_in_drawn_polygon(building_gdf, landmark_polygon, operation='within')
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
            landmark_ids = find_building_containing_point(building_gdf, target_point)

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

# def get_building_surface_svf(voxel_data, meshsize, **kwargs):
#     """
#     Compute and visualize the Sky View Factor (SVF) for building surface meshes.
    
#     Args:
#         voxel_data (ndarray): 3D array of voxel values.
#         meshsize (float): Size of each voxel in meters.
#         **kwargs: Additional parameters (colormap, ray counts, etc.)
    
#     Returns:
#         trimesh.Trimesh: Mesh of building surfaces with SVF values stored in metadata.
#     """
#     # Import required modules
#     import trimesh
#     import numpy as np
#     import time
    
#     # Default parameters
#     colormap = kwargs.get("colormap", 'BuPu_r')
#     vmin = kwargs.get("vmin", 0.0)
#     vmax = kwargs.get("vmax", 1.0)
#     N_azimuth = kwargs.get("N_azimuth", 60)
#     N_elevation = kwargs.get("N_elevation", 10)
#     debug = kwargs.get("debug", False)
#     progress_report = kwargs.get("progress_report", False)
    
#     # Tree transmittance parameters
#     tree_k = kwargs.get("tree_k", 0.6)
#     tree_lad = kwargs.get("tree_lad", 1.0)
    
#     # Sky detection parameters
#     hit_values = (0,)  # Sky is typically represented by 0
#     inclusion_mode = False  # We want rays that DON'T hit obstacles
    
#     # Extract building mesh (building voxels have value -3)
#     building_class_id = kwargs.get("building_class_id", -3)
#     start_time = time.time()
#     # print(f"Extracting building mesh for class ID {building_class_id}...")
#     try:
#         building_mesh = create_voxel_mesh(voxel_data, building_class_id, meshsize)
#         # print(f"Mesh extraction took {time.time() - start_time:.2f} seconds")
        
#         if building_mesh is None or len(building_mesh.faces) == 0:
#             print("No building surfaces found in voxel data.")
#             return None
            
#         # print(f"Successfully extracted mesh with {len(building_mesh.faces)} faces")
#     except Exception as e:
#         print(f"Error during mesh extraction: {e}")
#         return None
    
#     if progress_report:
#         print(f"Processing SVF for {len(building_mesh.faces)} building faces...")
    
#     try:
#         # Calculate face centers and normals
#         face_centers = building_mesh.triangles_center
#         face_normals = building_mesh.face_normals
        
#         # Initialize array to store SVF values for each face
#         face_svf_values = np.zeros(len(building_mesh.faces))
        
#         # Get voxel grid dimensions
#         grid_shape = voxel_data.shape
#         grid_bounds = np.array([
#             [0, 0, 0],  # Min bounds in voxel coordinates
#             [grid_shape[0], grid_shape[1], grid_shape[2]]  # Max bounds
#         ])
        
#         # Convert bounds to real-world coordinates
#         grid_bounds_real = grid_bounds * meshsize
        
#         # Small epsilon to detect boundary faces (within 0.5 voxel of boundary)
#         boundary_epsilon = meshsize * 0.05
        
#         # Create hemisphere directions for ray casting
#         hemisphere_dirs = []
#         azimuth_angles = np.linspace(0, 2 * np.pi, N_azimuth, endpoint=False)
#         elevation_angles = np.linspace(0, np.pi/2, N_elevation)  # 0 to 90 degrees
        
#         for elevation in elevation_angles:
#             sin_elev = np.sin(elevation)
#             cos_elev = np.cos(elevation)
#             for azimuth in azimuth_angles:
#                 x = cos_elev * np.cos(azimuth)
#                 y = cos_elev * np.sin(azimuth)
#                 z = sin_elev
#                 hemisphere_dirs.append([x, y, z])
        
#         hemisphere_dirs = np.array(hemisphere_dirs)
        
#         # Process each face
#         from scipy.spatial.transform import Rotation
#         processed_count = 0
#         boundary_count = 0
#         nan_boundary_count = 0
        
#         start_time = time.time()
#         for face_idx in range(len(building_mesh.faces)):
#             try:
#                 center = face_centers[face_idx]
#                 normal = face_normals[face_idx]
                
#                 # Check if this is a vertical surface (normal has no Z component)
#                 is_vertical = abs(normal[2]) < 0.01
                
#                 # Check if this face is on the boundary of the voxel grid
#                 on_x_min = abs(center[0] - grid_bounds_real[0, 0]) < boundary_epsilon
#                 on_y_min = abs(center[1] - grid_bounds_real[0, 1]) < boundary_epsilon
#                 on_x_max = abs(center[0] - grid_bounds_real[1, 0]) < boundary_epsilon
#                 on_y_max = abs(center[1] - grid_bounds_real[1, 1]) < boundary_epsilon
                
#                 # Check if this is a vertical surface on the boundary
#                 is_boundary_vertical = is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max)
                
#                 # Set NaN for all vertical surfaces on domain boundaries
#                 if is_boundary_vertical:
#                     face_svf_values[face_idx] = np.nan
#                     nan_boundary_count += 1
#                     processed_count += 1
#                     continue
                
#                 # For non-boundary surfaces, proceed with normal SVF calculation
#                 # Convert center to voxel coordinates (for ray origin)
#                 center_voxel = center / meshsize
                
#                 # IMPORTANT: Offset ray origin slightly to avoid self-intersection
#                 ray_origin = center_voxel + normal * 0.1  # Offset by 0.1 voxel units in normal direction
                
#                 # Create rotation from z-axis to face normal
#                 z_axis = np.array([0, 0, 1])
                
#                 # Handle special case where normal is parallel to z-axis
#                 if np.isclose(np.abs(np.dot(normal, z_axis)), 1.0, atol=1e-6):
#                     if np.dot(normal, z_axis) > 0:  # Normal points up
#                         rotation_matrix = np.eye(3)  # Identity matrix
#                     else:  # Normal points down
#                         rotation_matrix = np.array([
#                             [1, 0, 0],
#                             [0, -1, 0],
#                             [0, 0, -1]
#                         ])
#                     rotation = Rotation.from_matrix(rotation_matrix)
#                 else:
#                     # For all other cases, find rotation that aligns z-axis with normal
#                     rotation_axis = np.cross(z_axis, normal)
#                     rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
#                     angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))
#                     rotation = Rotation.from_rotvec(rotation_axis * angle)
                
#                 # Transform hemisphere directions to align with face normal
#                 local_dirs = rotation.apply(hemisphere_dirs)
                
#                 # Filter directions - keep only those that:
#                 # 1. Are pointing outward from the face (dot product with normal > 0)
#                 # 2. Have a positive z component (upward in world space)
#                 valid_dirs = []
#                 total_dirs = 0
                
#                 # Count total directions in the hemisphere (for normalization)
#                 for dir_vector in local_dirs:
#                     dot_product = np.dot(dir_vector, normal)
#                     # Count this direction if it's pointing outward from the face
#                     if dot_product > 0.01:  # Small threshold to avoid precision issues
#                         total_dirs += 1
#                         # Only trace rays that have a positive z component (can reach sky)
#                         if dir_vector[2] > 0:
#                             valid_dirs.append(dir_vector)
                
#                 # If no valid directions, SVF is 0
#                 if total_dirs == 0:
#                     face_svf_values[face_idx] = 0
#                     continue
                    
#                 # If no upward directions, SVF is 0 (all rays are blocked by ground)
#                 if len(valid_dirs) == 0:
#                     face_svf_values[face_idx] = 0
#                     continue
                    
#                 # Convert to numpy array for compute_vi_generic
#                 valid_dirs = np.array(valid_dirs, dtype=np.float64)
                
#                 # Calculate SVF using compute_vi_generic for the upward rays
#                 # Then scale by the fraction of upward rays to total rays
#                 upward_svf = compute_vi_generic(
#                     ray_origin,
#                     voxel_data,
#                     valid_dirs,
#                     hit_values,
#                     meshsize,
#                     tree_k,
#                     tree_lad,
#                     inclusion_mode
#                 )
                
#                 # Scale SVF by the fraction of rays that could potentially reach the sky
#                 # This accounts for downward rays that always have 0 SVF
#                 face_svf_values[face_idx] = upward_svf * (len(valid_dirs) / total_dirs)
            
#             except Exception as e:
#                 print(f"Error processing face {face_idx}: {e}")
#                 face_svf_values[face_idx] = 0
            
#             # Progress reporting
#             processed_count += 1
#             if progress_report:
#                 # Calculate frequency based on total number of faces, aiming for ~10 progress updates
#                 progress_frequency = max(1, len(building_mesh.faces) // 10)
#                 if processed_count % progress_frequency == 0 or processed_count == len(building_mesh.faces):
#                     elapsed = time.time() - start_time
#                     faces_per_second = processed_count / elapsed
#                     remaining = (len(building_mesh.faces) - processed_count) / faces_per_second if processed_count < len(building_mesh.faces) else 0
#                     print(f"Processed {processed_count}/{len(building_mesh.faces)} faces "
#                         f"({processed_count/len(building_mesh.faces)*100:.1f}%) - "
#                         f"{faces_per_second:.1f} faces/sec - "
#                         f"Est. remaining: {remaining:.1f} sec")
        
#         # print(f"Identified {nan_boundary_count} faces on domain vertical boundaries (set to NaN)")
        
#         # Store SVF values directly in mesh metadata
#         if not hasattr(building_mesh, 'metadata'):
#             building_mesh.metadata = {}
#         building_mesh.metadata['svf_values'] = face_svf_values
        
#         # Apply colors to the mesh based on SVF values (only for visualization)
#         if show_plot:
#             import matplotlib.cm as cm
#             import matplotlib.colors as mcolors
#             cmap = cm.get_cmap(colormap)
#             norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            
#             # Get a copy of face_svf_values with NaN replaced by a specific value outside the range
#             # This ensures NaN faces get a distinct color in the visualization
#             vis_values = face_svf_values.copy()
#             nan_mask = np.isnan(vis_values)
#             if np.any(nan_mask):
#                 # Use a color below vmin for NaN values (they'll be clipped to vmin in the colormap)
#                 # But we can see them as the minimum color
#                 vis_values[nan_mask] = vmin - 0.1
            
#             # Apply colors
#             face_colors = cmap(norm(vis_values))
#             building_mesh.visual.face_colors = face_colors
            
#             # Create a scene with the colored mesh
#             scene = trimesh.Scene()
#             scene.add_geometry(building_mesh)
#             scene.show()
            
#             # Also create a matplotlib figure with colorbar for reference
#             import matplotlib.pyplot as plt
            
#             fig, ax = plt.subplots(figsize=(8, 3))
#             cb = plt.colorbar(
#                 cm.ScalarMappable(norm=norm, cmap=cmap),
#                 ax=ax,
#                 orientation='horizontal',
#                 label='Sky View Factor'
#             )
#             ax.remove()  # Remove the axes, keep only colorbar
#             plt.tight_layout()
#             plt.show()
        
#             # Plot histogram of SVF values (excluding NaN)
#             valid_svf = face_svf_values[~np.isnan(face_svf_values)]
#             plt.figure(figsize=(10, 6))
#             plt.hist(valid_svf, bins=50, color='skyblue', alpha=0.7)
#             plt.title('Distribution of Sky View Factor on Building Surfaces')
#             plt.xlabel('Sky View Factor')
#             plt.ylabel('Frequency')
#             plt.grid(True, alpha=0.3)
#             plt.tight_layout()
#             plt.show()
        
#         # Handle optional OBJ export
#         obj_export = kwargs.get("obj_export", False)
#         if obj_export:
#             output_dir = kwargs.get("output_directory", "output")
#             output_file_name = kwargs.get("output_file_name", "building_surface_svf")
            
#             # Ensure output directory exists
#             import os
#             os.makedirs(output_dir, exist_ok=True)
            
#             # Export as OBJ with face colors
#             try:
#                 building_mesh.export(f"{output_dir}/{output_file_name}.obj")
#                 print(f"Exported building SVF mesh to {output_dir}/{output_file_name}.obj")
#             except Exception as e:
#                 print(f"Error exporting mesh: {e}")
        
#         return building_mesh
        
#     except Exception as e:
#         print(f"Error during SVF calculation: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

##############################################################################
# 1) New Numba helper: Rodrigues’ rotation formula for rotating vectors
##############################################################################
@njit
def rotate_vector_axis_angle(vec, axis, angle):
    """
    Rotate a 3D vector 'vec' around 'axis' by 'angle' (in radians),
    using Rodrigues’ rotation formula.
    """
    # Normalize rotation axis
    axis_len = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    if axis_len < 1e-12:
        # Axis is degenerate; return vec unchanged
        return vec
    
    ux, uy, uz = axis / axis_len
    c = np.cos(angle)
    s = np.sin(angle)
    dot = vec[0]*ux + vec[1]*uy + vec[2]*uz
    
    # cross = axis x vec
    cross_x = uy*vec[2] - uz*vec[1]
    cross_y = uz*vec[0] - ux*vec[2]
    cross_z = ux*vec[1] - uy*vec[0]
    
    # Rodrigues formula: v_rot = v*c + (k x v)*s + k*(k·v)*(1-c)
    v_rot = np.zeros(3, dtype=np.float64)
    # v*c
    v_rot[0] = vec[0] * c
    v_rot[1] = vec[1] * c
    v_rot[2] = vec[2] * c
    # + (k x v)*s
    v_rot[0] += cross_x * s
    v_rot[1] += cross_y * s
    v_rot[2] += cross_z * s
    # + k*(k·v)*(1-c)
    tmp = dot * (1.0 - c)
    v_rot[0] += ux * tmp
    v_rot[1] += uy * tmp
    v_rot[2] += uz * tmp
    
    return v_rot


# ##############################################################################
# # 2) New Numba helper: vectorized SVF computation for each face
# ##############################################################################
# @njit
# def compute_svf_for_all_faces(
#     face_centers,
#     face_normals,
#     hemisphere_dirs,
#     voxel_data,
#     meshsize,
#     tree_k,
#     tree_lad,
#     hit_values,
#     inclusion_mode,
#     grid_bounds_real,
#     boundary_epsilon
# ):
#     """
#     Per-face SVF calculation in Numba:
#       - Checks boundary conditions & sets NaN for boundary-vertical faces
#       - Builds local hemisphere (rotates from +Z to face normal)
#       - Filters directions that actually face outward (+ dot>0) and have z>0
#       - Calls compute_vi_generic to get fraction that sees sky
#       - Returns array of SVF values (same length as face_centers)
#     """
#     n_faces = face_centers.shape[0]
#     face_svf_values = np.zeros(n_faces, dtype=np.float64)
    
#     z_axis = np.array([0.0, 0.0, 1.0])
    
#     for fidx in range(n_faces):
#         center = face_centers[fidx]
#         normal = face_normals[fidx]
        
#         # -- 1) Check for boundary + vertical face => NaN
#         is_vertical = (abs(normal[2]) < 0.01)
        
#         on_x_min = (abs(center[0] - grid_bounds_real[0,0]) < boundary_epsilon)
#         on_y_min = (abs(center[1] - grid_bounds_real[0,1]) < boundary_epsilon)
#         on_x_max = (abs(center[0] - grid_bounds_real[1,0]) < boundary_epsilon)
#         on_y_max = (abs(center[1] - grid_bounds_real[1,1]) < boundary_epsilon)
        
#         is_boundary_vertical = is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max)
#         if is_boundary_vertical:
#             face_svf_values[fidx] = np.nan
#             continue
        
#         # -- 2) Compute rotation that aligns face normal -> +Z
#         norm_n = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
#         if norm_n < 1e-12:
#             # Degenerate normal
#             face_svf_values[fidx] = 0.0
#             continue
        
#         dot_zn = z_axis[0]*normal[0] + z_axis[1]*normal[1] + z_axis[2]*normal[2]
#         cos_angle = dot_zn / (norm_n)
#         if cos_angle >  1.0: cos_angle =  1.0
#         if cos_angle < -1.0: cos_angle = -1.0
#         angle = np.arccos(cos_angle)
        
#         # Distinguish near +Z vs near -Z vs general case
#         if abs(cos_angle - 1.0) < 1e-9:
#             # normal ~ +Z => no rotation
#             local_dirs = hemisphere_dirs
#         elif abs(cos_angle + 1.0) < 1e-9:
#             # normal ~ -Z => rotate 180 around X (or Y) axis
#             axis_180 = np.array([1.0, 0.0, 0.0])
#             local_dirs = np.empty_like(hemisphere_dirs)
#             for i in range(hemisphere_dirs.shape[0]):
#                 local_dirs[i] = rotate_vector_axis_angle(hemisphere_dirs[i], axis_180, np.pi)
#         else:
#             # normal is neither up nor down -> do standard axis-angle
#             axis_x = z_axis[1]*normal[2] - z_axis[2]*normal[1]
#             axis_y = z_axis[2]*normal[0] - z_axis[0]*normal[2]
#             axis_z = z_axis[0]*normal[1] - z_axis[1]*normal[0]
#             rot_axis = np.array([axis_x, axis_y, axis_z], dtype=np.float64)
            
#             local_dirs = np.empty_like(hemisphere_dirs)
#             for i in range(hemisphere_dirs.shape[0]):
#                 local_dirs[i] = rotate_vector_axis_angle(
#                     hemisphere_dirs[i],
#                     rot_axis,
#                     angle
#                 )
        
#         # -- 3) Count how many directions are outward & upward
#         total_outward = 0
#         num_upward = 0
#         for i in range(local_dirs.shape[0]):
#             dvec = local_dirs[i]
#             dp = dvec[0]*normal[0] + dvec[1]*normal[1] + dvec[2]*normal[2]
#             if dp > 0.0:
#                 total_outward += 1
#                 if dvec[2] > 0.0:
#                     num_upward += 1
        
#         # If no outward directions at all => SVF=0
#         if total_outward == 0:
#             face_svf_values[fidx] = 0.0
#             continue
        
#         # If no upward directions among them => SVF=0
#         if num_upward == 0:
#             face_svf_values[fidx] = 0.0
#             continue
        
#         # -- 4) Create an array for only the upward directions
#         valid_dirs_arr = np.empty((num_upward, 3), dtype=np.float64)
#         out_idx = 0
#         for i in range(local_dirs.shape[0]):
#             dvec = local_dirs[i]
#             dp = dvec[0]*normal[0] + dvec[1]*normal[1] + dvec[2]*normal[2]
#             if dp > 0.0 and dvec[2] > 0.0:
#                 valid_dirs_arr[out_idx, 0] = dvec[0]
#                 valid_dirs_arr[out_idx, 1] = dvec[1]
#                 valid_dirs_arr[out_idx, 2] = dvec[2]
#                 out_idx += 1
        
#         # -- 5) Ray origin in voxel coords, offset along face normal
#         offset_vox = 0.1
#         ray_origin = (center / meshsize) + (normal / norm_n) * offset_vox
        
#         # -- 6) Compute fraction of rays that see sky
#         upward_svf = compute_vi_generic(
#             ray_origin,
#             voxel_data,
#             valid_dirs_arr,
#             hit_values,
#             meshsize,
#             tree_k,
#             tree_lad,
#             inclusion_mode
#         )
        
#         # Scale by fraction of directions that were outward
#         fraction_up = num_upward / total_outward
#         face_svf_values[fidx] = upward_svf * fraction_up
    
#     return face_svf_values


# ##############################################################################
# # 3) Modified get_building_surface_svf (only numeric loop changed)
# ##############################################################################
# def get_building_surface_svf(voxel_data, meshsize, **kwargs):
#     """
#     Compute and visualize the Sky View Factor (SVF) for building surface meshes.
    
#     Args:
#         voxel_data (ndarray): 3D array of voxel values.
#         meshsize (float): Size of each voxel in meters.
#         **kwargs: Additional parameters (colormap, ray counts, etc.)
    
#     Returns:
#         trimesh.Trimesh: Mesh of building surfaces with SVF values stored in metadata.
#     """
#     import matplotlib.pyplot as plt
#     import matplotlib.cm as cm
#     import matplotlib.colors as mcolors
#     import os
    
#     # Default parameters
#     colormap = kwargs.get("colormap", 'BuPu_r')
#     vmin = kwargs.get("vmin", 0.0)
#     vmax = kwargs.get("vmax", 1.0)
#     N_azimuth = kwargs.get("N_azimuth", 60)
#     N_elevation = kwargs.get("N_elevation", 10)
#     debug = kwargs.get("debug", False)
#     progress_report = kwargs.get("progress_report", False)
#     building_id_grid = kwargs.get("building_id_grid", None)
    
#     # Tree parameters
#     tree_k = kwargs.get("tree_k", 0.6)
#     tree_lad = kwargs.get("tree_lad", 1.0)
    
#     # Sky detection parameters
#     hit_values = (0,)  # '0' is sky
#     inclusion_mode = False  # we want rays that DON'T hit obstacles (except sky)
    
#     # Building ID in voxel data
#     building_class_id = kwargs.get("building_class_id", -3)
    
#     start_time = time.time()
#     # 1) Extract building mesh from voxel_data
#     try:
#         # This function is presumably in your codebase (not shown):
#         building_mesh = create_voxel_mesh(voxel_data, building_class_id, meshsize, building_id_grid=building_id_grid)
#         if building_mesh is None or len(building_mesh.faces) == 0:
#             print("No building surfaces found in voxel data.")
#             return None
#     except Exception as e:
#         print(f"Error during mesh extraction: {e}")
#         return None
    
#     if progress_report:
#         print(f"Processing SVF for {len(building_mesh.faces)} building faces...")
    
#     # 2) Get face centers + normals as NumPy arrays
#     face_centers = building_mesh.triangles_center
#     face_normals = building_mesh.face_normals
    
#     # 3) Precompute hemisphere directions (global, pointing up)
#     azimuth_angles   = np.linspace(0, 2*np.pi, N_azimuth, endpoint=False)
#     elevation_angles = np.linspace(0, np.pi/2, N_elevation)
#     hemisphere_list = []
#     for elev in elevation_angles:
#         sin_elev = np.sin(elev)
#         cos_elev = np.cos(elev)
#         for az in azimuth_angles:
#             x = cos_elev * np.cos(az)
#             y = cos_elev * np.sin(az)
#             z = sin_elev
#             hemisphere_list.append([x, y, z])
#     hemisphere_dirs = np.array(hemisphere_list, dtype=np.float64)
    
#     # 4) Domain bounds in real coordinates
#     grid_shape = voxel_data.shape
#     grid_bounds_voxel = np.array([[0,0,0],[grid_shape[0],grid_shape[1],grid_shape[2]]], dtype=np.float64)
#     grid_bounds_real = grid_bounds_voxel * meshsize
#     boundary_epsilon = meshsize * 0.05
    
#     # 5) Call Numba-accelerated routine
#     face_svf_values = compute_svf_for_all_faces(
#         face_centers,
#         face_normals,
#         hemisphere_dirs,
#         voxel_data,
#         meshsize,
#         tree_k,
#         tree_lad,
#         hit_values,
#         inclusion_mode,
#         grid_bounds_real,
#         boundary_epsilon
#     )
    
#     # 6) Store SVF values in mesh metadata
#     if not hasattr(building_mesh, 'metadata'):
#         building_mesh.metadata = {}
#     building_mesh.metadata['svf_values'] = face_svf_values
       
#     # OBJ export if desired
#     obj_export = kwargs.get("obj_export", False)
#     if obj_export:
#         output_dir = kwargs.get("output_directory", "output")
#         output_file_name = kwargs.get("output_file_name", "building_surface_svf")
#         os.makedirs(output_dir, exist_ok=True)
#         try:
#             building_mesh.export(f"{output_dir}/{output_file_name}.obj")
#             print(f"Exported building SVF mesh to {output_dir}/{output_file_name}.obj")
#         except Exception as e:
#             print(f"Error exporting mesh: {e}")
    
#     return building_mesh

@njit
def compute_view_factor_for_all_faces(
    face_centers,
    face_normals,
    hemisphere_dirs,
    voxel_data,
    meshsize,
    tree_k,
    tree_lad,
    target_values,
    inclusion_mode,
    grid_bounds_real,
    boundary_epsilon
):
    """
    Compute a per-face "view factor" for a specified set of target voxel classes.

    By default (as in the old SVF case), you would pass:
        target_values = (0,)      # voxel value for 'sky'
        inclusion_mode = False    # i.e. any *non*-sky voxel will block the ray

    But you can pass any other combination:
        - E.g. target_values = (-2,), inclusion_mode=True
          to measure fraction of directions that intersect 'trees' (-2).
        - E.g. target_values = (-3,), inclusion_mode=True
          to measure fraction of directions that intersect 'buildings' (-3).
    
    Args:
        face_centers (np.ndarray): (n_faces, 3) face centroid positions.
        face_normals (np.ndarray): (n_faces, 3) face normals.
        hemisphere_dirs (np.ndarray): (N, 3) set of direction vectors in the hemisphere.
        voxel_data (np.ndarray): 3D array of voxel values.
        meshsize (float): Size of each voxel in meters.
        tree_k (float): Tree extinction coefficient.
        tree_lad (float): Leaf area density in m^-1.
        target_values (tuple[int]): Voxel classes that define a 'hit'.
        inclusion_mode (bool): If True, hitting any of target_values is considered "visible."
                               If False, hitting anything *not* in target_values (except -2 trees) blocks the ray.
        grid_bounds_real (np.ndarray): [[x_min,y_min,z_min],[x_max,y_max,z_max]] in real coords.
        boundary_epsilon (float): tolerance for marking boundary vertical faces.

    Returns:
        np.ndarray of shape (n_faces,):
            The computed view factor for each face (NaN for boundary‐vertical faces).
    """
    n_faces = face_centers.shape[0]
    face_vf_values = np.zeros(n_faces, dtype=np.float64)
    
    z_axis = np.array([0.0, 0.0, 1.0])
    
    for fidx in range(n_faces):
        center = face_centers[fidx]
        normal = face_normals[fidx]
        
        # -- 1) Check for boundary + vertical face => NaN
        is_vertical = (abs(normal[2]) < 0.01)
        
        on_x_min = (abs(center[0] - grid_bounds_real[0,0]) < boundary_epsilon)
        on_y_min = (abs(center[1] - grid_bounds_real[0,1]) < boundary_epsilon)
        on_x_max = (abs(center[0] - grid_bounds_real[1,0]) < boundary_epsilon)
        on_y_max = (abs(center[1] - grid_bounds_real[1,1]) < boundary_epsilon)
        
        is_boundary_vertical = is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max)
        if is_boundary_vertical:
            face_vf_values[fidx] = np.nan
            continue
        
        # -- 2) Compute rotation that aligns face normal -> +Z
        norm_n = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        if norm_n < 1e-12:
            # Degenerate normal
            face_vf_values[fidx] = 0.0
            continue
        
        dot_zn = z_axis[0]*normal[0] + z_axis[1]*normal[1] + z_axis[2]*normal[2]
        cos_angle = dot_zn / (norm_n)
        if cos_angle >  1.0: cos_angle =  1.0
        if cos_angle < -1.0: cos_angle = -1.0
        angle = np.arccos(cos_angle)
        
        # Distinguish near +Z vs near -Z vs general case
        if abs(cos_angle - 1.0) < 1e-9:
            # normal ~ +Z => no rotation needed
            local_dirs = hemisphere_dirs
        elif abs(cos_angle + 1.0) < 1e-9:
            # normal ~ -Z => rotate 180 around X (or Y) axis
            axis_180 = np.array([1.0, 0.0, 0.0])
            local_dirs = np.empty_like(hemisphere_dirs)
            for i in range(hemisphere_dirs.shape[0]):
                local_dirs[i] = rotate_vector_axis_angle(hemisphere_dirs[i], axis_180, np.pi)
        else:
            # normal is neither up nor down -> do standard axis-angle
            axis_x = z_axis[1]*normal[2] - z_axis[2]*normal[1]
            axis_y = z_axis[2]*normal[0] - z_axis[0]*normal[2]
            axis_z = z_axis[0]*normal[1] - z_axis[1]*normal[0]
            rot_axis = np.array([axis_x, axis_y, axis_z], dtype=np.float64)
            
            local_dirs = np.empty_like(hemisphere_dirs)
            for i in range(hemisphere_dirs.shape[0]):
                local_dirs[i] = rotate_vector_axis_angle(
                    hemisphere_dirs[i],
                    rot_axis,
                    angle
                )
        
        # -- 3) Count how many directions are outward & upward
        total_outward = 0
        num_upward = 0
        for i in range(local_dirs.shape[0]):
            dvec = local_dirs[i]
            dp = dvec[0]*normal[0] + dvec[1]*normal[1] + dvec[2]*normal[2]
            if dp > 0.0:
                total_outward += 1
                if dvec[2] > 0.0:
                    num_upward += 1
        
        # If no outward directions at all => view factor = 0
        if total_outward == 0:
            face_vf_values[fidx] = 0.0
            continue
        
        # If no upward directions => view factor = 0
        if num_upward == 0:
            face_vf_values[fidx] = 0.0
            continue
        
        # -- 4) Create an array for only the upward directions
        valid_dirs_arr = np.empty((num_upward, 3), dtype=np.float64)
        out_idx = 0
        for i in range(local_dirs.shape[0]):
            dvec = local_dirs[i]
            dp = dvec[0]*normal[0] + dvec[1]*normal[1] + dvec[2]*normal[2]
            if dp > 0.0 and dvec[2] > 0.0:
                valid_dirs_arr[out_idx, 0] = dvec[0]
                valid_dirs_arr[out_idx, 1] = dvec[1]
                valid_dirs_arr[out_idx, 2] = dvec[2]
                out_idx += 1
        
        # -- 5) Ray origin in voxel coords, offset along face normal
        offset_vox = 0.1
        ray_origin = (center / meshsize) + (normal / norm_n) * offset_vox
        
        # -- 6) Compute fraction of rays that "see" the target
        #    (in the old code, "seeing the sky" meant the ray was NOT blocked by non‐sky voxels)
        upward_vf = compute_vi_generic(
            ray_origin,
            voxel_data,
            valid_dirs_arr,
            target_values,
            meshsize,
            tree_k,
            tree_lad,
            inclusion_mode
        )
        
        # Scale by fraction of directions that were outward
        fraction_up = num_upward / total_outward
        face_vf_values[fidx] = upward_vf * fraction_up
    
    return face_vf_values

def get_surface_view_factor(voxel_data, meshsize, **kwargs):
    """
    Compute and optionally visualize the "view factor" for surface meshes
    with respect to a chosen target voxel class (or classes).
    
    By default, it computes Sky View Factor (target_values=(0,), inclusion_mode=False).
    But you can pass different arguments for other view factors:
      - target_values=(-2,), inclusion_mode=True  => Tree view factor
      - target_values=(-3,), inclusion_mode=True  => Building view factor
      etc.

    Args:
        voxel_data (ndarray): 3D array of voxel values
        meshsize (float): Size of each voxel in meters
        **kwargs: Additional parameters (colormap, ray counts, etc.)
                  including:
            target_values (tuple[int]): voxel classes that define 'hits'
            inclusion_mode (bool): interpretation of hits
            building_class_id (int): which class to mesh for surface extraction
            ...
    
    Returns:
        trimesh.Trimesh: The surface mesh with per-face view-factor values in metadata.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import os
    
    # Default parameters
    value_name     = kwargs.get("value_name", 'view_factor_values')
    colormap       = kwargs.get("colormap", 'BuPu_r')
    vmin           = kwargs.get("vmin", 0.0)
    vmax           = kwargs.get("vmax", 1.0)
    N_azimuth      = kwargs.get("N_azimuth", 60)
    N_elevation    = kwargs.get("N_elevation", 10)
    debug          = kwargs.get("debug", False)
    progress_report= kwargs.get("progress_report", False)
    building_id_grid = kwargs.get("building_id_grid", None)
    
    # Tree & bounding params
    tree_k         = kwargs.get("tree_k", 0.6)
    tree_lad       = kwargs.get("tree_lad", 1.0)
    
    # ----------------------------------------
    # NEW: user can override target classes
    # defaults for "sky" factor:
    target_values  = kwargs.get("target_values", (0,))  
    inclusion_mode = kwargs.get("inclusion_mode", False)
    # ----------------------------------------
    
    # Voxel class used for building (or other) surface
    building_class_id = kwargs.get("building_class_id", -3)
    
    # 1) Extract mesh from voxel_data
    try:
        building_mesh = create_voxel_mesh(
            voxel_data, 
            building_class_id, 
            meshsize,
            building_id_grid=building_id_grid,
            mesh_type='open_air'
        )
        if building_mesh is None or len(building_mesh.faces) == 0:
            print("No surfaces found in voxel data for the specified class.")
            return None
    except Exception as e:
        print(f"Error during mesh extraction: {e}")
        return None
    
    if progress_report:
        print(f"Processing view factor for {len(building_mesh.faces)} faces...")

    # 2) Get face centers + normals
    face_centers = building_mesh.triangles_center
    face_normals = building_mesh.face_normals
    
    # 3) Precompute hemisphere directions
    azimuth_angles   = np.linspace(0, 2*np.pi, N_azimuth, endpoint=False)
    elevation_angles = np.linspace(0, np.pi/2, N_elevation)
    hemisphere_list = []
    for elev in elevation_angles:
        sin_elev = np.sin(elev)
        cos_elev = np.cos(elev)
        for az in azimuth_angles:
            x = cos_elev * np.cos(az)
            y = cos_elev * np.sin(az)
            z = sin_elev
            hemisphere_list.append([x, y, z])
    hemisphere_dirs = np.array(hemisphere_list, dtype=np.float64)
    
    # 4) Domain bounds in real coordinates
    nx, ny, nz = voxel_data.shape
    grid_bounds_voxel = np.array([[0,0,0],[nx, ny, nz]], dtype=np.float64)
    grid_bounds_real = grid_bounds_voxel * meshsize
    boundary_epsilon = meshsize * 0.05
    
    # 5) Call the new Numba routine for per-face view factor
    face_vf_values = compute_view_factor_for_all_faces(
        face_centers,
        face_normals,
        hemisphere_dirs,
        voxel_data,
        meshsize,
        tree_k,
        tree_lad,
        target_values,   # <--- new
        inclusion_mode,  # <--- new
        grid_bounds_real,
        boundary_epsilon
    )
    
    # 6) Store these values in the mesh metadata
    if not hasattr(building_mesh, 'metadata'):
        building_mesh.metadata = {}
    building_mesh.metadata[value_name] = face_vf_values
       
    # Optionally export to OBJ
    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        output_dir      = kwargs.get("output_directory", "output")
        output_file_name= kwargs.get("output_file_name", "surface_view_factor")
        os.makedirs(output_dir, exist_ok=True)
        try:
            building_mesh.export(f"{output_dir}/{output_file_name}.obj")
            print(f"Exported surface mesh to {output_dir}/{output_file_name}.obj")
        except Exception as e:
            print(f"Error exporting mesh: {e}")
    
    return building_mesh