"""Functions for computing and visualizing various view indices in a voxel city model.

This module provides functionality to compute and visualize:
- Green View Index (GVI): Measures visibility of green elements like trees and vegetation
- Sky View Index (SVI): Measures visibility of open sky from street level
- Landmark Visibility: Measures visibility of specified landmark buildings from different locations

The module uses optimized ray tracing techniques with Numba JIT compilation for efficient computation.
Key features:
- Generic ray tracing framework that can be customized for different view indices
- Parallel processing for fast computation of view maps
- Visualization tools including matplotlib plots and OBJ exports
- Support for both inclusion and exclusion based visibility checks
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numba import njit, prange

from ..file.geojson import find_building_containing_point
from ..file.obj import grid_to_obj, export_obj

@njit
def trace_ray_generic(voxel_data, origin, direction, hit_values, inclusion_mode=True):
    """Trace a ray through a voxel grid and check for hits with specified values.
    
    Uses an optimized DDA (Digital Differential Analyzer) algorithm for ray traversal.

    Args:
        voxel_data (ndarray): 3D array of voxel values
        origin (tuple): Starting point (x,y,z) of ray
        direction (tuple): Direction vector of ray
        hit_values (tuple): Values to check for hits
        inclusion_mode (bool): If True, hit when value in hit_values. If False, hit when not in hit_values.

    Returns:
        bool: True if ray hits target value(s), False otherwise
    """
    nx, ny, nz = voxel_data.shape
    x0, y0, z0 = origin
    dx, dy, dz = direction

    # Normalize direction vector
    length = np.sqrt(dx*dx + dy*dy + dz*dz)
    if length == 0.0:
        return False
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
    while (0 <= i < nx) and (0 <= j < ny) and (0 <= k < nz):
        voxel_value = voxel_data[i, j, k]

        if inclusion_mode:
            # Inclusion mode: hit if voxel_value in hit_values
            for hv in hit_values:
                if voxel_value == hv:
                    return True
        else:
            # Exclusion mode: hit if voxel_value not in hit_values
            in_set = False
            for hv in hit_values:
                if voxel_value == hv:
                    in_set = True
                    break
            if not in_set:
                return True

        # Move to next voxel using DDA algorithm
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

    # No hit found within grid bounds
    return False

@njit
def compute_vi_generic(observer_location, voxel_data, ray_directions, hit_values, inclusion_mode=True):
    """Compute view index for a single observer location by casting multiple rays.

    Args:
        observer_location (ndarray): Position of observer (x,y,z)
        voxel_data (ndarray): 3D array of voxel values
        ray_directions (ndarray): Array of direction vectors for rays
        hit_values (tuple): Values to check for hits
        inclusion_mode (bool): If True, hit when value in hit_values. If False, hit when not in hit_values.

    Returns:
        float: Ratio of successful rays (0.0 to 1.0)
    """
    hit_count = 0
    total_rays = ray_directions.shape[0]

    # Cast rays in all directions and count hits
    for idx in range(total_rays):
        direction = ray_directions[idx]
        result = trace_ray_generic(voxel_data, observer_location, direction, hit_values, inclusion_mode)
        if inclusion_mode:
            if result:  # hit found
                hit_count += 1
        else:
            if not result:  # no hit means success in exclusion mode
                hit_count += 1

    return hit_count / total_rays

@njit(parallel=True)
def compute_vi_map_generic(voxel_data, ray_directions, view_height_voxel, hit_values, inclusion_mode=True):
    """Compute view index map for entire grid by placing observers at valid locations.

    Valid observer locations are empty voxels above ground level, excluding building roofs
    and vegetation surfaces.

    Args:
        voxel_data (ndarray): 3D array of voxel values
        ray_directions (ndarray): Array of direction vectors for rays
        view_height_voxel (int): Height offset for observer in voxels
        hit_values (tuple): Values to check for hits
        inclusion_mode (bool): If True, hit when value in hit_values. If False, hit when not in hit_values.

    Returns:
        ndarray: 2D array of view index values with y-axis flipped
    """
    nx, ny, nz = voxel_data.shape
    vi_map = np.full((nx, ny), np.nan)

    # Process each x,y position in parallel
    for x in prange(nx):
        for y in range(ny):
            found_observer = False
            # Find lowest empty voxel above ground
            for z in range(1, nz):
                if voxel_data[x, y, z] in (0, -2) and voxel_data[x, y, z - 1] not in (0, -2):
                    # Skip if standing on building or vegetation
                    if voxel_data[x, y, z - 1] in (-3, 7, 8, 9):
                        vi_map[x, y] = np.nan
                        found_observer = True
                        break
                    else:
                        # Place observer and compute view index
                        observer_location = np.array([x, y, z + view_height_voxel], dtype=np.float64)
                        vi_value = compute_vi_generic(observer_location, voxel_data, ray_directions, hit_values, inclusion_mode)
                        vi_map[x, y] = vi_value
                        found_observer = True
                        break
            if not found_observer:
                vi_map[x, y] = np.nan

    return np.flipud(vi_map)

def get_view_index(voxel_data, meshsize, mode=None, hit_values=None, inclusion_mode=True, **kwargs):
    """Calculate and visualize a generic view index for a voxel city model.

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
        # For other modes, user must specify hit_values
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

    # Compute the view index map
    vi_map = compute_vi_map_generic(voxel_data, ray_directions, view_height_voxel, hit_values, inclusion_mode)

    # Plot results
    cmap = plt.cm.get_cmap(colormap).copy()
    cmap.set_bad(color='lightgray')
    plt.figure(figsize=(10, 8))
    plt.imshow(vi_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='View Index')
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

def mark_building_by_id(voxcity_grid, building_id_grid_ori, ids, mark):
    """Mark specific buildings in the voxel grid with a given value.

    Args:
        voxcity_grid (ndarray): 3D array of voxel values
        building_id_grid_ori (ndarray): 2D array of building IDs
        ids (list): List of building IDs to mark
        mark (int): Value to mark the buildings with
    """
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

@njit
def trace_ray_to_target(voxel_data, origin, target, opaque_values):
    """Trace a ray from origin to target through voxel data.

    Uses DDA algorithm to efficiently traverse voxels along ray path.

    Args:
        voxel_data (ndarray): 3D array of voxel values
        origin (tuple): Starting point (x,y,z) of ray
        target (tuple): End point (x,y,z) of ray
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

    Args:
        observer_location (ndarray): Observer position (x,y,z)
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

    Args:
        voxel_data (ndarray): 3D array of voxel values
        landmark_positions (ndarray): Array of landmark positions
        opaque_values (ndarray): Array of voxel values that block visibility
        view_height_voxel (int): Height offset for observer in voxels

    Returns:
        ndarray: 2D array of visibility values (0 or 1)
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
                    if voxel_data[x, y, z - 1] in (-3, -2, 7, 8, 9):
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

    Args:
        voxel_data (ndarray): 3D array of voxel values
        target_value (int, optional): Value used to identify landmark voxels. Defaults to -30.
        view_height_voxel (int, optional): Height offset for observer in voxels. Defaults to 0.
        colormap (str, optional): Matplotlib colormap name. Defaults to 'viridis'.

    Returns:
        ndarray: 2D array of visibility values (0 or 1) with y-axis flipped

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
    
    plt.show()

    return np.flipud(visibility_map)

def get_landmark_visibility_map(voxcity_grid, building_id_grid, building_geojson, meshsize, **kwargs):
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
    if landmark_ids is None:
        rectangle_vertices = kwargs.get("rectangle_vertices", None)
        if rectangle_vertices is None:
            print("Cannot set landmark buildings. You need to input either of rectangle_vertices or landmark_ids.")
            return None
            
        # Calculate center point of rectangle
        lats = [coord[0] for coord in rectangle_vertices]
        lons = [coord[1] for coord in rectangle_vertices]
        center_lat = (min(lats) + max(lats)) / 2
        center_lon = (min(lons) + max(lons)) / 2
        target_point = (center_lat, center_lon)
        
        # Find buildings at center point
        landmark_ids = find_building_containing_point(features, target_point)

    # Mark landmark buildings in voxel grid with special value
    target_value = -30
    mark_building_by_id(voxcity_grid, building_id_grid, landmark_ids, target_value)
    
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

    return landmark_vis_map

def get_sky_view_factor_map(voxel_data, meshsize, **kwargs):
    """
    Compute and visualize the Sky View Factor (SVF) for each valid observer cell in the voxel grid.

    The SVF is computed similarly to how the 'sky' mode of the get_view_index function works:
    - Rays are cast from each observer position upward (within a specified angular range).
    - Any non-empty voxel encountered is considered an obstruction.
    - The ratio of unobstructed rays to total rays is the SVF.

    Args:
        voxel_data (ndarray): 3D array of voxel values.
        meshsize (float): Size of each voxel in meters.
        **kwargs: Additional parameters:
            - view_point_height (float): Observer height above ground in meters. Default: 1.5
            - colormap (str): Matplotlib colormap name. Default: 'viridis'
            - vmin (float): Minimum value for color bar. Default: 0.0
            - vmax (float): Maximum value for color bar. Default: 1.0
            - N_azimuth (int): Number of azimuth angles. Default: 60
            - N_elevation (int): Number of elevation angles. Default: 10
            - elevation_min_degrees (float): Minimum elevation angle in degrees. Typically 0 (horizon). Default: 0
            - elevation_max_degrees (float): Maximum elevation angle in degrees. Typically 90 for full hemisphere. Default: 90
            - obj_export (bool): Whether to export the result as an OBJ file. Default: False
            - output_directory (str), output_file_name (str), etc. are also supported if OBJ export is needed.

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

    # Define hit_values and inclusion_mode for sky detection
    # For sky: hit_values=(0,), inclusion_mode=False means:
    # A hit occurs whenever we encounter a voxel_value != 0 along the ray.
    # Thus, a ray that escapes with only 0's encountered is unobstructed sky.
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

    # Compute the SVF map using the same compute function but with the sky parameters
    vi_map = compute_vi_map_generic(voxel_data, ray_directions, view_height_voxel, hit_values, inclusion_mode)

    # vi_map now holds the fraction of rays that 'hit' the condition for sky (no obstruction)
    # Actually, since inclusion_mode=False and hit_values=(0,), vi_map gives the fraction of unobstructed rays.
    # This is essentially the sky view factor.

    # Plot results
    cmap = plt.cm.get_cmap(colormap).copy()
    cmap.set_bad(color='lightgray')
    plt.figure(figsize=(10, 8))
    plt.title("Sky View Factor Map")
    plt.imshow(vi_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Sky View Factor')
    plt.show()

    # Optional OBJ export
    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        from ..file.obj import grid_to_obj
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