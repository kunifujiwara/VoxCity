"""Functions for computing and visualizing various view indices in a voxel city model.

This module provides functionality to compute and visualize:
- Green View Index (GVI): Measures visibility of green elements
- Sky View Index (SVI): Measures visibility of the sky
- Landmark Visibility: Measures visibility of specified landmark buildings

The module uses ray tracing techniques optimized with Numba JIT compilation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numba import njit, prange

from ..file.geojson import find_building_containing_point
from ..file.obj import grid_to_obj, export_obj

@njit
def trace_ray_generic(voxel_data, origin, direction, hit_values, inclusion_mode=True):
    nx, ny, nz = voxel_data.shape
    x0, y0, z0 = origin
    dx, dy, dz = direction

    # Normalize direction
    length = np.sqrt(dx*dx + dy*dy + dz*dz)
    if length == 0.0:
        return False
    dx /= length
    dy /= length
    dz /= length

    x, y, z = x0 + 0.5, y0 + 0.5, z0 + 0.5
    i, j, k = int(x0), int(y0), int(z0)

    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1
    step_z = 1 if dz >= 0 else -1

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

    while (0 <= i < nx) and (0 <= j < ny) and (0 <= k < nz):
        voxel_value = voxel_data[i, j, k]

        # Check hit condition
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
                # voxel_value not in hit_values => hit
                return True

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

    # Exited the grid without finding a hit
    return False

@njit
def compute_gvi(observer_location, voxel_data, ray_directions):
    """Compute Green View Index for a single observer location."""
    green_hit_values = (-2, 2, 5, 7)  # Green-related voxel values
    green_count = 0
    total_rays = ray_directions.shape[0]

    for idx in range(total_rays):
        direction = ray_directions[idx]
        # Inclusion mode = True, hits if voxel_value in green_hit_values
        if trace_ray_generic(voxel_data, observer_location, direction, green_hit_values, inclusion_mode=True):
            green_count += 1

    return green_count / total_rays

@njit(parallel=True)
def compute_gvi_map(voxel_data, ray_directions, view_height_voxel=0):
    """Compute GVI map for entire voxel grid."""
    nx, ny, nz = voxel_data.shape
    gvi_map = np.full((nx, ny), np.nan)

    for x in prange(nx):
        for y in range(ny):
            found_observer = False
            for z in range(1, nz):
                # Check if current position is walkable (0 or -2) and below is solid
                if voxel_data[x, y, z] in (0, -2) and voxel_data[x, y, z - 1] not in (0, -2):
                    # If below is building or special element
                    if voxel_data[x, y, z - 1] in (-3, 7, 8, 9):
                        gvi_map[x, y] = np.nan
                        found_observer = True
                        break
                    else:
                        observer_location = np.array([x, y, z+view_height_voxel], dtype=np.float64)
                        gvi_value = compute_gvi(observer_location, voxel_data, ray_directions)
                        gvi_map[x, y] = gvi_value
                        found_observer = True
                        break
            if not found_observer:
                gvi_map[x, y] = np.nan

    return np.flipud(gvi_map)

def get_green_view_index(voxel_data, meshsize, **kwargs):
    """Calculate and visualize Green View Index for a voxel city model."""
    view_point_height = kwargs.get("view_point_height", 1.5)
    view_height_voxel = int(view_point_height / meshsize)
    colormap = kwargs.get("colormap", 'viridis')
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", 1.0)

    # Define ray sampling parameters for GVI
    N_azimuth = 60
    N_elevation = 10
    elevation_min_degrees = -30
    elevation_max_degrees = 30

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

    gvi_map = compute_gvi_map(voxel_data, ray_directions, view_height_voxel=view_height_voxel)

    cmap = plt.cm.get_cmap(colormap).copy()
    cmap.set_bad(color='lightgray')

    plt.figure(figsize=(10, 8))
    plt.imshow(gvi_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Green View Index')
    plt.show()

    obj_export = kwargs.get("obj_export")
    if obj_export == True:
        dem_grid = kwargs.get("dem_grid", np.zeros_like(gvi_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "view_index")        
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        grid_to_obj(
            gvi_map,
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

    return gvi_map

@njit
def compute_svi(observer_location, voxel_data, ray_directions):
    """Compute Sky View Index for a single observer location."""
    # For SVI, obstacles are any voxel != 0
    # We want to return True if we exit without hitting an obstacle (non-0).
    # Thus we set hit_values=(0,) and inclusion_mode=False
    # This means we only 'ignore' voxel_value 0. If we find any voxel !=0, we hit and return True,
    # but since inclusion_mode=False, a non-0 voxel triggers a hit and we return True immediately.
    # Actually, for SVI we want the opposite: if we hit a non-zero voxel, return False immediately.
    # Wait, let's clarify:
    # `trace_ray_generic` returns True if hit condition is met. We want:
    # - If we find a non-zero voxel, that's a "hit" under exclusion mode (since voxel_value not in [0]),
    #   return True from trace_ray_generic, meaning we found an obstacle.
    # Then `compute_svi` should interpret that as no sky.
    # If we exit the grid, trace_ray_generic returns True if exclusion_mode=False? Actually it returns True at the end, but we rely on logic:
    # - If no hit found in exclusion mode, it returns True, meaning no obstacle found.
    # Actually, that's correct. If we never hit a non-zero voxel, at the end we return not inclusion_mode => not False => True.
    # So trace_ray_generic will return True if no obstacle found and exit.
    # If obstacle found, it returns True early. 
    # To differentiate: We need to distinguish sky/no sky:
    # In SVI:
    # - If an obstacle is found: `trace_ray_generic` returns True (hit condition).
    #   That's actually the opposite of what we want. We want to return False for SVI if obstacle is found.
    #
    # Let's invert our logic:
    # For SVI:
    # - We can run `trace_ray_generic` with inclusion_mode=True and hit_values=[0],
    #   meaning we want to stop if we see a voxel in hit_values. But that's incorrect because we want to stop on non-zero voxel.
    #
    # Another approach:
    # Let's define a special mode for sky:
    # If we want no obstacles: If we find a non-0 voxel, that's a 'bad' hit (no sky).
    # If we never find a non-0 voxel and exit:
    # trace_ray_generic(inclusion_mode=False, hit_values=[0]) returns True at exit with no hits (no problem).
    # If we see a non-0 voxel: It's a hit (since not in [0]), returns True immediately.
    #
    # After calling trace_ray_generic:
    # - If returned True means we found a hit condition: that means an obstacle was found. For sky that means no sky.
    # - If returned False means no hit was found. For exclusion_mode=False we said `return not inclusion_mode` at the end,
    #   which is True for end. We must fix the logic in `trace_ray_generic`.
    #
    # Let's fix trace_ray_generic logic at the end:
    # Actually, let's define that `trace_ray_generic` only returns True if a voxel meets the hit condition inside the loop.
    # If it exits the loop, return False, meaning no hit found. 
    # This makes more sense: a "hit" should only occur if we find a condition inside the grid.
    # Exiting the grid means no hit.
    #
    # Revised logic in trace_ray_generic:
    # - If found condition: return True immediately.
    # - If exit: return False.
    #
    # For SVI:
    # - We want to return True if sky is visible (no obstacle)
    # - If an obstacle is found: trace_ray_generic returns True immediately = found hit (obstacle), no sky.
    #   We want to interpret True as obstacle found, so no sky = False.
    # - If exit without obstacle: trace_ray_generic returns False = no hit.
    #   No hit means no obstacle found, so sky visible = True.
    #
    # Conclusion:
    # After calling trace_ray_generic for SVI:
    # - if result == True: obstacle found => sky not visible => return False
    # - if result == False: no obstacle found => sky visible => return True

    sky_count = 0
    total_rays = ray_directions.shape[0]
    # We consider obstacles as anything not equal to 0
    # hit_values=(0,), inclusion_mode=False means:
    #   Hit if voxel_value not in (0,) => that means any non-zero voxel triggers a hit.
    # Exiting means no hit.
    hit_values = (0,)

    for idx in range(total_rays):
        direction = ray_directions[idx]
        hit_result = trace_ray_generic(voxel_data, observer_location, direction, hit_values, inclusion_mode=False)
        # hit_result == True: found an obstacle
        # hit_result == False: no obstacle found (exited grid)
        if hit_result == False:
            # No hit => sky visible along this ray
            sky_count += 1

    return sky_count / total_rays

@njit(parallel=True)
def compute_svi_map(voxel_data, ray_directions, view_height_voxel=0):
    """Compute SVI map for entire voxel grid."""
    nx, ny, nz = voxel_data.shape
    svi_map = np.full((nx, ny), np.nan)

    for x in prange(nx):
        for y in range(ny):
            found_observer = False
            for z in range(1, nz):
                if voxel_data[x, y, z] in (0, -2) and voxel_data[x, y, z - 1] not in (0, -2):
                    if voxel_data[x, y, z - 1] in (-3, 7, 8, 9):
                        svi_map[x, y] = np.nan
                        found_observer = True
                        break
                    else:
                        observer_location = np.array([x, y, z+view_height_voxel], dtype=np.float64)
                        svi_value = compute_svi(observer_location, voxel_data, ray_directions)
                        svi_map[x, y] = svi_value
                        found_observer = True
                        break
            if not found_observer:
                svi_map[x, y] = np.nan

    return np.flipud(svi_map)

def get_sky_view_index(voxel_data, meshsize, **kwargs):
    """Calculate and visualize Sky View Index for a voxel city model."""
    view_point_height = kwargs.get("view_point_height", 1.5)
    view_height_voxel = int(view_point_height / meshsize)
    colormap = kwargs.get("colormap", 'viridis')
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", 1.0)

    # Parameters for SVI rays (focused upward directions)
    N_azimuth_svi = 60
    N_elevation_svi = 5
    elevation_min_degrees_svi = 0
    elevation_max_degrees_svi = 30

    azimuth_angles_svi = np.linspace(0, 2 * np.pi, N_azimuth_svi, endpoint=False)
    elevation_angles_svi = np.deg2rad(np.linspace(elevation_min_degrees_svi, elevation_max_degrees_svi, N_elevation_svi))

    ray_directions_svi = []
    for elevation in elevation_angles_svi:
        cos_elev = np.cos(elevation)
        sin_elev = np.sin(elevation)
        for azimuth in azimuth_angles_svi:
            dx = cos_elev * np.cos(azimuth)
            dy = cos_elev * np.sin(azimuth)
            dz = sin_elev
            ray_directions_svi.append([dx, dy, dz])

    ray_directions_svi = np.array(ray_directions_svi, dtype=np.float64)
    svi_map = compute_svi_map(voxel_data, ray_directions_svi, view_height_voxel=view_height_voxel)

    cmap = plt.cm.get_cmap(colormap).copy()
    cmap.set_bad(color='lightgray')

    plt.figure(figsize=(10, 8))
    plt.imshow(svi_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Sky View Index')
    plt.show()

    obj_export = kwargs.get("obj_export")
    if obj_export == True:
        dem_grid = kwargs.get("dem_grid", np.zeros_like(svi_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "view_index")        
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        grid_to_obj(
            svi_map,
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

    return svi_map

def mark_building_by_id(voxelcity_grid, building_id_grid_ori, ids, mark):
    """Mark specific buildings in the voxel grid with a given value.

    Args:
        voxelcity_grid (ndarray): 3D array of voxel values
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
        z_mask = voxelcity_grid[x, y, :] == -3
        voxelcity_grid[x, y, z_mask] = mark

@njit
def trace_ray_to_target(voxel_data, origin, target, opaque_values):
    """Trace a ray from origin to target through voxel data.

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

        # Move to next voxel boundary using DDA algorithm
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

def get_landmark_visibility_map(voxelcity_grid, building_id_grid, building_geojson, meshsize, **kwargs):
    """Generate a visibility map for landmark buildings in a voxel city.

    Args:
        voxelcity_grid (ndarray): 3D array representing the voxel city
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

    # Mark landmark buildings in voxel grid
    target_value = -30
    mark_building_by_id(voxelcity_grid, building_id_grid, landmark_ids, target_value)
    
    # Compute visibility map
    landmark_vis_map = compute_landmark_visibility(voxelcity_grid, target_value=target_value, view_height_voxel=view_height_voxel, colormap=colormap)

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
        export_obj(voxelcity_grid, output_dir, output_file_name_vox, meshsize)

    return landmark_vis_map
