import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# JIT-compiled trace_ray function
@njit
def trace_ray(voxel_data, origin, direction):
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

    # Initialize variables
    x, y, z = x0 + 0.5, y0 + 0.5, z0 + 0.5  # Start at center of voxel
    i, j, k = int(x0), int(y0), int(z0)

    # Determine the step direction
    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1
    step_z = 1 if dz >= 0 else -1

    # Compute tMax and tDelta
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
        # Check voxel value
        voxel_value = voxel_data[i, j, k]
        if voxel_value in (-2, 2, 5, 7):  # Green voxel types
            return True

        # Move to next voxel boundary
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

    return False  # Did not hit a green voxel

# JIT-compiled compute_gvi function
@njit
def compute_gvi(observer_location, voxel_data, ray_directions):
    green_count = 0
    total_rays = ray_directions.shape[0]

    for idx in range(total_rays):
        direction = ray_directions[idx]
        if trace_ray(voxel_data, observer_location, direction):
            green_count += 1

    green_view_index = green_count / total_rays
    return green_view_index

# JIT-compiled function to compute GVI map
@njit(parallel=True)
def compute_gvi_map(voxel_data, ray_directions, view_height_voxel=0):
    nx, ny, nz = voxel_data.shape
    gvi_map = np.full((nx, ny), np.nan)

    for x in prange(nx):
        for y in range(ny):
            found_observer = False
            for z in range(1, nz):
                if voxel_data[x, y, z] in (0, -2) and voxel_data[x, y, z - 1] not in (0, -2):
                    if voxel_data[x, y, z - 1] in (-3, 6):
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

    return gvi_map

# Main script
# Load or define your voxel data (3D numpy array)
# For demonstration, let's create a small voxel_data array
# In practice, you would load your actual data
# voxel_data = np.random.randint(-3, 8, size=(100, 100, 50))
# Replace the above line with your actual voxel data
# voxel_data = voxelcity_grid  # Ensure voxelcity_grid is defined in your environment

def get_green_view_index(voxel_data, view_height_voxel=0):
    # Define parameters for ray emission
    N_azimuth = 60  # Number of horizontal angles
    N_elevation = 10  # Number of vertical angles within the specified range
    elevation_min_degrees = -30    # Minimum elevation angle (in degrees)
    elevation_max_degrees = 30   # Maximum elevation angle (in degrees)

    # Generate ray directions
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

    # Compute the GVI map using the optimized function
    gvi_map = compute_gvi_map(voxel_data, ray_directions, view_height_voxel=view_height_voxel)

    # Create a copy of the inverted 'BuPu' colormap
    cmap = plt.cm.get_cmap('viridis').copy()

    # Set the 'bad' color (for np.nan values) to gray
    # cmap.set_bad(color='#202020')
    cmap.set_bad(color='lightgray')

    # Visualization of the SVI map in 2D with inverted 'BuPu' colormap and gray for np.nan
    plt.figure(figsize=(10, 8))
    plt.imshow(gvi_map.T, origin='lower', cmap=cmap)
    plt.colorbar(label='Green View Index')
    plt.title('Green View Index Map with Inverted BuPu Colormap (NaN as Gray)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

    # Visualization of the SVI map in 2D with inverted 'BuPu' colormap and gray for np.nan
    plt.figure(figsize=(10, 8))
    plt.imshow(np.flipud(gvi_map), origin='lower', cmap=cmap)
    plt.axis('off')  # Remove axes, ticks, and tick numbers
    # plt.colorbar(label='Sky View Index')
    # plt.title('Sky View Index Map with Inverted BuPu Colormap (NaN as Gray)')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    plt.show()

    return np.flipud(gvi_map)

# JIT-compiled trace_ray_sky function
@njit
def trace_ray_sky(voxel_data, origin, direction):
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

    # Initialize variables
    x, y, z = x0 + 0.5, y0 + 0.5, z0 + 0.5  # Start at center of voxel
    i, j, k = int(x0), int(y0), int(z0)

    # Determine the step direction
    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1
    step_z = 1 if dz >= 0 else -1

    # Compute tMax and tDelta
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
        # Check voxel value
        voxel_value = voxel_data[i, j, k]
        if voxel_value != 0:  # Non-void voxel types (obstacles)
            return False  # Ray is blocked by an obstacle

        # Move to next voxel boundary
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

    # Ray has reached outside the voxel grid without hitting any obstacles
    return True

# JIT-compiled compute_svi function
@njit
def compute_svi(observer_location, voxel_data, ray_directions):
    sky_count = 0
    total_rays = ray_directions.shape[0]

    for idx in range(total_rays):
        direction = ray_directions[idx]
        if trace_ray_sky(voxel_data, observer_location, direction):
            sky_count += 1

    sky_view_index = sky_count / total_rays
    return sky_view_index

# JIT-compiled function to compute SVI map
@njit(parallel=True)
def compute_svi_map(voxel_data, ray_directions, view_height_voxel=0):
    nx, ny, nz = voxel_data.shape
    svi_map = np.full((nx, ny), np.nan)

    for x in prange(nx):
        for y in range(ny):
            found_observer = False
            for z in range(1, nz):
                if voxel_data[x, y, z] in (0, -2) and voxel_data[x, y, z - 1] not in (0, -2):
                    if voxel_data[x, y, z - 1] in (-3, 6):
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

    return svi_map

# Main script modifications
# Load or define your voxel data (3D numpy array)
# voxel_data = voxelcity_grid  # Ensure voxelcity_grid is defined in your environment

def get_sky_view_index(voxel_data, view_height_voxel=0):
    # Define parameters for ray emission for SVI
    # For SVI, we focus on upward directions
    N_azimuth_svi = 60  # Number of horizontal angles
    N_elevation_svi = 5  # Number of vertical angles within the specified range
    elevation_min_degrees_svi = 0   # Minimum elevation angle (in degrees)
    elevation_max_degrees_svi = 30   # Maximum elevation angle (in degrees)

    # Generate ray directions for SVI
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

    # Compute the SVI map using the optimized function
    svi_map = compute_svi_map(voxel_data, ray_directions_svi, view_height_voxel=view_height_voxel)

    # Create a copy of the inverted 'BuPu' colormap
    cmap = plt.cm.get_cmap('BuPu_r').copy()

    # Set the 'bad' color (for np.nan values) to gray
    # cmap.set_bad(color='#202020')
    cmap.set_bad(color='lightgray')

    # Visualization of the SVI map in 2D with inverted 'BuPu' colormap and gray for np.nan
    plt.figure(figsize=(10, 8))
    plt.imshow(np.flipud(svi_map), origin='lower', cmap=cmap)
    plt.colorbar(label='Sky View Index')
    plt.title('Sky View Index Map with Inverted BuPu Colormap (NaN as Gray)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

    # Visualization of the SVI map in 2D with inverted 'BuPu' colormap and gray for np.nan
    plt.figure(figsize=(10, 8))
    plt.imshow(np.flipud(svi_map), origin='lower', cmap=cmap)
    plt.axis('off')  # Remove axes, ticks, and tick numbers
    # plt.colorbar(label='Sky View Index')
    # plt.title('Sky View Index Map with Inverted BuPu Colormap (NaN as Gray)')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    plt.show()

    return np.flipud(svi_map)