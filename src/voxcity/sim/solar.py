import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

from .view import trace_ray_generic, compute_vi_map_generic, get_sky_view_factor_map

@njit(parallel=True)
def compute_direct_solar_irradiance_map_binary(voxel_data, sun_direction, view_height_voxel, hit_values, inclusion_mode):
    """
    Compute a binary map of direct solar irradiation: 1.0 if cell is sunlit, 0.0 if shaded.

    Args:
        voxel_data (ndarray): 3D array of voxel values.
        sun_direction (tuple): Direction vector of the sun.
        view_height_voxel (int): Observer height in voxel units.
        hit_values (tuple): Values considered non-obstacles if inclusion_mode=False (here we only use (0,)).
        inclusion_mode (bool): False here, meaning any voxel not in hit_values is an obstacle.

    Returns:
        ndarray: 2D array where 1.0 = sunlit, 0.0 = shaded, NaN = invalid observer.
    """
    nx, ny, nz = voxel_data.shape
    irradiance_map = np.full((nx, ny), np.nan, dtype=np.float64)

    # Normalize sun direction
    sd = np.array(sun_direction, dtype=np.float64)
    sd_len = np.sqrt(sd[0]**2 + sd[1]**2 + sd[2]**2)
    if sd_len == 0.0:
        return np.flipud(irradiance_map)
    sd /= sd_len

    for x in prange(nx):
        for y in range(ny):
            found_observer = False
            # Find lowest empty voxel above ground
            for z in range(1, nz):
                # Check if this position is a valid observer location:
                # voxel_data[x, y, z] in (0, -2) means it's air or ground-air interface (open)
                # voxel_data[x, y, z-1] not in (0, -2) means below it is some ground or structure
                if voxel_data[x, y, z] in (0, -2) and voxel_data[x, y, z - 1] not in (0, -2):
                    # Check if standing on building or vegetation
                    if voxel_data[x, y, z - 1] in (-3, 7, 8, 9):
                        # Invalid observer location
                        irradiance_map[x, y] = np.nan
                        found_observer = True
                        break
                    else:
                        # Place observer and cast a ray in sun direction
                        observer_location = np.array([x, y, z + view_height_voxel], dtype=np.float64)
                        hit = trace_ray_generic(voxel_data, observer_location, sd, hit_values, inclusion_mode)
                        irradiance_map[x, y] = 0.0 if hit else 1.0
                        found_observer = True
                        break
            if not found_observer:
                irradiance_map[x, y] = np.nan

    return np.flipud(irradiance_map)


def get_direct_solar_irradiance_map(voxel_data, meshsize, azimuth_degrees_ori, elevation_degrees, direct_normal_irradiance, **kwargs):
    view_point_height = kwargs.get("view_point_height", 1.5)
    view_height_voxel = int(view_point_height / meshsize)
    colormap = kwargs.get("colormap", 'viridis')
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", direct_normal_irradiance)

    # Convert angles to direction with the adjusted formula
    azimuth_degrees = 180 - azimuth_degrees_ori
    azimuth_radians = np.deg2rad(azimuth_degrees)
    elevation_radians = np.deg2rad(elevation_degrees)
    dx = np.cos(elevation_radians) * np.cos(azimuth_radians)
    dy = np.cos(elevation_radians) * np.sin(azimuth_radians)
    dz = np.sin(elevation_radians)
    sun_direction = (dx, dy, dz)

    # All non-zero voxels are obstacles
    hit_values = (0,)
    inclusion_mode = False

    binary_map = compute_direct_solar_irradiance_map_binary(
        voxel_data, sun_direction, view_height_voxel, hit_values, inclusion_mode
    )

    sin_elev = dz
    direct_map = binary_map * direct_normal_irradiance * sin_elev

    # Visualization
    cmap = plt.cm.get_cmap(colormap).copy()
    cmap.set_bad(color='lightgray')
    plt.figure(figsize=(10, 8))
    plt.title("Horizontal Direct Solar Irradiance Map (0° = North)")
    plt.imshow(direct_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Direct Solar Irradiance (W/m²)')
    plt.show()

    # Optional OBJ export
    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        from ..file.obj import grid_to_obj
        dem_grid = kwargs.get("dem_grid", np.zeros_like(direct_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "direct_solar_irradiance")
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        grid_to_obj(
            direct_map,
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

    return direct_map


def get_diffuse_solar_irradiance_map(voxel_data, meshsize, diffuse_irradiance=1.0, **kwargs):
    """
    Compute diffuse solar irradiance map using the Sky View Factor (SVF).
    Diffuse = SVF * diffuse_irradiance.

    No mode or hit_values needed since this calculation relies on the SVF which is internally computed.

    Args:
        voxel_data (ndarray): 3D voxel array.
        meshsize (float): Voxel size in meters.
        diffuse_irradiance (float): Diffuse irradiance in W/m².

    Returns:
        ndarray: 2D array of diffuse solar irradiance (W/m²).
    """
    # SVF computation does not require mode/hit_values/inclusion_mode, 
    # it's already defined to consider all non-empty voxels as obstacles internally.
    svf_kwargs = kwargs.copy()
    svf_kwargs["colormap"] = "BuPu_r"
    svf_kwargs["vmin"] = 0
    svf_kwargs["vmax"] = 1
    SVF_map = get_sky_view_factor_map(voxel_data, meshsize, **svf_kwargs)
    diffuse_map = SVF_map * diffuse_irradiance

    colormap = kwargs.get("colormap", 'viridis')
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", diffuse_irradiance)
    cmap = plt.cm.get_cmap(colormap).copy()
    cmap.set_bad(color='lightgray')
    plt.figure(figsize=(10, 8))
    plt.title("Diffuse Solar Irradiance Map")
    plt.imshow(diffuse_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Diffuse Solar Irradiance (W/m²)')
    plt.show()

    # Optional OBJ export
    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        from ..file.obj import grid_to_obj
        dem_grid = kwargs.get("dem_grid", np.zeros_like(diffuse_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "diffuse_solar_irradiance")
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        view_point_height = kwargs.get("view_point_height", 1.5)
        grid_to_obj(
            diffuse_map,
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

    return diffuse_map


def get_global_solar_irradiance_map(
    voxel_data,
    meshsize,
    azimuth_degrees,
    elevation_degrees,
    direct_normal_irradiance,
    diffuse_irradiance,
    **kwargs
):
    """
    Compute global solar irradiance (direct + diffuse) on a horizontal plane at each valid observer location.

    No mode/hit_values/inclusion_mode needed. Uses the updated direct and diffuse functions.

    Args:
        voxel_data (ndarray): 3D voxel array.
        meshsize (float): Voxel size in meters.
        azimuth_degrees (float): Sun azimuth angle in degrees.
        elevation_degrees (float): Sun elevation angle in degrees.
        direct_normal_irradiance (float): DNI in W/m².
        diffuse_irradiance (float): Diffuse irradiance in W/m².

    Returns:
        ndarray: 2D array of global solar irradiance (W/m²).
    """
    # Compute direct irradiance map (no mode/hit_values/inclusion_mode needed)
    direct_map = get_direct_solar_irradiance_map(
        voxel_data,
        meshsize,
        azimuth_degrees,
        elevation_degrees,
        direct_normal_irradiance,
        **kwargs
    )

    # Compute diffuse irradiance map
    diffuse_map = get_diffuse_solar_irradiance_map(
        voxel_data,
        meshsize,
        diffuse_irradiance=diffuse_irradiance,
        **kwargs
    )

    # Sum the two
    global_map = direct_map + diffuse_map

    colormap = kwargs.get("colormap", 'viridis')
    vmin = kwargs.get("vmin", np.nanmin(global_map))
    vmax = kwargs.get("vmax", np.nanmax(global_map))
    cmap = plt.cm.get_cmap(colormap).copy()
    cmap.set_bad(color='lightgray')
    plt.figure(figsize=(10, 8))
    plt.title("Global Solar Irradiance Map")
    plt.imshow(global_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Global Solar Irradiance (W/m²)')
    plt.show()

    # Optional OBJ export
    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        from ..file.obj import grid_to_obj
        dem_grid = kwargs.get("dem_grid", np.zeros_like(global_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "global_solar_irradiance")
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        meshsize_param = kwargs.get("meshsize", meshsize)
        view_point_height = kwargs.get("view_point_height", 1.5)
        grid_to_obj(
            global_map,
            dem_grid,
            output_dir,
            output_file_name,
            meshsize_param,
            view_point_height,
            colormap_name=colormap,
            num_colors=num_colors,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax
        )

    return global_map