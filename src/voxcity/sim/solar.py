import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange
from datetime import datetime, timezone
import pytz
from astral import Observer
from astral.sun import elevation, azimuth

from .view import trace_ray_generic, compute_vi_map_generic, get_sky_view_factor_map
from ..utils.weather import get_nearest_epw_from_climate_onebuilding, read_epw_for_solar_simulation
from ..file.obj import grid_to_obj, export_obj

@njit(parallel=True)
def compute_direct_solar_irradiance_map_binary(voxel_data, sun_direction, view_point_height, hit_values, meshsize, tree_k, tree_lad, inclusion_mode):
    """
    Compute a map of direct solar irradiation accounting for tree transmittance.

    Args:
        voxel_data (ndarray): 3D array of voxel values.
        sun_direction (tuple): Direction vector of the sun.
        view_height_voxel (int): Observer height in voxel units.
        hit_values (tuple): Values considered non-obstacles if inclusion_mode=False.
        meshsize (float): Size of each voxel in meters.
        tree_k (float): Tree extinction coefficient.
        tree_lad (float): Leaf area density in m^-1.
        inclusion_mode (bool): False here, meaning any voxel not in hit_values is an obstacle.

    Returns:
        ndarray: 2D array of transmittance values (0.0-1.0), NaN = invalid observer.
    """
    
    view_height_voxel = int(view_point_height / meshsize)
    
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
            for z in range(1, nz):
                if voxel_data[x, y, z] in (0, -2) and voxel_data[x, y, z - 1] not in (0, -2):
                    if voxel_data[x, y, z - 1] in (-30, -3, -2):
                        irradiance_map[x, y] = np.nan
                        found_observer = True
                        break
                    else:
                        # Place observer and cast a ray in sun direction
                        observer_location = np.array([x, y, z + view_height_voxel], dtype=np.float64)
                        hit, transmittance = trace_ray_generic(voxel_data, observer_location, sd, 
                                                             hit_values, meshsize, tree_k, tree_lad, inclusion_mode)
                        irradiance_map[x, y] = transmittance if not hit else 0.0
                        found_observer = True
                        break
            if not found_observer:
                irradiance_map[x, y] = np.nan

    return np.flipud(irradiance_map)

def get_direct_solar_irradiance_map(voxel_data, meshsize, azimuth_degrees_ori, elevation_degrees, 
                                  direct_normal_irradiance, show_plot=False, **kwargs):
    """
    Compute direct solar irradiance map with tree transmittance.
    """
    view_point_height = kwargs.get("view_point_height", 1.5)
    colormap = kwargs.get("colormap", 'magma')
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", direct_normal_irradiance)
    
    # Get tree transmittance parameters
    tree_k = kwargs.get("tree_k", 0.6)
    tree_lad = kwargs.get("tree_lad", 1.0)

    # Convert angles to direction
    azimuth_degrees = 180 - azimuth_degrees_ori
    azimuth_radians = np.deg2rad(azimuth_degrees)
    elevation_radians = np.deg2rad(elevation_degrees)
    dx = np.cos(elevation_radians) * np.cos(azimuth_radians)
    dy = np.cos(elevation_radians) * np.sin(azimuth_radians)
    dz = np.sin(elevation_radians)
    sun_direction = (dx, dy, dz)

    # All non-zero voxels are obstacles except for trees which have transmittance
    hit_values = (0,)
    inclusion_mode = False

    transmittance_map = compute_direct_solar_irradiance_map_binary(
        voxel_data, sun_direction, view_point_height, hit_values, 
        meshsize, tree_k, tree_lad, inclusion_mode
    )

    sin_elev = dz
    direct_map = transmittance_map * direct_normal_irradiance * sin_elev

    if show_plot:
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

def get_diffuse_solar_irradiance_map(voxel_data, meshsize, diffuse_irradiance=1.0, show_plot=False, **kwargs):
    """
    Compute diffuse solar irradiance map using the Sky View Factor (SVF) with tree transmittance.
    """

    view_point_height = kwargs.get("view_point_height", 1.5)
    colormap = kwargs.get("colormap", 'magma')
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", diffuse_irradiance)
    
    # Pass tree transmittance parameters to SVF calculation
    svf_kwargs = kwargs.copy()
    svf_kwargs["colormap"] = "BuPu_r"
    svf_kwargs["vmin"] = 0
    svf_kwargs["vmax"] = 1

    # SVF calculation now handles tree transmittance internally
    SVF_map = get_sky_view_factor_map(voxel_data, meshsize, **svf_kwargs)
    diffuse_map = SVF_map * diffuse_irradiance

    if show_plot:
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
        dem_grid = kwargs.get("dem_grid", np.zeros_like(diffuse_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "diffuse_solar_irradiance")
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
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
    show_plot=False,
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
    
    colormap = kwargs.get("colormap", 'magma')

    # Create kwargs for diffuse calculation
    direct_diffuse_kwargs = kwargs.copy()
    direct_diffuse_kwargs.update({
        'show_plot': False,
        'obj_export': False
    })

    # Compute direct irradiance map (no mode/hit_values/inclusion_mode needed)
    direct_map = get_direct_solar_irradiance_map(
        voxel_data,
        meshsize,
        azimuth_degrees,
        elevation_degrees,
        direct_normal_irradiance,
        **direct_diffuse_kwargs
    )

    # Compute diffuse irradiance map
    diffuse_map = get_diffuse_solar_irradiance_map(
        voxel_data,
        meshsize,
        diffuse_irradiance=diffuse_irradiance,
        **direct_diffuse_kwargs
    )

    # Sum the two
    global_map = direct_map + diffuse_map

    vmin = kwargs.get("vmin", np.nanmin(global_map))
    vmax = kwargs.get("vmax", np.nanmax(global_map))

    if show_plot:
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

def get_solar_positions_astral(times, lat, lon):
    """
    Compute solar azimuth and elevation using Astral for given times and location.
    Times must be timezone-aware.
    """
    observer = Observer(latitude=lat, longitude=lon)
    df_pos = pd.DataFrame(index=times, columns=['azimuth', 'elevation'], dtype=float)

    for t in times:
        # t is already timezone-aware; no need to replace tzinfo
        el = elevation(observer=observer, dateandtime=t)
        az = azimuth(observer=observer, dateandtime=t)
        df_pos.at[t, 'elevation'] = el
        df_pos.at[t, 'azimuth'] = az

    return df_pos

def get_cumulative_global_solar_irradiance(
    voxel_data,
    meshsize,
    start_time,
    end_time,
    direct_normal_irradiance_scaling=1.0,
    diffuse_irradiance_scaling=1.0,
    **kwargs
):
    """
    Compute cumulative global solar irradiance over a specified period using data from an EPW file,
    accounting for tree transmittance.

    Args:
        voxel_data (ndarray): 3D array of voxel values.
        meshsize (float): Size of each voxel in meters.
        start_time (str): Start time in format 'MM-DD HH:MM:SS' (no year).
        end_time (str): End time in format 'MM-DD HH:MM:SS' (no year).
        direct_normal_irradiance_scaling (float): Scaling factor for DNI.
        diffuse_irradiance_scaling (float): Scaling factor for DHI.
        **kwargs: Additional arguments including:
            - view_point_height (float): Observer height in meters
            - tree_k (float): Tree extinction coefficient (default: 0.5)
            - tree_lad (float): Leaf area density in m^-1 (default: 1.0)
            - download_nearest_epw (bool): Whether to download nearest EPW file
            - epw_file_path (str): Path to EPW file
            - show_plot (bool): Whether to show final plot
            - show_each_timestep (bool): Whether to show plots for each timestep

    Returns:
        ndarray: 2D array of cumulative global solar irradiance (W/m²·hour).
    """
    view_point_height = kwargs.get("view_point_height", 1.5)
    colormap = kwargs.get("colormap", 'magma')

    # Get EPW file
    download_nearest_epw = kwargs.get("download_nearest_epw", False)
    rectangle_vertices = kwargs.get("rectangle_vertices", None)
    epw_file_path = kwargs.get("epw_file_path", None)
    if download_nearest_epw:
        if rectangle_vertices is None:
            print("rectangle_vertices is required to download nearest EPW file")
            return None
        else:
            # Calculate center point of rectangle
            lats = [coord[0] for coord in rectangle_vertices]
            lons = [coord[1] for coord in rectangle_vertices]
            center_lat = (min(lats) + max(lats)) / 2
            center_lon = (min(lons) + max(lons)) / 2
            target_point = (center_lat, center_lon)

            # Optional: specify maximum distance in kilometers
            max_distance = 100  # None for no limit

            output_dir = kwargs.get("output_dir", "output")

            epw_file_path, weather_data, metadata = get_nearest_epw_from_climate_onebuilding(
                latitude=center_lat,
                longitude=center_lon,
                output_dir=output_dir,
                max_distance=max_distance,
                extract_zip=True,
                load_data=True
            )

    # Read EPW data
    df, lat, lon, tz, elevation_m = read_epw_for_solar_simulation(epw_file_path)
    if df.empty:
        raise ValueError("No data in EPW file.")

    # Parse start and end times without year
    try:
        start_dt = datetime.strptime(start_time, "%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end_time, "%m-%d %H:%M:%S")
    except ValueError as ve:
        raise ValueError("start_time and end_time must be in format 'MM-DD HH:MM:SS'") from ve

    # Add hour of year column and filter data as before...
    df['hour_of_year'] = (df.index.dayofyear - 1) * 24 + df.index.hour + 1
    
    start_doy = datetime(2000, start_dt.month, start_dt.day).timetuple().tm_yday
    end_doy = datetime(2000, end_dt.month, end_dt.day).timetuple().tm_yday
    
    start_hour = (start_doy - 1) * 24 + start_dt.hour + 1
    end_hour = (end_doy - 1) * 24 + end_dt.hour + 1

    if start_hour <= end_hour:
        df_period = df[(df['hour_of_year'] >= start_hour) & (df['hour_of_year'] <= end_hour)]
    else:
        df_period = df[(df['hour_of_year'] >= start_hour) | (df['hour_of_year'] <= end_hour)]

    df_period = df_period[
        ((df_period.index.hour != start_dt.hour) | (df_period.index.minute >= start_dt.minute)) &
        ((df_period.index.hour != end_dt.hour) | (df_period.index.minute <= end_dt.minute))
    ]

    if df_period.empty:
        raise ValueError("No EPW data in the specified period.")

    # Prepare timezone conversion
    offset_minutes = int(tz * 60)
    local_tz = pytz.FixedOffset(offset_minutes)
    df_period_local = df_period.copy()
    df_period_local.index = df_period_local.index.tz_localize(local_tz)
    df_period_utc = df_period_local.tz_convert(pytz.UTC)

    # Compute solar positions
    solar_positions = get_solar_positions_astral(df_period_utc.index, lat, lon)

    # Create kwargs for diffuse calculation
    diffuse_kwargs = kwargs.copy()
    diffuse_kwargs.update({
        'show_plot': False,
        'obj_export': False
    })

    # Compute base diffuse map once with diffuse_irradiance=1.0
    base_diffuse_map = get_diffuse_solar_irradiance_map(
        voxel_data,
        meshsize,
        diffuse_irradiance=1.0,
        **diffuse_kwargs
    )

    # Initialize maps
    cumulative_map = np.zeros((voxel_data.shape[0], voxel_data.shape[1]))
    mask_map = np.ones((voxel_data.shape[0], voxel_data.shape[1]), dtype=bool)

    # Create kwargs for direct calculation
    direct_kwargs = kwargs.copy()
    direct_kwargs.update({
        'show_plot': False,
        'view_point_height': view_point_height,
        'obj_export': False
    })

    # Iterate through each time step
    for idx, (time_utc, row) in enumerate(df_period_utc.iterrows()):
        DNI = row['DNI'] * direct_normal_irradiance_scaling
        DHI = row['DHI'] * diffuse_irradiance_scaling
        time_local = df_period_local.index[idx]

        # Get solar position
        solpos = solar_positions.loc[time_utc]
        azimuth_degrees = solpos['azimuth']
        elevation_degrees = solpos['elevation']        

        # Compute direct irradiance map with transmittance
        direct_map = get_direct_solar_irradiance_map(
            voxel_data,
            meshsize,
            azimuth_degrees,
            elevation_degrees,
            direct_normal_irradiance=DNI,
            **direct_kwargs
        )

        # Scale base_diffuse_map by actual DHI
        diffuse_map = base_diffuse_map * DHI

        # Combine direct and diffuse
        global_map = direct_map + diffuse_map

        # Update mask_map
        mask_map &= ~np.isnan(global_map)

        # Replace NaN with 0 for accumulation
        global_map_filled = np.nan_to_num(global_map, nan=0.0)
        cumulative_map += global_map_filled

        # Optional timestep visualization
        show_each_timestep = kwargs.get("show_each_timestep", False)
        if show_each_timestep:
            colormap = kwargs.get("colormap", 'viridis')
            vmin = kwargs.get("vmin", 0.0)
            vmax = kwargs.get("vmax", max(direct_normal_irradiance_scaling, diffuse_irradiance_scaling) * 1000)
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_bad(color='lightgray')
            plt.figure(figsize=(8, 6))
            plt.title(f"Global Solar Irradiance at {time_local.strftime('%Y-%m-%d %H:%M:%S')}")
            plt.imshow(global_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label='Global Solar Irradiance (W/m²)')
            plt.show()

    # Apply mask
    cumulative_map[~mask_map] = np.nan

    # Final visualization
    show_plot = kwargs.get("show_plot", True)
    if show_plot:
        colormap = kwargs.get("colormap", 'magma')
        vmin = kwargs.get("vmin", np.nanmin(cumulative_map))
        vmax = kwargs.get("vmax", np.nanmax(cumulative_map))
        cmap = plt.cm.get_cmap(colormap).copy()
        cmap.set_bad(color='lightgray')
        plt.figure(figsize=(8, 6))
        plt.title("Cumulative Global Solar Irradiance Map")
        plt.imshow(cumulative_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Cumulative Global Solar Irradiance (W/m²·hour)')
        plt.show()

    # Optional OBJ export
    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        colormap = kwargs.get("colormap", "magma")
        vmin = kwargs.get("vmin", np.nanmin(cumulative_map))
        vmax = kwargs.get("vmax", np.nanmax(cumulative_map))
        dem_grid = kwargs.get("dem_grid", np.zeros_like(cumulative_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "cummurative_global_solar_irradiance")
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        grid_to_obj(
            cumulative_map,
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

    return cumulative_map