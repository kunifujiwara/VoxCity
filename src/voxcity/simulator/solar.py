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
from ..exporter.obj import grid_to_obj, export_obj

@njit(parallel=True)
def compute_direct_solar_irradiance_map_binary(voxel_data, sun_direction, view_point_height, hit_values, meshsize, tree_k, tree_lad, inclusion_mode):
    """
    Compute a map of direct solar irradiation accounting for tree transmittance.

    The function:
    1. Places observers at valid locations (empty voxels above ground)
    2. Casts rays from each observer in the sun direction
    3. Computes transmittance through trees using Beer-Lambert law
    4. Returns a 2D map of transmittance values

    Args:
        voxel_data (ndarray): 3D array of voxel values.
        sun_direction (tuple): Direction vector of the sun.
        view_point_height (float): Observer height in meters.
        hit_values (tuple): Values considered non-obstacles if inclusion_mode=False.
        meshsize (float): Size of each voxel in meters.
        tree_k (float): Tree extinction coefficient.
        tree_lad (float): Leaf area density in m^-1.
        inclusion_mode (bool): False here, meaning any voxel not in hit_values is an obstacle.

    Returns:
        ndarray: 2D array of transmittance values (0.0-1.0), NaN = invalid observer position.
    """
    
    view_height_voxel = int(view_point_height / meshsize)
    
    nx, ny, nz = voxel_data.shape
    irradiance_map = np.full((nx, ny), np.nan, dtype=np.float64)

    # Normalize sun direction vector for ray tracing
    sd = np.array(sun_direction, dtype=np.float64)
    sd_len = np.sqrt(sd[0]**2 + sd[1]**2 + sd[2]**2)
    if sd_len == 0.0:
        return np.flipud(irradiance_map)
    sd /= sd_len

    # Process each x,y position in parallel
    for x in prange(nx):
        for y in range(ny):
            found_observer = False
            # Search upward for valid observer position
            for z in range(1, nz):
                # Check if current voxel is empty/tree and voxel below is solid
                if voxel_data[x, y, z] in (0, -2) and voxel_data[x, y, z - 1] not in (0, -2):
                    # Skip if standing on building/vegetation/water
                    if (voxel_data[x, y, z - 1] in (7, 8, 9)) or (voxel_data[x, y, z - 1] < 0):
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

    # Flip map vertically to match visualization conventions
    return np.flipud(irradiance_map)

def get_direct_solar_irradiance_map(voxel_data, meshsize, azimuth_degrees_ori, elevation_degrees, 
                                  direct_normal_irradiance, show_plot=False, **kwargs):
    """
    Compute direct solar irradiance map with tree transmittance.
    
    The function:
    1. Converts sun angles to direction vector
    2. Computes binary transmittance map
    3. Scales by direct normal irradiance and sun elevation
    4. Optionally visualizes and exports results
    
    Args:
        voxel_data (ndarray): 3D array of voxel values.
        meshsize (float): Size of each voxel in meters.
        azimuth_degrees_ori (float): Sun azimuth angle in degrees (0° = North, 90° = East).
        elevation_degrees (float): Sun elevation angle in degrees above horizon.
        direct_normal_irradiance (float): Direct normal irradiance in W/m².
        show_plot (bool): Whether to display visualization.
        **kwargs: Additional arguments including:
            - view_point_height (float): Observer height in meters (default: 1.5)
            - colormap (str): Matplotlib colormap name (default: 'magma')
            - vmin (float): Minimum value for colormap
            - vmax (float): Maximum value for colormap
            - tree_k (float): Tree extinction coefficient (default: 0.6)
            - tree_lad (float): Leaf area density in m^-1 (default: 1.0)
            - obj_export (bool): Whether to export as OBJ file
            - output_directory (str): Directory for OBJ export
            - output_file_name (str): Filename for OBJ export
            - dem_grid (ndarray): DEM grid for OBJ export
            - num_colors (int): Number of colors for OBJ export
            - alpha (float): Alpha value for OBJ export

    Returns:
        ndarray: 2D array of direct solar irradiance values (W/m²).
    """
    view_point_height = kwargs.get("view_point_height", 1.5)
    colormap = kwargs.get("colormap", 'magma')
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", direct_normal_irradiance)
    
    # Get tree transmittance parameters
    tree_k = kwargs.get("tree_k", 0.6)
    tree_lad = kwargs.get("tree_lad", 1.0)

    # Convert sun angles to direction vector
    # Note: azimuth is adjusted by 180° to match coordinate system
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

    # Compute transmittance map
    transmittance_map = compute_direct_solar_irradiance_map_binary(
        voxel_data, sun_direction, view_point_height, hit_values, 
        meshsize, tree_k, tree_lad, inclusion_mode
    )

    # Scale by direct normal irradiance and sun elevation
    sin_elev = dz
    direct_map = transmittance_map * direct_normal_irradiance * sin_elev

    # Optional visualization
    if show_plot:
        cmap = plt.cm.get_cmap(colormap).copy()
        cmap.set_bad(color='lightgray')
        plt.figure(figsize=(10, 8))
        # plt.title("Horizontal Direct Solar Irradiance Map (0° = North)")
        plt.imshow(direct_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Direct Solar Irradiance (W/m²)')
        plt.axis('off')
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

    The function:
    1. Computes SVF map accounting for tree transmittance
    2. Scales SVF by diffuse horizontal irradiance
    3. Optionally visualizes and exports results

    Args:
        voxel_data (ndarray): 3D array of voxel values.
        meshsize (float): Size of each voxel in meters.
        diffuse_irradiance (float): Diffuse horizontal irradiance in W/m².
        show_plot (bool): Whether to display visualization.
        **kwargs: Additional arguments including:
            - view_point_height (float): Observer height in meters (default: 1.5)
            - colormap (str): Matplotlib colormap name (default: 'magma')
            - vmin (float): Minimum value for colormap
            - vmax (float): Maximum value for colormap
            - tree_k (float): Tree extinction coefficient
            - tree_lad (float): Leaf area density in m^-1
            - obj_export (bool): Whether to export as OBJ file
            - output_directory (str): Directory for OBJ export
            - output_file_name (str): Filename for OBJ export
            - dem_grid (ndarray): DEM grid for OBJ export
            - num_colors (int): Number of colors for OBJ export
            - alpha (float): Alpha value for OBJ export

    Returns:
        ndarray: 2D array of diffuse solar irradiance values (W/m²).
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

    # Optional visualization
    if show_plot:
        vmin = kwargs.get("vmin", 0.0)
        vmax = kwargs.get("vmax", diffuse_irradiance)
        cmap = plt.cm.get_cmap(colormap).copy()
        cmap.set_bad(color='lightgray')
        plt.figure(figsize=(10, 8))
        # plt.title("Diffuse Solar Irradiance Map")
        plt.imshow(diffuse_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Diffuse Solar Irradiance (W/m²)')
        plt.axis('off')
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

    The function:
    1. Computes direct solar irradiance map
    2. Computes diffuse solar irradiance map
    3. Combines maps and optionally visualizes/exports results

    Args:
        voxel_data (ndarray): 3D voxel array.
        meshsize (float): Voxel size in meters.
        azimuth_degrees (float): Sun azimuth angle in degrees (0° = North, 90° = East).
        elevation_degrees (float): Sun elevation angle in degrees above horizon.
        direct_normal_irradiance (float): Direct normal irradiance in W/m².
        diffuse_irradiance (float): Diffuse horizontal irradiance in W/m².
        show_plot (bool): Whether to display visualization.
        **kwargs: Additional arguments including:
            - view_point_height (float): Observer height in meters (default: 1.5)
            - colormap (str): Matplotlib colormap name (default: 'magma')
            - vmin (float): Minimum value for colormap
            - vmax (float): Maximum value for colormap
            - tree_k (float): Tree extinction coefficient
            - tree_lad (float): Leaf area density in m^-1
            - obj_export (bool): Whether to export as OBJ file
            - output_directory (str): Directory for OBJ export
            - output_file_name (str): Filename for OBJ export
            - dem_grid (ndarray): DEM grid for OBJ export
            - num_colors (int): Number of colors for OBJ export
            - alpha (float): Alpha value for OBJ export

    Returns:
        ndarray: 2D array of global solar irradiance values (W/m²).
    """    
    
    colormap = kwargs.get("colormap", 'magma')

    # Create kwargs for diffuse calculation
    direct_diffuse_kwargs = kwargs.copy()
    direct_diffuse_kwargs.update({
        'show_plot': True,
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

    # Sum the two components
    global_map = direct_map + diffuse_map

    vmin = kwargs.get("vmin", np.nanmin(global_map))
    vmax = kwargs.get("vmax", np.nanmax(global_map))

    # Optional visualization
    if show_plot:
        cmap = plt.cm.get_cmap(colormap).copy()
        cmap.set_bad(color='lightgray')
        plt.figure(figsize=(10, 8))
        # plt.title("Global Solar Irradiance Map")
        plt.imshow(global_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Global Solar Irradiance (W/m²)')
        plt.axis('off')
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

def get_solar_positions_astral(times, lon, lat):
    """
    Compute solar azimuth and elevation using Astral for given times and location.
    
    The function:
    1. Creates an Astral observer at the specified location
    2. Computes sun position for each timestamp
    3. Returns DataFrame with azimuth and elevation angles
    
    Args:
        times (DatetimeIndex): Array of timezone-aware datetime objects.
        lon (float): Longitude in degrees.
        lat (float): Latitude in degrees.

    Returns:
        DataFrame: DataFrame with columns 'azimuth' and 'elevation' containing solar positions.
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
    df, lon, lat, tz,
    direct_normal_irradiance_scaling=1.0,
    diffuse_irradiance_scaling=1.0,
    **kwargs
):
    """
    Compute cumulative global solar irradiance over a specified period using data from an EPW file.

    The function:
    1. Filters EPW data for specified time period
    2. Computes sun positions for each timestep
    3. Calculates and accumulates global irradiance maps
    4. Handles tree transmittance and visualization

    Args:
        voxel_data (ndarray): 3D array of voxel values.
        meshsize (float): Size of each voxel in meters.
        df (DataFrame): EPW weather data.
        lon (float): Longitude in degrees.
        lat (float): Latitude in degrees.
        tz (float): Timezone offset in hours.
        direct_normal_irradiance_scaling (float): Scaling factor for direct normal irradiance.
        diffuse_irradiance_scaling (float): Scaling factor for diffuse horizontal irradiance.
        **kwargs: Additional arguments including:
            - view_point_height (float): Observer height in meters (default: 1.5)
            - start_time (str): Start time in format 'MM-DD HH:MM:SS'
            - end_time (str): End time in format 'MM-DD HH:MM:SS'
            - tree_k (float): Tree extinction coefficient
            - tree_lad (float): Leaf area density in m^-1
            - show_plot (bool): Whether to show final plot
            - show_each_timestep (bool): Whether to show plots for each timestep
            - colormap (str): Matplotlib colormap name
            - vmin (float): Minimum value for colormap
            - vmax (float): Maximum value for colormap
            - obj_export (bool): Whether to export as OBJ file
            - output_directory (str): Directory for OBJ export
            - output_file_name (str): Filename for OBJ export
            - dem_grid (ndarray): DEM grid for OBJ export
            - num_colors (int): Number of colors for OBJ export
            - alpha (float): Alpha value for OBJ export

    Returns:
        ndarray: 2D array of cumulative global solar irradiance values (W/m²·hour).
    """
    view_point_height = kwargs.get("view_point_height", 1.5)
    colormap = kwargs.get("colormap", 'magma')
    start_time = kwargs.get("start_time", "01-01 05:00:00")
    end_time = kwargs.get("end_time", "01-01 20:00:00")

    if df.empty:
        raise ValueError("No data in EPW file.")

    # Parse start and end times without year
    try:
        start_dt = datetime.strptime(start_time, "%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end_time, "%m-%d %H:%M:%S")
    except ValueError as ve:
        raise ValueError("start_time and end_time must be in format 'MM-DD HH:MM:SS'") from ve

    # Add hour of year column and filter data
    df['hour_of_year'] = (df.index.dayofyear - 1) * 24 + df.index.hour + 1
    
    # Convert dates to day of year and hour
    start_doy = datetime(2000, start_dt.month, start_dt.day).timetuple().tm_yday
    end_doy = datetime(2000, end_dt.month, end_dt.day).timetuple().tm_yday
    
    start_hour = (start_doy - 1) * 24 + start_dt.hour + 1
    end_hour = (end_doy - 1) * 24 + end_dt.hour + 1

    # Handle period crossing year boundary
    if start_hour <= end_hour:
        df_period = df[(df['hour_of_year'] >= start_hour) & (df['hour_of_year'] <= end_hour)]
    else:
        df_period = df[(df['hour_of_year'] >= start_hour) | (df['hour_of_year'] <= end_hour)]

    # Filter by minutes within start/end hours
    df_period = df_period[
        ((df_period.index.hour != start_dt.hour) | (df_period.index.minute >= start_dt.minute)) &
        ((df_period.index.hour != end_dt.hour) | (df_period.index.minute <= end_dt.minute))
    ]

    if df_period.empty:
        raise ValueError("No EPW data in the specified period.")

    # Handle timezone conversion
    offset_minutes = int(tz * 60)
    local_tz = pytz.FixedOffset(offset_minutes)
    df_period_local = df_period.copy()
    df_period_local.index = df_period_local.index.tz_localize(local_tz)
    df_period_utc = df_period_local.tz_convert(pytz.UTC)

    # Compute solar positions for period
    solar_positions = get_solar_positions_astral(df_period_utc.index, lon, lat)

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

    # Initialize accumulation maps
    cumulative_map = np.zeros((voxel_data.shape[0], voxel_data.shape[1]))
    mask_map = np.ones((voxel_data.shape[0], voxel_data.shape[1]), dtype=bool)

    # Create kwargs for direct calculation
    direct_kwargs = kwargs.copy()
    direct_kwargs.update({
        'show_plot': False,
        'view_point_height': view_point_height,
        'obj_export': False
    })

    # Process each timestep
    for idx, (time_utc, row) in enumerate(df_period_utc.iterrows()):
        # Get scaled irradiance values
        DNI = row['DNI'] * direct_normal_irradiance_scaling
        DHI = row['DHI'] * diffuse_irradiance_scaling
        time_local = df_period_local.index[idx]

        # Get solar position for timestep
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

        # Scale base diffuse map by actual DHI
        diffuse_map = base_diffuse_map * DHI

        # Combine direct and diffuse components
        global_map = direct_map + diffuse_map

        # Update valid pixel mask
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
            plt.figure(figsize=(10, 8))
            # plt.title(f"Global Solar Irradiance at {time_local.strftime('%Y-%m-%d %H:%M:%S')}")
            plt.imshow(global_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.axis('off')
            plt.colorbar(label='Global Solar Irradiance (W/m²)')
            plt.show()

    # Apply mask to final result
    cumulative_map[~mask_map] = np.nan

    # Final visualization
    show_plot = kwargs.get("show_plot", True)
    if show_plot:
        colormap = kwargs.get("colormap", 'magma')
        vmin = kwargs.get("vmin", np.nanmin(cumulative_map))
        vmax = kwargs.get("vmax", np.nanmax(cumulative_map))
        cmap = plt.cm.get_cmap(colormap).copy()
        cmap.set_bad(color='lightgray')
        plt.figure(figsize=(10, 8))
        # plt.title("Cumulative Global Solar Irradiance Map")
        plt.imshow(cumulative_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Cumulative Global Solar Irradiance (W/m²·hour)')
        plt.axis('off')
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

def get_global_solar_irradiance_using_epw(
    voxel_data,
    meshsize,
    calc_type='instantaneous',
    direct_normal_irradiance_scaling=1.0,
    diffuse_irradiance_scaling=1.0,
    **kwargs
):
    """
    Compute global solar irradiance using EPW weather data, either for a single time or cumulatively over a period.

    The function:
    1. Optionally downloads and reads EPW weather data
    2. Handles timezone conversions and solar position calculations
    3. Computes either instantaneous or cumulative irradiance maps
    4. Supports visualization and export options

    Args:
        voxel_data (ndarray): 3D array of voxel values.
        meshsize (float): Size of each voxel in meters.
        calc_type (str): 'instantaneous' or 'cumulative'.
        direct_normal_irradiance_scaling (float): Scaling factor for direct normal irradiance.
        diffuse_irradiance_scaling (float): Scaling factor for diffuse horizontal irradiance.
        **kwargs: Additional arguments including:
            - download_nearest_epw (bool): Whether to download nearest EPW file
            - epw_file_path (str): Path to EPW file
            - rectangle_vertices (list): List of (lat,lon) coordinates for EPW download
            - output_dir (str): Directory for EPW download
            - calc_time (str): Time for instantaneous calculation ('MM-DD HH:MM:SS')
            - start_time (str): Start time for cumulative calculation
            - end_time (str): End time for cumulative calculation
            - start_hour (int): Starting hour for daily time window (0-23)
            - end_hour (int): Ending hour for daily time window (0-23)
            - view_point_height (float): Observer height in meters
            - tree_k (float): Tree extinction coefficient
            - tree_lad (float): Leaf area density in m^-1
            - show_plot (bool): Whether to show visualization
            - show_each_timestep (bool): Whether to show timestep plots
            - colormap (str): Matplotlib colormap name
            - obj_export (bool): Whether to export as OBJ file

    Returns:
        ndarray: 2D array of solar irradiance values (W/m²).
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
            lons = [coord[0] for coord in rectangle_vertices]
            lats = [coord[1] for coord in rectangle_vertices]
            center_lon = (min(lons) + max(lons)) / 2
            center_lat = (min(lats) + max(lats)) / 2
            target_point = (center_lon, center_lat)

            # Optional: specify maximum distance in kilometers
            max_distance = 100  # None for no limit

            output_dir = kwargs.get("output_dir", "output")

            epw_file_path, weather_data, metadata = get_nearest_epw_from_climate_onebuilding(
                longitude=center_lon,
                latitude=center_lat,
                output_dir=output_dir,
                max_distance=max_distance,
                extract_zip=True,
                load_data=True
            )

    # Read EPW data
    df, lat, lon, tz, elevation_m = read_epw_for_solar_simulation(epw_file_path)
    if df.empty:
        raise ValueError("No data in EPW file.")

    if calc_type == 'instantaneous':
        if df.empty:
            raise ValueError("No data in EPW file.")

        calc_time = kwargs.get("calc_time", "01-01 12:00:00")

        # Parse start and end times without year
        try:
            calc_dt = datetime.strptime(calc_time, "%m-%d %H:%M:%S")
        except ValueError as ve:
            raise ValueError("calc_time must be in format 'MM-DD HH:MM:SS'") from ve

        df_period = df[
            (df.index.month == calc_dt.month) & (df.index.day == calc_dt.day) & (df.index.hour == calc_dt.hour)
        ]

        if df_period.empty:
            raise ValueError("No EPW data at the specified time.")

        # Prepare timezone conversion
        offset_minutes = int(tz * 60)
        local_tz = pytz.FixedOffset(offset_minutes)
        df_period_local = df_period.copy()
        df_period_local.index = df_period_local.index.tz_localize(local_tz)
        df_period_utc = df_period_local.tz_convert(pytz.UTC)

        # Compute solar positions
        solar_positions = get_solar_positions_astral(df_period_utc.index, lat, lon)
        direct_normal_irradiance = df_period_utc.iloc[0]['DNI']
        diffuse_irradiance = df_period_utc.iloc[0]['DHI']
        azimuth_degrees = solar_positions.iloc[0]['azimuth']
        elevation_degrees = solar_positions.iloc[0]['elevation']    
        solar_map = get_global_solar_irradiance_map(
            voxel_data,                 # 3D voxel grid representing the urban environment
            meshsize,                   # Size of each grid cell in meters
            azimuth_degrees,            # Sun's azimuth angle
            elevation_degrees,          # Sun's elevation angle
            direct_normal_irradiance,   # Direct Normal Irradiance value
            diffuse_irradiance,         # Diffuse irradiance value
            show_plot=True,             # Display visualization of results
            **kwargs
        )
    if calc_type == 'cumulative':
        # Get time window parameters
        start_hour = kwargs.get("start_hour", 0)  # Default to midnight
        end_hour = kwargs.get("end_hour", 23)     # Default to 11 PM
        
        # Filter dataframe for specified hours
        df_filtered = df[(df.index.hour >= start_hour) & (df.index.hour <= end_hour)]
        
        solar_map = get_cumulative_global_solar_irradiance(
            voxel_data,
            meshsize,
            df_filtered, lat, lon, tz,
            **kwargs
        )
    
    return solar_map 