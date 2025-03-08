import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange
from datetime import datetime, timezone
import pytz
from astral import Observer
from astral.sun import elevation, azimuth

from .view import trace_ray_generic, compute_vi_map_generic, get_sky_view_factor_map, get_building_surface_svf
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

import numpy as np
import trimesh
import time
from numba import njit

##############################################################################
# 1) New Numba helper: per-face solar irradiance computation
##############################################################################
@njit
def compute_solar_irradiance_for_all_faces(
    face_centers,
    face_normals,
    face_svf_values,
    sun_direction,
    direct_normal_irradiance,
    diffuse_irradiance,
    voxel_data,
    meshsize,
    tree_k,
    tree_lad,
    hit_values,
    inclusion_mode,
    grid_bounds_real,
    boundary_epsilon
):
    """
    Numba-compiled function to compute direct, diffuse, and global solar irradiance
    for each face in the mesh.
    
    Args:
        face_centers (float64[:, :]): (N x 3) array of face center points
        face_normals (float64[:, :]): (N x 3) array of face normals
        face_svf_values (float64[:]): (N) array of SVF values for each face
        sun_direction (float64[:]): (3) array for sun direction (dx, dy, dz)
        direct_normal_irradiance (float): Direct normal irradiance (DNI) in W/m²
        diffuse_irradiance (float): Diffuse horizontal irradiance (DHI) in W/m²
        voxel_data (ndarray): 3D array of voxel values
        meshsize (float): Size of each voxel in meters
        tree_k (float): Tree extinction coefficient
        tree_lad (float): Leaf area density
        hit_values (tuple): Values considered 'sky' (e.g. (0,))
        inclusion_mode (bool): Whether we want to "include" or "exclude" these hit_values
        grid_bounds_real (float64[2,3]): [[x_min, y_min, z_min],[x_max, y_max, z_max]]
        boundary_epsilon (float): Distance threshold for bounding-box check
    
    Returns:
        (direct_irr, diffuse_irr, global_irr) as three float64[N] arrays
    """
    n_faces = face_centers.shape[0]
    
    face_direct = np.zeros(n_faces, dtype=np.float64)
    face_diffuse = np.zeros(n_faces, dtype=np.float64)
    face_global = np.zeros(n_faces, dtype=np.float64)
    
    x_min, y_min, z_min = grid_bounds_real[0, 0], grid_bounds_real[0, 1], grid_bounds_real[0, 2]
    x_max, y_max, z_max = grid_bounds_real[1, 0], grid_bounds_real[1, 1], grid_bounds_real[1, 2]
    
    for fidx in range(n_faces):
        center = face_centers[fidx]
        normal = face_normals[fidx]
        svf    = face_svf_values[fidx]
        
        # -- 1) Check for vertical boundary face
        is_vertical = (abs(normal[2]) < 0.01)
        
        on_x_min = (abs(center[0] - x_min) < boundary_epsilon)
        on_y_min = (abs(center[1] - y_min) < boundary_epsilon)
        on_x_max = (abs(center[0] - x_max) < boundary_epsilon)
        on_y_max = (abs(center[1] - y_max) < boundary_epsilon)
        
        is_boundary_vertical = is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max)
        
        if is_boundary_vertical:
            face_direct[fidx]  = np.nan
            face_diffuse[fidx] = np.nan
            face_global[fidx]  = np.nan
            continue
        
        # If SVF is NaN, skip (means it was set to boundary or invalid earlier)
        if svf != svf:  # NaN check in Numba
            face_direct[fidx]  = np.nan
            face_diffuse[fidx] = np.nan
            face_global[fidx]  = np.nan
            continue
        
        # -- 2) Direct irradiance (if face is oriented towards sun)
        cos_incidence = normal[0]*sun_direction[0] + \
                        normal[1]*sun_direction[1] + \
                        normal[2]*sun_direction[2]
        
        direct_val = 0.0
        if cos_incidence > 0.0:
            # Offset ray origin slightly to avoid self-intersection
            offset_vox = 0.1
            ray_origin_x = center[0]/meshsize + normal[0]*offset_vox
            ray_origin_y = center[1]/meshsize + normal[1]*offset_vox
            ray_origin_z = center[2]/meshsize + normal[2]*offset_vox
            
            # Single ray toward the sun            
            hit_detected, transmittance = trace_ray_generic(
                voxel_data,
                np.array([ray_origin_x, ray_origin_y, ray_origin_z], dtype=np.float64),
                sun_direction,
                hit_values,
                meshsize,
                tree_k,
                tree_lad,
                inclusion_mode
            )
            if not hit_detected:
                direct_val = direct_normal_irradiance * cos_incidence * transmittance
        
        # -- 3) Diffuse irradiance from sky: use SVF * DHI
        diffuse_val = svf * diffuse_irradiance
        if diffuse_val > diffuse_irradiance:
            diffuse_val = diffuse_irradiance
        
        # -- 4) Sum up
        face_direct[fidx]  = direct_val
        face_diffuse[fidx] = diffuse_val
        face_global[fidx]  = direct_val + diffuse_val
    
    return face_direct, face_diffuse, face_global


##############################################################################
# 2) Modified get_building_solar_irradiance: main Python wrapper
##############################################################################
def get_building_solar_irradiance(
    voxel_data,
    meshsize,
    building_svf_mesh,
    azimuth_degrees,
    elevation_degrees,
    direct_normal_irradiance,
    diffuse_irradiance,
    **kwargs
):
    """
    Calculate solar irradiance on building surfaces using SVF,
    with the numeric per-face loop accelerated by Numba.
    
    Args:
        voxel_data (ndarray): 3D array of voxel values.
        meshsize (float): Size of each voxel in meters.
        building_svf_mesh (trimesh.Trimesh): Building mesh with SVF values in metadata.
        azimuth_degrees (float): Sun azimuth angle in degrees (0=North, 90=East).
        elevation_degrees (float): Sun elevation angle in degrees above horizon.
        direct_normal_irradiance (float): DNI in W/m².
        diffuse_irradiance (float): DHI in W/m².
        **kwargs: Additional parameters, e.g. tree_k, tree_lad, progress_report, obj_export, etc.
    
    Returns:
        trimesh.Trimesh: A copy of the input mesh with direct/diffuse/global irradiance stored in metadata.
    """
    import time
    
    tree_k          = kwargs.get("tree_k", 0.6)
    tree_lad        = kwargs.get("tree_lad", 1.0)
    progress_report = kwargs.get("progress_report", False)
    
    # Sky detection
    hit_values     = (0,)    # '0' = sky
    inclusion_mode = False
    
    # Convert angles -> direction
    az_rad = np.deg2rad(180 - azimuth_degrees)
    el_rad = np.deg2rad(elevation_degrees)
    sun_dx = np.cos(el_rad) * np.sin(az_rad)
    sun_dy = np.cos(el_rad) * np.cos(az_rad)
    sun_dz = np.sin(el_rad)
    sun_direction = np.array([sun_dx, sun_dy, sun_dz], dtype=np.float64)
    
    # Extract mesh data
    face_centers = building_svf_mesh.triangles_center
    face_normals = building_svf_mesh.face_normals
    
    # Get SVF from metadata
    if hasattr(building_svf_mesh, 'metadata') and ('svf_values' in building_svf_mesh.metadata):
        face_svf_values = building_svf_mesh.metadata['svf_values']
    else:
        face_svf_values = np.zeros(len(building_svf_mesh.faces), dtype=np.float64)
    
    # Prepare boundary checks
    grid_shape = voxel_data.shape
    grid_bounds_voxel = np.array([[0,0,0],[grid_shape[0], grid_shape[1], grid_shape[2]]], dtype=np.float64)
    grid_bounds_real = grid_bounds_voxel * meshsize
    boundary_epsilon = meshsize * 0.05
    
    # Call Numba-compiled function
    t0 = time.time()
    face_direct, face_diffuse, face_global = compute_solar_irradiance_for_all_faces(
        face_centers,
        face_normals,
        face_svf_values,
        sun_direction,
        direct_normal_irradiance,
        diffuse_irradiance,
        voxel_data,
        meshsize,
        tree_k,
        tree_lad,
        hit_values,
        inclusion_mode,
        grid_bounds_real,
        boundary_epsilon
    )
    if progress_report:
        elapsed = time.time() - t0
        print(f"Numba-based solar irradiance calculation took {elapsed:.2f} seconds")
    
    # Create a copy of the mesh
    irradiance_mesh = building_svf_mesh.copy()
    if not hasattr(irradiance_mesh, 'metadata'):
        irradiance_mesh.metadata = {}
    
    # Store results
    irradiance_mesh.metadata['svf']    = face_svf_values
    irradiance_mesh.metadata['direct'] = face_direct
    irradiance_mesh.metadata['diffuse'] = face_diffuse
    irradiance_mesh.metadata['global'] = face_global
    
    irradiance_mesh.name = "Solar Irradiance (W/m²)"
    
    # # Optional OBJ export
    # obj_export = kwargs.get("obj_export", False)
    # if obj_export:
    #     _export_solar_irradiance_mesh(
    #         irradiance_mesh,
    #         face_global,
    #         **kwargs
    #     )
    
    return irradiance_mesh


# ##############################################################################
# # 3) Small helper for OBJ export & color-mapping
# ##############################################################################
# def _export_solar_irradiance_mesh(mesh_obj, face_global_values, **kwargs):
#     import os
#     import matplotlib.cm as cm
#     import matplotlib.colors as mcolors
    
#     output_dir       = kwargs.get("output_directory", "output")
#     output_file_name = kwargs.get("output_file_name", "building_solar_irradiance")
#     os.makedirs(output_dir, exist_ok=True)
    
#     vmin = kwargs.get("vmin", 0.0)
#     vmax = kwargs.get("vmax", None)
#     valid_mask = ~np.isnan(face_global_values)
#     if vmax is None and np.any(valid_mask):
#         vmax = max(np.nanmax(face_global_values), 0.1)
#     elif vmax is None:
#         vmax = 1.0  # fallback
    
#     colormap = kwargs.get("colormap", "magma")
#     cmap = cm.get_cmap(colormap)
#     norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
#     face_colors = np.zeros((len(face_global_values), 4))
#     nan_mask = np.isnan(face_global_values)
    
#     # Use 'gray' or user-specified color for NaN
#     nan_color = kwargs.get("nan_color", "gray")
#     if isinstance(nan_color, str):
#         nan_rgba = np.array(mcolors.to_rgba(nan_color))
#     else:
#         nan_rgba = np.array(nan_color)
    
#     face_colors[valid_mask] = cmap(norm(face_global_values[valid_mask]))
#     face_colors[nan_mask]   = nan_rgba
    
#     # Apply and export
#     export_mesh = mesh_obj.copy()
#     export_mesh.visual.face_colors = face_colors
#     try:
#         export_mesh.export(f"{output_dir}/{output_file_name}.obj")
#         print(f"Exported solar irradiance mesh to {output_dir}/{output_file_name}.obj")
#     except Exception as e:
#         print(f"Error exporting mesh: {e}")


##############################################################################
# 4) Modified get_cumulative_building_solar_irradiance
##############################################################################
def get_cumulative_building_solar_irradiance(
    voxel_data,
    meshsize,
    building_svf_mesh,
    weather_df,
    lon, lat, tz,
    **kwargs
):
    """
    Calculate cumulative solar irradiance on building surfaces over a time period.
    Uses the Numba-accelerated get_building_solar_irradiance for each time step.
    
    Args:
        voxel_data (ndarray): 3D array of voxel values.
        meshsize (float): Size of each voxel in meters.
        building_svf_mesh (trimesh.Trimesh): Mesh with pre-calculated SVF in metadata.
        weather_df (DataFrame): Weather data with DNI (W/m²) and DHI (W/m²).
        lon (float): Longitude in degrees.
        lat (float): Latitude in degrees.
        tz (float): Timezone offset in hours.
        **kwargs: Additional parameters for time range, scaling, OBJ export, etc.
    
    Returns:
        trimesh.Trimesh: A mesh with cumulative (Wh/m²) irradiance in metadata.
    """
    import pytz
    from datetime import datetime
    
    period_start = kwargs.get("period_start", "01-01 00:00:00")
    period_end   = kwargs.get("period_end",   "12-31 23:59:59")
    time_step_hours = kwargs.get("time_step_hours", 1.0)
    direct_normal_irradiance_scaling = kwargs.get("direct_normal_irradiance_scaling", 1.0)
    diffuse_irradiance_scaling       = kwargs.get("diffuse_irradiance_scaling", 1.0)
    
    # Parse times, create local tz
    try:
        start_dt = datetime.strptime(period_start, "%m-%d %H:%M:%S")
        end_dt   = datetime.strptime(period_end,   "%m-%d %H:%M:%S")
    except ValueError as ve:
        raise ValueError("Time must be in format 'MM-DD HH:MM:SS'") from ve
    
    offset_minutes = int(tz * 60)
    local_tz = pytz.FixedOffset(offset_minutes)
    
    # Filter weather_df
    df_period = weather_df[
        ((weather_df.index.month > start_dt.month) |
         ((weather_df.index.month == start_dt.month) &
          (weather_df.index.day >= start_dt.day) &
          (weather_df.index.hour >= start_dt.hour))) &
        ((weather_df.index.month < end_dt.month) |
         ((weather_df.index.month == end_dt.month) &
          (weather_df.index.day <= end_dt.day) &
          (weather_df.index.hour <= end_dt.hour)))
    ]
    if df_period.empty:
        raise ValueError("No weather data in specified period.")
    
    # Convert to local time, then to UTC
    df_period_local = df_period.copy()
    df_period_local.index = df_period_local.index.tz_localize(local_tz)
    df_period_utc = df_period_local.tz_convert(pytz.UTC)
    
    # Get solar positions
    # You presumably have a get_solar_positions_astral(...) that returns az/elev
    solar_positions = get_solar_positions_astral(df_period_utc.index, lon, lat)
    
    # Prepare arrays for accumulation
    n_faces = len(building_svf_mesh.faces)
    face_cum_direct  = np.zeros(n_faces, dtype=np.float64)
    face_cum_diffuse = np.zeros(n_faces, dtype=np.float64)
    face_cum_global  = np.zeros(n_faces, dtype=np.float64)
    
    boundary_mask = None
    
    # Iterate over each timestep
    for idx, (time_utc, row) in enumerate(df_period_utc.iterrows()):
        DNI = row['DNI'] * direct_normal_irradiance_scaling
        DHI = row['DHI'] * diffuse_irradiance_scaling
        
        # Sun angles
        az_deg = solar_positions.loc[time_utc, 'azimuth']
        el_deg = solar_positions.loc[time_utc, 'elevation']
        
        # Skip if sun below horizon
        if el_deg <= 0:
            continue
        
        # Call instantaneous function (Numba-accelerated inside)
        irr_mesh = get_building_solar_irradiance(
            voxel_data,
            meshsize,
            building_svf_mesh,
            az_deg,
            el_deg,
            DNI,
            DHI,
            show_plot=False,  # or any other flags
            **kwargs
        )
        
        # Extract arrays
        face_dir  = irr_mesh.metadata['direct']
        face_diff = irr_mesh.metadata['diffuse']
        face_glob = irr_mesh.metadata['global']
        
        # If first time, note boundary mask from NaNs
        if boundary_mask is None:
            boundary_mask = np.isnan(face_glob)
        
        # Convert from W/m² to Wh/m² by multiplying time_step_hours
        face_cum_direct  += np.nan_to_num(face_dir)  * time_step_hours
        face_cum_diffuse += np.nan_to_num(face_diff) * time_step_hours
        face_cum_global  += np.nan_to_num(face_glob) * time_step_hours
    
    # Reapply NaN for boundary
    if boundary_mask is not None:
        face_cum_direct[boundary_mask]  = np.nan
        face_cum_diffuse[boundary_mask] = np.nan
        face_cum_global[boundary_mask]  = np.nan
    
    # Create a new mesh with cumulative results
    cumulative_mesh = building_svf_mesh.copy()
    if not hasattr(cumulative_mesh, 'metadata'):
        cumulative_mesh.metadata = {}
    
    # If original mesh had SVF
    if 'svf_values' in building_svf_mesh.metadata:
        cumulative_mesh.metadata['svf'] = building_svf_mesh.metadata['svf_values']
    
    cumulative_mesh.metadata['direct']  = face_cum_direct
    cumulative_mesh.metadata['diffuse'] = face_cum_diffuse
    cumulative_mesh.metadata['global']  = face_cum_global
    
    cumulative_mesh.name = "Cumulative Solar Irradiance (Wh/m²)"
    
    # Optional export
    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        _export_solar_irradiance_mesh(
            cumulative_mesh,
            face_cum_global,
            **kwargs
        )
    
    return cumulative_mesh

def get_building_global_solar_irradiance_using_epw(
    voxel_data,
    meshsize,
    calc_type='instantaneous',
    direct_normal_irradiance_scaling=1.0,
    diffuse_irradiance_scaling=1.0,
    **kwargs
):
    """
    Compute global solar irradiance on building surfaces using EPW weather data, either for a single time or cumulatively.

    The function:
    1. Optionally downloads and reads EPW weather data
    2. Handles timezone conversions and solar position calculations
    3. Computes either instantaneous or cumulative irradiance on building surfaces
    4. Supports visualization and export options

    Args:
        voxel_data (ndarray): 3D array of voxel values.
        meshsize (float): Size of each voxel in meters.
        building_svf_mesh (trimesh.Trimesh): Building mesh with pre-calculated SVF values in metadata.
        calc_type (str): 'instantaneous' or 'cumulative'.
        direct_normal_irradiance_scaling (float): Scaling factor for direct normal irradiance.
        diffuse_irradiance_scaling (float): Scaling factor for diffuse horizontal irradiance.
        **kwargs: Additional arguments including:
            - download_nearest_epw (bool): Whether to download nearest EPW file
            - epw_file_path (str): Path to EPW file
            - rectangle_vertices (list): List of (lon,lat) coordinates for EPW download
            - output_dir (str): Directory for EPW download
            - calc_time (str): Time for instantaneous calculation ('MM-DD HH:MM:SS')
            - period_start (str): Start time for cumulative calculation ('MM-DD HH:MM:SS')
            - period_end (str): End time for cumulative calculation ('MM-DD HH:MM:SS')
            - time_step_hours (float): Time step for cumulative calculation
            - tree_k (float): Tree extinction coefficient
            - tree_lad (float): Leaf area density in m^-1
            - show_each_timestep (bool): Whether to show plots for each timestep
            - nan_color (str): Color for NaN values in visualization
            - colormap (str): Matplotlib colormap name
            - vmin (float): Minimum value for colormap
            - vmax (float): Maximum value for colormap
            - obj_export (bool): Whether to export as OBJ file
            - output_directory (str): Directory for OBJ export
            - output_file_name (str): Filename for OBJ export

    Returns:
        trimesh.Trimesh: Building mesh with irradiance values stored in metadata.
    """
    import numpy as np
    import pytz
    from datetime import datetime
    
    # Get EPW file
    download_nearest_epw = kwargs.get("download_nearest_epw", False)
    rectangle_vertices = kwargs.get("rectangle_vertices", None)
    epw_file_path = kwargs.get("epw_file_path", None)
    building_id_grid = kwargs.get("building_id_grid", None)

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
            
            # Optional: specify maximum distance in kilometers
            max_distance = kwargs.get("max_distance", 100)  # None for no limit
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
    
    # Step 1: Calculate Sky View Factor for building surfaces
    print(f"Processing Sky View Factor for building surfaces...")
    building_svf_mesh = get_building_surface_svf(
        voxel_data,  # Your 3D voxel grid
        meshsize,      # Size of each voxel in meters
        building_id_grid=building_id_grid,
    )

    print(f"Processing Solar Irradiance for building surfaces...")
    if calc_type == 'instantaneous':
        calc_time = kwargs.get("calc_time", "01-01 12:00:00")

        # Parse calculation time without year
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
        solar_positions = get_solar_positions_astral(df_period_utc.index, lon, lat)
        
        # Scale irradiance values
        direct_normal_irradiance = df_period_utc.iloc[0]['DNI'] * direct_normal_irradiance_scaling
        diffuse_irradiance = df_period_utc.iloc[0]['DHI'] * diffuse_irradiance_scaling
        
        # Get solar position
        azimuth_degrees = solar_positions.iloc[0]['azimuth']
        elevation_degrees = solar_positions.iloc[0]['elevation']
        
        print(f"Time: {df_period_local.index[0].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Sun position: Azimuth {azimuth_degrees:.1f}°, Elevation {elevation_degrees:.1f}°")
        print(f"DNI: {direct_normal_irradiance:.1f} W/m², DHI: {diffuse_irradiance:.1f} W/m²")
        
        # Skip if sun is below horizon
        if elevation_degrees <= 0:
            print("Sun is below horizon, skipping calculation.")
            return building_svf_mesh.copy()
        
        # Compute irradiance
        irradiance_mesh = get_building_solar_irradiance(
            voxel_data,
            meshsize,
            building_svf_mesh,
            azimuth_degrees,
            elevation_degrees,
            direct_normal_irradiance,
            diffuse_irradiance,
            **kwargs
        )
        
        return irradiance_mesh

    elif calc_type == 'cumulative':
        # Set default parameters
        period_start = kwargs.get("period_start", "01-01 00:00:00")
        period_end = kwargs.get("period_end", "12-31 23:59:59")
        time_step_hours = kwargs.get("time_step_hours", 1.0)
        
        # Parse start and end times without year
        try:
            start_dt = datetime.strptime(period_start, "%m-%d %H:%M:%S")
            end_dt = datetime.strptime(period_end, "%m-%d %H:%M:%S")
        except ValueError as ve:
            raise ValueError("Time must be in format 'MM-DD HH:MM:SS'") from ve
        
        # Create local timezone
        offset_minutes = int(tz * 60)
        local_tz = pytz.FixedOffset(offset_minutes)
        
        # Filter weather data by month, day, hour
        df_period = df[
            ((df.index.month > start_dt.month) | 
             ((df.index.month == start_dt.month) & (df.index.day >= start_dt.day) & 
              (df.index.hour >= start_dt.hour))) &
            ((df.index.month < end_dt.month) | 
             ((df.index.month == end_dt.month) & (df.index.day <= end_dt.day) & 
              (df.index.hour <= end_dt.hour)))
        ]
        
        if df_period.empty:
            raise ValueError("No weather data available for the specified period.")
        
        # Convert to local timezone and then to UTC for solar position calculation
        df_period_local = df_period.copy()
        df_period_local.index = df_period_local.index.tz_localize(local_tz)
        df_period_utc = df_period_local.tz_convert(pytz.UTC)
        
        # Get solar positions for all times
        solar_positions = get_solar_positions_astral(df_period_utc.index, lon, lat)
        
        # Create a copy of kwargs without time_step_hours to avoid duplicate argument
        kwargs_copy = kwargs.copy()
        if 'time_step_hours' in kwargs_copy:
            del kwargs_copy['time_step_hours']
        
        # Get cumulative irradiance - adapt to match expected function signature
        cumulative_mesh = get_cumulative_building_solar_irradiance(
            voxel_data,
            meshsize,
            building_svf_mesh,
            df, lon, lat, tz,  # Pass only the required 7 positional arguments
            period_start=period_start,
            period_end=period_end,
            time_step_hours=time_step_hours,
            direct_normal_irradiance_scaling=direct_normal_irradiance_scaling,
            diffuse_irradiance_scaling=diffuse_irradiance_scaling,
            colormap=kwargs.get('colormap', 'jet'),
            show_each_timestep=kwargs.get('show_each_timestep', False),
            obj_export=kwargs.get('obj_export', False),
            output_directory=kwargs.get('output_directory', 'output'),
            output_file_name=kwargs.get('output_file_name', 'cumulative_solar')
        )
        
        return cumulative_mesh
    
    else:
        raise ValueError("calc_type must be either 'instantaneous' or 'cumulative'")