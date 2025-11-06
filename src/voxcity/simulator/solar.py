"""
Solar Irradiance Simulation Module

This module provides comprehensive solar irradiance calculations for urban environments,
including direct and diffuse solar radiation analysis with consideration for tree transmittance
and building shadows. It supports both instantaneous and cumulative irradiance calculations
using weather data from EPW files.

Key Features:
- Direct solar irradiance with ray tracing and shadow analysis
- Diffuse solar irradiance using Sky View Factor (SVF)
- Tree transmittance modeling using Beer-Lambert law
- Building surface irradiance calculation with 3D mesh support
- Weather data integration from EPW files
- Visualization and export capabilities

The module uses numba for performance optimization in computationally intensive calculations.

Orientation contract:
- Simulation inputs (2D grids and voxel arrays) follow the canonical orientation:
  north_up (row 0 = north/top), columns increase eastward (col 0 = west/left), and
  3D indexing (row, col, z) = (north→south, west→east, ground→up).
- Any internal flips are implementation details for rendering or mapping and do not
  change the semantic orientation of inputs/outputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange
import os
import numba
from datetime import datetime, timezone
import pytz
from astral import Observer
from astral.sun import elevation, azimuth

# Import custom modules for view analysis and weather data processing
from .view import trace_ray_generic, compute_vi_map_generic, get_sky_view_factor_map, get_surface_view_factor
from ..utils.weather import get_nearest_epw_from_climate_onebuilding, read_epw_for_solar_simulation
from ..exporter.obj import grid_to_obj, export_obj
from ..models import VoxCity

@njit(parallel=True)
def compute_direct_solar_irradiance_map_binary(voxel_data, sun_direction, view_point_height, hit_values, meshsize, tree_k, tree_lad, inclusion_mode):
    """
    Compute a map of direct solar irradiation accounting for tree transmittance.

    This function performs ray tracing from observer positions on the ground surface
    towards the sun to determine direct solar irradiance at each location. It accounts
    for shadows cast by buildings and vegetation, with special consideration for
    tree transmittance using the Beer-Lambert law.

    The function:
    1. Places observers at valid locations (empty voxels above ground)
    2. Casts rays from each observer in the sun direction
    3. Computes transmittance through trees using Beer-Lambert law
    4. Returns a 2D map of transmittance values

    Observer Placement Rules:
    - Observers are placed in empty voxels (value 0 or -2 for trees) above solid ground
    - Observers are NOT placed on buildings, vegetation, or water surfaces
    - Observer height is added above the detected ground surface

    Ray Tracing Process:
    - Rays are cast from each valid observer position toward the sun
    - Intersections with obstacles (non-sky voxels) are detected
    - Tree voxels provide partial transmittance rather than complete blocking
    - Final transmittance value represents solar energy reaching the surface

    Args:
        voxel_data (ndarray): 3D array of voxel values representing the urban environment.
                             Common values: 0=sky, 1-6=buildings, 7-9=special surfaces, -2=trees
        sun_direction (tuple): Direction vector of the sun (dx, dy, dz), should be normalized
        view_point_height (float): Observer height above ground surface in meters
        hit_values (tuple): Values considered non-obstacles if inclusion_mode=False
                           Typically (0,) meaning only sky voxels are transparent
        meshsize (float): Size of each voxel in meters (spatial resolution)
        tree_k (float): Tree extinction coefficient for Beer-Lambert law (higher = more opaque)
        tree_lad (float): Leaf area density in m^-1 (affects light attenuation through trees)
        inclusion_mode (bool): False here, meaning any voxel not in hit_values is an obstacle

    Returns:
        ndarray: 2D array of transmittance values (0.0-1.0)
                - 1.0 = full sun exposure
                - 0.0 = complete shadow
                - 0.0-1.0 = partial transmittance through trees
                - NaN = invalid observer position (cannot place observer)
    
    Note:
        The returned map is vertically flipped to match standard visualization conventions
        where the origin is at the bottom-left corner.
    """
    
    # Convert observer height from meters to voxel units
    view_height_voxel = int(view_point_height / meshsize)
    
    # Get dimensions of the voxel grid
    nx, ny, nz = voxel_data.shape
    
    # Initialize irradiance map with NaN (invalid positions)
    irradiance_map = np.full((nx, ny), np.nan, dtype=np.float64)

    # Normalize sun direction vector for consistent ray tracing
    # This ensures rays travel at unit speed through the voxel grid
    sd = np.array(sun_direction, dtype=np.float64)
    sd_len = np.sqrt(sd[0]**2 + sd[1]**2 + sd[2]**2)
    if sd_len == 0.0:
        return np.flipud(irradiance_map)
    sd /= sd_len

    # Process each x,y position in parallel for performance
    # This is the main computational loop optimized with numba
    for x in prange(nx):
        for y in range(ny):
            found_observer = False
            
            # Search upward through the vertical column to find valid observer position
            # Start from z=1 to ensure we can check the voxel below
            for z in range(1, nz):
                
                # Check if current voxel is empty/tree and voxel below is solid
                # This identifies the ground surface where observers can be placed
                if voxel_data[x, y, z] in (0, -2) and voxel_data[x, y, z - 1] not in (0, -2):
                    
                    # Skip if standing on building/vegetation/water surfaces
                    # These are considered invalid observer locations
                    if (voxel_data[x, y, z - 1] in (7, 8, 9)) or (voxel_data[x, y, z - 1] < 0):
                        irradiance_map[x, y] = np.nan
                        found_observer = True
                        break
                    else:
                        # Place observer at valid ground location and cast ray toward sun
                        observer_location = np.array([x, y, z + view_height_voxel], dtype=np.float64)
                        
                        # Trace ray from observer to sun, accounting for obstacles and tree transmittance
                        hit, transmittance = trace_ray_generic(voxel_data, observer_location, sd, 
                                                             hit_values, meshsize, tree_k, tree_lad, inclusion_mode)
                        
                        # Store transmittance value (0 if hit solid obstacle, 0-1 if through trees)
                        irradiance_map[x, y] = transmittance if not hit else 0.0
                        found_observer = True
                        break
            
            # If no valid observer position found in this column, mark as invalid
            if not found_observer:
                irradiance_map[x, y] = np.nan

    # Flip map vertically to match visualization conventions (origin at bottom-left)
    return np.flipud(irradiance_map)

def get_direct_solar_irradiance_map(voxcity: VoxCity, azimuth_degrees_ori, elevation_degrees, 
                                  direct_normal_irradiance, show_plot=False, **kwargs):
    """
    Compute direct solar irradiance map with tree transmittance.
    
    This function converts solar angles to a 3D direction vector, computes the binary
    transmittance map using ray tracing, and scales the results by actual solar irradiance
    values to produce physically meaningful irradiance measurements.
    
    Solar Geometry:
    - Azimuth: Horizontal angle measured from North (0°) clockwise to East (90°)
    - Elevation: Vertical angle above the horizon (0° = horizon, 90° = zenith)
    - The coordinate system is adjusted by 180° to match the voxel grid orientation
    
    Physics Background:
    - Direct Normal Irradiance (DNI): Solar energy on a surface perpendicular to sun rays
    - Horizontal irradiance: DNI scaled by sine of elevation angle
    - Tree transmittance: Applied using Beer-Lambert law for realistic light attenuation
    
    The function:
    1. Converts sun angles to direction vector using spherical coordinates
    2. Computes binary transmittance map accounting for shadows and tree effects
    3. Scales by direct normal irradiance and sun elevation for horizontal surfaces
    4. Optionally visualizes and exports results in various formats
    
    Args:
        voxcity (VoxCity): VoxCity object containing voxel data and metadata
        azimuth_degrees_ori (float): Sun azimuth angle in degrees (0° = North, 90° = East)
        elevation_degrees (float): Sun elevation angle in degrees above horizon (0-90°)
        direct_normal_irradiance (float): Direct normal irradiance in W/m² (from weather data)
        show_plot (bool): Whether to display visualization of results
        **kwargs: Additional arguments including:
            - view_point_height (float): Observer height in meters (default: 1.5)
                Height above ground where irradiance is measured
            - colormap (str): Matplotlib colormap name for visualization (default: 'magma')
            - vmin (float): Minimum value for colormap scaling
            - vmax (float): Maximum value for colormap scaling  
            - tree_k (float): Tree extinction coefficient (default: 0.6)
                Higher values mean trees block more light
            - tree_lad (float): Leaf area density in m^-1 (default: 1.0)
                Affects light attenuation through tree canopies
            - obj_export (bool): Whether to export results as 3D OBJ file
            - output_directory (str): Directory for file exports
            - output_file_name (str): Base filename for exports
            - num_colors (int): Number of discrete colors for OBJ export
            - alpha (float): Transparency value for 3D visualization

    Returns:
        ndarray: 2D array of direct solar irradiance values in W/m²
                - Values represent energy flux on horizontal surfaces
                - NaN indicates invalid measurement locations
                - Range typically 0 to direct_normal_irradiance * sin(elevation)
    
    Note:
        The azimuth is internally adjusted by 180° to match the coordinate system
        where the voxel grid's y-axis points in the opposite direction from geographic north.
    """
    # Extract voxel data and meshsize from VoxCity object
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    
    # Extract parameters with defaults for observer and visualization settings
    view_point_height = kwargs.get("view_point_height", 1.5)
    colormap = kwargs.get("colormap", 'magma')
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", direct_normal_irradiance)
    
    # Get tree transmittance parameters for Beer-Lambert law calculations
    tree_k = kwargs.get("tree_k", 0.6)
    tree_lad = kwargs.get("tree_lad", 1.0)

    # Convert sun angles to 3D direction vector using spherical coordinates
    # Note: azimuth is adjusted by 180° to match coordinate system orientation
    azimuth_degrees = 180 - azimuth_degrees_ori
    azimuth_radians = np.deg2rad(azimuth_degrees)
    elevation_radians = np.deg2rad(elevation_degrees)
    
    # Calculate direction vector components
    # dx, dy: horizontal components, dz: vertical component (upward positive)
    dx = np.cos(elevation_radians) * np.cos(azimuth_radians)
    dy = np.cos(elevation_radians) * np.sin(azimuth_radians)
    dz = np.sin(elevation_radians)
    sun_direction = (dx, dy, dz)

    # Define obstacle detection parameters for ray tracing
    # All non-zero voxels are obstacles except for trees which have transmittance
    hit_values = (0,)    # Only sky voxels (value 0) are transparent
    inclusion_mode = False  # Values NOT in hit_values are considered obstacles

    # Compute transmittance map using optimized ray tracing
    transmittance_map = compute_direct_solar_irradiance_map_binary(
        voxel_data, sun_direction, view_point_height, hit_values, 
        meshsize, tree_k, tree_lad, inclusion_mode
    )

    # Scale transmittance by solar irradiance and geometry
    # For horizontal surfaces: multiply by sine of elevation angle
    sin_elev = dz
    direct_map = transmittance_map * direct_normal_irradiance * sin_elev

    # Optional visualization of results
    if show_plot:
        # Set up colormap with special handling for invalid data
        cmap = plt.cm.get_cmap(colormap).copy()
        cmap.set_bad(color='lightgray')  # NaN values shown in gray
        
        plt.figure(figsize=(10, 8))
        # plt.title("Horizontal Direct Solar Irradiance Map (0° = North)")
        plt.imshow(direct_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Direct Solar Irradiance (W/m²)')
        plt.axis('off')
        plt.show()

    # Optional export to 3D OBJ format for external visualization
    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        # Get export parameters with defaults
        dem_grid = kwargs.get("dem_grid", voxcity.dem.elevation if voxcity.dem else np.zeros_like(direct_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "direct_solar_irradiance")
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        
        # Export as colored 3D mesh
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

def get_diffuse_solar_irradiance_map(voxcity: VoxCity, diffuse_irradiance=1.0, show_plot=False, **kwargs):
    """
    Compute diffuse solar irradiance map using the Sky View Factor (SVF) with tree transmittance.

    This function calculates the diffuse component of solar radiation, which consists of
    sunlight scattered by the atmosphere and reaches surfaces from all directions across
    the sky hemisphere. The calculation is based on the Sky View Factor (SVF), which
    quantifies how much of the sky dome is visible from each location.
    
    Physics Background:
    - Diffuse radiation: Solar energy scattered by atmospheric particles and clouds
    - Sky View Factor (SVF): Fraction of sky hemisphere visible from a point (0.0 to 1.0)
    - Isotropic sky model: Assumes uniform diffuse radiation distribution across the sky
    - Tree effects: Partial transmittance through canopies reduces effective sky visibility
    
    SVF Characteristics:
    - SVF = 1.0: Completely open sky (maximum diffuse radiation)
    - SVF = 0.0: Completely blocked sky (no diffuse radiation)  
    - SVF = 0.5: Half of sky visible (typical for urban canyons)
    - Trees reduce SVF through partial light attenuation rather than complete blocking

    The function:
    1. Computes SVF map accounting for building shadows and tree transmittance
    2. Scales SVF by diffuse horizontal irradiance from weather data
    3. Optionally visualizes and exports results for analysis

    Args:
        voxcity (VoxCity): VoxCity object containing voxel data and metadata
        diffuse_irradiance (float): Diffuse horizontal irradiance in W/m² (from weather data)
                                  Default 1.0 for normalized calculations
        show_plot (bool): Whether to display visualization of results
        **kwargs: Additional arguments including:
            - view_point_height (float): Observer height in meters (default: 1.5)
                Height above ground where measurements are taken
            - colormap (str): Matplotlib colormap name for visualization (default: 'magma')
            - vmin (float): Minimum value for colormap scaling
            - vmax (float): Maximum value for colormap scaling
            - tree_k (float): Tree extinction coefficient for transmittance calculations
                Higher values mean trees block more diffuse light
            - tree_lad (float): Leaf area density in m^-1
                Affects light attenuation through tree canopies
            - obj_export (bool): Whether to export results as 3D OBJ file
            - output_directory (str): Directory for file exports
            - output_file_name (str): Base filename for exports
            - num_colors (int): Number of discrete colors for OBJ export  
            - alpha (float): Transparency value for 3D visualization

    Returns:
        ndarray: 2D array of diffuse solar irradiance values in W/m²
                - Values represent diffuse energy flux on horizontal surfaces
                - Range: 0.0 to diffuse_irradiance (input parameter)
                - NaN indicates invalid measurement locations
    
    Note:
        The SVF calculation internally handles tree transmittance effects, so trees
        contribute partial sky visibility rather than complete obstruction.
    """
    # Extract voxel data and meshsize from VoxCity object
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize

    # Extract parameters with defaults for observer and visualization settings
    view_point_height = kwargs.get("view_point_height", 1.5)
    colormap = kwargs.get("colormap", 'magma')
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", diffuse_irradiance)
    
    # Prepare parameters for SVF calculation with appropriate visualization settings
    # Pass tree transmittance parameters to SVF calculation
    svf_kwargs = kwargs.copy()
    svf_kwargs["colormap"] = "BuPu_r"  # Purple colormap for SVF visualization
    svf_kwargs["vmin"] = 0             # SVF ranges from 0 to 1
    svf_kwargs["vmax"] = 1

    # Calculate Sky View Factor map accounting for all obstructions
    # SVF calculation now handles tree transmittance internally
    SVF_map = get_sky_view_factor_map(voxcity, **svf_kwargs)
    
    # Convert SVF to diffuse irradiance by scaling with weather data
    # Each location receives diffuse radiation proportional to its sky visibility
    diffuse_map = SVF_map * diffuse_irradiance

    # Optional visualization of diffuse irradiance results
    if show_plot:
        # Use parameters from kwargs for consistent visualization
        vmin = kwargs.get("vmin", 0.0)
        vmax = kwargs.get("vmax", diffuse_irradiance)
        
        # Set up colormap with special handling for invalid data
        cmap = plt.cm.get_cmap(colormap).copy()
        cmap.set_bad(color='lightgray')  # NaN values shown in gray
        
        plt.figure(figsize=(10, 8))
        # plt.title("Diffuse Solar Irradiance Map")
        plt.imshow(diffuse_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Diffuse Solar Irradiance (W/m²)')
        plt.axis('off')
        plt.show()

    # Optional export to 3D OBJ format for external visualization
    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        # Get export parameters with defaults
        dem_grid = kwargs.get("dem_grid", voxcity.dem.elevation if voxcity.dem else np.zeros_like(diffuse_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "diffuse_solar_irradiance")
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        
        # Export as colored 3D mesh
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
    voxcity: VoxCity,
    azimuth_degrees,
    elevation_degrees,
    direct_normal_irradiance,
    diffuse_irradiance,
    show_plot=False,
    **kwargs
):
    """
    Compute global solar irradiance (direct + diffuse) on a horizontal plane at each valid observer location.

    This function combines both direct and diffuse components of solar radiation to calculate
    the total solar irradiance at each location. Global horizontal irradiance (GHI) is the
    most commonly used metric for solar energy assessment and represents the total solar
    energy available on a horizontal surface.
    
    Global Irradiance Components:
    - Direct component: Solar radiation from the sun's disk, affected by shadows and obstacles
    - Diffuse component: Solar radiation scattered by the atmosphere, affected by sky view
    - Total irradiance: Sum of direct and diffuse components at each location
    
    Physical Considerations:
    - Direct radiation varies with sun position and local obstructions
    - Diffuse radiation varies with sky visibility (Sky View Factor)
    - Both components are affected by tree transmittance using Beer-Lambert law
    - Invalid locations (e.g., on water, buildings) are marked as NaN

    The function:
    1. Computes direct solar irradiance map accounting for sun position and shadows
    2. Computes diffuse solar irradiance map based on Sky View Factor
    3. Combines maps and optionally visualizes/exports results for analysis

    Args:
        voxcity (VoxCity): VoxCity object containing voxel data and metadata
        azimuth_degrees (float): Sun azimuth angle in degrees (0° = North, 90° = East)
        elevation_degrees (float): Sun elevation angle in degrees above horizon (0-90°)
        direct_normal_irradiance (float): Direct normal irradiance in W/m² (from weather data)
        diffuse_irradiance (float): Diffuse horizontal irradiance in W/m² (from weather data)
        show_plot (bool): Whether to display visualization of results
        **kwargs: Additional arguments including:
            - view_point_height (float): Observer height in meters (default: 1.5)
                Height above ground where measurements are taken
            - colormap (str): Matplotlib colormap name for visualization (default: 'magma')
            - vmin (float): Minimum value for colormap scaling
            - vmax (float): Maximum value for colormap scaling
            - tree_k (float): Tree extinction coefficient for transmittance calculations
                Higher values mean trees block more light
            - tree_lad (float): Leaf area density in m^-1
                Affects light attenuation through tree canopies
            - obj_export (bool): Whether to export results as 3D OBJ file
            - output_directory (str): Directory for file exports
            - output_file_name (str): Base filename for exports
            - num_colors (int): Number of discrete colors for OBJ export
            - alpha (float): Transparency value for 3D visualization

    Returns:
        ndarray: 2D array of global solar irradiance values in W/m²
                - Values represent total solar energy flux on horizontal surfaces
                - Range: 0.0 to (direct_normal_irradiance * sin(elevation) + diffuse_irradiance)
                - NaN indicates invalid measurement locations
    
    Note:
        Global irradiance is the standard metric used for solar energy assessment
        and represents the maximum solar energy available at each location.
    """
    # Extract voxel data and meshsize from VoxCity object
    meshsize = voxcity.voxels.meta.meshsize
    
    # Extract visualization parameters
    colormap = kwargs.get("colormap", 'magma')

    # Create kwargs for individual component calculations
    # Both direct and diffuse calculations use the same base parameters
    direct_diffuse_kwargs = kwargs.copy()
    direct_diffuse_kwargs.update({
        'show_plot': True,   # Show intermediate results for debugging
        'obj_export': False  # Don't export intermediate results
    })

    # Compute direct irradiance component
    # Accounts for sun position, shadows, and tree transmittance
    direct_map = get_direct_solar_irradiance_map(
        voxcity,
        azimuth_degrees,
        elevation_degrees,
        direct_normal_irradiance,
        **direct_diffuse_kwargs
    )

    # Compute diffuse irradiance component  
    # Based on Sky View Factor and atmospheric scattering
    diffuse_map = get_diffuse_solar_irradiance_map(
        voxcity,
        diffuse_irradiance=diffuse_irradiance,
        **direct_diffuse_kwargs
    )

    # Sum the two components to get total global irradiance
    # This represents the total solar energy available at each location
    global_map = direct_map + diffuse_map

    # Determine colormap scaling range from actual data
    vmin = kwargs.get("vmin", np.nanmin(global_map))
    vmax = kwargs.get("vmax", np.nanmax(global_map))

    # Optional visualization of combined results
    if show_plot:
        # Set up colormap with special handling for invalid data
        cmap = plt.cm.get_cmap(colormap).copy()
        cmap.set_bad(color='lightgray')  # NaN values shown in gray
        
        plt.figure(figsize=(10, 8))
        # plt.title("Global Solar Irradiance Map")
        plt.imshow(global_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Global Solar Irradiance (W/m²)')
        plt.axis('off')
        plt.show()

    # Optional export to 3D OBJ format for external visualization
    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        # Get export parameters with defaults
        dem_grid = kwargs.get("dem_grid", voxcity.dem.elevation if voxcity.dem else np.zeros_like(global_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "global_solar_irradiance")
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        view_point_height = kwargs.get("view_point_height", 1.5)
        
        # Export as colored 3D mesh
        grid_to_obj(
            global_map,
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

    return global_map

def get_solar_positions_astral(times, lon, lat):
    """
    Compute solar azimuth and elevation using Astral for given times and location.
    
    This function uses the Astral astronomical library to calculate precise solar positions
    based on location coordinates and timestamps. The calculations account for Earth's
    orbital mechanics, axial tilt, and atmospheric refraction effects.
    
    Astronomical Background:
    - Solar position depends on date, time, and geographic location
    - Azimuth: Horizontal angle measured clockwise from North (0°-360°)
    - Elevation: Vertical angle above the horizon (-90° to +90°)
    - Calculations use standard astronomical algorithms (e.g., NREL SPA)
    
    Coordinate System:
    - Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
    - Elevation: 0° = horizon, 90° = zenith, negative values = below horizon
    - All angles are in degrees for consistency with weather data formats
    
    The function:
    1. Creates an Astral observer at the specified geographic location
    2. Computes sun position for each timestamp in the input array
    3. Returns DataFrame with azimuth and elevation angles for further processing
    
    Args:
        times (DatetimeIndex): Array of timezone-aware datetime objects
                              Must include timezone information for accurate calculations
        lon (float): Longitude in degrees (positive = East, negative = West)
                    Range: -180° to +180°
        lat (float): Latitude in degrees (positive = North, negative = South)
                    Range: -90° to +90°

    Returns:
        DataFrame: DataFrame with columns 'azimuth' and 'elevation' containing solar positions
                  - Index: Input timestamps (timezone-aware)
                  - 'azimuth': Solar azimuth angles in degrees (0°-360°)
                  - 'elevation': Solar elevation angles in degrees (-90° to +90°)
                  - All values are float type for numerical calculations
    
    Note:
        Input times must be timezone-aware. The function preserves the original
        timezone information and performs calculations in the specified timezone.
    """
    # Create an astronomical observer at the specified geographic location
    observer = Observer(latitude=lat, longitude=lon)
    
    # Initialize result DataFrame with appropriate structure
    df_pos = pd.DataFrame(index=times, columns=['azimuth', 'elevation'], dtype=float)

    # Calculate solar position for each timestamp
    for t in times:
        # t is already timezone-aware; no need to replace tzinfo
        # Calculate solar elevation (vertical angle above horizon)
        el = elevation(observer=observer, dateandtime=t)
        
        # Calculate solar azimuth (horizontal angle from North)
        az = azimuth(observer=observer, dateandtime=t)
        
        # Store results in DataFrame
        df_pos.at[t, 'elevation'] = el
        df_pos.at[t, 'azimuth'] = az

    return df_pos

def _configure_num_threads(desired_threads=None, progress=False):
    try:
        cores = os.cpu_count() or 4
    except Exception:
        cores = 4
    used = desired_threads if desired_threads is not None else cores
    try:
        numba.set_num_threads(int(used))
    except Exception:
        pass
    # Best-effort oversubscription guards (only set defaults if unset)
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = str(int(used))
    if progress:
        try:
            print(f"Numba threads: {numba.get_num_threads()} (requested {used})")
        except Exception:
            print(f"Numba threads set to {used}")
    return used

def _auto_time_batch_size(n_faces, total_steps, user_value=None):
    if user_value is not None:
        return max(1, int(user_value))
    # Heuristic based on face count
    if total_steps <= 0:
        return 1
    if n_faces <= 5_000:
        batches = 2
    elif n_faces <= 50_000:
        batches = 8
    elif n_faces <= 200_000:
        batches = 16
    else:
        batches = 32
    batches = min(batches, total_steps)
    return max(1, total_steps // batches)

def get_cumulative_global_solar_irradiance(
    voxcity: VoxCity,
    df, lon, lat, tz,
    direct_normal_irradiance_scaling=1.0,
    diffuse_irradiance_scaling=1.0,
    **kwargs
):
    """
    Compute cumulative global solar irradiance over a specified period using data from an EPW file.

    This function performs time-series analysis of solar irradiance by processing weather data
    over a user-defined period and accumulating irradiance values at each location. The result
    represents the total solar energy received during the specified time period, which is
    essential for seasonal analysis, solar panel positioning, and energy yield predictions.
    
    Cumulative Analysis Concept:
    - Instantaneous irradiance (W/m²): Power at a specific moment
    - Cumulative irradiance (Wh/m²): Energy accumulated over time
    - Integration: Sum of (irradiance × time_step) for all timesteps
    - Applications: Annual energy yield, seasonal variations, optimal siting
    
    Time Period Processing:
    - Supports flexible time ranges (daily, seasonal, annual analysis)
    - Handles timezone conversions between local and UTC time
    - Filters weather data based on user-specified start/end times
    - Accounts for leap years and varying daylight hours
    
    Performance Optimization:
    - Pre-calculates diffuse map once (scales linearly with DHI)
    - Processes direct component for each timestep (varies with sun position)
    - Uses efficient memory management for large time series
    - Provides optional progress monitoring for long calculations

    The function:
    1. Filters EPW data for specified time period with timezone handling
    2. Computes sun positions for each timestep using astronomical calculations
    3. Calculates and accumulates global irradiance maps over the entire period
    4. Handles tree transmittance and provides visualization/export options

    Args:
        voxcity (VoxCity): VoxCity object containing voxel data and metadata
        df (DataFrame): EPW weather data with columns 'DNI', 'DHI' and datetime index
                       Must include complete meteorological dataset
        lon (float): Longitude in degrees for solar position calculations
        lat (float): Latitude in degrees for solar position calculations  
        tz (float): Timezone offset in hours from UTC (positive = East of UTC)
        direct_normal_irradiance_scaling (float): Scaling factor for direct normal irradiance
                                                 Allows sensitivity analysis or unit conversions
        diffuse_irradiance_scaling (float): Scaling factor for diffuse horizontal irradiance
                                          Allows sensitivity analysis or unit conversions
        **kwargs: Additional arguments including:
            - view_point_height (float): Observer height in meters (default: 1.5)
                Height above ground where measurements are taken
            - start_time (str): Start time in format 'MM-DD HH:MM:SS'
                Defines beginning of analysis period (default: "01-01 05:00:00")
            - end_time (str): End time in format 'MM-DD HH:MM:SS'  
                Defines end of analysis period (default: "01-01 20:00:00")
            - tree_k (float): Tree extinction coefficient for transmittance calculations
                Higher values mean trees block more light
            - tree_lad (float): Leaf area density in m^-1
                Affects light attenuation through tree canopies
            - show_plot (bool): Whether to show final accumulated results
            - show_each_timestep (bool): Whether to show plots for each timestep
                Useful for debugging but significantly increases computation time
            - colormap (str): Matplotlib colormap name for visualization
            - vmin (float): Minimum value for colormap scaling
            - vmax (float): Maximum value for colormap scaling
            - obj_export (bool): Whether to export results as 3D OBJ file
            - output_directory (str): Directory for file exports
            - output_file_name (str): Base filename for exports
            - num_colors (int): Number of discrete colors for OBJ export
            - alpha (float): Transparency value for 3D visualization

    Returns:
        ndarray: 2D array of cumulative global solar irradiance values in W/m²·hour
                - Values represent total solar energy received during the analysis period
                - Range depends on period length and local climate conditions
                - NaN indicates invalid measurement locations (e.g., on buildings, water)
    
    Note:
        The function efficiently handles large time series by pre-computing the diffuse
        component once and scaling it for each timestep, significantly reducing
        computation time for long-term analysis.
    """
    # Extract voxel data and meshsize from VoxCity object
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    
    # Extract parameters with defaults for observer positioning and visualization
    view_point_height = kwargs.get("view_point_height", 1.5)
    colormap = kwargs.get("colormap", 'magma')
    start_time = kwargs.get("start_time", "01-01 05:00:00")
    end_time = kwargs.get("end_time", "01-01 20:00:00")
    # Optional: configure num threads here as well when called directly
    desired_threads = kwargs.get("numba_num_threads", None)
    progress_report = kwargs.get("progress_report", False)
    _configure_num_threads(desired_threads, progress=progress_report)

    # Validate input data
    if df.empty:
        raise ValueError("No data in EPW file.")

    # Parse start and end times without year (supports multi-year analysis)
    try:
        start_dt = datetime.strptime(start_time, "%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end_time, "%m-%d %H:%M:%S")
    except ValueError as ve:
        raise ValueError("start_time and end_time must be in format 'MM-DD HH:MM:SS'") from ve

    # Add hour of year column for efficient time filtering
    # Hour 1 = January 1st, 00:00; Hour 8760 = December 31st, 23:00
    df['hour_of_year'] = (df.index.dayofyear - 1) * 24 + df.index.hour + 1
    
    # Convert parsed dates to day of year and hour for filtering
    start_doy = datetime(2000, start_dt.month, start_dt.day).timetuple().tm_yday
    end_doy = datetime(2000, end_dt.month, end_dt.day).timetuple().tm_yday
    
    start_hour = (start_doy - 1) * 24 + start_dt.hour + 1
    end_hour = (end_doy - 1) * 24 + end_dt.hour + 1

    # Handle period crossing year boundary (e.g., Dec 15 to Jan 15)
    if start_hour <= end_hour:
        # Normal period within single year
        df_period = df[(df['hour_of_year'] >= start_hour) & (df['hour_of_year'] <= end_hour)]
    else:
        # Period crosses year boundary - include end and beginning of year
        df_period = df[(df['hour_of_year'] >= start_hour) | (df['hour_of_year'] <= end_hour)]

    # Apply minute-level filtering within start/end hours for precision
    df_period = df_period[
        ((df_period.index.hour != start_dt.hour) | (df_period.index.minute >= start_dt.minute)) &
        ((df_period.index.hour != end_dt.hour) | (df_period.index.minute <= end_dt.minute))
    ]

    # Validate filtered data
    if df_period.empty:
        raise ValueError("No EPW data in the specified period.")

    # Handle timezone conversion for accurate solar position calculations
    # Convert local time (from EPW) to UTC for astronomical calculations
    offset_minutes = int(tz * 60)
    local_tz = pytz.FixedOffset(offset_minutes)
    df_period_local = df_period.copy()
    df_period_local.index = df_period_local.index.tz_localize(local_tz)
    df_period_utc = df_period_local.tz_convert(pytz.UTC)

    # Compute solar positions for entire analysis period
    # This is done once to optimize performance
    solar_positions = get_solar_positions_astral(df_period_utc.index, lon, lat)

    # Prepare parameters for efficient diffuse irradiance calculation
    # Create kwargs for diffuse calculation with visualization disabled
    diffuse_kwargs = kwargs.copy()
    diffuse_kwargs.update({
        'show_plot': False,
        'obj_export': False
    })

    # Pre-compute base diffuse map once with unit irradiance
    # This map will be scaled by actual DHI values for each timestep
    base_diffuse_map = get_diffuse_solar_irradiance_map(
        voxcity,
        diffuse_irradiance=1.0,
        **diffuse_kwargs
    )

    # Initialize accumulation arrays for energy integration
    cumulative_map = np.zeros((voxel_data.shape[0], voxel_data.shape[1]))
    mask_map = np.ones((voxel_data.shape[0], voxel_data.shape[1]), dtype=bool)

    # Prepare parameters for direct irradiance calculations
    # Create kwargs for direct calculation with visualization disabled
    direct_kwargs = kwargs.copy()
    direct_kwargs.update({
        'show_plot': False,
        'view_point_height': view_point_height,
        'obj_export': False
    })

    # Main processing loop: iterate through each timestep in the analysis period
    for idx, (time_utc, row) in enumerate(df_period_utc.iterrows()):
        # Apply scaling factors to weather data
        # Allows for sensitivity analysis or unit conversions
        DNI = row['DNI'] * direct_normal_irradiance_scaling
        DHI = row['DHI'] * diffuse_irradiance_scaling
        time_local = df_period_local.index[idx]

        # Get solar position for timestep
        solpos = solar_positions.loc[time_utc]
        azimuth_degrees = solpos['azimuth']
        elevation_degrees = solpos['elevation']        

        # Compute direct irradiance map with transmittance
        direct_map = get_direct_solar_irradiance_map(
            voxcity,
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
        dem_grid = kwargs.get("dem_grid", voxcity.dem.elevation if voxcity.dem else np.zeros_like(cumulative_map))
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
    voxcity: VoxCity,
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
        voxcity (VoxCity): VoxCity object containing voxel data and metadata
        calc_type (str): 'instantaneous' or 'cumulative'.
        direct_normal_irradiance_scaling (float): Scaling factor for direct normal irradiance.
        diffuse_irradiance_scaling (float): Scaling factor for diffuse horizontal irradiance.
        **kwargs: Additional arguments including:
            - download_nearest_epw (bool): Whether to download nearest EPW file
            - epw_file_path (str): Path to EPW file
            - rectangle_vertices (list, optional): List of (lat,lon) coordinates for EPW download. If not provided, extracted from voxcity.extras
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
    # Extract voxel data and meshsize from VoxCity object
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    
    view_point_height = kwargs.get("view_point_height", 1.5)
    colormap = kwargs.get("colormap", 'magma')

    # Get EPW file
    download_nearest_epw = kwargs.get("download_nearest_epw", False)
    # Extract rectangle_vertices from VoxCity object if not provided in kwargs
    rectangle_vertices = kwargs.get("rectangle_vertices", None)
    if rectangle_vertices is None:
        rectangle_vertices = voxcity.extras.get("rectangle_vertices", None)
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
                load_data=True,
                allow_insecure_ssl=kwargs.get("allow_insecure_ssl", False),
                allow_http_fallback=kwargs.get("allow_http_fallback", False),
                ssl_verify=kwargs.get("ssl_verify", True)
            )

    # Read EPW data
    if epw_file_path is None:
        raise RuntimeError("EPW file path is None. Set 'epw_file_path' or enable 'download_nearest_epw' and ensure network succeeds.")
    df, lon, lat, tz, elevation_m = read_epw_for_solar_simulation(epw_file_path)
    if df.empty:
        raise ValueError("No data in EPW file.")

    if calc_type == 'instantaneous':

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
        solar_positions = get_solar_positions_astral(df_period_utc.index, lon, lat)
        direct_normal_irradiance = df_period_utc.iloc[0]['DNI']
        diffuse_irradiance = df_period_utc.iloc[0]['DHI']
        azimuth_degrees = solar_positions.iloc[0]['azimuth']
        elevation_degrees = solar_positions.iloc[0]['elevation']    
        solar_map = get_global_solar_irradiance_map(
            voxcity,                    # VoxCity object
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
            voxcity,
            df_filtered, lon, lat, tz,
            **kwargs
        )
    
    return solar_map 

import numpy as np
import trimesh
import time
from numba import njit, prange

##############################################################################
# 1) New Numba helper: per-face solar irradiance computation
##############################################################################
@njit(parallel=True)
def compute_solar_irradiance_for_all_faces(
    face_centers,
    face_normals,
    face_svf,
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
    for each face in a 3D building mesh.
    
    This optimized function processes all mesh faces in parallel to calculate solar
    irradiance components. It handles both direct radiation (dependent on sun position
    and surface orientation) and diffuse radiation (dependent on sky visibility).
    The function is compiled with Numba for high-performance computation on large meshes.
    
    Surface Irradiance Physics:
    - Direct component: DNI × cos(incidence_angle) × transmittance
    - Diffuse component: DHI × sky_view_factor
    - Incidence angle: Angle between sun direction and surface normal
    - Transmittance: Attenuation factor from obstacles and vegetation
    
    Boundary Condition Handling:
    - Vertical boundary faces are excluded (mesh edges touching domain boundaries)
    - Invalid faces (NaN SVF) are skipped to maintain data consistency
    - Surface orientation affects direct radiation calculation
    
    Performance Optimizations:
    - Numba JIT compilation for near C-speed execution
    - Parallel processing of face calculations
    - Efficient geometric computations using vectorized operations
    - Memory-optimized array operations
    
    Args:
        face_centers (float64[:, :]): (N x 3) array of face center coordinates in real-world units
        face_normals (float64[:, :]): (N x 3) array of outward-pointing unit normal vectors
        face_svf (float64[:]): (N,) array of Sky View Factor values for each face (0.0-1.0)
        sun_direction (float64[:]): (3,) array for normalized sun direction vector (dx, dy, dz)
        direct_normal_irradiance (float): Direct normal irradiance (DNI) in W/m²
        diffuse_irradiance (float): Diffuse horizontal irradiance (DHI) in W/m²
        voxel_data (ndarray): 3D array of voxel values for obstacle detection
        meshsize (float): Size of each voxel in meters (spatial resolution)
        tree_k (float): Tree extinction coefficient for Beer-Lambert law
        tree_lad (float): Leaf area density in m^-1
        hit_values (tuple): Values considered 'sky' for ray tracing (e.g. (0,))
        inclusion_mode (bool): Whether hit_values are included (True) or excluded (False)
        grid_bounds_real (float64[2,3]): Domain boundaries [[x_min,y_min,z_min],[x_max,y_max,z_max]]
        boundary_epsilon (float): Distance threshold for boundary face detection
    
    Returns:
        tuple: Three float64[N] arrays containing:
            - direct_irr: Direct solar irradiance for each face (W/m²)
            - diffuse_irr: Diffuse solar irradiance for each face (W/m²)  
            - global_irr: Global solar irradiance for each face (W/m²)
    
    Note:
        This function is optimized with Numba and should not be called directly.
        Use the higher-level wrapper functions for normal operation.
    """
    n_faces = face_centers.shape[0]
    
    # Initialize output arrays for each irradiance component
    face_direct = np.zeros(n_faces, dtype=np.float64)
    face_diffuse = np.zeros(n_faces, dtype=np.float64)
    face_global = np.zeros(n_faces, dtype=np.float64)
    
    # Extract domain boundaries for boundary face detection
    x_min, y_min, z_min = grid_bounds_real[0, 0], grid_bounds_real[0, 1], grid_bounds_real[0, 2]
    x_max, y_max, z_max = grid_bounds_real[1, 0], grid_bounds_real[1, 1], grid_bounds_real[1, 2]
    
    # Process each face individually (Numba optimizes this loop)
    for fidx in prange(n_faces):
        center = face_centers[fidx]
        normal = face_normals[fidx]
        svf    = face_svf[fidx]
        
        # Check for vertical boundary faces that should be excluded
        # These are mesh edges at domain boundaries, not actual building surfaces
        is_vertical = (abs(normal[2]) < 0.01)  # Nearly vertical normal
        
        # Check if face center is at domain boundary
        on_x_min = (abs(center[0] - x_min) < boundary_epsilon)
        on_y_min = (abs(center[1] - y_min) < boundary_epsilon)
        on_x_max = (abs(center[0] - x_max) < boundary_epsilon)
        on_y_max = (abs(center[1] - y_max) < boundary_epsilon)
        
        is_boundary_vertical = is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max)
        
        # Skip boundary faces to avoid artifacts
        if is_boundary_vertical:
            face_direct[fidx]  = np.nan
            face_diffuse[fidx] = np.nan
            face_global[fidx]  = np.nan
            continue
        
        # Skip faces with invalid SVF data
        if svf != svf:  # NaN check in Numba-compatible way
            face_direct[fidx]  = np.nan
            face_diffuse[fidx] = np.nan
            face_global[fidx]  = np.nan
            continue
        
        # Calculate direct irradiance component
        # Only surfaces oriented towards the sun receive direct radiation
        cos_incidence = normal[0]*sun_direction[0] + \
                        normal[1]*sun_direction[1] + \
                        normal[2]*sun_direction[2]
        
        direct_val = 0.0
        if cos_incidence > 0.0:  # Surface faces towards sun
            # Offset ray origin slightly along normal to avoid self-intersection
            offset_vox = 0.1  # Small offset in voxel units
            ray_origin_x = center[0]/meshsize + normal[0]*offset_vox
            ray_origin_y = center[1]/meshsize + normal[1]*offset_vox
            ray_origin_z = center[2]/meshsize + normal[2]*offset_vox
            
            # Cast ray toward the sun to check for obstructions
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
            
            # Calculate direct irradiance if path to sun is clear/partially clear
            if not hit_detected:
                direct_val = direct_normal_irradiance * cos_incidence * transmittance
        
        # Calculate diffuse irradiance component using Sky View Factor
        # All surfaces receive diffuse radiation proportional to their sky visibility
        diffuse_val = svf * diffuse_irradiance
        
        # Ensure diffuse irradiance doesn't exceed theoretical maximum
        if diffuse_val > diffuse_irradiance:
            diffuse_val = diffuse_irradiance
        
        # Store results for this face
        face_direct[fidx]  = direct_val
        face_diffuse[fidx] = diffuse_val
        face_global[fidx]  = direct_val + diffuse_val
    
    return face_direct, face_diffuse, face_global


##############################################################################
# 2) Modified get_building_solar_irradiance: main Python wrapper
##############################################################################
def get_building_solar_irradiance(
    voxcity: VoxCity,
    building_svf_mesh,
    azimuth_degrees,
    elevation_degrees,
    direct_normal_irradiance,
    diffuse_irradiance,
    **kwargs
):
    """
    Calculate solar irradiance on building surfaces using Sky View Factor (SVF) analysis,
    with high-performance computation accelerated by Numba JIT compilation.
    
    This function performs detailed solar irradiance analysis on 3D building surfaces
    represented as triangulated meshes. It calculates both direct and diffuse components
    of solar radiation for each mesh face, accounting for surface orientation, shadows,
    and sky visibility. The computation is optimized for large urban models using
    efficient algorithms and parallel processing.
    
    Mesh-Based Analysis Advantages:
    - Surface-specific calculations for facades, roofs, and complex geometries
    - Accurate accounting of surface orientation and local shading effects
    - Integration with 3D visualization and CAD workflows
    - Detailed irradiance data for building energy modeling
    
    Performance Features:
    - Numba JIT compilation for near C-speed execution
    - Parallel processing of mesh faces
    - Efficient ray tracing with tree transmittance
    - Memory-optimized operations for large datasets
    
    Physical Modeling:
    - Direct irradiance: Based on sun position and surface orientation
    - Diffuse irradiance: Based on Sky View Factor from each surface
    - Tree effects: Partial transmittance using Beer-Lambert law
    - Boundary handling: Automatic exclusion of domain boundary artifacts
    
    Args:
        voxcity (VoxCity): VoxCity object containing voxel data and metadata
        building_svf_mesh (trimesh.Trimesh): Building mesh with pre-calculated SVF values in metadata
                                           Must have 'svf' array in mesh.metadata
        azimuth_degrees (float): Sun azimuth angle in degrees (0=North, 90=East)
        elevation_degrees (float): Sun elevation angle in degrees above horizon (0-90°)
        direct_normal_irradiance (float): Direct normal irradiance (DNI) in W/m² from weather data
        diffuse_irradiance (float): Diffuse horizontal irradiance (DHI) in W/m² from weather data
        **kwargs: Additional parameters including:
            - tree_k (float): Tree extinction coefficient (default: 0.6)
                Higher values mean trees block more light
            - tree_lad (float): Leaf area density in m^-1 (default: 1.0)
                Affects light attenuation through tree canopies
            - progress_report (bool): Whether to print timing information (default: False)
            - obj_export (bool): Whether to export results as OBJ file
            - output_directory (str): Directory for file exports
            - output_file_name (str): Base filename for exports
    
    Returns:
        trimesh.Trimesh: A copy of the input mesh with irradiance data stored in metadata:
                        - 'svf': Sky View Factor for each face (preserved from input)
                        - 'direct': Direct solar irradiance for each face (W/m²)
                        - 'diffuse': Diffuse solar irradiance for each face (W/m²)
                        - 'global': Global solar irradiance for each face (W/m²)
    
    Note:
        The input mesh must have SVF values pre-calculated and stored in metadata.
        Use get_surface_view_factor() to compute SVF before calling this function.
    """
    import time
    
    # Extract voxel data and meshsize from VoxCity object
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    
    # Extract tree transmittance parameters with defaults
    tree_k          = kwargs.get("tree_k", 0.6)
    tree_lad        = kwargs.get("tree_lad", 1.0)
    progress_report = kwargs.get("progress_report", False)
    
    # Define sky detection parameters for ray tracing
    hit_values     = (0,)    # '0' = sky voxel value
    inclusion_mode = False   # Treat non-sky values as obstacles
    
    # Convert solar angles to 3D direction vector using spherical coordinates
    az_rad = np.deg2rad(180 - azimuth_degrees)  # Adjust for coordinate system
    el_rad = np.deg2rad(elevation_degrees)
    sun_dx = np.cos(el_rad) * np.cos(az_rad)
    sun_dy = np.cos(el_rad) * np.sin(az_rad)
    sun_dz = np.sin(el_rad)
    sun_direction = np.array([sun_dx, sun_dy, sun_dz], dtype=np.float64)
    
    # Extract mesh geometry data for processing (optionally from cache)
    precomputed_geometry = kwargs.get("precomputed_geometry", None)
    if precomputed_geometry is not None:
        face_centers = precomputed_geometry.get("face_centers", building_svf_mesh.triangles_center)
        face_normals = precomputed_geometry.get("face_normals", building_svf_mesh.face_normals)
        face_svf_opt = precomputed_geometry.get("face_svf", None)
        grid_bounds_real = precomputed_geometry.get("grid_bounds_real", None)
        boundary_epsilon = precomputed_geometry.get("boundary_epsilon", None)
    else:
        face_centers = building_svf_mesh.triangles_center  # Center point of each face
        face_normals = building_svf_mesh.face_normals      # Normal vector for each face
    
    # Extract Sky View Factor data from mesh metadata
    if hasattr(building_svf_mesh, 'metadata') and ('svf' in building_svf_mesh.metadata):
        face_svf = building_svf_mesh.metadata['svf']
    else:
        # Initialize with zeros if SVF not available (should be pre-calculated)
        face_svf = np.zeros(len(building_svf_mesh.faces), dtype=np.float64)
    
    # Set up domain boundaries for boundary face detection (use cache if available)
    if grid_bounds_real is None or boundary_epsilon is None:
        grid_shape = voxel_data.shape
        grid_bounds_voxel = np.array([[0,0,0],[grid_shape[0], grid_shape[1], grid_shape[2]]], dtype=np.float64)
        grid_bounds_real = grid_bounds_voxel * meshsize
        boundary_epsilon = meshsize * 0.05  # Small tolerance for boundary detection
    
    # Optional fast path using masked DDA kernel
    fast_path = kwargs.get("fast_path", True)
    precomputed_masks = kwargs.get("precomputed_masks", None)
    t0 = time.time()
    if fast_path:
        # Prepare masks (reuse cache when possible)
        if precomputed_masks is not None:
            vox_is_tree = precomputed_masks.get("vox_is_tree", (voxel_data == -2))
            vox_is_opaque = precomputed_masks.get("vox_is_opaque", (voxel_data != 0) & (voxel_data != -2))
            att = float(precomputed_masks.get("att", np.exp(-tree_k * tree_lad * meshsize)))
        else:
            vox_is_tree = (voxel_data == -2)
            vox_is_opaque = (voxel_data != 0) & (~vox_is_tree)
            att = float(np.exp(-tree_k * tree_lad * meshsize))

        face_direct, face_diffuse, face_global = compute_solar_irradiance_for_all_faces_masked(
            face_centers.astype(np.float64),
            face_normals.astype(np.float64),
            face_svf.astype(np.float64),
            sun_direction.astype(np.float64),
            float(direct_normal_irradiance),
            float(diffuse_irradiance),
            vox_is_tree,
            vox_is_opaque,
            float(meshsize),
            att,
            float(grid_bounds_real[0,0]), float(grid_bounds_real[0,1]), float(grid_bounds_real[0,2]),
            float(grid_bounds_real[1,0]), float(grid_bounds_real[1,1]), float(grid_bounds_real[1,2]),
            float(boundary_epsilon)
        )
    else:
        face_direct, face_diffuse, face_global = compute_solar_irradiance_for_all_faces(
            face_centers,
            face_normals,
            face_svf_opt if (precomputed_geometry is not None and face_svf_opt is not None) else face_svf,
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
    
    # Report performance timing if requested
    if progress_report:
        elapsed = time.time() - t0
        print(f"Numba-based solar irradiance calculation took {elapsed:.2f} seconds")
    
    # Create a copy of the input mesh to store results
    irradiance_mesh = building_svf_mesh.copy()
    if not hasattr(irradiance_mesh, 'metadata'):
        irradiance_mesh.metadata = {}
    
    # Store results
    irradiance_mesh.metadata['svf']    = face_svf
    irradiance_mesh.metadata['direct'] = face_direct
    irradiance_mesh.metadata['diffuse'] = face_diffuse
    irradiance_mesh.metadata['global'] = face_global
    
    irradiance_mesh.name = "Solar Irradiance (W/m²)"
    
    # # Optional OBJ export
    # obj_export = kwargs.get("obj_export", False)
    # if obj_export:
    #     # Get export parameters
    #     output_dir = kwargs.get("output_directory", "output")
    #     output_file_name = kwargs.get("output_file_name", "solar_irradiance")

    #     # Export the mesh directly
    #     irradiance_mesh.export(f"{output_dir}/{output_file_name}.obj")
    
    return irradiance_mesh

##############################################################################
# 2.5) Specialized masked DDA for per-face irradiance (parallel)
##############################################################################
@njit(cache=True, fastmath=True, nogil=True)
def _trace_direct_masked(vox_is_tree, vox_is_opaque, origin, direction, att, att_cutoff=0.01):
    nx, ny, nz = vox_is_opaque.shape
    x0 = origin[0]; y0 = origin[1]; z0 = origin[2]
    dx = direction[0]; dy = direction[1]; dz = direction[2]

    # Normalize
    L = (dx*dx + dy*dy + dz*dz) ** 0.5
    if L == 0.0:
        return False, 1.0
    invL = 1.0 / L
    dx *= invL; dy *= invL; dz *= invL

    # Start at voxel centers
    x = x0 + 0.5; y = y0 + 0.5; z = z0 + 0.5
    i = int(x0); j = int(y0); k = int(z0)

    step_x = 1 if dx >= 0.0 else -1
    step_y = 1 if dy >= 0.0 else -1
    step_z = 1 if dz >= 0.0 else -1

    BIG = 1e30
    if dx != 0.0:
        t_max_x = (((i + (1 if step_x > 0 else 0)) - x) / dx)
        t_delta_x = abs(1.0 / dx)
    else:
        t_max_x = BIG; t_delta_x = BIG
    if dy != 0.0:
        t_max_y = (((j + (1 if step_y > 0 else 0)) - y) / dy)
        t_delta_y = abs(1.0 / dy)
    else:
        t_max_y = BIG; t_delta_y = BIG
    if dz != 0.0:
        t_max_z = (((k + (1 if step_z > 0 else 0)) - z) / dz)
        t_delta_z = abs(1.0 / dz)
    else:
        t_max_z = BIG; t_delta_z = BIG

    T = 1.0
    while True:
        if (i < 0) or (i >= nx) or (j < 0) or (j >= ny) or (k < 0) or (k >= nz):
            # Exited grid: not blocked
            return False, T

        if vox_is_opaque[i, j, k]:
            # Hit opaque (non-sky, non-tree)
            return True, T

        if vox_is_tree[i, j, k]:
            T *= att
            if T < att_cutoff:
                # Consider fully attenuated
                return True, T

        # Step DDA
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                t_max_x += t_delta_x; i += step_x
            else:
                t_max_z += t_delta_z; k += step_z
        else:
            if t_max_y < t_max_z:
                t_max_y += t_delta_y; j += step_y
            else:
                t_max_z += t_delta_z; k += step_z

@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def compute_solar_irradiance_for_all_faces_masked(
    face_centers,
    face_normals,
    face_svf,
    sun_direction,
    direct_normal_irradiance,
    diffuse_irradiance,
    vox_is_tree,
    vox_is_opaque,
    meshsize,
    att,
    x_min, y_min, z_min,
    x_max, y_max, z_max,
    boundary_epsilon
):
    n_faces = face_centers.shape[0]
    face_direct = np.zeros(n_faces, dtype=np.float64)
    face_diffuse = np.zeros(n_faces, dtype=np.float64)
    face_global = np.zeros(n_faces, dtype=np.float64)

    for fidx in prange(n_faces):
        center = face_centers[fidx]
        normal = face_normals[fidx]
        svf = face_svf[fidx]

        # Boundary vertical exclusion
        is_vertical = (abs(normal[2]) < 0.01)
        on_x_min = (abs(center[0] - x_min) < boundary_epsilon)
        on_y_min = (abs(center[1] - y_min) < boundary_epsilon)
        on_x_max = (abs(center[0] - x_max) < boundary_epsilon)
        on_y_max = (abs(center[1] - y_max) < boundary_epsilon)
        if is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max):
            face_direct[fidx] = np.nan
            face_diffuse[fidx] = np.nan
            face_global[fidx] = np.nan
            continue

        if svf != svf:
            face_direct[fidx] = np.nan
            face_diffuse[fidx] = np.nan
            face_global[fidx] = np.nan
            continue

        # Direct component
        cos_incidence = normal[0]*sun_direction[0] + normal[1]*sun_direction[1] + normal[2]*sun_direction[2]
        direct_val = 0.0
        if cos_incidence > 0.0 and direct_normal_irradiance > 0.0:
            offset_vox = 0.1
            ox = center[0]/meshsize + normal[0]*offset_vox
            oy = center[1]/meshsize + normal[1]*offset_vox
            oz = center[2]/meshsize + normal[2]*offset_vox
            blocked, T = _trace_direct_masked(
                vox_is_tree,
                vox_is_opaque,
                np.array((ox, oy, oz), dtype=np.float64),
                sun_direction,
                att
            )
            if not blocked:
                direct_val = direct_normal_irradiance * cos_incidence * T

        # Diffuse component
        diffuse_val = svf * diffuse_irradiance
        if diffuse_val > diffuse_irradiance:
            diffuse_val = diffuse_irradiance

        face_direct[fidx] = direct_val
        face_diffuse[fidx] = diffuse_val
        face_global[fidx] = direct_val + diffuse_val

    return face_direct, face_diffuse, face_global

@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def compute_cumulative_solar_irradiance_faces_masked_timeseries(
    face_centers,
    face_normals,
    face_svf,
    sun_dirs_arr,      # shape (T, 3)
    DNI_arr,           # shape (T,)
    DHI_arr,           # shape (T,)
    vox_is_tree,
    vox_is_opaque,
    meshsize,
    att,
    x_min, y_min, z_min,
    x_max, y_max, z_max,
    boundary_epsilon,
    t_start, t_end,              # [start, end) indices
    time_step_hours
):
    n_faces = face_centers.shape[0]
    out_dir  = np.zeros(n_faces, dtype=np.float64)
    out_diff = np.zeros(n_faces, dtype=np.float64)
    out_glob = np.zeros(n_faces, dtype=np.float64)

    for fidx in prange(n_faces):
        center = face_centers[fidx]
        normal = face_normals[fidx]
        svf = face_svf[fidx]

        # Boundary vertical exclusion
        is_vertical = (abs(normal[2]) < 0.01)
        on_x_min = (abs(center[0] - x_min) < boundary_epsilon)
        on_y_min = (abs(center[1] - y_min) < boundary_epsilon)
        on_x_max = (abs(center[0] - x_max) < boundary_epsilon)
        on_y_max = (abs(center[1] - y_max) < boundary_epsilon)
        if is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max):
            out_dir[fidx]  = np.nan
            out_diff[fidx] = np.nan
            out_glob[fidx] = np.nan
            continue

        if svf != svf:
            out_dir[fidx]  = np.nan
            out_diff[fidx] = np.nan
            out_glob[fidx] = np.nan
            continue

        accum_dir = 0.0
        accum_diff = 0.0
        accum_glob = 0.0

        # Precompute ray origin (voxel coords) once per face
        offset_vox = 0.1
        ox = center[0]/meshsize + normal[0]*offset_vox
        oy = center[1]/meshsize + normal[1]*offset_vox
        oz = center[2]/meshsize + normal[2]*offset_vox
        origin = np.array((ox, oy, oz), dtype=np.float64)

        for t in range(t_start, t_end):
            dni = DNI_arr[t]
            dhi = DHI_arr[t]
            sd0 = sun_dirs_arr[t, 0]
            sd1 = sun_dirs_arr[t, 1]
            sd2 = sun_dirs_arr[t, 2]
            # Skip below horizon quickly: dz <= 0 implies elevation<=0
            if sd2 <= 0.0:
                # diffuse only
                diff_val = svf * dhi
                if diff_val > dhi:
                    diff_val = dhi
                accum_diff += diff_val * time_step_hours
                accum_glob += diff_val * time_step_hours
                continue

            # Direct
            cos_inc = normal[0]*sd0 + normal[1]*sd1 + normal[2]*sd2
            direct_val = 0.0
            if (dni > 0.0) and (cos_inc > 0.0):
                blocked, T = _trace_direct_masked(
                    vox_is_tree,
                    vox_is_opaque,
                    origin,
                    np.array((sd0, sd1, sd2), dtype=np.float64),
                    att
                )
                if not blocked:
                    direct_val = dni * cos_inc * T

            diff_val = svf * dhi
            if diff_val > dhi:
                diff_val = dhi

            accum_dir  += direct_val * time_step_hours
            accum_diff += diff_val   * time_step_hours
            accum_glob += (direct_val + diff_val) * time_step_hours

        out_dir[fidx]  = accum_dir
        out_diff[fidx] = accum_diff
        out_glob[fidx] = accum_glob

    return out_dir, out_diff, out_glob

##############################################################################
# 4) Modified get_cumulative_building_solar_irradiance
##############################################################################
def get_cumulative_building_solar_irradiance(
    voxcity: VoxCity,
    building_svf_mesh,
    weather_df,
    lon, lat, tz,
    **kwargs
):
    """
    Calculate cumulative solar irradiance on building surfaces over a time period.
    Uses the Numba-accelerated get_building_solar_irradiance for each time step.
    
    Args:
        voxcity (VoxCity): VoxCity object containing voxel data and metadata
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
    import numpy as np
    
    # Extract voxel data and meshsize from VoxCity object
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    
    period_start = kwargs.get("period_start", "01-01 00:00:00")
    period_end   = kwargs.get("period_end",   "12-31 23:59:59")
    time_step_hours = kwargs.get("time_step_hours", 1.0)
    direct_normal_irradiance_scaling = kwargs.get("direct_normal_irradiance_scaling", 1.0)
    diffuse_irradiance_scaling       = kwargs.get("diffuse_irradiance_scaling", 1.0)
    progress_report = kwargs.get("progress_report", False)
    fast_path = kwargs.get("fast_path", True)
    
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
    
    # Get solar positions (allow precomputed to avoid recomputation)
    precomputed_solar_positions = kwargs.get("precomputed_solar_positions", None)
    if precomputed_solar_positions is not None and len(precomputed_solar_positions) == len(df_period_utc.index):
        solar_positions = precomputed_solar_positions
    else:
        solar_positions = get_solar_positions_astral(df_period_utc.index, lon, lat)

    # Precompute arrays to avoid per-iteration pandas operations
    times_len = len(df_period_utc.index)
    azimuth_deg_arr = solar_positions['azimuth'].to_numpy()
    elev_deg_arr = solar_positions['elevation'].to_numpy()
    az_rad_arr = np.deg2rad(180.0 - azimuth_deg_arr)
    el_rad_arr = np.deg2rad(elev_deg_arr)
    sun_dx_arr = np.cos(el_rad_arr) * np.cos(az_rad_arr)
    sun_dy_arr = np.cos(el_rad_arr) * np.sin(az_rad_arr)
    sun_dz_arr = np.sin(el_rad_arr)
    sun_dirs_arr = np.stack([sun_dx_arr, sun_dy_arr, sun_dz_arr], axis=1).astype(np.float64)
    DNI_arr = (df_period_utc['DNI'].to_numpy() * direct_normal_irradiance_scaling).astype(np.float64)
    DHI_arr = (df_period_utc['DHI'].to_numpy() * diffuse_irradiance_scaling).astype(np.float64)
    sun_above_mask = elev_deg_arr > 0.0
    
    # Prepare arrays for accumulation
    n_faces = len(building_svf_mesh.faces)
    face_cum_direct  = np.zeros(n_faces, dtype=np.float64)
    face_cum_diffuse = np.zeros(n_faces, dtype=np.float64)
    face_cum_global  = np.zeros(n_faces, dtype=np.float64)

    # Pre-extract mesh face arrays and domain bounds for fast path
    # Optionally reuse precomputed geometry/bounds
    precomputed_geometry = kwargs.get("precomputed_geometry", None)
    if precomputed_geometry is not None:
        face_centers = precomputed_geometry.get("face_centers", building_svf_mesh.triangles_center)
        face_normals = precomputed_geometry.get("face_normals", building_svf_mesh.face_normals)
        face_svf = precomputed_geometry.get(
            "face_svf",
            building_svf_mesh.metadata['svf'] if ('svf' in building_svf_mesh.metadata) else np.zeros(n_faces, dtype=np.float64)
        )
        grid_bounds_real = precomputed_geometry.get("grid_bounds_real", None)
        boundary_epsilon = precomputed_geometry.get("boundary_epsilon", None)
    else:
        face_centers = building_svf_mesh.triangles_center
        face_normals = building_svf_mesh.face_normals
        face_svf = building_svf_mesh.metadata['svf'] if ('svf' in building_svf_mesh.metadata) else np.zeros(n_faces, dtype=np.float64)
        grid_bounds_real = None
        boundary_epsilon = None

    if grid_bounds_real is None or boundary_epsilon is None:
        grid_shape = voxel_data.shape
        grid_bounds_voxel = np.array([[0, 0, 0], [grid_shape[0], grid_shape[1], grid_shape[2]]], dtype=np.float64)
        grid_bounds_real = grid_bounds_voxel * meshsize
        boundary_epsilon = meshsize * 0.05

    # Params used in Numba kernel
    hit_values = (0,)      # sky
    inclusion_mode = False # any non-sky is obstacle but trees transmit
    tree_k = kwargs.get("tree_k", 0.6)
    tree_lad = kwargs.get("tree_lad", 1.0)

    boundary_mask = None
    instant_kwargs = kwargs.copy()
    instant_kwargs['obj_export'] = False

    total_steps = times_len
    progress_every = max(1, total_steps // 20)  # ~5% steps

    # Pre-cast stable arrays to avoid repeated allocations
    face_centers64 = (face_centers if isinstance(face_centers, np.ndarray) else building_svf_mesh.triangles_center).astype(np.float64)
    face_normals64 = (face_normals if isinstance(face_normals, np.ndarray) else building_svf_mesh.face_normals).astype(np.float64)
    face_svf64 = face_svf.astype(np.float64)
    x_min, y_min, z_min = grid_bounds_real[0, 0], grid_bounds_real[0, 1], grid_bounds_real[0, 2]
    x_max, y_max, z_max = grid_bounds_real[1, 0], grid_bounds_real[1, 1], grid_bounds_real[1, 2]

    if fast_path:
        # Use masked cumulative kernel with chunking to minimize Python overhead
        precomputed_masks = kwargs.get("precomputed_masks", None)
        if precomputed_masks is not None:
            vox_is_tree = precomputed_masks.get("vox_is_tree", (voxel_data == -2))
            vox_is_opaque = precomputed_masks.get("vox_is_opaque", (voxel_data != 0) & (voxel_data != -2))
            att = float(precomputed_masks.get("att", np.exp(-tree_k * tree_lad * meshsize)))
        else:
            vox_is_tree = (voxel_data == -2)
            vox_is_opaque = (voxel_data != 0) & (~vox_is_tree)
            att = float(np.exp(-tree_k * tree_lad * meshsize))

        # Auto-tune chunk size if user didn't pass one
        time_batch_size = _auto_time_batch_size(n_faces, total_steps, kwargs.get("time_batch_size", None))
        if progress_report:
            print(f"Faces: {n_faces:,}, Timesteps: {total_steps:,}, Batch size: {time_batch_size}")

        for start in range(0, total_steps, time_batch_size):
            end = min(start + time_batch_size, total_steps)
            # Accumulate Wh/m² over this chunk inside the kernel
            ch_dir, ch_diff, ch_glob = compute_cumulative_solar_irradiance_faces_masked_timeseries(
                face_centers64,
                face_normals64,
                face_svf64,
                sun_dirs_arr,
                DNI_arr,
                DHI_arr,
                vox_is_tree,
                vox_is_opaque,
                float(meshsize),
                float(att),
                float(x_min), float(y_min), float(z_min),
                float(x_max), float(y_max), float(z_max),
                float(boundary_epsilon),
                int(start), int(end),
                float(time_step_hours)
            )
            face_cum_direct  += ch_dir
            face_cum_diffuse += ch_diff
            face_cum_global  += ch_glob

            if progress_report:
                pct = (end * 100.0) / total_steps
                print(f"Cumulative irradiance: {end}/{total_steps} ({pct:.1f}%)")
    else:
        # Iterate per timestep (fallback)
        for idx in range(total_steps):
            DNI = float(DNI_arr[idx])
            DHI = float(DHI_arr[idx])

            # Skip if sun below horizon
            if not sun_above_mask[idx]:
                # Only diffuse term contributes (still based on SVF)
                if boundary_mask is None:
                    boundary_mask = np.isnan(face_svf)
                # Accumulate diffuse only
                face_cum_diffuse += np.nan_to_num(face_svf * DHI) * time_step_hours
                face_cum_global  += np.nan_to_num(face_svf * DHI) * time_step_hours
                # progress
                if progress_report:
                    if ((idx + 1) % progress_every == 0) or (idx == total_steps - 1):
                        pct = (idx + 1) * 100.0 / total_steps
                        print(f"Cumulative irradiance: {idx+1}/{total_steps} ({pct:.1f}%)")
                continue

            # Fallback to wrapper per-timestep
            irr_mesh = get_building_solar_irradiance(
                voxcity,
                building_svf_mesh,
                float(azimuth_deg_arr[idx]),
                float(elev_deg_arr[idx]),
                DNI,
                DHI,
                show_plot=False,
                **instant_kwargs
            )
            face_direct = irr_mesh.metadata['direct']
            face_diffuse = irr_mesh.metadata['diffuse']
            face_global  = irr_mesh.metadata['global']

            # If first time, note boundary mask from NaNs
            if boundary_mask is None:
                boundary_mask = np.isnan(face_global)

            # Convert from W/m² to Wh/m² by multiplying time_step_hours
            face_cum_direct  += np.nan_to_num(face_direct)  * time_step_hours
            face_cum_diffuse += np.nan_to_num(face_diffuse) * time_step_hours
            face_cum_global  += np.nan_to_num(face_global)  * time_step_hours

            if progress_report and (((idx + 1) % progress_every == 0) or (idx == total_steps - 1)):
                pct = (idx + 1) * 100.0 / total_steps
                print(f"Cumulative irradiance: {idx+1}/{total_steps} ({pct:.1f}%)")
    
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
    if 'svf' in building_svf_mesh.metadata:
        cumulative_mesh.metadata['svf'] = building_svf_mesh.metadata['svf']
    
    cumulative_mesh.metadata['direct']  = face_cum_direct
    cumulative_mesh.metadata['diffuse'] = face_cum_diffuse
    cumulative_mesh.metadata['global']  = face_cum_global
    
    cumulative_mesh.name = "Cumulative Solar Irradiance (Wh/m²)"
    
    # # Optional OBJ export
    # obj_export = kwargs.get("obj_export", False)
    # if obj_export:
    #     # Get export parameters
    #     output_dir = kwargs.get("output_directory", "output")
    #     output_file_name = kwargs.get("output_file_name", "solar_irradiance")

    #     # Export the mesh directly
    #     irradiance_mesh.export(f"{output_dir}/{output_file_name}.obj")
    
    return cumulative_mesh

def get_building_global_solar_irradiance_using_epw(
    voxcity: VoxCity,
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
        voxcity (VoxCity): VoxCity object containing voxel data and metadata
        calc_type (str): 'instantaneous' or 'cumulative'.
        direct_normal_irradiance_scaling (float): Scaling factor for direct normal irradiance.
        diffuse_irradiance_scaling (float): Scaling factor for diffuse horizontal irradiance.
        **kwargs: Additional arguments including:
            - download_nearest_epw (bool): Whether to download nearest EPW file
            - epw_file_path (str): Path to EPW file
            - rectangle_vertices (list, optional): List of (lon,lat) coordinates for EPW download. If not provided, extracted from voxcity.extras
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
            - save_mesh (bool): Whether to save the mesh data using pickle
            - mesh_output_path (str): Path to save the mesh data (if save_mesh is True)

    Returns:
        trimesh.Trimesh: Building mesh with irradiance values stored in metadata.
    """
    import numpy as np
    import pytz
    from datetime import datetime
    
    # Extract voxel data and meshsize from VoxCity object
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    
    # Get EPW file
    download_nearest_epw = kwargs.get("download_nearest_epw", False)
    # Extract rectangle_vertices from VoxCity object if not provided in kwargs
    rectangle_vertices = kwargs.get("rectangle_vertices", None)
    if rectangle_vertices is None:
        rectangle_vertices = voxcity.extras.get("rectangle_vertices", None)
    epw_file_path = kwargs.get("epw_file_path", None)
    building_id_grid = kwargs.get("building_id_grid", None)
    building_svf_mesh = kwargs.get("building_svf_mesh", None)
    progress_report = kwargs.get("progress_report", False)
    fast_path = kwargs.get("fast_path", True)

    # Threading tuning (auto): choose sensible defaults based on hardware
    desired_threads = kwargs.get("numba_num_threads", None)
    _configure_num_threads(desired_threads, progress=kwargs.get("progress_report", False))

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
                load_data=True,
                allow_insecure_ssl=kwargs.get("allow_insecure_ssl", False),
                allow_http_fallback=kwargs.get("allow_http_fallback", False),
                ssl_verify=kwargs.get("ssl_verify", True)
            )

    # Read EPW data
    if epw_file_path is None:
        raise RuntimeError("EPW file path is None. Set 'epw_file_path' or enable 'download_nearest_epw' and ensure network succeeds.")
    df, lon, lat, tz, elevation_m = read_epw_for_solar_simulation(epw_file_path)
    if df.empty:
        raise ValueError("No data in EPW file.")
    
    # Step 1: Calculate Sky View Factor for building surfaces (unless provided)
    if building_svf_mesh is None:
        if progress_report:
            print("Processing Sky View Factor for building surfaces...")
        # Allow passing through specific SVF parameters via kwargs
        svf_kwargs = {
            'value_name': 'svf',
            'target_values': (0,),
            'inclusion_mode': False,
            'building_id_grid': building_id_grid,
            'progress_report': progress_report,
            'fast_path': fast_path,
        }
        # Permit overrides
        for k in ("N_azimuth","N_elevation","tree_k","tree_lad","debug"):
            if k in kwargs:
                svf_kwargs[k] = kwargs[k]
        building_svf_mesh = get_surface_view_factor(
            voxcity,
            **svf_kwargs
        )

    if progress_report:
        print(f"SVF ready. Faces: {len(building_svf_mesh.faces):,}")

    # Step 2: Build precomputed caches (geometry, masks, attenuation) for speed
    precomputed_geometry = {}
    try:
        grid_shape = voxel_data.shape
        grid_bounds_voxel = np.array([[0, 0, 0], [grid_shape[0], grid_shape[1], grid_shape[2]]], dtype=np.float64)
        grid_bounds_real = grid_bounds_voxel * meshsize
        boundary_epsilon = meshsize * 0.05
        precomputed_geometry = {
            'face_centers': building_svf_mesh.triangles_center,
            'face_normals': building_svf_mesh.face_normals,
            'face_svf': building_svf_mesh.metadata['svf'] if ('svf' in building_svf_mesh.metadata) else None,
            'grid_bounds_real': grid_bounds_real,
            'boundary_epsilon': boundary_epsilon,
        }
    except Exception:
        # Fallback silently
        precomputed_geometry = {}

    tree_k = kwargs.get("tree_k", 0.6)
    tree_lad = kwargs.get("tree_lad", 1.0)
    precomputed_masks = {
        'vox_is_tree': (voxel_data == -2),
        'vox_is_opaque': (voxel_data != 0) & (voxel_data != -2),
        'att': float(np.exp(-tree_k * tree_lad * meshsize)),
    }

    if progress_report:
        t_cnt = int(np.count_nonzero(precomputed_masks['vox_is_tree']))
        b_cnt = int(np.count_nonzero(voxel_data == -3)) if hasattr(voxel_data, 'shape') else 0
        print(f"Precomputed caches: trees={t_cnt:,}, buildings={b_cnt:,}, tree_att_per_voxel={precomputed_masks['att']:.4f}")
        print(f"Processing Solar Irradiance for building surfaces...")
    result_mesh = None
    
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
        
        if progress_report:
            print(f"Time: {df_period_local.index[0].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Sun position: Azimuth {azimuth_degrees:.1f}°, Elevation {elevation_degrees:.1f}°")
            print(f"DNI: {direct_normal_irradiance:.1f} W/m², DHI: {diffuse_irradiance:.1f} W/m²")
        
        # Skip if sun is below horizon
        if elevation_degrees <= 0:
            if progress_report:
                print("Sun is below horizon, skipping calculation.")
            result_mesh = building_svf_mesh.copy()
        else:
            # Compute irradiance
            _call_kwargs = kwargs.copy()
            if 'progress_report' in _call_kwargs:
                _call_kwargs.pop('progress_report')
            result_mesh = get_building_solar_irradiance(
                voxcity,
                building_svf_mesh,
                azimuth_degrees,
                elevation_degrees,
                direct_normal_irradiance,
                diffuse_irradiance,
                progress_report=progress_report,
                fast_path=fast_path,
                precomputed_geometry=precomputed_geometry,
                precomputed_masks=precomputed_masks,
                **_call_kwargs
            )

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
        if progress_report:
            print(f"Calculating cumulative irradiance from {period_start} to {period_end}...")
        result_mesh = get_cumulative_building_solar_irradiance(
            voxcity,
            building_svf_mesh,
            df, lon, lat, tz,
            period_start=period_start,
            period_end=period_end,
            time_step_hours=time_step_hours,
            direct_normal_irradiance_scaling=direct_normal_irradiance_scaling,
            diffuse_irradiance_scaling=diffuse_irradiance_scaling,
            progress_report=progress_report,
            fast_path=fast_path,
            precomputed_solar_positions=solar_positions,
            precomputed_geometry=precomputed_geometry,
            precomputed_masks=precomputed_masks,
            colormap=kwargs.get('colormap', 'jet'),
            show_each_timestep=kwargs.get('show_each_timestep', False),
            obj_export=kwargs.get('obj_export', False),
            output_directory=kwargs.get('output_directory', 'output'),
            output_file_name=kwargs.get('output_file_name', 'cumulative_solar')
        )
    
    else:
        raise ValueError("calc_type must be either 'instantaneous' or 'cumulative'")
    
    # Save mesh data if requested
    save_mesh = kwargs.get("save_mesh", False)
    if save_mesh:
        mesh_output_path = kwargs.get("mesh_output_path", None)
        if mesh_output_path is None:
            # Generate default path if none provided
            output_directory = kwargs.get("output_directory", "output")
            output_file_name = kwargs.get("output_file_name", f"{calc_type}_solar_irradiance")
            mesh_output_path = f"{output_directory}/{output_file_name}.pkl"
        
        save_irradiance_mesh(result_mesh, mesh_output_path)
        print(f"Saved irradiance mesh data to: {mesh_output_path}")
    
    return result_mesh

def save_irradiance_mesh(irradiance_mesh, output_file_path):
    """
    Save the irradiance mesh data to a file using pickle serialization.
    
    This function provides persistent storage for computed irradiance results,
    enabling reuse of expensive calculations and sharing of results between
    analysis sessions. The mesh data includes all geometry, irradiance values,
    and metadata required for visualization and further analysis.
    
    Serialization Benefits:
    - Preserves complete mesh structure with all computed data
    - Enables offline analysis and visualization workflows
    - Supports sharing results between different tools and users
    - Avoids recomputation of expensive irradiance calculations
    
    Data Preservation:
    - All mesh geometry (vertices, faces, normals)
    - Computed irradiance values (direct, diffuse, global)
    - Sky View Factor data and other metadata
    - Material properties and visualization settings
    
    Args:
        irradiance_mesh (trimesh.Trimesh): Mesh with irradiance data in metadata
                                          Should contain computed irradiance results
        output_file_path (str): Path to save the mesh data file
                               Recommended extension: .pkl for clarity
    
    Note:
        The function automatically creates the output directory if it doesn't exist.
        Use pickle format for maximum compatibility with Python data structures.
    """
    import pickle
    import os

    # Create output directory structure if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Serialize mesh data using pickle for complete data preservation
    with open(output_file_path, 'wb') as f:
        pickle.dump(irradiance_mesh, f)

def load_irradiance_mesh(input_file_path):
    """
    Load previously saved irradiance mesh data from a file.
    
    This function restores complete mesh data including geometry, computed
    irradiance values, and all associated metadata. It enables continuation
    of analysis workflows and reuse of expensive computation results.
    
    Restoration Capabilities:
    - Complete mesh geometry with all topological information
    - All computed irradiance data (direct, diffuse, global components)
    - Sky View Factor values and analysis metadata
    - Visualization settings and material properties
    
    Workflow Integration:
    - Load results from previous analysis sessions
    - Share computed data between team members
    - Perform post-processing and visualization
    - Compare results from different scenarios
    
    Args:
        input_file_path (str): Path to the saved mesh data file
                              Should be a file created by save_irradiance_mesh()
    
    Returns:
        trimesh.Trimesh: Complete mesh with all irradiance data in metadata
                        Ready for visualization, analysis, or further processing
    
    Note:
        The loaded mesh maintains all original data structure and can be used
        immediately for visualization or additional analysis operations.
    """
    import pickle
    
    # Deserialize mesh data preserving all original structure
    with open(input_file_path, 'rb') as f:
        irradiance_mesh = pickle.load(f)
    
    return irradiance_mesh
