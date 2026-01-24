"""
Volumetric Solar Irradiance Module

This module provides functions for computing volumetric (3D) solar irradiance
fields and extracting 2D slices at specified heights above terrain. Useful for:
- Mean Radiant Temperature (MRT) calculations
- Pedestrian thermal comfort analysis
- Light availability assessment

Functions:
    - get_volumetric_solar_irradiance_map: Single-timestep volumetric irradiance
    - get_cumulative_volumetric_solar_irradiance: Time-integrated volumetric irradiance
    - get_volumetric_solar_irradiance_using_epw: High-level EPW-based interface
    - get_global_solar_irradiance_using_epw: Unified interface for ground/volumetric modes
"""

from __future__ import annotations

import math
import time
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import pytz

from .utils import (
    get_location_from_voxcity,
    convert_voxel_data_to_arrays,
    compute_valid_ground_vectorized,
    get_solar_positions_astral,
    load_epw_data,
)
from .caching import (
    get_or_create_radiation_model,
    get_or_create_volumetric_calculator,
)

if TYPE_CHECKING:
    import pandas as pd


# Voxel class codes
VOXCITY_GROUND_CODE = -1
VOXCITY_TREE_CODE = -2
VOXCITY_BUILDING_CODE = -3


def _compute_ground_k_from_voxels(voxel_data: np.ndarray) -> np.ndarray:
    """
    Compute ground surface k-level for each (i,j) cell from voxel data.
    
    This finds the terrain top - the highest k where the cell below the first air
    cell is solid ground (not building). This is used for terrain-following
    height extraction in volumetric calculations.
    
    Water areas (voxel classes 7, 8, 9) and building/underground cells (negative codes)
    are excluded and marked as -1.
    
    Args:
        voxel_data: 3D array of voxel class codes
        
    Returns:
        2D array of ground k-levels (ni, nj). -1 means no valid ground found.
    """
    _, ground_k = compute_valid_ground_vectorized(voxel_data)
    return ground_k


def get_volumetric_solar_irradiance_map(
    voxcity,
    azimuth_degrees_ori: float,
    elevation_degrees: float,
    direct_normal_irradiance: float,
    diffuse_irradiance: float,
    volumetric_height: float = 1.5,
    with_reflections: bool = False,
    show_plot: bool = False,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated volumetric solar irradiance map at a specified height.
    
    Computes the 3D radiation field at each grid cell and extracts a 2D horizontal
    slice at the specified height. This is useful for:
    - Mean Radiant Temperature (MRT) calculations
    - Pedestrian thermal comfort analysis
    - Light availability assessment
    
    Args:
        voxcity: VoxCity object
        azimuth_degrees_ori: Solar azimuth in degrees (0=North, clockwise)
        elevation_degrees: Solar elevation in degrees above horizon
        direct_normal_irradiance: DNI in W/m²
        diffuse_irradiance: DHI in W/m²
        volumetric_height: Height above ground for irradiance extraction (meters)
        with_reflections: If True, include reflected radiation from surfaces.
            If False (default), only direct + diffuse sky radiation.
        show_plot: Whether to display a matplotlib plot
        **kwargs: Additional parameters:
            - n_azimuth (int): Number of azimuthal directions for SVF (default: 36)
            - n_zenith (int): Number of zenith angles for SVF (default: 9)
            - computation_mask (np.ndarray): Optional 2D boolean mask of shape (nx, ny).
                Grid cells outside the masked region are set to NaN.
            - progress_report (bool): Print progress (default: False)
            - colormap (str): Colormap for plot (default: 'magma')
            - vmin, vmax (float): Colormap bounds
            - n_reflection_steps (int): Reflection bounces when with_reflections=True (default: 2)
    
    Returns:
        2D numpy array of volumetric irradiance at the specified height (W/m²)
    """
    kwargs = kwargs.copy()
    progress_report = kwargs.pop('progress_report', False)
    n_azimuth = kwargs.pop('n_azimuth', 36)
    n_zenith = kwargs.pop('n_zenith', 9)
    computation_mask = kwargs.pop('computation_mask', None)
    
    # Get or create cached calculator
    calculator, domain = get_or_create_volumetric_calculator(
        voxcity,
        n_azimuth=n_azimuth,
        n_zenith=n_zenith,
        progress_report=progress_report,
        **kwargs
    )
    
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    ni, nj, nk = voxel_data.shape
    
    # Convert solar angles to direction vector
    # Match the coordinate system used in ground-level functions:
    # azimuth_degrees_ori: 0=North, 90=East (clockwise from North)
    # Transform to model coordinates: 180 - azimuth_degrees_ori
    azimuth_degrees = 180 - azimuth_degrees_ori
    azimuth_rad = math.radians(azimuth_degrees)
    elevation_rad = math.radians(elevation_degrees)
    
    cos_elev = math.cos(elevation_rad)
    sin_elev = math.sin(elevation_rad)
    
    # Direction toward sun (matching ground-level function convention)
    sun_dir_x = cos_elev * math.cos(azimuth_rad)
    sun_dir_y = cos_elev * math.sin(azimuth_rad)
    sun_dir_z = sin_elev
    sun_direction = (sun_dir_x, sun_dir_y, sun_dir_z)
    
    cos_zenith = sin_elev  # cos(zenith) = sin(elevation)
    
    # Compute ground_k for terrain-following extraction
    ground_k = _compute_ground_k_from_voxels(voxel_data)
    
    # Compute volumetric flux
    if with_reflections:
        # Use full reflection model
        if progress_report:
            print("Computing volumetric flux with reflections...")
        
        # Get radiation model for surface reflections
        n_reflection_steps = kwargs.pop('n_reflection_steps', 2)
        model, valid_ground, model_ground_k = get_or_create_radiation_model(
            voxcity,
            n_reflection_steps=n_reflection_steps,
            progress_report=progress_report,
            **kwargs
        )
        
        # Manually set solar position on the model's solar_calc
        model.solar_calc.cos_zenith[None] = cos_zenith
        model.solar_calc.sun_direction[None] = [sun_direction[0], sun_direction[1], sun_direction[2]]
        model.solar_calc.sun_up[None] = 1 if cos_zenith > 0 else 0
        
        # Compute shortwave radiation with the set solar position
        model.compute_shortwave_radiation(
            sw_direct=direct_normal_irradiance,
            sw_diffuse=diffuse_irradiance
        )
        
        # Get surface outgoing radiation
        n_surfaces = model.surfaces.count
        surf_outgoing = model.surfaces.sw_out.to_numpy()[:n_surfaces]
        
        # Compute volumetric flux including reflections
        if calculator.c2s_matrix_cached:
            calculator.compute_swflux_vol_with_reflections_cached(
                sw_direct=direct_normal_irradiance,
                sw_diffuse=diffuse_irradiance,
                cos_zenith=cos_zenith,
                sun_direction=sun_direction,
                surf_outgoing=surf_outgoing,
                lad=domain.lad
            )
        else:
            calculator.compute_swflux_vol_with_reflections(
                sw_direct=direct_normal_irradiance,
                sw_diffuse=diffuse_irradiance,
                cos_zenith=cos_zenith,
                sun_direction=sun_direction,
                surfaces=model.surfaces,
                surf_outgoing=surf_outgoing,
                lad=domain.lad
            )
    else:
        # Simple direct + diffuse only
        if progress_report:
            print("Computing volumetric flux (direct + diffuse)...")
        
        calculator.compute_swflux_vol(
            sw_direct=direct_normal_irradiance,
            sw_diffuse=diffuse_irradiance,
            cos_zenith=cos_zenith,
            sun_direction=sun_direction,
            lad=domain.lad
        )
    
    # Extract terrain-following horizontal slice at specified height above ground
    height_offset_k = max(1, int(round(volumetric_height / meshsize)))
    if progress_report:
        print(f"Extracting volumetric irradiance at {volumetric_height}m above terrain (offset={height_offset_k} cells)")
    
    # Get full 3D volumetric flux and is_solid arrays
    swflux_3d = calculator.get_swflux_vol()
    is_solid = domain.is_solid.to_numpy()
    
    # Create output array
    volumetric_map = np.full((ni, nj), np.nan, dtype=np.float64)
    
    # Extract terrain-following values
    for i in range(ni):
        for j in range(nj):
            gk = ground_k[i, j]
            if gk < 0:
                continue
            k_extract = gk + height_offset_k
            if k_extract >= nk:
                continue
            if is_solid[i, j, k_extract] == 1:
                continue
            volumetric_map[i, j] = swflux_3d[i, j, k_extract]
    
    # Flip to match VoxCity coordinate system
    volumetric_map = np.flipud(volumetric_map)
    
    # Apply computation_mask if provided
    if computation_mask is not None:
        if computation_mask.shape == volumetric_map.shape:
            flipped_mask = np.flipud(computation_mask)
            volumetric_map = np.where(flipped_mask, volumetric_map, np.nan)
        elif computation_mask.T.shape == volumetric_map.shape:
            flipped_mask = np.flipud(computation_mask.T)
            volumetric_map = np.where(flipped_mask, volumetric_map, np.nan)
        else:
            if computation_mask.shape == volumetric_map.shape:
                volumetric_map = np.where(computation_mask, volumetric_map, np.nan)
        
        if progress_report:
            n_masked = np.sum(np.isnan(volumetric_map))
            total = volumetric_map.size
            print(f"  Cells outside computation_mask set to NaN: {n_masked}/{total} ({100*n_masked/total:.1f}%)")

    if show_plot:
        colormap = kwargs.get('colormap', 'magma')
        vmin = kwargs.get('vmin', 0.0)
        vmax = kwargs.get('vmax', max(float(np.nanmax(volumetric_map)), 1.0))
        try:
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_bad(color='lightgray')
            plt.figure(figsize=(10, 8))
            plt.imshow(volumetric_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label=f'Volumetric Solar Irradiance at {volumetric_height}m (W/m²)')
            plt.title(f'Volumetric Irradiance (reflections={"on" if with_reflections else "off"})')
            plt.axis('off')
            plt.show()
        except ImportError:
            pass
    
    return volumetric_map


def get_cumulative_volumetric_solar_irradiance(
    voxcity,
    df: 'pd.DataFrame',
    lon: float,
    lat: float,
    tz: float,
    direct_normal_irradiance_scaling: float = 1.0,
    diffuse_irradiance_scaling: float = 1.0,
    volumetric_height: float = 1.5,
    with_reflections: bool = False,
    show_plot: bool = False,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated cumulative volumetric solar irradiance over a period.
    
    Integrates the 3D radiation field over time and extracts a 2D horizontal
    slice at the specified height.
    
    Args:
        voxcity: VoxCity object
        df: pandas DataFrame with 'DNI' and 'DHI' columns, datetime-indexed
        lon: Longitude in degrees
        lat: Latitude in degrees
        tz: Timezone offset in hours
        direct_normal_irradiance_scaling: Scaling factor for DNI
        diffuse_irradiance_scaling: Scaling factor for DHI
        volumetric_height: Height above ground for irradiance extraction (meters)
        with_reflections: If True, include reflected radiation from buildings,
            ground, and tree canopy surfaces. If False (default), only direct
            + diffuse sky radiation.
        show_plot: Whether to display a matplotlib plot
        **kwargs: Additional parameters:
            - start_time (str): Start time 'MM-DD HH:MM:SS' (default: '01-01 05:00:00')
            - end_time (str): End time 'MM-DD HH:MM:SS' (default: '01-01 20:00:00')
            - use_sky_patches (bool): Use sky patch optimization (default: True)
            - sky_discretization (str): 'tregenza', 'reinhart', 'uniform', 'fibonacci'
            - computation_mask (np.ndarray): Optional 2D boolean mask of shape (nx, ny).
                Grid cells outside the masked region are set to NaN.
            - progress_report (bool): Print progress (default: False)
            - n_reflection_steps (int): Reflection bounces when with_reflections=True (default: 2)
    
    Returns:
        2D numpy array of cumulative volumetric irradiance at the specified height (Wh/m²)
    """
    # Extract parameters
    kwargs = kwargs.copy()
    progress_report = kwargs.pop('progress_report', False)
    start_time = kwargs.pop('start_time', '01-01 05:00:00')
    end_time = kwargs.pop('end_time', '01-01 20:00:00')
    use_sky_patches = kwargs.pop('use_sky_patches', True)
    sky_discretization = kwargs.pop('sky_discretization', 'tregenza')
    n_azimuth = kwargs.pop('n_azimuth', 36)
    n_zenith = kwargs.pop('n_zenith', 9)
    computation_mask = kwargs.pop('computation_mask', None)
    
    if df.empty:
        raise ValueError("No data in EPW dataframe.")
    
    # Parse time range
    try:
        start_dt = datetime.strptime(start_time, '%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_time, '%m-%d %H:%M:%S')
    except ValueError as ve:
        raise ValueError("start_time and end_time must be in format 'MM-DD HH:MM:SS'") from ve
    
    # Filter dataframe to period
    df = df.copy()
    df['hour_of_year'] = (df.index.dayofyear - 1) * 24 + df.index.hour + 1
    start_doy = datetime(2000, start_dt.month, start_dt.day).timetuple().tm_yday
    end_doy = datetime(2000, end_dt.month, end_dt.day).timetuple().tm_yday
    start_hour = (start_doy - 1) * 24 + start_dt.hour + 1
    end_hour = (end_doy - 1) * 24 + end_dt.hour + 1
    
    if start_hour <= end_hour:
        df_period = df[(df['hour_of_year'] >= start_hour) & (df['hour_of_year'] <= end_hour)]
    else:
        df_period = df[(df['hour_of_year'] >= start_hour) | (df['hour_of_year'] <= end_hour)]
    
    if df_period.empty:
        raise ValueError("No EPW data in the specified period.")
    
    # Localize and convert to UTC
    offset_minutes = int(tz * 60)
    local_tz = pytz.FixedOffset(offset_minutes)
    df_period_local = df_period.copy()
    df_period_local.index = df_period_local.index.tz_localize(local_tz)
    df_period_utc = df_period_local.tz_convert(pytz.UTC)
    
    # Get solar positions
    solar_positions = get_solar_positions_astral(df_period_utc.index, lon, lat)
    
    # Get or create cached calculator
    calculator, domain = get_or_create_volumetric_calculator(
        voxcity,
        n_azimuth=n_azimuth,
        n_zenith=n_zenith,
        progress_report=progress_report,
        **kwargs
    )
    
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    ni, nj, nk = voxel_data.shape
    
    # Compute terrain-following extraction parameters
    height_offset_k = max(1, int(round(volumetric_height / meshsize)))
    if progress_report:
        print(f"Extracting volumetric irradiance at {volumetric_height}m above terrain (offset={height_offset_k} cells)")
    
    # Get is_solid array for masking
    is_solid = domain.is_solid.to_numpy()
    
    # Initialize cumulative map
    time_step_hours = kwargs.get('time_step_hours', 1.0)
    
    # Get radiation model for reflections if needed
    model = None
    ground_k = None
    if with_reflections:
        n_reflection_steps = kwargs.pop('n_reflection_steps', 2)
        model, valid_ground, ground_k = get_or_create_radiation_model(
            voxcity,
            n_reflection_steps=n_reflection_steps,
            progress_report=progress_report,
            **kwargs
        )
    else:
        ground_k = _compute_ground_k_from_voxels(voxel_data)
    
    # Initialize GPU-side cumulative accumulation
    calculator.init_cumulative_accumulation(
        ground_k=ground_k,
        height_offset_k=height_offset_k,
        is_solid=is_solid
    )
    
    # Pre-compute Terrain-to-Surface VF matrix for cached reflections
    if with_reflections and model is not None:
        t2s_start = time.perf_counter() if progress_report else 0
        calculator.compute_t2s_matrix(
            surfaces=model.surfaces,
            min_vf_threshold=1e-6,
            progress_report=progress_report
        )
        if progress_report:
            t2s_elapsed = time.perf_counter() - t2s_start
            print(f"  T2S matrix pre-computation: {t2s_elapsed:.2f}s")
    
    # Extract arrays
    azimuth_arr = solar_positions['azimuth'].to_numpy()
    elevation_arr = solar_positions['elevation'].to_numpy()
    dni_arr = df_period_utc['DNI'].to_numpy() * direct_normal_irradiance_scaling
    dhi_arr = df_period_utc['DHI'].to_numpy() * diffuse_irradiance_scaling
    
    n_timesteps = len(azimuth_arr)
    
    if progress_report:
        print(f"Computing cumulative volumetric irradiance for {n_timesteps} timesteps...")
        print(f"  Height: {volumetric_height}m, Reflections: {'on' if with_reflections else 'off'}")
        print(f"  Using GPU-optimized terrain-following accumulation")
    
    t0 = time.perf_counter() if progress_report else 0
    
    if use_sky_patches:
        # Use sky patch aggregation for efficiency
        from ..sky import (
            generate_tregenza_patches,
            generate_reinhart_patches,
            generate_uniform_grid_patches,
            generate_fibonacci_patches,
            get_tregenza_patch_index
        )
        
        # Generate sky patches
        if sky_discretization.lower() == 'tregenza':
            patches, directions, solid_angles = generate_tregenza_patches()
        elif sky_discretization.lower() == 'reinhart':
            mf = kwargs.get('reinhart_mf', kwargs.get('mf', 4))
            patches, directions, solid_angles = generate_reinhart_patches(mf=mf)
        elif sky_discretization.lower() == 'uniform':
            n_az = kwargs.get('sky_n_azimuth', 36)
            n_el = kwargs.get('sky_n_elevation', 9)
            patches, directions, solid_angles = generate_uniform_grid_patches(n_az, n_el)
        elif sky_discretization.lower() == 'fibonacci':
            n_patches = kwargs.get('sky_n_patches', 145)
            patches, directions, solid_angles = generate_fibonacci_patches(n_patches=n_patches)
        else:
            raise ValueError(f"Unknown sky discretization method: {sky_discretization}")
        
        n_patches = len(patches)
        cumulative_dni = np.zeros(n_patches, dtype=np.float64)
        total_dhi = 0.0
        
        # Bin sun positions to patches
        for i in range(n_timesteps):
            elev = elevation_arr[i]
            if elev <= 0:
                continue
            
            az = azimuth_arr[i]
            dni = dni_arr[i]
            dhi = dhi_arr[i]
            
            if dni > 0:
                patch_idx = get_tregenza_patch_index(az, elev)
                if 0 <= patch_idx < n_patches:
                    cumulative_dni[patch_idx] += dni * time_step_hours
            
            if dhi > 0:
                total_dhi += dhi * time_step_hours
        
        # Process each patch with accumulated DNI
        patches_with_dni = np.where(cumulative_dni > 0)[0]
        
        if progress_report:
            print(f"  Processing {len(patches_with_dni)} sky patches with accumulated DNI...")
        
        for idx, patch_idx in enumerate(patches_with_dni):
            patch_dni = cumulative_dni[patch_idx]
            patch_dir = directions[patch_idx]
            
            # Convert patch direction to azimuth/elevation
            patch_azimuth_ori = math.degrees(math.atan2(patch_dir[0], patch_dir[1]))
            if patch_azimuth_ori < 0:
                patch_azimuth_ori += 360
            patch_elevation = math.degrees(math.asin(patch_dir[2]))
            
            # Apply same coordinate transform as ground-level functions
            patch_azimuth = 180 - patch_azimuth_ori
            azimuth_rad = math.radians(patch_azimuth)
            elevation_rad = math.radians(patch_elevation)
            cos_elev = math.cos(elevation_rad)
            sin_elev = math.sin(elevation_rad)
            
            cos_zenith = sin_elev
            sun_dir_x = cos_elev * math.cos(azimuth_rad)
            sun_dir_y = cos_elev * math.sin(azimuth_rad)
            sun_dir_z = sin_elev
            sun_direction = (sun_dir_x, sun_dir_y, sun_dir_z)
            
            if with_reflections and model is not None:
                model.solar_calc.cos_zenith[None] = cos_zenith
                model.solar_calc.sun_direction[None] = [sun_direction[0], sun_direction[1], sun_direction[2]]
                model.solar_calc.sun_up[None] = 1 if cos_zenith > 0 else 0
                
                model.compute_shortwave_radiation(
                    sw_direct=patch_dni / time_step_hours,
                    sw_diffuse=0.0
                )
                
                n_surfaces = model.surfaces.count
                surf_outgoing = model.surfaces.sw_out.to_numpy()[:n_surfaces]
                
                calculator.compute_swflux_vol(
                    sw_direct=patch_dni / time_step_hours,
                    sw_diffuse=0.0,
                    cos_zenith=cos_zenith,
                    sun_direction=sun_direction,
                    lad=domain.lad
                )
                
                calculator.compute_reflected_flux_terrain_cached(surf_outgoing=surf_outgoing)
                calculator._add_reflected_to_total()
            else:
                calculator.compute_swflux_vol(
                    sw_direct=patch_dni / time_step_hours,
                    sw_diffuse=0.0,
                    cos_zenith=cos_zenith,
                    sun_direction=sun_direction,
                    lad=domain.lad
                )
            
            calculator.accumulate_terrain_following_slice_gpu(weight=time_step_hours)
            
            if progress_report and (idx + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(f"    Processed {idx + 1}/{len(patches_with_dni)} patches ({elapsed:.1f}s)")
        
        # Add diffuse contribution
        if total_dhi > 0:
            calculator.accumulate_svf_diffuse_gpu(total_dhi=total_dhi)
    
    else:
        # Process each timestep individually
        for i in range(n_timesteps):
            elev = elevation_arr[i]
            if elev <= 0:
                continue
            
            az = azimuth_arr[i]
            dni = dni_arr[i]
            dhi = dhi_arr[i]
            
            if dni <= 0 and dhi <= 0:
                continue
            
            azimuth_degrees = 180 - az
            azimuth_rad = math.radians(azimuth_degrees)
            elevation_rad = math.radians(elev)
            cos_elev = math.cos(elevation_rad)
            sin_elev = math.sin(elevation_rad)
            
            sun_dir_x = cos_elev * math.cos(azimuth_rad)
            sun_dir_y = cos_elev * math.sin(azimuth_rad)
            sun_dir_z = sin_elev
            sun_direction = (sun_dir_x, sun_dir_y, sun_dir_z)
            cos_zenith = sin_elev
            
            if with_reflections and model is not None:
                model.solar_calc.cos_zenith[None] = cos_zenith
                model.solar_calc.sun_direction[None] = [sun_direction[0], sun_direction[1], sun_direction[2]]
                model.solar_calc.sun_up[None] = 1 if cos_zenith > 0 else 0
                
                model.compute_shortwave_radiation(
                    sw_direct=dni,
                    sw_diffuse=dhi
                )
                
                n_surfaces = model.surfaces.count
                surf_outgoing = model.surfaces.sw_out.to_numpy()[:n_surfaces]
                
                calculator.compute_swflux_vol(
                    sw_direct=dni,
                    sw_diffuse=dhi,
                    cos_zenith=cos_zenith,
                    sun_direction=sun_direction,
                    lad=domain.lad
                )
                
                calculator.compute_reflected_flux_terrain_cached(surf_outgoing=surf_outgoing)
                calculator._add_reflected_to_total()
            else:
                calculator.compute_swflux_vol(
                    sw_direct=dni,
                    sw_diffuse=dhi,
                    cos_zenith=cos_zenith,
                    sun_direction=sun_direction,
                    lad=domain.lad
                )
            
            calculator.accumulate_terrain_following_slice_gpu(weight=time_step_hours)
            
            if progress_report and (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  Processed {i + 1}/{n_timesteps} timesteps ({elapsed:.1f}s)")
    
    if progress_report:
        elapsed = time.perf_counter() - t0
        print(f"Cumulative volumetric irradiance complete in {elapsed:.2f}s")
    
    # Get final cumulative map from GPU with NaN masking
    cumulative_map = calculator.finalize_cumulative_map(apply_nan_mask=True)
    
    # Flip to match VoxCity coordinate system
    cumulative_map = np.flipud(cumulative_map)
    
    # Apply computation_mask if provided
    if computation_mask is not None:
        if computation_mask.shape == cumulative_map.shape:
            flipped_mask = np.flipud(computation_mask)
            cumulative_map = np.where(flipped_mask, cumulative_map, np.nan)
        elif computation_mask.T.shape == cumulative_map.shape:
            flipped_mask = np.flipud(computation_mask.T)
            cumulative_map = np.where(flipped_mask, cumulative_map, np.nan)
        else:
            if computation_mask.shape == cumulative_map.shape:
                cumulative_map = np.where(computation_mask, cumulative_map, np.nan)
        
        if progress_report:
            n_masked = np.sum(np.isnan(cumulative_map))
            total = cumulative_map.size
            print(f"  Cells outside computation_mask set to NaN: {n_masked}/{total} ({100*n_masked/total:.1f}%)")

    if show_plot:
        colormap = kwargs.get('colormap', 'magma')
        vmin = kwargs.get('vmin', 0.0)
        vmax = kwargs.get('vmax', max(float(np.nanmax(cumulative_map)), 1.0))
        try:
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_bad(color='lightgray')
            plt.figure(figsize=(10, 8))
            plt.imshow(cumulative_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label=f'Cumulative Volumetric Irradiance at {volumetric_height}m (Wh/m²)')
            plt.title(f'Cumulative Volumetric Irradiance (reflections={"on" if with_reflections else "off"})')
            plt.axis('off')
            plt.show()
        except ImportError:
            pass
    
    return cumulative_map


def get_volumetric_solar_irradiance_using_epw(
    voxcity,
    calc_type: str = 'instantaneous',
    direct_normal_irradiance_scaling: float = 1.0,
    diffuse_irradiance_scaling: float = 1.0,
    volumetric_height: float = 1.5,
    with_reflections: bool = False,
    show_plot: bool = False,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated volumetric solar irradiance from EPW file.
    
    Computes 3D radiation fields and extracts a 2D horizontal slice at the 
    specified height above ground. This is useful for:
    - Mean Radiant Temperature (MRT) calculations
    - Pedestrian thermal comfort analysis
    - Light availability assessment
    
    Args:
        voxcity: VoxCity object
        calc_type: 'instantaneous' or 'cumulative'
        direct_normal_irradiance_scaling: Scaling factor for DNI
        diffuse_irradiance_scaling: Scaling factor for DHI
        volumetric_height: Height above ground for irradiance extraction (meters)
        with_reflections: If True, include reflected radiation from buildings,
            ground, and tree canopy surfaces. If False (default), only direct
            + diffuse sky radiation.
        show_plot: Whether to display a matplotlib plot
        **kwargs: Additional parameters including:
            - epw_file_path (str): Path to EPW file
            - download_nearest_epw (bool): Download nearest EPW (default: False)
            - calc_time (str): For instantaneous: 'MM-DD HH:MM:SS'
            - start_time, end_time (str): For cumulative: 'MM-DD HH:MM:SS'
            - rectangle_vertices: Location vertices (for EPW download)
            - computation_mask (np.ndarray): Optional 2D boolean mask of shape (nx, ny).
                Grid cells outside the masked region are set to NaN.
            - n_reflection_steps (int): Reflection bounces when with_reflections=True (default: 2)
    
    Returns:
        2D numpy array of volumetric irradiance at the specified height (W/m² or Wh/m²)
    """
    # Load EPW data using helper function
    kwargs_copy = dict(kwargs)
    df, lon, lat, tz = load_epw_data(
        epw_file_path=kwargs_copy.pop('epw_file_path', None),
        download_nearest_epw=kwargs_copy.pop('download_nearest_epw', False),
        voxcity=voxcity,
        **kwargs_copy
    )
    
    if calc_type == 'instantaneous':
        calc_time = kwargs.get('calc_time', '01-01 12:00:00')
        try:
            calc_dt = datetime.strptime(calc_time, '%m-%d %H:%M:%SS')
        except ValueError:
            try:
                calc_dt = datetime.strptime(calc_time, '%m-%d %H:%M:%S')
            except ValueError:
                raise ValueError("calc_time must be in format 'MM-DD HH:MM:SS'")
        
        df_period = df[
            (df.index.month == calc_dt.month) &
            (df.index.day == calc_dt.day) &
            (df.index.hour == calc_dt.hour)
        ]
        if df_period.empty:
            raise ValueError("No EPW data at the specified time.")
        
        # Get solar position
        offset_minutes = int(tz * 60)
        local_tz = pytz.FixedOffset(offset_minutes)
        df_local = df_period.copy()
        df_local.index = df_local.index.tz_localize(local_tz)
        df_utc = df_local.tz_convert(pytz.UTC)
        
        solar_positions = get_solar_positions_astral(df_utc.index, lon, lat)
        DNI = float(df_utc.iloc[0]['DNI']) * direct_normal_irradiance_scaling
        DHI = float(df_utc.iloc[0]['DHI']) * diffuse_irradiance_scaling
        azimuth_degrees = float(solar_positions.iloc[0]['azimuth'])
        elevation_degrees = float(solar_positions.iloc[0]['elevation'])
        
        return get_volumetric_solar_irradiance_map(
            voxcity,
            azimuth_degrees,
            elevation_degrees,
            DNI,
            DHI,
            volumetric_height=volumetric_height,
            with_reflections=with_reflections,
            show_plot=show_plot,
            **kwargs
        )
    
    elif calc_type == 'cumulative':
        return get_cumulative_volumetric_solar_irradiance(
            voxcity,
            df,
            lon,
            lat,
            tz,
            direct_normal_irradiance_scaling=direct_normal_irradiance_scaling,
            diffuse_irradiance_scaling=diffuse_irradiance_scaling,
            volumetric_height=volumetric_height,
            with_reflections=with_reflections,
            show_plot=show_plot,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown calc_type: {calc_type}. Use 'instantaneous' or 'cumulative'.")


def get_global_solar_irradiance_using_epw(
    voxcity,
    temporal_mode: str = 'instantaneous',
    spatial_mode: str = 'horizontal',
    direct_normal_irradiance_scaling: float = 1.0,
    diffuse_irradiance_scaling: float = 1.0,
    show_plot: bool = False,
    calc_type: str = None,  # Deprecated, for backward compatibility
    computation_mask: np.ndarray = None,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated global irradiance from EPW file.
    
    This function provides a unified interface for both ground-level and volumetric
    irradiance calculations using Taichi GPU acceleration.
    
    Args:
        voxcity: VoxCity object
        temporal_mode: Time integration mode:
            - 'instantaneous': Single time point (requires calc_time)
            - 'cumulative': Integrate over time range (requires start_time, end_time)
        spatial_mode: Spatial computation mode:
            - 'horizontal': 2D ground-level irradiance at view_point_height
            - 'volumetric': 3D radiation field extracted at volumetric_height above terrain
        direct_normal_irradiance_scaling: Scaling factor for DNI
        diffuse_irradiance_scaling: Scaling factor for DHI
        show_plot: Whether to display a matplotlib plot
        calc_type: DEPRECATED. Use temporal_mode and spatial_mode instead.
            Legacy values 'instantaneous', 'cumulative', 'volumetric' are still supported.
        computation_mask: Optional 2D boolean numpy array of shape (nx, ny).
            If provided, only cells where mask is True will be computed.
            Cells where mask is False will be set to NaN in the output.
        **kwargs: Additional parameters including:
            - epw_file_path (str): Path to EPW file
            - download_nearest_epw (bool): Download nearest EPW (default: False)
            - calc_time (str): For instantaneous: 'MM-DD HH:MM:SS'
            - start_time, end_time (str): For cumulative: 'MM-DD HH:MM:SS'
            - rectangle_vertices: Location vertices (for EPW download)
            - view_point_height (float): Height for horizontal mode (default: 1.5)
            - volumetric_height (float): Height for volumetric mode (default: 1.5)
            - with_reflections (bool): Include reflections (default: False)
            - n_reflection_steps (int): Reflection bounces (default: 2)
    
    Returns:
        2D numpy array of irradiance (W/m² for instantaneous, Wh/m² for cumulative)
    
    Examples:
        # Instantaneous ground-level irradiance
        grid = get_global_solar_irradiance_using_epw(
            voxcity,
            temporal_mode='instantaneous',
            spatial_mode='horizontal',
            calc_time='08-03 10:00:00',
            epw_file_path='weather.epw'
        )
        
        # Cumulative volumetric irradiance with reflections
        grid = get_global_solar_irradiance_using_epw(
            voxcity,
            temporal_mode='cumulative',
            spatial_mode='volumetric',
            start_time='01-01 09:00:00',
            end_time='01-31 19:00:00',
            volumetric_height=1.5,
            with_reflections=True,
            epw_file_path='weather.epw'
        )
    """
    # Import ground-level functions for horizontal mode
    from .ground import (
        get_global_solar_irradiance_map,
        get_cumulative_global_solar_irradiance,
    )
    
    # Handle backward compatibility with calc_type parameter
    if calc_type is not None:
        warnings.warn(
            "calc_type parameter is deprecated. Use temporal_mode and spatial_mode instead. "
            "Example: temporal_mode='cumulative', spatial_mode='volumetric'",
            DeprecationWarning,
            stacklevel=2
        )
        if calc_type == 'instantaneous':
            temporal_mode = 'instantaneous'
            spatial_mode = 'horizontal'
        elif calc_type == 'cumulative':
            temporal_mode = 'cumulative'
            spatial_mode = 'horizontal'
        elif calc_type == 'volumetric':
            spatial_mode = 'volumetric'
            calc_time = kwargs.get('calc_time', None)
            start_time = kwargs.get('start_time', None)
            if calc_time is not None and start_time is None:
                temporal_mode = 'instantaneous'
            else:
                temporal_mode = 'cumulative'
        else:
            raise ValueError(f"Unknown calc_type: {calc_type}. Use temporal_mode/spatial_mode instead.")
    
    # Validate parameters
    if temporal_mode not in ('instantaneous', 'cumulative'):
        raise ValueError(f"temporal_mode must be 'instantaneous' or 'cumulative', got '{temporal_mode}'")
    if spatial_mode not in ('horizontal', 'volumetric'):
        raise ValueError(f"spatial_mode must be 'horizontal' or 'volumetric', got '{spatial_mode}'")
    
    # Load EPW data using helper function
    kwargs_copy = dict(kwargs)
    df, lon, lat, tz = load_epw_data(
        epw_file_path=kwargs_copy.pop('epw_file_path', None),
        download_nearest_epw=kwargs_copy.pop('download_nearest_epw', False),
        voxcity=voxcity,
        **kwargs_copy
    )
    
    # Add computation_mask to kwargs for passing to underlying functions
    if computation_mask is not None:
        kwargs['computation_mask'] = computation_mask
    
    # Route to appropriate function based on temporal_mode × spatial_mode
    if spatial_mode == 'horizontal':
        # Ground-level horizontal irradiance
        if temporal_mode == 'instantaneous':
            calc_time = kwargs.get('calc_time', '01-01 12:00:00')
            try:
                calc_dt = datetime.strptime(calc_time, '%m-%d %H:%M:%S')
            except ValueError:
                raise ValueError("calc_time must be in format 'MM-DD HH:MM:SS'")
            
            df_period = df[
                (df.index.month == calc_dt.month) &
                (df.index.day == calc_dt.day) &
                (df.index.hour == calc_dt.hour)
            ]
            if df_period.empty:
                raise ValueError("No EPW data at the specified time.")
            
            offset_minutes = int(tz * 60)
            local_tz = pytz.FixedOffset(offset_minutes)
            df_local = df_period.copy()
            df_local.index = df_local.index.tz_localize(local_tz)
            df_utc = df_local.tz_convert(pytz.UTC)
            
            solar_positions = get_solar_positions_astral(df_utc.index, lon, lat)
            DNI = float(df_utc.iloc[0]['DNI']) * direct_normal_irradiance_scaling
            DHI = float(df_utc.iloc[0]['DHI']) * diffuse_irradiance_scaling
            azimuth_degrees = float(solar_positions.iloc[0]['azimuth'])
            elevation_degrees = float(solar_positions.iloc[0]['elevation'])
            
            return get_global_solar_irradiance_map(
                voxcity,
                azimuth_degrees,
                elevation_degrees,
                DNI,
                DHI,
                show_plot=show_plot,
                **kwargs
            )
        
        else:  # cumulative
            return get_cumulative_global_solar_irradiance(
                voxcity,
                df,
                lon,
                lat,
                tz,
                direct_normal_irradiance_scaling=direct_normal_irradiance_scaling,
                diffuse_irradiance_scaling=diffuse_irradiance_scaling,
                show_plot=show_plot,
                **kwargs
            )
    
    else:  # volumetric
        volumetric_height = kwargs.pop('volumetric_height', kwargs.pop('view_point_height', 1.5))
        with_reflections = kwargs.pop('with_reflections', False)
        
        if temporal_mode == 'instantaneous':
            calc_time = kwargs.get('calc_time', '01-01 12:00:00')
            try:
                calc_dt = datetime.strptime(calc_time, '%m-%d %H:%M:%S')
            except ValueError:
                raise ValueError("calc_time must be in format 'MM-DD HH:MM:SS'")
            
            df_period = df[
                (df.index.month == calc_dt.month) &
                (df.index.day == calc_dt.day) &
                (df.index.hour == calc_dt.hour)
            ]
            if df_period.empty:
                raise ValueError("No EPW data at the specified time.")
            
            offset_minutes = int(tz * 60)
            local_tz = pytz.FixedOffset(offset_minutes)
            df_local = df_period.copy()
            df_local.index = df_local.index.tz_localize(local_tz)
            df_utc = df_local.tz_convert(pytz.UTC)
            
            solar_positions = get_solar_positions_astral(df_utc.index, lon, lat)
            DNI = float(df_utc.iloc[0]['DNI']) * direct_normal_irradiance_scaling
            DHI = float(df_utc.iloc[0]['DHI']) * diffuse_irradiance_scaling
            azimuth_degrees = float(solar_positions.iloc[0]['azimuth'])
            elevation_degrees = float(solar_positions.iloc[0]['elevation'])
            
            return get_volumetric_solar_irradiance_map(
                voxcity,
                azimuth_degrees,
                elevation_degrees,
                DNI,
                DHI,
                volumetric_height=volumetric_height,
                with_reflections=with_reflections,
                show_plot=show_plot,
                **kwargs
            )
        
        else:  # cumulative
            return get_cumulative_volumetric_solar_irradiance(
                voxcity,
                df,
                lon,
                lat,
                tz,
                direct_normal_irradiance_scaling=direct_normal_irradiance_scaling,
                diffuse_irradiance_scaling=diffuse_irradiance_scaling,
                volumetric_height=volumetric_height,
                with_reflections=with_reflections,
                show_plot=show_plot,
                **kwargs
            )


# Mesh I/O utilities
def save_irradiance_mesh(mesh, filepath: str) -> None:
    """
    Save irradiance mesh to pickle file.
    
    Args:
        mesh: Trimesh object with irradiance metadata
        filepath: Output file path
    """
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(mesh, f)


def load_irradiance_mesh(filepath: str):
    """
    Load irradiance mesh from pickle file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Trimesh object with irradiance metadata
    """
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)
