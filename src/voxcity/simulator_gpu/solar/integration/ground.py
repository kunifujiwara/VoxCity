"""
Ground-level solar irradiance functions for VoxCity.

This module provides GPU-accelerated ground-level solar irradiance calculations:
- Direct solar irradiance map
- Diffuse solar irradiance map (SVF-based)
- Global solar irradiance map (direct + diffuse)
- Cumulative solar irradiance over time periods
- Sunlight hours (PSH and DSH modes)

These functions match the voxcity.simulator.solar API signatures for 
drop-in replacement with GPU acceleration.
"""

import numpy as np
from typing import Optional, Tuple

from .utils import (
    VOXCITY_BUILDING_CODE,
    VOXCITY_GROUND_CODE,
    VOXCITY_TREE_CODE,
    compute_sun_direction,
    filter_df_to_period,
    parse_time_period,
    get_hour_range_from_period,
    load_epw_data,
    get_solar_positions_astral,
    add_metadata_to_array,
    get_location_from_voxcity,
    get_timezone_offset_from_location,
    generate_annual_hourly_dataframe,
)

from .caching import (
    get_radiation_model_cache,
    get_or_create_radiation_model,
    get_or_create_gpu_ray_tracer,
    compute_direct_transmittance_map_gpu,
)


# =============================================================================
# Ground Irradiance with Reflections (Internal)
# =============================================================================

def _compute_ground_irradiance_with_reflections(
    voxcity,
    azimuth_degrees_ori: float,
    elevation_degrees: float,
    direct_normal_irradiance: float,
    diffuse_irradiance: float,
    view_point_height: float = 1.5,
    n_reflection_steps: int = 2,
    progress_report: bool = False,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ground-level irradiance using full RadiationModel with reflections.
    
    Uses a cached RadiationModel to avoid recomputing SVF/CSF matrices for each
    solar position. The geometry-dependent matrices are computed once and reused.
    
    Args:
        voxcity: VoxCity object
        azimuth_degrees_ori: Solar azimuth in degrees (0=North, clockwise)
        elevation_degrees: Solar elevation in degrees above horizon
        direct_normal_irradiance: DNI in W/m²
        diffuse_irradiance: DHI in W/m²
        view_point_height: Observer height above ground (default: 1.5)
        n_reflection_steps: Number of reflection bounces (default: 2)
        progress_report: Print progress (default: False)
        **kwargs: Additional parameters
    
    Returns:
        Tuple of (direct_map, diffuse_map, reflected_map) as 2D numpy arrays
    """
    from ..domain import IUP
    
    voxel_data = voxcity.voxels.classes
    ni, nj, nk = voxel_data.shape
    
    # Remove parameters that we pass explicitly to avoid duplicates
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ('n_reflection_steps', 'progress_report', 'view_point_height')}
    
    # Get or create cached RadiationModel (SVF/CSF only computed once)
    model, valid_ground, ground_k = get_or_create_radiation_model(
        voxcity,
        n_reflection_steps=n_reflection_steps,
        progress_report=progress_report,
        **filtered_kwargs
    )
    
    # Set solar position for this timestep
    sun_dir_x, sun_dir_y, sun_dir_z, cos_zenith = compute_sun_direction(
        azimuth_degrees_ori, elevation_degrees
    )
    
    # Set sun direction and cos_zenith directly on the SolarCalculator fields
    model.solar_calc.sun_direction[None] = (sun_dir_x, sun_dir_y, sun_dir_z)
    model.solar_calc.cos_zenith[None] = cos_zenith
    model.solar_calc.sun_up[None] = 1 if elevation_degrees > 0 else 0
    
    # Compute shortwave radiation (uses cached SVF/CSF matrices)
    model.compute_shortwave_radiation(
        sw_direct=direct_normal_irradiance,
        sw_diffuse=diffuse_irradiance
    )
    
    # Extract surface irradiance
    n_surfaces = model.surfaces.count
    
    # Initialize output arrays
    direct_map = np.full((ni, nj), np.nan, dtype=np.float32)
    diffuse_map = np.full((ni, nj), np.nan, dtype=np.float32)
    reflected_map = np.zeros((ni, nj), dtype=np.float32)
    
    # Use pre-computed surface-to-grid mapping if available (from cache)
    cache = get_radiation_model_cache()
    if (cache is not None and 
        cache.grid_indices is not None and 
        len(cache.grid_indices) > 0):
        
        grid_indices = cache.grid_indices
        surface_indices = cache.surface_indices
        
        # Extract only the irradiance values we need (vectorized)
        sw_in_direct = model.surfaces.sw_in_direct.to_numpy()
        sw_in_diffuse = model.surfaces.sw_in_diffuse.to_numpy()
        
        # Vectorized assignment using pre-computed indices
        direct_map[grid_indices[:, 0], grid_indices[:, 1]] = sw_in_direct[surface_indices]
        diffuse_map[grid_indices[:, 0], grid_indices[:, 1]] = sw_in_diffuse[surface_indices]
    else:
        # Fallback to original loop if no cached mapping
        from ..domain import IUP
        positions = model.surfaces.position.to_numpy()[:n_surfaces]
        directions = model.surfaces.direction.to_numpy()[:n_surfaces]
        sw_in_direct = model.surfaces.sw_in_direct.to_numpy()[:n_surfaces]
        sw_in_diffuse = model.surfaces.sw_in_diffuse.to_numpy()[:n_surfaces]
        
        for idx in range(n_surfaces):
            pos_i, pos_j, k = positions[idx]
            direction = directions[idx]
            
            if direction == IUP:
                ii, jj = int(pos_i), int(pos_j)
                if 0 <= ii < ni and 0 <= jj < nj:
                    if not valid_ground[ii, jj]:
                        continue
                    if int(k) == ground_k[ii, jj]:
                        if np.isnan(direct_map[ii, jj]):
                            direct_map[ii, jj] = sw_in_direct[idx]
                            diffuse_map[ii, jj] = sw_in_diffuse[idx]
    
    # Flip to match VoxCity coordinate system
    direct_map = np.flipud(direct_map)
    diffuse_map = np.flipud(diffuse_map)
    reflected_map = np.flipud(reflected_map)
    
    return direct_map, diffuse_map, reflected_map


# =============================================================================
# Public API Functions
# =============================================================================

def get_direct_solar_irradiance_map(
    voxcity,
    azimuth_degrees_ori: float,
    elevation_degrees: float,
    direct_normal_irradiance: float,
    show_plot: bool = False,
    with_reflections: bool = False,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated direct horizontal irradiance map computation.
    
    Args:
        voxcity: VoxCity object
        azimuth_degrees_ori: Solar azimuth in degrees (0=North, clockwise)
        elevation_degrees: Solar elevation in degrees above horizon
        direct_normal_irradiance: DNI in W/m²
        show_plot: Whether to display a matplotlib plot
        with_reflections: If True, use full RadiationModel with multi-bounce 
            reflections. If False (default), use simple ray-tracing for 
            faster but less accurate results.
        **kwargs: Additional parameters including:
            - view_point_height (float): Observer height above ground (default: 1.5)
            - tree_k (float): Tree extinction coefficient (default: 0.6)
            - tree_lad (float): Leaf area density (default: 1.0)
            - colormap (str): Matplotlib colormap name (default: 'magma')
            - vmin, vmax (float): Colormap limits
            - obj_export (bool): Export to OBJ file (default: False)
            - n_reflection_steps (int): Number of reflection bounces when 
                with_reflections=True (default: 2)
            - progress_report (bool): Print progress (default: False)
    
    Returns:
        2D numpy array of direct horizontal irradiance (W/m²)
    """
    import taichi as ti
    from ...init_taichi import ensure_initialized
    ensure_initialized()
    
    colormap = kwargs.get('colormap', 'magma')
    vmin = kwargs.get('vmin', 0.0)
    vmax = kwargs.get('vmax', direct_normal_irradiance)
    
    if with_reflections:
        # Use full RadiationModel with reflections
        direct_map, _, _ = _compute_ground_irradiance_with_reflections(
            voxcity=voxcity,
            azimuth_degrees_ori=azimuth_degrees_ori,
            elevation_degrees=elevation_degrees,
            direct_normal_irradiance=direct_normal_irradiance,
            diffuse_irradiance=0.0,
            **kwargs
        )
    else:
        # Use simple ray-tracing (faster but no reflections)
        voxel_data = voxcity.voxels.classes
        meshsize = voxcity.voxels.meta.meshsize
        
        view_point_height = kwargs.get('view_point_height', 1.5)
        tree_k = kwargs.get('tree_k', 0.6)
        tree_lad = kwargs.get('tree_lad', 1.0)
        
        # Convert to sun direction vector
        azimuth_degrees = 180 - azimuth_degrees_ori
        azimuth_radians = np.deg2rad(azimuth_degrees)
        elevation_radians = np.deg2rad(elevation_degrees)
        
        dx_dir = np.cos(elevation_radians) * np.cos(azimuth_radians)
        dy_dir = np.cos(elevation_radians) * np.sin(azimuth_radians)
        dz_dir = np.sin(elevation_radians)
        
        # Compute transmittance map using ray tracing
        transmittance_map = compute_direct_transmittance_map_gpu(
            voxel_data=voxel_data,
            sun_direction=(dx_dir, dy_dir, dz_dir),
            view_point_height=view_point_height,
            meshsize=meshsize,
            tree_k=tree_k,
            tree_lad=tree_lad
        )
        
        # Convert to horizontal irradiance
        sin_elev = np.sin(elevation_radians)
        direct_map = transmittance_map * direct_normal_irradiance * sin_elev
        
        # Flip to match VoxCity coordinate system
        direct_map = np.flipud(direct_map)
    
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_bad(color='lightgray')
            plt.figure(figsize=(10, 8))
            plt.imshow(direct_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label='Direct Solar Irradiance (W/m²)')
            plt.axis('off')
            plt.show()
        except ImportError:
            pass
    
    if kwargs.get('obj_export', False):
        _export_irradiance_to_obj(
            voxcity, direct_map, 
            output_name=kwargs.get('output_file_name', 'direct_solar_irradiance'),
            **kwargs
        )
    
    return direct_map


def get_diffuse_solar_irradiance_map(
    voxcity,
    diffuse_irradiance: float = 1.0,
    show_plot: bool = False,
    with_reflections: bool = False,
    azimuth_degrees_ori: float = 180.0,
    elevation_degrees: float = 45.0,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated diffuse horizontal irradiance map computation using SVF.
    
    Args:
        voxcity: VoxCity object
        diffuse_irradiance: Diffuse horizontal irradiance in W/m²
        show_plot: Whether to display a matplotlib plot
        with_reflections: If True, use full RadiationModel with multi-bounce 
            reflections. If False (default), use simple SVF-based computation.
        azimuth_degrees_ori: Solar azimuth (only used when with_reflections=True)
        elevation_degrees: Solar elevation (only used when with_reflections=True)
        **kwargs: Additional parameters
    
    Returns:
        2D numpy array of diffuse horizontal irradiance (W/m²)
    """
    colormap = kwargs.get('colormap', 'magma')
    vmin = kwargs.get('vmin', 0.0)
    vmax = kwargs.get('vmax', diffuse_irradiance)
    
    if with_reflections:
        # Use full RadiationModel with reflections
        refl_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ('direct_normal_irradiance', 'diffuse_irradiance')}
        _, diffuse_map, reflected_map = _compute_ground_irradiance_with_reflections(
            voxcity=voxcity,
            azimuth_degrees_ori=azimuth_degrees_ori,
            elevation_degrees=elevation_degrees,
            direct_normal_irradiance=kwargs.get('direct_normal_irradiance', 0.0),
            diffuse_irradiance=diffuse_irradiance,
            **refl_kwargs
        )
        diffuse_map = np.where(np.isnan(diffuse_map), np.nan, diffuse_map + reflected_map)
    else:
        # Use simple SVF-based computation (faster but no reflections)
        from ...visibility.integration import get_sky_view_factor_map as get_svf_map
        
        svf_kwargs = kwargs.copy()
        svf_kwargs['colormap'] = 'BuPu_r'
        svf_kwargs['vmin'] = 0
        svf_kwargs['vmax'] = 1
        
        SVF_map = get_svf_map(voxcity, show_plot=False, **svf_kwargs)
        diffuse_map = SVF_map * diffuse_irradiance
    
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_bad(color='lightgray')
            plt.figure(figsize=(10, 8))
            plt.imshow(diffuse_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label='Diffuse Solar Irradiance (W/m²)')
            plt.axis('off')
            plt.show()
        except ImportError:
            pass
    
    if kwargs.get('obj_export', False):
        _export_irradiance_to_obj(
            voxcity, diffuse_map,
            output_name=kwargs.get('output_file_name', 'diffuse_solar_irradiance'),
            **kwargs
        )
    
    return diffuse_map


def get_global_solar_irradiance_map(
    voxcity,
    azimuth_degrees_ori: float,
    elevation_degrees: float,
    direct_normal_irradiance: float,
    diffuse_irradiance: float,
    show_plot: bool = False,
    with_reflections: bool = False,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated global (direct + diffuse) horizontal irradiance map.
    
    Args:
        voxcity: VoxCity object
        azimuth_degrees_ori: Solar azimuth in degrees (0=North, clockwise)
        elevation_degrees: Solar elevation in degrees above horizon
        direct_normal_irradiance: DNI in W/m²
        diffuse_irradiance: DHI in W/m²
        show_plot: Whether to display a matplotlib plot
        with_reflections: If True, use full RadiationModel with multi-bounce 
            reflections. If False (default), use simple ray-tracing/SVF.
        **kwargs: Additional parameters including:
            - computation_mask (np.ndarray): Optional 2D boolean mask
            - n_reflection_steps (int): Number of reflection bounces
            - progress_report (bool): Print progress (default: False)
    
    Returns:
        2D numpy array of global horizontal irradiance (W/m²)
    """
    computation_mask = kwargs.pop('computation_mask', None)
    
    if with_reflections:
        # Use full RadiationModel with reflections
        direct_map, diffuse_map, reflected_map = _compute_ground_irradiance_with_reflections(
            voxcity=voxcity,
            azimuth_degrees_ori=azimuth_degrees_ori,
            elevation_degrees=elevation_degrees,
            direct_normal_irradiance=direct_normal_irradiance,
            diffuse_irradiance=diffuse_irradiance,
            **kwargs
        )
        global_map = np.where(
            np.isnan(direct_map), 
            np.nan, 
            direct_map + diffuse_map + reflected_map
        )
    else:
        # Compute direct and diffuse separately (no reflections)
        direct_map = get_direct_solar_irradiance_map(
            voxcity,
            azimuth_degrees_ori,
            elevation_degrees,
            direct_normal_irradiance,
            show_plot=False,
            with_reflections=False,
            **kwargs
        )
        
        diffuse_map = get_diffuse_solar_irradiance_map(
            voxcity,
            diffuse_irradiance=diffuse_irradiance,
            show_plot=False,
            with_reflections=False,
            **kwargs
        )
        
        global_map = np.where(np.isnan(direct_map), diffuse_map, direct_map + diffuse_map)
    
    if show_plot:
        colormap = kwargs.get('colormap', 'magma')
        vmin = kwargs.get('vmin', 0.0)
        vmax = kwargs.get('vmax', max(float(np.nanmax(global_map)), 1.0))
        try:
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_bad(color='lightgray')
            plt.figure(figsize=(10, 8))
            plt.imshow(global_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label='Global Solar Irradiance (W/m²)')
            plt.axis('off')
            plt.show()
        except ImportError:
            pass
    
    if kwargs.get('obj_export', False):
        _export_irradiance_to_obj(
            voxcity, global_map,
            output_name=kwargs.get('output_file_name', 'global_solar_irradiance'),
            **kwargs
        )
    
    # Apply computation mask if provided
    if computation_mask is not None:
        if computation_mask.shape == global_map.shape:
            global_map = np.where(np.flipud(computation_mask), global_map, np.nan)
        elif computation_mask.T.shape == global_map.shape:
            global_map = np.where(np.flipud(computation_mask.T), global_map, np.nan)
        else:
            if computation_mask.shape == global_map.shape:
                global_map = np.where(computation_mask, global_map, np.nan)
    
    return global_map


def get_cumulative_global_solar_irradiance(
    voxcity,
    df,
    lon: float,
    lat: float,
    tz: float,
    direct_normal_irradiance_scaling: float = 1.0,
    diffuse_irradiance_scaling: float = 1.0,
    show_plot: bool = False,
    with_reflections: bool = False,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated cumulative global solar irradiance over a period.
    
    Uses sky patch optimization for efficient multi-timestep calculations.
    
    Args:
        voxcity: VoxCity object
        df: pandas DataFrame with 'DNI' and 'DHI' columns, datetime-indexed
        lon: Longitude in degrees
        lat: Latitude in degrees
        tz: Timezone offset in hours
        direct_normal_irradiance_scaling: Scaling factor for DNI
        diffuse_irradiance_scaling: Scaling factor for DHI
        show_plot: Whether to display a matplotlib plot
        with_reflections: If True, use full RadiationModel with reflections
        **kwargs: Additional parameters including:
            - computation_mask (np.ndarray): Optional 2D boolean mask
            - start_time (str): Start time 'MM-DD HH:MM:SS'
            - end_time (str): End time 'MM-DD HH:MM:SS'
            - view_point_height (float): Observer height
            - use_sky_patches (bool): Use sky patch optimization (default: True)
            - sky_discretization (str): 'tregenza', 'reinhart', etc.
            - progress_report (bool): Print progress
    
    Returns:
        2D numpy array of cumulative irradiance (Wh/m²)
    """
    import time
    from datetime import datetime
    import pytz
    
    kwargs = kwargs.copy()
    computation_mask = kwargs.pop('computation_mask', None)
    view_point_height = kwargs.pop('view_point_height', 1.5)
    colormap = kwargs.pop('colormap', 'magma')
    start_time = kwargs.pop('start_time', '01-01 05:00:00')
    end_time = kwargs.pop('end_time', '01-01 20:00:00')
    progress_report = kwargs.pop('progress_report', False)
    use_sky_patches = kwargs.pop('use_sky_patches', True)
    sky_discretization = kwargs.pop('sky_discretization', 'tregenza')
    
    if df.empty:
        raise ValueError("No data in EPW dataframe.")
    
    # Filter dataframe to period
    df_period_utc = filter_df_to_period(df, start_time, end_time, tz)
    
    # Get solar positions
    solar_positions = get_solar_positions_astral(df_period_utc.index, lon, lat)
    
    # Compute base diffuse map
    diffuse_kwargs = kwargs.copy()
    diffuse_kwargs.update({'show_plot': False, 'obj_export': False})
    base_diffuse_map = get_diffuse_solar_irradiance_map(
        voxcity,
        diffuse_irradiance=1.0,
        with_reflections=False,
        **diffuse_kwargs
    )
    
    voxel_data = voxcity.voxels.classes
    nx, ny, _ = voxel_data.shape
    cumulative_map = np.zeros((nx, ny))
    mask_map = np.ones((nx, ny), dtype=bool)
    
    direct_kwargs = kwargs.copy()
    direct_kwargs.update({
        'show_plot': False, 
        'view_point_height': view_point_height, 
        'obj_export': False,
        'with_reflections': with_reflections
    })
    
    if use_sky_patches:
        # Use sky patch aggregation for efficiency
        from ..sky import (
            generate_tregenza_patches,
            generate_reinhart_patches,
            generate_uniform_grid_patches,
            generate_fibonacci_patches,
            get_tregenza_patch_index
        )
        
        t0 = time.perf_counter() if progress_report else 0
        
        # Extract arrays
        azimuth_arr = solar_positions['azimuth'].to_numpy()
        elevation_arr = solar_positions['elevation'].to_numpy()
        dni_arr = df_period_utc['DNI'].to_numpy() * direct_normal_irradiance_scaling
        dhi_arr = df_period_utc['DHI'].to_numpy() * diffuse_irradiance_scaling
        time_step_hours = kwargs.get('time_step_hours', 1.0)
        
        # Generate sky patches
        if sky_discretization.lower() == 'tregenza':
            patches, directions, solid_angles = generate_tregenza_patches()
        elif sky_discretization.lower() == 'reinhart':
            mf = kwargs.get('reinhart_mf', kwargs.get('mf', 4))
            patches, directions, solid_angles = generate_reinhart_patches(mf=mf)
        elif sky_discretization.lower() == 'uniform':
            n_az = kwargs.get('sky_n_azimuth', kwargs.get('n_azimuth', 36))
            n_el = kwargs.get('sky_n_elevation', kwargs.get('n_elevation', 9))
            patches, directions, solid_angles = generate_uniform_grid_patches(n_az, n_el)
        elif sky_discretization.lower() == 'fibonacci':
            n_patches = kwargs.get('sky_n_patches', kwargs.get('n_patches', 145))
            patches, directions, solid_angles = generate_fibonacci_patches(n_patches=n_patches)
        else:
            raise ValueError(f"Unknown sky discretization method: {sky_discretization}")
        
        n_patches = len(patches)
        n_timesteps = len(azimuth_arr)
        cumulative_dni = np.zeros(n_patches, dtype=np.float64)
        
        # Vectorized DHI accumulation
        valid_dhi_mask = dhi_arr > 0
        total_cumulative_dhi = np.sum(dhi_arr[valid_dhi_mask]) * time_step_hours
        
        # DNI binning
        for i in range(n_timesteps):
            elev = elevation_arr[i]
            if elev <= 0:
                continue
            
            az = azimuth_arr[i]
            dni = dni_arr[i]
            
            if dni <= 0:
                continue
            
            patch_idx = int(get_tregenza_patch_index(float(az), float(elev)))
            if patch_idx >= 0 and patch_idx < n_patches:
                cumulative_dni[patch_idx] += dni * time_step_hours
        
        active_mask = cumulative_dni > 0
        n_active = int(np.sum(active_mask))
        
        if progress_report:
            bin_time = time.perf_counter() - t0
            print(f"Sky patch optimization: {n_timesteps} timesteps -> {n_active} active patches ({sky_discretization})")
            print(f"  Sun position binning: {bin_time:.3f}s")
            print(f"  Total cumulative DHI: {total_cumulative_dhi:.1f} Wh/m²")
            if with_reflections:
                print("  Using RadiationModel with multi-bounce reflections")
        
        # Diffuse component
        cumulative_diffuse = base_diffuse_map * total_cumulative_dhi
        cumulative_map += np.nan_to_num(cumulative_diffuse, nan=0.0)
        mask_map &= ~np.isnan(cumulative_diffuse)
        
        # Direct component - loop over active patches
        active_indices = np.where(active_mask)[0]
        
        # Pre-warm the model
        if with_reflections and len(active_indices) > 0:
            n_reflection_steps = kwargs.get('n_reflection_steps', 2)
            _ = get_or_create_radiation_model(
                voxcity,
                n_reflection_steps=n_reflection_steps,
                progress_report=progress_report
            )
        
        if progress_report:
            t_patch_start = time.perf_counter()
        
        for i, patch_idx in enumerate(active_indices):
            az_deg = patches[patch_idx, 0]
            el_deg = patches[patch_idx, 1]
            cumulative_dni_patch = cumulative_dni[patch_idx]
            
            if with_reflections:
                direct_map, _, reflected_map = _compute_ground_irradiance_with_reflections(
                    voxcity=voxcity,
                    azimuth_degrees_ori=az_deg,
                    elevation_degrees=el_deg,
                    direct_normal_irradiance=1.0,
                    diffuse_irradiance=0.0,
                    view_point_height=view_point_height,
                    **kwargs
                )
                patch_contribution = (direct_map + np.nan_to_num(reflected_map, nan=0.0)) * cumulative_dni_patch
            else:
                direct_map = get_direct_solar_irradiance_map(
                    voxcity,
                    az_deg,
                    el_deg,
                    direct_normal_irradiance=1.0,
                    **direct_kwargs
                )
                patch_contribution = direct_map * cumulative_dni_patch
            
            mask_map &= ~np.isnan(patch_contribution)
            cumulative_map += np.nan_to_num(patch_contribution, nan=0.0)
            
            if progress_report and ((i + 1) % max(1, len(active_indices) // 10) == 0 or i == len(active_indices) - 1):
                elapsed = time.perf_counter() - t_patch_start
                pct = (i + 1) * 100.0 / len(active_indices)
                avg_per_patch = elapsed / (i + 1)
                eta = avg_per_patch * (len(active_indices) - i - 1)
                print(f"  Patch {i+1}/{len(active_indices)} ({pct:.1f}%) - elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
        
        if progress_report:
            total_patch_time = time.perf_counter() - t_patch_start
            print(f"  Total patch processing: {total_patch_time:.2f}s ({n_active} patches)")
    
    else:
        # Per-timestep path
        if progress_report and with_reflections:
            print("  Using RadiationModel with multi-bounce reflections (per-timestep)")
        
        for idx, (time_utc, row) in enumerate(df_period_utc.iterrows()):
            DNI = float(row['DNI']) * direct_normal_irradiance_scaling
            DHI = float(row['DHI']) * diffuse_irradiance_scaling
            
            solpos = solar_positions.loc[time_utc]
            azimuth_degrees = float(solpos['azimuth'])
            elevation_degrees_val = float(solpos['elevation'])
            
            if with_reflections:
                direct_map, diffuse_map_ts, reflected_map = _compute_ground_irradiance_with_reflections(
                    voxcity=voxcity,
                    azimuth_degrees_ori=azimuth_degrees,
                    elevation_degrees=elevation_degrees_val,
                    direct_normal_irradiance=DNI,
                    diffuse_irradiance=DHI,
                    view_point_height=view_point_height,
                    **kwargs
                )
                combined = (np.nan_to_num(direct_map, nan=0.0) + 
                           np.nan_to_num(diffuse_map_ts, nan=0.0) + 
                           np.nan_to_num(reflected_map, nan=0.0))
                mask_map &= ~np.isnan(direct_map)
            else:
                direct_map = get_direct_solar_irradiance_map(
                    voxcity,
                    azimuth_degrees,
                    elevation_degrees_val,
                    direct_normal_irradiance=DNI,
                    **direct_kwargs
                )
                
                diffuse_contrib = base_diffuse_map * DHI
                combined = np.nan_to_num(direct_map, nan=0.0) + np.nan_to_num(diffuse_contrib, nan=0.0)
                mask_map &= ~np.isnan(direct_map) & ~np.isnan(diffuse_contrib)
            
            cumulative_map += combined
            
            if progress_report and (idx + 1) % max(1, len(df_period_utc) // 10) == 0:
                pct = (idx + 1) * 100.0 / len(df_period_utc)
                print(f"  Timestep {idx+1}/{len(df_period_utc)} ({pct:.1f}%)")
    
    # Apply mask
    cumulative_map = np.where(mask_map, cumulative_map, np.nan)
    
    # Apply computation mask
    if computation_mask is not None:
        if computation_mask.shape == cumulative_map.shape:
            cumulative_map = np.where(np.flipud(computation_mask), cumulative_map, np.nan)
        elif computation_mask.T.shape == cumulative_map.shape:
            cumulative_map = np.where(np.flipud(computation_mask.T), cumulative_map, np.nan)
    
    if show_plot:
        vmax = kwargs.get('vmax', float(np.nanmax(cumulative_map)) if not np.all(np.isnan(cumulative_map)) else 1.0)
        vmin = kwargs.get('vmin', 0.0)
        try:
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_bad(color='lightgray')
            plt.figure(figsize=(10, 8))
            plt.imshow(cumulative_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label='Cumulative Global Solar Irradiance (Wh/m²)')
            plt.axis('off')
            plt.show()
        except ImportError:
            pass
    
    return cumulative_map


def get_sunlight_hours(
    voxcity,
    mode: str = 'PSH',
    epw_file_path: str = None,
    download_nearest_epw: bool = False,
    dni_threshold: float = 120.0,
    show_plot: bool = False,
    lon: float = None,
    lat: float = None,
    tz: float = None,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated ground-level sunlight hours computation.
    
    Supports two modes:
    
    **PSH (Probable Sunlight Hours)**: Uses EPW weather data to account for cloud cover.
      Requires an EPW file (via epw_file_path or download_nearest_epw).
    
    **DSH (Direct Sun Hours)**: Assumes clear sky for all hours.
      Does NOT require an EPW file. Location (lon/lat) is automatically
      extracted from the VoxCity object, and timezone is inferred from
      the location. These can be overridden via the lon, lat, tz parameters.
    
    Args:
        voxcity: VoxCity object
        mode: 'PSH' (Probable Sunlight Hours) or 'DSH' (Direct Sun Hours)
        epw_file_path: Path to EPW file (required for PSH, optional for DSH)
        download_nearest_epw: If True, download nearest EPW based on location
        dni_threshold: DNI threshold for PSH mode (default: 120.0 W/m², WMO standard)
        show_plot: Whether to display a matplotlib plot
        lon: Longitude in degrees (optional, extracted from voxcity if not provided)
        lat: Latitude in degrees (optional, extracted from voxcity if not provided)
        tz: Timezone offset in hours (optional, inferred from location if not provided)
        **kwargs: Additional parameters
    
    Returns:
        2D numpy array with sunlight hours and metadata attribute
    """
    from datetime import datetime
    import pytz
    
    mode = mode.upper()
    if mode not in ('PSH', 'DSH'):
        raise ValueError(f"mode must be 'PSH' or 'DSH', got '{mode}'")
    
    kwargs = dict(kwargs)
    period_start = kwargs.pop('period_start', '01-01 00:00:00')
    period_end = kwargs.pop('period_end', '12-31 23:59:59')
    time_step_hours = float(kwargs.pop('time_step_hours', 1.0))
    progress_report = kwargs.pop('progress_report', False)
    computation_mask = kwargs.pop('computation_mask', None)
    min_elevation = float(kwargs.pop('min_elevation', 0.0))
    view_point_height = kwargs.pop('view_point_height', 1.5)
    colormap = kwargs.pop('colormap', 'magma')
    use_sky_patches = kwargs.pop('use_sky_patches', True)
    sky_discretization = kwargs.pop('sky_discretization', 'tregenza')
    
    # Load data depending on mode
    if mode == 'DSH' and epw_file_path is None and not download_nearest_epw:
        # DSH mode without EPW: derive location and timezone from voxcity
        if lat is None or lon is None:
            _lat, _lon = get_location_from_voxcity(voxcity)
            if lat is None:
                lat = _lat
            if lon is None:
                lon = _lon
        if tz is None:
            tz = get_timezone_offset_from_location(lon, lat)
        
        # Generate synthetic annual hourly timestamps
        weather_df = generate_annual_hourly_dataframe()
        
        if progress_report:
            print(f"  Mode: {mode} (no EPW file needed)")
            print(f"  Location: lon={lon:.4f}, lat={lat:.4f}, tz={tz}")
    else:
        # PSH mode or DSH with EPW provided
        weather_df, lon_epw, lat_epw, tz_epw = load_epw_data(
            epw_file_path=epw_file_path,
            download_nearest_epw=download_nearest_epw,
            voxcity=voxcity,
            **kwargs
        )
        # Use EPW values as defaults, but allow user overrides
        if lon is None:
            lon = lon_epw
        if lat is None:
            lat = lat_epw
        if tz is None:
            tz = tz_epw
        
        if progress_report:
            print(f"  Mode: {mode}")
            print(f"  Location: lon={lon:.4f}, lat={lat:.4f}, tz={tz}")
        
        if mode == 'PSH' and 'DNI' not in weather_df.columns:
            raise ValueError("Weather dataframe must have 'DNI' column for PSH mode.")
    
    # Filter dataframe to period
    df_period_utc = filter_df_to_period(weather_df, period_start, period_end, tz)
    
    # Get solar positions
    solar_positions = get_solar_positions_astral(df_period_utc.index, lon, lat)
    
    # Get grid dimensions
    voxel_data = voxcity.voxels.classes
    nx, ny, _ = voxel_data.shape
    
    sunlight_hours_map = np.zeros((nx, ny), dtype=np.float64)
    mask_map = np.ones((nx, ny), dtype=bool)
    potential_hours = 0.0
    
    # Extract arrays
    elevation_arr = solar_positions['elevation'].to_numpy()
    azimuth_arr = solar_positions['azimuth'].to_numpy()
    n_timesteps = len(elevation_arr)
    
    if mode == 'PSH':
        dni_arr = df_period_utc['DNI'].to_numpy()
    
    # Select sunshine timesteps
    sunshine_timesteps = []
    for t_idx in range(n_timesteps):
        elev = elevation_arr[t_idx]
        
        if mode == 'PSH':
            dni = dni_arr[t_idx]
            if elev > 0 and dni >= dni_threshold:
                sunshine_timesteps.append(t_idx)
                potential_hours += time_step_hours
        else:
            if elev > min_elevation:
                sunshine_timesteps.append(t_idx)
                potential_hours += time_step_hours
    
    n_sunshine = len(sunshine_timesteps)
    
    if progress_report:
        print(f"  Timesteps in period: {n_timesteps}")
        print(f"  Sunshine timesteps: {n_sunshine}")
        print(f"  Potential sunlight hours: {potential_hours:.1f} h")
    
    if n_sunshine == 0:
        result = np.zeros((nx, ny), dtype=np.float64)
        result = add_metadata_to_array(result, {
            'potential_sunlight_hours': potential_hours,
            'mode': mode,
            'dni_threshold': dni_threshold if mode == 'PSH' else None,
            'min_elevation': min_elevation if mode == 'DSH' else None
        })
        return result
    
    direct_kwargs = kwargs.copy()
    direct_kwargs.update({
        'show_plot': False,
        'view_point_height': view_point_height,
        'obj_export': False
    })
    
    if use_sky_patches:
        # Use sky patch aggregation
        import time as time_module
        from ..sky import (
            generate_tregenza_patches,
            generate_reinhart_patches,
            generate_uniform_grid_patches,
            generate_fibonacci_patches,
            get_tregenza_patch_index
        )
        
        t0 = time_module.perf_counter() if progress_report else 0
        
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
            n_patches_fib = kwargs.get('sky_n_patches', 145)
            patches, directions, solid_angles = generate_fibonacci_patches(n_patches=n_patches_fib)
        else:
            raise ValueError(f"Unknown sky discretization method: {sky_discretization}")
        
        n_patches = len(patches)
        hours_per_patch = np.zeros(n_patches, dtype=np.float64)
        
        # Bin sunshine timesteps to patches
        for t_idx in sunshine_timesteps:
            elev = elevation_arr[t_idx]
            az = azimuth_arr[t_idx]
            
            patch_idx = int(get_tregenza_patch_index(float(az), float(elev)))
            if 0 <= patch_idx < n_patches:
                hours_per_patch[patch_idx] += time_step_hours
        
        active_mask = hours_per_patch > 0
        n_active = int(np.sum(active_mask))
        active_indices = np.where(active_mask)[0]
        
        if progress_report:
            bin_time = time_module.perf_counter() - t0
            print(f"Sky patch optimization: {n_sunshine} timesteps -> {n_active} active patches")
            print(f"  Sun position binning: {bin_time:.3f}s")
        
        if progress_report:
            t_patch_start = time_module.perf_counter()
        
        for i, patch_idx in enumerate(active_indices):
            az_deg = patches[patch_idx, 0]
            el_deg = patches[patch_idx, 1]
            patch_hours = hours_per_patch[patch_idx]
            
            direct_map = get_direct_solar_irradiance_map(
                voxcity,
                azimuth_degrees_ori=az_deg,
                elevation_degrees=el_deg,
                direct_normal_irradiance=1.0,
                **direct_kwargs
            )
            
            receives_sun = np.nan_to_num(direct_map, nan=0.0) > 0.0
            sunlight_hours_map += receives_sun.astype(np.float64) * patch_hours
            
            mask_map &= ~np.isnan(direct_map)
            
            if progress_report and ((i + 1) % max(1, n_active // 10) == 0 or i == n_active - 1):
                elapsed = time_module.perf_counter() - t_patch_start
                pct = (i + 1) * 100.0 / n_active
                print(f"  Patch {i+1}/{n_active} ({pct:.1f}%) - elapsed: {elapsed:.1f}s")
        
        if progress_report:
            total_patch_time = time_module.perf_counter() - t_patch_start
            print(f"  Total patch processing: {total_patch_time:.2f}s")
    
    else:
        # Per-timestep path
        for i, t_idx in enumerate(sunshine_timesteps):
            elev = elevation_arr[t_idx]
            az = azimuth_arr[t_idx]
            
            direct_map = get_direct_solar_irradiance_map(
                voxcity,
                azimuth_degrees_ori=az,
                elevation_degrees=elev,
                direct_normal_irradiance=1.0,
                **direct_kwargs
            )
            
            receives_sun = np.nan_to_num(direct_map, nan=0.0) > 0.0
            sunlight_hours_map += receives_sun.astype(np.float64) * time_step_hours
            
            mask_map &= ~np.isnan(direct_map)
            
            if progress_report and ((i + 1) % max(1, n_sunshine // 10) == 0 or i == n_sunshine - 1):
                pct = (i + 1) * 100.0 / n_sunshine
                print(f"  Processed {i+1}/{n_sunshine} ({pct:.1f}%)")
    
    sunlight_hours_map = np.where(mask_map, sunlight_hours_map, np.nan)
    
    # Apply computation mask
    if computation_mask is not None:
        if computation_mask.shape == sunlight_hours_map.shape:
            sunlight_hours_map = np.where(np.flipud(computation_mask), sunlight_hours_map, np.nan)
        elif computation_mask.T.shape == sunlight_hours_map.shape:
            sunlight_hours_map = np.where(np.flipud(computation_mask.T), sunlight_hours_map, np.nan)
    
    if progress_report:
        print(f"Sunlight hours complete:")
        print(f"  Potential hours: {potential_hours:.1f} h")
        print(f"  Mean sunlight hours: {np.nanmean(sunlight_hours_map):.1f} h")
    
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            vmax = kwargs.get('vmax', potential_hours)
            vmin = kwargs.get('vmin', 0.0)
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_bad(color='lightgray')
            plt.figure(figsize=(10, 8))
            plt.imshow(sunlight_hours_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            mode_label = "PSH" if mode == 'PSH' else "DSH"
            plt.colorbar(label=f'{mode_label} Sunlight Hours (h)')
            plt.axis('off')
            plt.title(f"Ground-Level {mode_label}")
            plt.show()
        except ImportError:
            pass
    
    # Add metadata
    sunlight_fraction_map = sunlight_hours_map / potential_hours if potential_hours > 0 else np.zeros_like(sunlight_hours_map)
    result = add_metadata_to_array(sunlight_hours_map, {
        'potential_sunlight_hours': potential_hours,
        'sunlight_fraction': sunlight_fraction_map,
        'mode': mode,
        'dni_threshold': dni_threshold if mode == 'PSH' else None,
        'min_elevation': min_elevation if mode == 'DSH' else None
    })
    
    return result


# =============================================================================
# Export Helper
# =============================================================================

def _export_irradiance_to_obj(voxcity, irradiance_map: np.ndarray, output_name: str = 'irradiance', **kwargs):
    """Export irradiance map to OBJ file using VoxCity utilities."""
    try:
        from voxcity.exporter.obj import grid_to_obj
        meshsize = voxcity.voxels.meta.meshsize
        dem_grid = voxcity.dem.elevation if hasattr(voxcity, 'dem') and voxcity.dem else np.zeros_like(irradiance_map)
        output_dir = kwargs.get('output_directory', 'output')
        view_point_height = kwargs.get('view_point_height', 1.5)
        colormap = kwargs.get('colormap', 'magma')
        vmin = kwargs.get('vmin', 0.0)
        vmax = kwargs.get('vmax', float(np.nanmax(irradiance_map)) if not np.all(np.isnan(irradiance_map)) else 1.0)
        num_colors = kwargs.get('num_colors', 10)
        alpha = kwargs.get('alpha', 1.0)
        
        grid_to_obj(
            irradiance_map,
            dem_grid,
            output_dir,
            output_name,
            meshsize,
            view_point_height,
            colormap_name=colormap,
            num_colors=num_colors,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax
        )
    except ImportError:
        print("VoxCity exporter.obj required for OBJ export")
