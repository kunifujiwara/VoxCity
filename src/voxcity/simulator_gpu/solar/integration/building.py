"""
Building surface solar irradiance functions for VoxCity.

This module provides GPU-accelerated building surface solar irradiance calculations:
- Building solar irradiance (instantaneous)
- Cumulative building solar irradiance
- Building sunlight hours (PSH and DSH modes)
- EPW-based building irradiance wrapper

These functions match the voxcity.simulator.solar API signatures for 
drop-in replacement with GPU acceleration.
"""

import numpy as np
from typing import Optional, Tuple, Dict

from .utils import (
    VOXCITY_BUILDING_CODE,
    get_location_from_voxcity,
    compute_sun_direction,
    filter_df_to_period,
    parse_time_period,
    load_epw_data,
    get_solar_positions_astral,
    compute_boundary_vertical_mask,
    apply_computation_mask_to_faces,
)

from .caching import (
    get_building_radiation_model_cache,
    get_or_create_building_radiation_model,
    CachedBuildingRadiationModel,
)


# =============================================================================
# Public API Functions
# =============================================================================

def get_building_solar_irradiance(
    voxcity,
    building_svf_mesh=None,
    azimuth_degrees_ori: float = None,
    elevation_degrees: float = None,
    direct_normal_irradiance: float = None,
    diffuse_irradiance: float = None,
    **kwargs
):
    """
    GPU-accelerated building surface solar irradiance computation.
    
    Uses cached RadiationModel to avoid recomputing SVF/CSF matrices for each timestep.
    
    Args:
        voxcity: VoxCity object
        building_svf_mesh: Pre-computed mesh with SVF values (optional)
        azimuth_degrees_ori: Solar azimuth in degrees (0=North, clockwise)
        elevation_degrees: Solar elevation in degrees above horizon
        direct_normal_irradiance: DNI in W/m²
        diffuse_irradiance: DHI in W/m²
        **kwargs: Additional parameters including:
            - with_reflections (bool): Enable multi-bounce reflections (default: False)
            - n_reflection_steps (int): Number of reflection bounces (default: 2)
            - building_class_id (int): Building voxel class code (default: -3)
            - computation_mask (np.ndarray): Optional 2D boolean mask
            - progress_report (bool): Print progress (default: False)
    
    Returns:
        Trimesh object with irradiance values in metadata
    """
    # Handle positional argument order from VoxCity API
    if isinstance(building_svf_mesh, (int, float)):
        diffuse_irradiance = direct_normal_irradiance
        direct_normal_irradiance = elevation_degrees
        elevation_degrees = azimuth_degrees_ori
        azimuth_degrees_ori = building_svf_mesh
        building_svf_mesh = None
    
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    building_id_grid = voxcity.buildings.ids
    ny_vc, nx_vc, nz = voxel_data.shape
    
    # Extract parameters
    progress_report = kwargs.pop('progress_report', False)
    building_class_id = kwargs.pop('building_class_id', -3)
    n_reflection_steps = kwargs.pop('n_reflection_steps', 2)
    with_reflections = kwargs.pop('with_reflections', False)
    computation_mask = kwargs.pop('computation_mask', None)
    
    if not with_reflections:
        n_reflection_steps = 0
    
    # Get cached or create new RadiationModel
    model, is_building_surf = get_or_create_building_radiation_model(
        voxcity,
        n_reflection_steps=n_reflection_steps,
        progress_report=progress_report,
        building_class_id=building_class_id,
        **kwargs
    )
    
    # Set solar position
    sun_dir_x, sun_dir_y, sun_dir_z, cos_zenith = compute_sun_direction(
        azimuth_degrees_ori, elevation_degrees
    )
    
    model.solar_calc.sun_direction[None] = (sun_dir_x, sun_dir_y, sun_dir_z)
    model.solar_calc.cos_zenith[None] = cos_zenith
    model.solar_calc.sun_up[None] = 1 if elevation_degrees > 0 else 0
    
    # Compute radiation
    model.compute_shortwave_radiation(
        sw_direct=direct_normal_irradiance,
        sw_diffuse=diffuse_irradiance
    )
    
    # Extract surface irradiance
    n_surfaces = model.surfaces.count
    sw_in_direct_all = model.surfaces.sw_in_direct.to_numpy()
    sw_in_diffuse_all = model.surfaces.sw_in_diffuse.to_numpy()
    
    if hasattr(model.surfaces, 'sw_in_reflected'):
        sw_in_reflected_all = model.surfaces.sw_in_reflected.to_numpy()
    else:
        sw_in_reflected_all = np.zeros_like(sw_in_direct_all)
    
    total_sw_all = sw_in_direct_all + sw_in_diffuse_all + sw_in_reflected_all
    
    # Get building indices from cache
    cache = get_building_radiation_model_cache()
    bldg_indices = cache.bldg_indices if cache else np.where(is_building_surf)[0]
    
    # Get or create building mesh
    if building_svf_mesh is not None:
        building_mesh = building_svf_mesh
        face_svf = building_mesh.metadata.get('svf') if hasattr(building_mesh, 'metadata') else None
    elif cache is not None and cache.cached_building_mesh is not None:
        building_mesh = cache.cached_building_mesh
        face_svf = None
    else:
        try:
            from voxcity.geoprocessor.mesh import create_voxel_mesh
            if progress_report:
                print("  Creating building mesh (first call, will be cached)...")
            building_mesh = create_voxel_mesh(
                voxel_data,
                building_class_id,
                meshsize,
                building_id_grid=building_id_grid,
                mesh_type='open_air'
            )
            if building_mesh is None or len(building_mesh.faces) == 0:
                print("No building surfaces found.")
                return None
            if cache is not None:
                cache.cached_building_mesh = building_mesh
        except ImportError:
            print("VoxCity geoprocessor.mesh required for mesh creation")
            return None
        face_svf = None
    
    n_mesh_faces = len(building_mesh.faces)
    
    # Map palm_solar values to mesh faces
    if len(bldg_indices) > 0:
        if (cache is not None and 
            cache.mesh_to_surface_idx is not None and 
            len(cache.mesh_to_surface_idx) == n_mesh_faces):
            mesh_to_surface_idx = cache.mesh_to_surface_idx
        else:
            from scipy.spatial import cKDTree
            
            surf_centers_all = model.surfaces.center.to_numpy()[:n_surfaces]
            bldg_centers = surf_centers_all[bldg_indices]
            
            if cache is not None and cache.mesh_face_centers is not None:
                mesh_face_centers = cache.mesh_face_centers
            else:
                mesh_face_centers = building_mesh.triangles_center
                if cache is not None:
                    cache.mesh_face_centers = mesh_face_centers.copy()
                    cache.mesh_face_normals = building_mesh.face_normals.copy()
            
            tree = cKDTree(bldg_centers)
            distances, nearest_idx = tree.query(mesh_face_centers, k=1)
            
            mesh_to_surface_idx = bldg_indices[nearest_idx]
            
            if cache is not None:
                cache.mesh_to_surface_idx = mesh_to_surface_idx
        
        sw_in_direct = sw_in_direct_all[mesh_to_surface_idx]
        sw_in_diffuse = sw_in_diffuse_all[mesh_to_surface_idx]
        sw_in_reflected = sw_in_reflected_all[mesh_to_surface_idx]
        total_sw = total_sw_all[mesh_to_surface_idx]
    else:
        sw_in_direct = np.zeros(n_mesh_faces, dtype=np.float32)
        sw_in_diffuse = np.zeros(n_mesh_faces, dtype=np.float32)
        sw_in_reflected = np.zeros(n_mesh_faces, dtype=np.float32)
        total_sw = np.zeros(n_mesh_faces, dtype=np.float32)
    
    # Handle boundary faces
    if cache is not None and cache.boundary_mask is not None and len(cache.boundary_mask) == n_mesh_faces:
        is_boundary_vertical = cache.boundary_mask
    else:
        grid_bounds_real = np.array([
            [0.0, 0.0, 0.0],
            [nx_vc * meshsize, ny_vc * meshsize, nz * meshsize]
        ], dtype=np.float64)
        boundary_epsilon = meshsize * 0.05
        
        if cache is not None and cache.mesh_face_centers is not None:
            mesh_face_centers = cache.mesh_face_centers
            mesh_face_normals = cache.mesh_face_normals
        else:
            mesh_face_centers = building_mesh.triangles_center
            mesh_face_normals = building_mesh.face_normals
        
        is_boundary_vertical = compute_boundary_vertical_mask(
            mesh_face_centers, mesh_face_normals, grid_bounds_real, boundary_epsilon
        )
        
        if cache is not None:
            cache.boundary_mask = is_boundary_vertical
    
    sw_in_direct = np.where(is_boundary_vertical, np.nan, sw_in_direct)
    sw_in_diffuse = np.where(is_boundary_vertical, np.nan, sw_in_diffuse)
    sw_in_reflected = np.where(is_boundary_vertical, np.nan, sw_in_reflected)
    total_sw = np.where(is_boundary_vertical, np.nan, total_sw)
    
    # Apply computation mask
    if computation_mask is not None:
        if cache is not None and cache.mesh_face_centers is not None:
            mesh_face_centers = cache.mesh_face_centers
        else:
            mesh_face_centers = building_mesh.triangles_center
        
        sw_in_direct = apply_computation_mask_to_faces(
            sw_in_direct, mesh_face_centers, computation_mask, meshsize, (ny_vc, nx_vc)
        )
        sw_in_diffuse = apply_computation_mask_to_faces(
            sw_in_diffuse, mesh_face_centers, computation_mask, meshsize, (ny_vc, nx_vc)
        )
        sw_in_reflected = apply_computation_mask_to_faces(
            sw_in_reflected, mesh_face_centers, computation_mask, meshsize, (ny_vc, nx_vc)
        )
        total_sw = apply_computation_mask_to_faces(
            total_sw, mesh_face_centers, computation_mask, meshsize, (ny_vc, nx_vc)
        )
    
    building_mesh.metadata = {
        'irradiance_direct': sw_in_direct,
        'irradiance_diffuse': sw_in_diffuse,
        'irradiance_reflected': sw_in_reflected,
        'irradiance_total': total_sw,
        'direct': sw_in_direct,
        'diffuse': sw_in_diffuse,
        'global': total_sw,
    }
    if face_svf is not None:
        building_mesh.metadata['svf'] = face_svf
    
    if kwargs.get('obj_export', False):
        import os
        output_dir = kwargs.get('output_directory', 'output')
        output_file_name = kwargs.get('output_file_name', 'building_solar_irradiance')
        os.makedirs(output_dir, exist_ok=True)
        try:
            building_mesh.export(f"{output_dir}/{output_file_name}.obj")
        except Exception as e:
            print(f"Error exporting mesh: {e}")
    
    return building_mesh


def get_cumulative_building_solar_irradiance(
    voxcity,
    building_svf_mesh,
    weather_df,
    lon: float,
    lat: float,
    tz: float,
    direct_normal_irradiance_scaling: float = 1.0,
    diffuse_irradiance_scaling: float = 1.0,
    **kwargs
):
    """
    GPU-accelerated cumulative solar irradiance on building surfaces.
    
    Args:
        voxcity: VoxCity object
        building_svf_mesh: Trimesh object with SVF in metadata
        weather_df: pandas DataFrame with 'DNI' and 'DHI' columns
        lon: Longitude in degrees
        lat: Latitude in degrees
        tz: Timezone offset in hours
        direct_normal_irradiance_scaling: Scaling factor for DNI
        diffuse_irradiance_scaling: Scaling factor for DHI
        **kwargs: Additional parameters
    
    Returns:
        Trimesh object with cumulative irradiance (Wh/m²) in metadata
    """
    from datetime import datetime
    import pytz
    
    kwargs = dict(kwargs)
    period_start = kwargs.pop('period_start', '01-01 00:00:00')
    period_end = kwargs.pop('period_end', '12-31 23:59:59')
    time_step_hours = float(kwargs.pop('time_step_hours', 1.0))
    progress_report = kwargs.pop('progress_report', False)
    use_sky_patches = kwargs.pop('use_sky_patches', False)
    computation_mask = kwargs.pop('computation_mask', None)
    
    if weather_df.empty:
        raise ValueError("No data in weather dataframe.")
    
    # Filter dataframe
    df_period_utc = filter_df_to_period(weather_df, period_start, period_end, tz)
    
    # Get solar positions
    solar_positions = get_solar_positions_astral(df_period_utc.index, lon, lat)
    
    # Initialize
    result_mesh = building_svf_mesh.copy() if hasattr(building_svf_mesh, 'copy') else building_svf_mesh
    n_faces = len(result_mesh.faces) if hasattr(result_mesh, 'faces') else 0
    
    if n_faces == 0:
        raise ValueError("Building mesh has no faces")
    
    cumulative_direct = np.zeros(n_faces, dtype=np.float64)
    cumulative_diffuse = np.zeros(n_faces, dtype=np.float64)
    cumulative_global = np.zeros(n_faces, dtype=np.float64)
    
    face_svf = result_mesh.metadata.get('svf') if hasattr(result_mesh, 'metadata') else None
    
    # Extract arrays
    azimuth_arr = solar_positions['azimuth'].to_numpy()
    elevation_arr = solar_positions['elevation'].to_numpy()
    dni_arr = df_period_utc['DNI'].to_numpy() * direct_normal_irradiance_scaling
    dhi_arr = df_period_utc['DHI'].to_numpy() * diffuse_irradiance_scaling
    n_timesteps = len(azimuth_arr)
    
    if use_sky_patches:
        from ..sky import generate_sky_patches, get_tregenza_patch_index
        
        sky_discretization = kwargs.pop('sky_discretization', 'tregenza')
        sky_patches = generate_sky_patches(sky_discretization)
        patches = sky_patches.patches
        n_patches = sky_patches.n_patches
        cumulative_dni_per_patch = np.zeros(n_patches, dtype=np.float64)
        total_cumulative_dhi = 0.0
        
        # Bin sun positions
        for i in range(n_timesteps):
            elev = elevation_arr[i]
            dhi = dhi_arr[i]
            
            if dhi > 0:
                total_cumulative_dhi += dhi * time_step_hours
            
            if elev <= 0:
                continue
            
            az = azimuth_arr[i]
            dni = dni_arr[i]
            
            if dni <= 0:
                continue
            
            patch_idx = int(get_tregenza_patch_index(float(az), float(elev)))
            if 0 <= patch_idx < n_patches:
                cumulative_dni_per_patch[patch_idx] += dni * time_step_hours
        
        active_mask = cumulative_dni_per_patch > 0
        n_active = int(np.sum(active_mask))
        
        if progress_report:
            print(f"  Sky patch optimization: {n_timesteps} -> {n_active} active patches")
        
        # Diffuse component
        if face_svf is not None and len(face_svf) == n_faces:
            cumulative_diffuse = face_svf * total_cumulative_dhi
        else:
            diffuse_mesh = get_building_solar_irradiance(
                voxcity,
                building_svf_mesh=building_svf_mesh,
                azimuth_degrees_ori=180.0,
                elevation_degrees=45.0,
                direct_normal_irradiance=0.0,
                diffuse_irradiance=1.0,
                progress_report=False,
                **kwargs
            )
            if diffuse_mesh is not None and 'diffuse' in diffuse_mesh.metadata:
                base_diffuse = diffuse_mesh.metadata['diffuse']
                cumulative_diffuse = np.nan_to_num(base_diffuse, nan=0.0) * total_cumulative_dhi
        
        # Direct component
        active_indices = np.where(active_mask)[0]
        for i, patch_idx in enumerate(active_indices):
            az_deg = patches[patch_idx, 0]
            el_deg = patches[patch_idx, 1]
            cumulative_dni_patch = cumulative_dni_per_patch[patch_idx]
            
            irradiance_mesh = get_building_solar_irradiance(
                voxcity,
                building_svf_mesh=building_svf_mesh,
                azimuth_degrees_ori=az_deg,
                elevation_degrees=el_deg,
                direct_normal_irradiance=1.0,
                diffuse_irradiance=0.0,
                progress_report=False,
                **kwargs
            )
            
            if irradiance_mesh is not None and 'direct' in irradiance_mesh.metadata:
                direct_vals = irradiance_mesh.metadata['direct']
                if len(direct_vals) == n_faces:
                    cumulative_direct += np.nan_to_num(direct_vals, nan=0.0) * cumulative_dni_patch
            
            if progress_report and ((i + 1) % max(1, len(active_indices) // 10) == 0):
                print(f"  Patch {i+1}/{len(active_indices)} ({100*(i+1)/len(active_indices):.1f}%)")
        
        cumulative_global = cumulative_direct + cumulative_diffuse
    
    else:
        # Per-timestep
        for t_idx, (timestamp, row) in enumerate(df_period_utc.iterrows()):
            dni = float(row['DNI']) * direct_normal_irradiance_scaling
            dhi = float(row['DHI']) * diffuse_irradiance_scaling
            
            elevation = float(solar_positions.loc[timestamp, 'elevation'])
            azimuth = float(solar_positions.loc[timestamp, 'azimuth'])
            
            if elevation <= 0 or (dni <= 0 and dhi <= 0):
                continue
            
            irradiance_mesh = get_building_solar_irradiance(
                voxcity,
                building_svf_mesh=building_svf_mesh,
                azimuth_degrees_ori=azimuth,
                elevation_degrees=elevation,
                direct_normal_irradiance=dni,
                diffuse_irradiance=dhi,
                progress_report=False,
                **kwargs
            )
            
            if irradiance_mesh is not None and hasattr(irradiance_mesh, 'metadata'):
                if 'direct' in irradiance_mesh.metadata:
                    cumulative_direct += np.nan_to_num(irradiance_mesh.metadata['direct'], nan=0.0) * time_step_hours
                if 'diffuse' in irradiance_mesh.metadata:
                    cumulative_diffuse += np.nan_to_num(irradiance_mesh.metadata['diffuse'], nan=0.0) * time_step_hours
                if 'global' in irradiance_mesh.metadata:
                    cumulative_global += np.nan_to_num(irradiance_mesh.metadata['global'], nan=0.0) * time_step_hours
            
            if progress_report and (t_idx + 1) % max(1, n_timesteps // 10) == 0:
                print(f"  Processed {t_idx + 1}/{n_timesteps} ({100*(t_idx+1)/n_timesteps:.1f}%)")
    
    # Apply boundary handling
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    ny_vc, nx_vc, nz = voxel_data.shape
    grid_bounds_real = np.array([
        [0.0, 0.0, 0.0],
        [ny_vc * meshsize, nx_vc * meshsize, nz * meshsize]
    ], dtype=np.float64)
    boundary_epsilon = meshsize * 0.05
    
    mesh_face_centers = result_mesh.triangles_center
    mesh_face_normals = result_mesh.face_normals
    
    is_boundary_vertical = compute_boundary_vertical_mask(
        mesh_face_centers, mesh_face_normals, grid_bounds_real, boundary_epsilon
    )
    
    cumulative_direct[is_boundary_vertical] = np.nan
    cumulative_diffuse[is_boundary_vertical] = np.nan
    cumulative_global[is_boundary_vertical] = np.nan
    
    # Apply computation mask
    if computation_mask is not None:
        cumulative_direct = apply_computation_mask_to_faces(
            cumulative_direct, mesh_face_centers, computation_mask, meshsize, (ny_vc, nx_vc)
        )
        cumulative_diffuse = apply_computation_mask_to_faces(
            cumulative_diffuse, mesh_face_centers, computation_mask, meshsize, (ny_vc, nx_vc)
        )
        cumulative_global = apply_computation_mask_to_faces(
            cumulative_global, mesh_face_centers, computation_mask, meshsize, (ny_vc, nx_vc)
        )
    
    # Store results
    result_mesh.metadata = getattr(result_mesh, 'metadata', {})
    result_mesh.metadata['cumulative_direct'] = cumulative_direct
    result_mesh.metadata['cumulative_diffuse'] = cumulative_diffuse
    result_mesh.metadata['cumulative_global'] = cumulative_global
    result_mesh.metadata['direct'] = cumulative_direct
    result_mesh.metadata['diffuse'] = cumulative_diffuse
    result_mesh.metadata['global'] = cumulative_global
    if face_svf is not None:
        result_mesh.metadata['svf'] = face_svf
    
    return result_mesh


def get_building_sunlight_hours(
    voxcity,
    building_svf_mesh=None,
    mode: str = 'PSH',
    epw_file_path: str = None,
    download_nearest_epw: bool = False,
    dni_threshold: float = 120.0,
    **kwargs
):
    """
    GPU-accelerated sunlight hours computation for building surfaces.
    
    Supports PSH (Probable Sunlight Hours) and DSH (Direct Sun Hours) modes.
    
    Args:
        voxcity: VoxCity object
        building_svf_mesh: Trimesh object with building surfaces (optional)
        mode: 'PSH' or 'DSH'
        epw_file_path: Path to EPW file
        download_nearest_epw: If True, download nearest EPW
        dni_threshold: DNI threshold for PSH mode (default: 120.0 W/m²)
        **kwargs: Additional parameters
    
    Returns:
        Trimesh object with sunlight hours in metadata
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
    use_sky_patches = kwargs.pop('use_sky_patches', True)
    sky_discretization = kwargs.pop('sky_discretization', 'tregenza')
    
    # Load EPW data
    weather_df, lon, lat, tz = load_epw_data(
        epw_file_path=epw_file_path,
        download_nearest_epw=download_nearest_epw,
        voxcity=voxcity,
        **kwargs
    )
    
    if mode == 'PSH' and 'DNI' not in weather_df.columns:
        raise ValueError("Weather dataframe must have 'DNI' column for PSH mode.")
    
    # Filter dataframe
    df_period_utc = filter_df_to_period(weather_df, period_start, period_end, tz)
    
    # Get solar positions
    solar_positions = get_solar_positions_astral(df_period_utc.index, lon, lat)
    
    # Create building mesh if needed
    if building_svf_mesh is None:
        try:
            from voxcity.geoprocessor.mesh import create_voxel_mesh
            voxel_data = voxcity.voxels.classes
            meshsize = voxcity.voxels.meta.meshsize
            building_id_grid = voxcity.buildings.ids
            building_class_id = kwargs.pop('building_class_id', -3)
            building_svf_mesh = create_voxel_mesh(
                voxel_data,
                building_class_id,
                meshsize,
                building_id_grid=building_id_grid,
                mesh_type='open_air'
            )
        except ImportError:
            raise ImportError("VoxCity geoprocessor.mesh required for mesh creation")
    
    result_mesh = building_svf_mesh.copy() if hasattr(building_svf_mesh, 'copy') else building_svf_mesh
    n_faces = len(result_mesh.faces) if hasattr(result_mesh, 'faces') else 0
    
    if n_faces == 0:
        raise ValueError("Building mesh has no faces")
    
    sunlight_hours = np.zeros(n_faces, dtype=np.float64)
    potential_hours = 0.0
    
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
        print(f"  Mode: {mode}, Sunshine timesteps: {n_sunshine}, Potential hours: {potential_hours:.1f}")
    
    if n_sunshine == 0:
        result_mesh.metadata = getattr(result_mesh, 'metadata', {})
        result_mesh.metadata['sunlight_hours'] = sunlight_hours
        result_mesh.metadata['potential_sunlight_hours'] = potential_hours
        result_mesh.metadata['sunlight_fraction'] = np.zeros(n_faces, dtype=np.float64)
        result_mesh.metadata['mode'] = mode
        return result_mesh
    
    if use_sky_patches:
        from ..sky import (
            generate_tregenza_patches,
            generate_reinhart_patches,
            generate_uniform_grid_patches,
            generate_fibonacci_patches,
            get_tregenza_patch_index
        )
        
        if sky_discretization.lower() == 'tregenza':
            patches, directions, solid_angles = generate_tregenza_patches()
        elif sky_discretization.lower() == 'reinhart':
            mf = kwargs.get('reinhart_mf', 4)
            patches, directions, solid_angles = generate_reinhart_patches(mf=mf)
        elif sky_discretization.lower() == 'uniform':
            n_az = kwargs.get('sky_n_azimuth', 36)
            n_el = kwargs.get('sky_n_elevation', 9)
            patches, directions, solid_angles = generate_uniform_grid_patches(n_az, n_el)
        else:
            n_patches_fib = kwargs.get('sky_n_patches', 145)
            patches, directions, solid_angles = generate_fibonacci_patches(n_patches=n_patches_fib)
        
        n_patches_sky = len(patches)
        hours_per_patch = np.zeros(n_patches_sky, dtype=np.float64)
        
        for t_idx in sunshine_timesteps:
            elev = elevation_arr[t_idx]
            az = azimuth_arr[t_idx]
            patch_idx = int(get_tregenza_patch_index(float(az), float(elev)))
            if 0 <= patch_idx < n_patches_sky:
                hours_per_patch[patch_idx] += time_step_hours
        
        active_mask = hours_per_patch > 0
        active_indices = np.where(active_mask)[0]
        
        for i, patch_idx in enumerate(active_indices):
            az_deg = patches[patch_idx, 0]
            el_deg = patches[patch_idx, 1]
            patch_hours = hours_per_patch[patch_idx]
            
            irradiance_mesh = get_building_solar_irradiance(
                voxcity,
                building_svf_mesh=building_svf_mesh,
                azimuth_degrees_ori=az_deg,
                elevation_degrees=el_deg,
                direct_normal_irradiance=1.0,
                diffuse_irradiance=0.0,
                progress_report=False,
                **kwargs
            )
            
            if irradiance_mesh is not None and 'direct' in irradiance_mesh.metadata:
                direct_vals = irradiance_mesh.metadata['direct']
                if len(direct_vals) == n_faces:
                    receives_sun = np.nan_to_num(direct_vals, nan=0.0) > 0.0
                    sunlight_hours += receives_sun.astype(np.float64) * patch_hours
            
            if progress_report and ((i + 1) % max(1, len(active_indices) // 10) == 0):
                print(f"  Patch {i+1}/{len(active_indices)} ({100*(i+1)/len(active_indices):.1f}%)")
    else:
        for i, t_idx in enumerate(sunshine_timesteps):
            elev = elevation_arr[t_idx]
            az = azimuth_arr[t_idx]
            
            irradiance_mesh = get_building_solar_irradiance(
                voxcity,
                building_svf_mesh=building_svf_mesh,
                azimuth_degrees_ori=az,
                elevation_degrees=elev,
                direct_normal_irradiance=1.0,
                diffuse_irradiance=0.0,
                progress_report=False,
                **kwargs
            )
            
            if irradiance_mesh is not None and 'direct' in irradiance_mesh.metadata:
                direct_vals = irradiance_mesh.metadata['direct']
                if len(direct_vals) == n_faces:
                    receives_sun = np.nan_to_num(direct_vals, nan=0.0) > 0.0
                    sunlight_hours += receives_sun.astype(np.float64) * time_step_hours
            
            if progress_report and ((i + 1) % max(1, n_sunshine // 10) == 0):
                print(f"  Processed {i+1}/{n_sunshine} ({100*(i+1)/n_sunshine:.1f}%)")
    
    # Apply boundary handling
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    ny_vc, nx_vc, nz = voxel_data.shape
    grid_bounds_real = np.array([
        [0.0, 0.0, 0.0],
        [ny_vc * meshsize, nx_vc * meshsize, nz * meshsize]
    ], dtype=np.float64)
    
    mesh_face_centers = result_mesh.triangles_center
    mesh_face_normals = result_mesh.face_normals
    
    is_boundary_vertical = compute_boundary_vertical_mask(
        mesh_face_centers, mesh_face_normals, grid_bounds_real, meshsize * 0.05
    )
    sunlight_hours[is_boundary_vertical] = np.nan
    
    if computation_mask is not None:
        sunlight_hours = apply_computation_mask_to_faces(
            sunlight_hours, mesh_face_centers, computation_mask, meshsize, (ny_vc, nx_vc)
        )
    
    sunlight_fraction = sunlight_hours / potential_hours if potential_hours > 0 else np.zeros(n_faces)
    
    result_mesh.metadata = getattr(result_mesh, 'metadata', {})
    result_mesh.metadata['sunlight_hours'] = sunlight_hours
    result_mesh.metadata['potential_sunlight_hours'] = potential_hours
    result_mesh.metadata['sunlight_fraction'] = sunlight_fraction
    result_mesh.metadata['mode'] = mode
    if mode == 'PSH':
        result_mesh.metadata['dni_threshold'] = dni_threshold
    else:
        result_mesh.metadata['min_elevation'] = min_elevation
    
    return result_mesh


def get_building_global_solar_irradiance_using_epw(
    voxcity,
    calc_type: str = 'instantaneous',
    direct_normal_irradiance_scaling: float = 1.0,
    diffuse_irradiance_scaling: float = 1.0,
    building_svf_mesh=None,
    **kwargs
):
    """
    GPU-accelerated building surface irradiance using EPW weather data.
    
    Args:
        voxcity: VoxCity object
        calc_type: 'instantaneous' or 'cumulative'
        direct_normal_irradiance_scaling: Scaling factor for DNI
        diffuse_irradiance_scaling: Scaling factor for DHI
        building_svf_mesh: Pre-computed building mesh (optional)
        **kwargs: Additional parameters
    
    Returns:
        Trimesh object with irradiance values in metadata
    """
    from datetime import datetime
    import pytz
    
    progress_report = kwargs.get('progress_report', False)
    kwargs = dict(kwargs)
    kwargs.pop('progress_report', None)
    
    # Load EPW data
    df, lon, lat, tz = load_epw_data(
        epw_file_path=kwargs.pop('epw_file_path', None),
        download_nearest_epw=kwargs.pop('download_nearest_epw', False),
        voxcity=voxcity,
        **kwargs
    )
    
    # Create building mesh if needed
    if building_svf_mesh is None:
        try:
            from voxcity.geoprocessor.mesh import create_voxel_mesh
            building_class_id = kwargs.get('building_class_id', -3)
            voxel_data = voxcity.voxels.classes
            meshsize = voxcity.voxels.meta.meshsize
            building_id_grid = voxcity.buildings.ids
            
            building_svf_mesh = create_voxel_mesh(
                voxel_data,
                building_class_id,
                meshsize,
                building_id_grid=building_id_grid,
                mesh_type='open_air'
            )
        except ImportError:
            pass
    
    if calc_type == 'instantaneous':
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
        
        return get_building_solar_irradiance(
            voxcity,
            building_svf_mesh=building_svf_mesh,
            azimuth_degrees_ori=azimuth_degrees,
            elevation_degrees=elevation_degrees,
            direct_normal_irradiance=DNI,
            diffuse_irradiance=DHI,
            **kwargs
        )
    
    elif calc_type == 'cumulative':
        period_start = kwargs.get('period_start', '01-01 00:00:00')
        period_end = kwargs.get('period_end', '12-31 23:59:59')
        time_step_hours = float(kwargs.get('time_step_hours', 1.0))
        
        kwargs.pop('period_start', None)
        kwargs.pop('period_end', None)
        kwargs.pop('time_step_hours', None)
        
        return get_cumulative_building_solar_irradiance(
            voxcity,
            building_svf_mesh=building_svf_mesh,
            weather_df=df,
            lon=lon,
            lat=lat,
            tz=tz,
            direct_normal_irradiance_scaling=direct_normal_irradiance_scaling,
            diffuse_irradiance_scaling=diffuse_irradiance_scaling,
            period_start=period_start,
            period_end=period_end,
            time_step_hours=time_step_hours,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown calc_type: {calc_type}. Use 'instantaneous' or 'cumulative'.")
