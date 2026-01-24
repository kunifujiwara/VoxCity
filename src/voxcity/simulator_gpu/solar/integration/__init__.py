"""
VoxCity GPU-accelerated Solar Irradiance Integration Package

This package provides GPU-accelerated solar irradiance calculations for VoxCity
voxel grids using Taichi. It supports:

- Ground-level (horizontal) solar irradiance maps
- Building surface irradiance with directional faces
- Volumetric 3D radiation fields for thermal comfort analysis
- Cumulative time-integrated irradiance and sunlight hours
- Sky patch discretization (Tregenza, Reinhart, uniform, Fibonacci)
- Surface reflections (optional)

Modules:
    - utils: Common helper functions
    - caching: Cache infrastructure for models and calculators
    - ground: Ground-level solar irradiance functions
    - building: Building surface irradiance functions
    - volumetric: Volumetric 3D radiation field functions

Public API:
    The package re-exports all public functions for backward compatibility.
    Import directly from this package:
    
        from voxcity.simulator_gpu.solar.integration import (
            get_global_solar_irradiance_map,
            get_cumulative_global_solar_irradiance,
            get_sunlight_hours,
            get_building_solar_irradiance,
            get_volumetric_solar_irradiance_map,
            # ... etc
        )
"""

from __future__ import annotations

# Re-export utility functions
from .utils import (
    get_location_from_voxcity,
    convert_voxel_data_to_arrays,
    compute_valid_ground_vectorized,
    compute_sun_direction,
    parse_time_period,
    filter_df_to_period,
    load_epw_data,
    get_solar_positions_astral,
    extract_terrain_following_slice,
    accumulate_terrain_following_slice,
    ArrayWithMetadata,
    compute_boundary_vertical_mask,
    apply_computation_mask_to_faces,
)

# Re-export caching infrastructure
from .caching import (
    # Dataclasses
    LandCoverAlbedo,
    VoxCityDomainResult,
    CachedRadiationModel,
    CachedBuildingRadiationModel,
    CachedGPURayTracer,
    # Cache management
    clear_radiation_model_cache,
    clear_building_radiation_model_cache,
    clear_gpu_ray_tracer_cache,
    clear_volumetric_flux_cache,
    clear_all_caches,
    clear_all_radiation_caches,
    reset_solar_taichi_cache,
    get_radiation_model_cache,
    get_building_radiation_model_cache,
    set_radiation_model_cache,
    set_building_radiation_model_cache,
    # Creator functions
    get_or_create_radiation_model,
    get_or_create_building_radiation_model,
    get_or_create_gpu_ray_tracer,
    get_or_create_volumetric_calculator,
    # Low-level GPU helpers
    compute_direct_transmittance_map_gpu,
    # VoxCity load/convert utilities
    load_voxcity,
    convert_voxcity_to_domain,
    apply_voxcity_albedo,
    create_radiation_config_for_voxcity,
)

# Re-export ground-level irradiance functions
from .ground import (
    get_direct_solar_irradiance_map,
    get_diffuse_solar_irradiance_map,
    get_global_solar_irradiance_map,
    get_cumulative_global_solar_irradiance,
    get_sunlight_hours,
)

# Re-export building surface irradiance functions
from .building import (
    get_building_solar_irradiance,
    get_cumulative_building_solar_irradiance,
    get_building_sunlight_hours,
    get_building_global_solar_irradiance_using_epw,
)

# Re-export volumetric irradiance functions
from .volumetric import (
    get_volumetric_solar_irradiance_map,
    get_cumulative_volumetric_solar_irradiance,
    get_volumetric_solar_irradiance_using_epw,
    get_global_solar_irradiance_using_epw,
    save_irradiance_mesh,
    load_irradiance_mesh,
)


# Voxel class codes - expose for external use
VOXCITY_GROUND_CODE = -1
VOXCITY_TREE_CODE = -2
VOXCITY_BUILDING_CODE = -3


# Define public API
__all__ = [
    # Constants
    'VOXCITY_GROUND_CODE',
    'VOXCITY_TREE_CODE',
    'VOXCITY_BUILDING_CODE',
    
    # Utils
    'get_location_from_voxcity',
    'convert_voxel_data_to_arrays',
    'compute_valid_ground_vectorized',
    'compute_sun_direction',
    'parse_time_period',
    'filter_df_to_period',
    'load_epw_data',
    'get_solar_positions_astral',
    'extract_terrain_following_slice',
    'accumulate_terrain_following_slice',
    'ArrayWithMetadata',
    'compute_boundary_vertical_mask',
    'apply_computation_mask_to_faces',
    
    # Caching - Dataclasses
    'LandCoverAlbedo',
    'VoxCityDomainResult',
    'CachedRadiationModel',
    'CachedBuildingRadiationModel',
    'CachedGPURayTracer',
    
    # Caching - Management
    'clear_radiation_model_cache',
    'clear_building_radiation_model_cache',
    'clear_gpu_ray_tracer_cache',
    'clear_volumetric_flux_cache',
    'clear_all_caches',
    'get_radiation_model_cache',
    'get_building_radiation_model_cache',
    'set_radiation_model_cache',
    'set_building_radiation_model_cache',
    
    # Caching - Creators
    'get_or_create_radiation_model',
    'get_or_create_building_radiation_model',
    'get_or_create_gpu_ray_tracer',
    'get_or_create_volumetric_calculator',
    
    # Caching - GPU helpers
    'compute_direct_transmittance_map_gpu',
    
    # Ground-level functions
    'get_direct_solar_irradiance_map',
    'get_diffuse_solar_irradiance_map',
    'get_global_solar_irradiance_map',
    'get_cumulative_global_solar_irradiance',
    'get_sunlight_hours',
    
    # Building surface functions
    'get_building_solar_irradiance',
    'get_cumulative_building_solar_irradiance',
    'get_building_sunlight_hours',
    'get_building_global_solar_irradiance_using_epw',
    
    # Volumetric functions
    'get_volumetric_solar_irradiance_map',
    'get_cumulative_volumetric_solar_irradiance',
    'get_volumetric_solar_irradiance_using_epw',
    'get_global_solar_irradiance_using_epw',
    
    # I/O utilities
    'save_irradiance_mesh',
    'load_irradiance_mesh',
]
