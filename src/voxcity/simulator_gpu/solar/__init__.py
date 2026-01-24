"""
palm-solar: GPU-accelerated solar radiation simulation for urban environments

This package emulates PALM's Radiative Transfer Model (RTM) using Taichi for
GPU acceleration. It computes:
- Direct and diffuse solar radiation on surfaces
- Shadows from buildings and vegetation
- Sky View Factors (SVF) 
- Canopy Sink Factors (CSF) for plant canopy absorption
- Surface-to-surface radiative exchange

Coordinate System
-----------------
VoxCity uses a grid-index coordinate system where:
- x (index i): Row direction, increases from North to South
- y (index j): Column direction, increases from West to East  
- z (index k): Vertical direction, increases upward

This differs from standard ENU (East-North-Up) coordinates:
- ENU: x=East, y=North, z=Up
- VoxCity grid: x=South, y=East, z=Up

The relationship is:
    grid_x = -enu_north
    grid_y = +enu_east
    grid_z = +enu_up

Sun Direction Vector
--------------------
Sun direction vectors in this module are in VoxCity grid coordinates:
- sun_x > 0: Sun is in the South (azimuth ~180°)
- sun_x < 0: Sun is in the North (azimuth ~0°)
- sun_y > 0: Sun is in the East (azimuth ~90°)
- sun_y < 0: Sun is in the West (azimuth ~270°)
- sun_z > 0: Sun is above horizon

Surface Direction Indices (PALM convention)
--------------------------------------------
Direction indices follow PALM naming but map to VoxCity grid:
- IUP (0): +z, upward-facing surfaces
- IDOWN (1): -z, downward-facing surfaces
- INORTH (2): +y normal = East-facing in geographic terms
- ISOUTH (3): -y normal = West-facing in geographic terms
- IEAST (4): +x normal = South-facing in geographic terms
- IWEST (5): -x normal = North-facing in geographic terms

The naming (INORTH, ISOUTH, etc.) is legacy from PALM. In VoxCity's grid:
- "IEAST" surfaces receive sun when sun_x > 0 (sun in South)
- "INORTH" surfaces receive sun when sun_y > 0 (sun in East)

References:
- Resler et al., GMD 2017: https://doi.org/10.5194/gmd-10-3635-2017
- Krč et al., GMD 2021: https://doi.org/10.5194/gmd-14-3095-2021
"""

from .core import (
    Vector3, Point3, 
    SOLAR_CONSTANT, EXT_COEF, MIN_STABLE_COSZEN,
    PI, TWO_PI, DEG_TO_RAD, RAD_TO_DEG,
    normalize, dot, cross, spherical_to_cartesian
)

from .domain import Domain, Surfaces, extract_surfaces_from_domain

from .solar import (
    SolarPosition, SolarCalculator,
    calc_zenith, calc_solar_position_datetime,
    discretize_sky_directions
)

from .raytracing import (
    RayTracer,
    ray_aabb_intersect,
    ray_voxel_first_hit,
    ray_canopy_absorption,
    ray_voxel_transmissivity,
    ray_trace_to_target,
    ray_point_to_point_transmissivity,
    sample_hemisphere_direction,
    hemisphere_solid_angle,
)

from .svf import SVFCalculator

from .csf import CSFCalculator

from .radiation import RadiationModel, RadiationConfig

from .volumetric import VolumetricFluxCalculator, VolumetricFluxMode

# EPW file processing for cumulative irradiance
from .epw import (
    EPWLocation,
    EPWSolarData,
    read_epw_header,
    read_epw_solar_data,
    prepare_cumulative_simulation_input,
    get_typical_days,
    estimate_annual_irradiance,
)

# Sky discretization for cumulative irradiance
from .sky import (
    SkyPatches,
    BinnedSolarData,
    generate_tregenza_patches,
    generate_reinhart_patches,
    generate_uniform_grid_patches,
    generate_fibonacci_patches,
    generate_sky_patches,
    bin_sun_positions_to_patches,
    get_tregenza_patch_index,
    get_tregenza_patch_index_fast,
    bin_sun_positions_to_tregenza_fast,
    get_patch_info,
    calculate_cumulative_irradiance_weights,
    visualize_sky_patches,
    TREGENZA_BANDS,
    TREGENZA_BAND_BOUNDARIES,
)

# Computation mask utilities
from .mask import (
    create_computation_mask,
    draw_computation_mask,
    get_mask_from_drawing,
    visualize_computation_mask,
    get_mask_info,
)

# VoxCity integration
from .integration import (
    load_voxcity,
    convert_voxcity_to_domain,
    apply_voxcity_albedo,
    create_radiation_config_for_voxcity,
    LandCoverAlbedo,
    VoxCityDomainResult,
    VOXCITY_GROUND_CODE,
    VOXCITY_TREE_CODE,
    VOXCITY_BUILDING_CODE,
    # VoxCity API-compatible solar functions
    get_direct_solar_irradiance_map,
    get_diffuse_solar_irradiance_map,
    get_global_solar_irradiance_map,
    get_cumulative_global_solar_irradiance,
    get_sunlight_hours,
    get_building_solar_irradiance,
    get_cumulative_building_solar_irradiance,
    get_building_sunlight_hours,
    get_global_solar_irradiance_using_epw,
    get_building_global_solar_irradiance_using_epw,
    # Volumetric solar irradiance functions
    get_volumetric_solar_irradiance_map,
    get_cumulative_volumetric_solar_irradiance,
    get_volumetric_solar_irradiance_using_epw,
    clear_volumetric_flux_cache,
    save_irradiance_mesh,
    load_irradiance_mesh,
    # Temporal utilities
    get_solar_positions_astral,
    # Cache management
    clear_radiation_model_cache,
    clear_building_radiation_model_cache,
    clear_all_radiation_caches,
    clear_all_caches,
    reset_solar_taichi_cache,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    'Vector3', 'Point3',
    'SOLAR_CONSTANT', 'EXT_COEF', 'MIN_STABLE_COSZEN',
    'PI', 'TWO_PI', 'DEG_TO_RAD', 'RAD_TO_DEG',
    'normalize', 'dot', 'cross', 'spherical_to_cartesian',
    # Domain
    'Domain', 'Surfaces', 'extract_surfaces_from_domain',
    # Solar
    'SolarPosition', 'SolarCalculator',
    'calc_zenith', 'calc_solar_position_datetime',
    'discretize_sky_directions',
    # Ray tracing
    'RayTracer',
    'ray_aabb_intersect',
    'ray_voxel_first_hit',
    'ray_canopy_absorption',
    'ray_voxel_transmissivity',
    'ray_trace_to_target',
    'ray_point_to_point_transmissivity',
    'sample_hemisphere_direction',
    'hemisphere_solid_angle',
    # SVF
    'SVFCalculator',
    # CSF
    'CSFCalculator',
    # Radiation
    'RadiationModel', 'RadiationConfig',
    # Volumetric flux
    'VolumetricFluxCalculator',
    # EPW file processing
    'EPWLocation',
    'EPWSolarData',
    'read_epw_header',
    'read_epw_solar_data',
    'prepare_cumulative_simulation_input',
    'get_typical_days',
    'estimate_annual_irradiance',
    # Sky discretization (VoxCity API compatible)
    'SkyPatches',
    'BinnedSolarData',
    'generate_tregenza_patches',
    'generate_reinhart_patches',
    'generate_uniform_grid_patches',
    'generate_fibonacci_patches',
    'generate_sky_patches',
    'bin_sun_positions_to_patches',
    'bin_sun_positions_to_tregenza_fast',
    'get_tregenza_patch_index',
    'get_tregenza_patch_index_fast',
    'get_patch_info',
    'calculate_cumulative_irradiance_weights',
    'visualize_sky_patches',
    'TREGENZA_BANDS',
    'TREGENZA_BAND_BOUNDARIES',
    # Computation mask utilities
    'create_computation_mask',
    'draw_computation_mask',
    'get_mask_from_drawing',
    'visualize_computation_mask',
    'get_mask_info',
    # VoxCity integration
    'load_voxcity',
    'convert_voxcity_to_domain',
    'apply_voxcity_albedo',
    'create_radiation_config_for_voxcity',
    'LandCoverAlbedo',
    'VoxCityDomainResult',
    'VOXCITY_GROUND_CODE',
    'VOXCITY_TREE_CODE',
    'VOXCITY_BUILDING_CODE',
    # VoxCity API-compatible solar functions
    'get_direct_solar_irradiance_map',
    'get_diffuse_solar_irradiance_map',
    'get_global_solar_irradiance_map',
    'get_cumulative_global_solar_irradiance',
    'get_sunlight_hours',
    'get_building_solar_irradiance',
    'get_cumulative_building_solar_irradiance',
    'get_building_sunlight_hours',
    'get_global_solar_irradiance_using_epw',
    'get_building_global_solar_irradiance_using_epw',
    # Volumetric solar irradiance functions
    'get_volumetric_solar_irradiance_map',
    'get_cumulative_volumetric_solar_irradiance',
    'get_volumetric_solar_irradiance_using_epw',
    'clear_volumetric_flux_cache',
    'save_irradiance_mesh',
    'load_irradiance_mesh',
    # Cache management
    'clear_radiation_model_cache',
    'clear_building_radiation_model_cache',
    'clear_all_radiation_caches',
    'clear_all_caches',
    'reset_solar_taichi_cache',
    # Temporal utilities (VoxCity API compatible)
    'get_solar_positions_astral',
]
