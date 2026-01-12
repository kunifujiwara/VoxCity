"""
palm-solar: GPU-accelerated solar radiation simulation for urban environments

This package emulates PALM's Radiative Transfer Model (RTM) using Taichi for
GPU acceleration. It computes:
- Direct and diffuse solar radiation on surfaces
- Shadows from buildings and vegetation
- Sky View Factors (SVF) 
- Canopy Sink Factors (CSF) for plant canopy absorption
- Surface-to-surface radiative exchange

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

from .raytracing import RayTracer

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

# VoxCity integration
from .voxcity import (
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
    get_building_solar_irradiance,
    get_cumulative_building_solar_irradiance,
    get_global_solar_irradiance_using_epw,
    get_building_global_solar_irradiance_using_epw,
    save_irradiance_mesh,
    load_irradiance_mesh,
    # Temporal utilities
    get_solar_positions_astral,
    # Cache management
    clear_radiation_model_cache,
    clear_building_radiation_model_cache,
    clear_all_radiation_caches,
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
    'get_building_solar_irradiance',
    'get_cumulative_building_solar_irradiance',
    'get_global_solar_irradiance_using_epw',
    'get_building_global_solar_irradiance_using_epw',
    'save_irradiance_mesh',
    'load_irradiance_mesh',
    # Cache management
    'clear_radiation_model_cache',
    'clear_building_radiation_model_cache',
    'clear_all_radiation_caches',
    # Temporal utilities (VoxCity API compatible)
    'get_solar_positions_astral',
]
