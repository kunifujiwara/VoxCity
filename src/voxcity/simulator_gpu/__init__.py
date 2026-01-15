"""simulator_gpu: GPU-accelerated urban simulation using Taichi.

This package provides GPU-accelerated implementations for:
- Solar radiation simulation (direct, diffuse, cumulative)
- View analysis (green view index, sky view factor)
- Landmark visibility analysis

Submodules:
    solar: Solar radiation calculations
    visibility: View and visibility analysis

Example:
    from voxcity.simulator_gpu import solar, visibility
    
    # Solar radiation
    irradiance = solar.get_global_solar_irradiance_using_epw(voxcity, ...)
    
    # View analysis
    gvi = visibility.get_view_index(voxcity, mode='green')
"""

import os

# Disable Numba caching to prevent stale cache issues
os.environ.setdefault("NUMBA_CACHE_DIR", "")
os.environ.setdefault("NUMBA_DISABLE_JIT", "0")

# Taichi initialization
from .init_taichi import init_taichi, ensure_initialized, is_initialized

# Core utilities
from .core import (
    Vector3, Point3,
    PI, TWO_PI, DEG_TO_RAD, RAD_TO_DEG,
    SOLAR_CONSTANT, EXT_COEF,
)

# Domain (shared between solar and visibility)
from .domain import Domain, Surfaces, extract_surfaces_from_domain
from .domain import IUP, IDOWN, INORTH, ISOUTH, IEAST, IWEST

# Submodules
from . import solar
from . import visibility

# Convenience imports from solar
from .solar import (
    get_global_solar_irradiance_using_epw,
    get_building_global_solar_irradiance_using_epw,
    get_direct_solar_irradiance_map,
    get_diffuse_solar_irradiance_map,
    get_global_solar_irradiance_map,
)

# Convenience imports from visibility
from .visibility import (
    get_view_index,
    get_sky_view_factor_map,
    get_surface_view_factor,
    get_landmark_visibility_map,
    get_surface_landmark_visibility,
)

__version__ = "0.1.0"

__all__ = [
    # Initialization
    'init_taichi', 'ensure_initialized', 'is_initialized',
    # Core
    'Vector3', 'Point3',
    'PI', 'TWO_PI', 'DEG_TO_RAD', 'RAD_TO_DEG',
    'SOLAR_CONSTANT', 'EXT_COEF',
    # Domain
    'Domain', 'Surfaces', 'extract_surfaces_from_domain',
    'IUP', 'IDOWN', 'INORTH', 'ISOUTH', 'IEAST', 'IWEST',
    # Submodules
    'solar', 'visibility',
    # Solar (convenience)
    'get_global_solar_irradiance_using_epw',
    'get_building_global_solar_irradiance_using_epw',
    'get_direct_solar_irradiance_map',
    'get_diffuse_solar_irradiance_map',
    'get_global_solar_irradiance_map',
    # Visibility (convenience)
    'get_view_index',
    'get_sky_view_factor_map',
    'get_surface_view_factor',
    'get_landmark_visibility_map',
    'get_surface_landmark_visibility',
]
