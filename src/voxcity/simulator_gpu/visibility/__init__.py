"""
simulator_gpu.visibility: GPU-accelerated visibility analysis module.

This package emulates voxcity.simulator.visibility using Taichi GPU acceleration.

Features:
- View Index calculation (green view, sky view, custom targets)
- Sky View Factor calculation
- Landmark visibility analysis
- Surface view factor computation

API Compatibility:
    This module provides GPU-accelerated versions of the voxcity.simulator.visibility
    API functions. The main functions mirror the original API:
    
    - get_view_index() -> GPU version of voxcity.simulator.visibility.get_view_index
    - get_sky_view_factor_map() -> GPU version of voxcity.simulator.visibility.get_sky_view_factor_map
    - get_surface_view_factor() -> GPU version of voxcity.simulator.visibility.get_surface_view_factor
    - get_landmark_visibility_map() -> GPU version of voxcity.simulator.visibility.get_landmark_visibility_map
    - get_surface_landmark_visibility() -> GPU version of voxcity.simulator.visibility.get_surface_landmark_visibility
    - mark_building_by_id() -> Same as voxcity.simulator.visibility.mark_building_by_id

Usage:
    from simulator_gpu.visibility import get_view_index, get_sky_view_factor_map
    
    vi_map = get_view_index(voxcity, mode='green')
    svf_map = get_sky_view_factor_map(voxcity)
    
    # Or use the class-based API:
    from simulator_gpu.visibility import ViewCalculator
    calc = ViewCalculator(domain)
    view_map = calc.compute_view_index(mode='green')
"""

from .view import (
    ViewCalculator,
    compute_view_index_map,
    compute_sky_view_factor_map,
    SurfaceViewFactorCalculator,
)

from .landmark import (
    LandmarkVisibilityCalculator,
    compute_landmark_visibility_map,
    mark_building_by_id,
    SurfaceLandmarkVisibilityCalculator,
    compute_landmark_visibility,
)

from .geometry import (
    generate_ray_directions_grid,
    generate_ray_directions_fibonacci,
    rotate_vector_axis_angle,
)

from .voxcity import (
    # VoxCity API-compatible functions (main interface)
    get_view_index,
    get_sky_view_factor_map,
    get_surface_view_factor,
    get_landmark_visibility_map,
    get_surface_landmark_visibility,
    # Legacy GPU-suffixed functions (backward compatibility)
    get_view_index_gpu,
    get_sky_view_factor_map_gpu,
    get_landmark_visibility_map_gpu,
    # Utility functions
    create_domain_from_voxcity,
    mark_building_by_id,
    # Constants
    VOXCITY_GROUND_CODE,
    VOXCITY_TREE_CODE,
    VOXCITY_BUILDING_CODE,
    GREEN_VIEW_CODES,
)

__all__ = [
    # VoxCity API-compatible functions (recommended)
    'get_view_index',
    'get_sky_view_factor_map',
    'get_surface_view_factor',
    'get_landmark_visibility_map',
    'get_surface_landmark_visibility',
    'mark_building_by_id',
    'compute_landmark_visibility',
    # Geometry helpers (matches voxcity.simulator.common.geometry)
    'rotate_vector_axis_angle',
    # Main calculators (class-based API)
    'ViewCalculator',
    'LandmarkVisibilityCalculator',
    'SurfaceViewFactorCalculator',
    'SurfaceLandmarkVisibilityCalculator',
    # Functions
    'compute_view_index_map',
    'compute_sky_view_factor_map',
    'compute_landmark_visibility_map',
    # Geometry helpers
    'generate_ray_directions_grid',
    'generate_ray_directions_fibonacci',
    # VoxCity integration (legacy, backward compatibility)
    'create_domain_from_voxcity',
    'get_view_index_gpu',
    'get_sky_view_factor_map_gpu',
    'get_landmark_visibility_map_gpu',
    'VOXCITY_GROUND_CODE',
    'VOXCITY_TREE_CODE',
    'VOXCITY_BUILDING_CODE',
    'GREEN_VIEW_CODES',
]
