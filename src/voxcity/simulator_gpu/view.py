"""Compatibility wrapper for the legacy VoxCity `view` module.

VoxCity exposes view-related functions both under:
- `voxcity.simulator.visibility.*` (newer)
- `voxcity.simulator.view.*` (legacy wrapper)

This module mirrors that pattern for `simulator_gpu` so code can do:
    import simulator_gpu as simulator
    simulator.view.get_view_index(...)
"""

from .visibility import (
    get_view_index,
    get_sky_view_factor_map,
    get_surface_view_factor,
    get_landmark_visibility_map,
    get_surface_landmark_visibility,
    mark_building_by_id,
    compute_landmark_visibility,
    rotate_vector_axis_angle,
)

__all__ = [
    "get_view_index",
    "get_sky_view_factor_map",
    "get_surface_view_factor",
    "mark_building_by_id",
    "compute_landmark_visibility",
    "get_landmark_visibility_map",
    "get_surface_landmark_visibility",
    "rotate_vector_axis_angle",
]
